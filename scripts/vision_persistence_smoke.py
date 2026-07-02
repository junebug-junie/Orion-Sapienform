#!/usr/bin/env python3
"""
Helpers for scripts/smoke_vision_persistence_live.sh.

Live mode exercises the real bus + Postgres mesh. Contract mode validates payload
shape, SQL coercion, RDF N-Triples, and channel/kind constants without claiming
live persistence.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Optional
from uuid import UUID

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orion.core.bus.async_service import OrionBusAsync  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.rdf import RdfWriteRequest  # noqa: E402
from orion.schemas.vision import (  # noqa: E402
    VisionEventBundleItem,
    VisionEventPayload,
    VisionScribeAckPayload,
)

CHANNEL_VISION_EVENTS = "orion:vision:events"
CHANNEL_SCRIBE_PUB = "orion:vision:scribe:pub"
CHANNEL_SQL_WRITE = "orion:vision:events:sql-write"
CHANNEL_RDF_ENQUEUE = "orion:rdf:enqueue"
CHANNEL_RDF_CONFIRM = "orion:rdf:confirm"

KIND_VISION_EVENT_BUNDLE = "vision.event.bundle"
KIND_VISION_SCRIBE_ACK = "vision.scribe.ack"
KIND_VISION_EVENT_V1 = "vision.event.v1"
KIND_RDF_WRITE_REQUEST = "rdf.write.request"

SMOKE_SERVICE = ServiceRef(name="vision-persistence-smoke", version="0.1.0")


def new_smoke_ids() -> tuple[str, UUID, str]:
    ts = int(time.time())
    event_id = f"vision-smoke-{ts}-{uuid.uuid4().hex[:8]}"
    correlation_id = uuid.uuid4()
    narrative = f"Vision persistence live smoke event {event_id}"
    return event_id, correlation_id, narrative


def build_smoke_bundle_item(
    event_id: str,
    narrative: str,
    *,
    event_type: str = "smoke_test",
) -> VisionEventBundleItem:
    return VisionEventBundleItem(
        event_id=event_id,
        event_type=event_type,
        narrative=narrative,
        entities=["orion", "vision", "smoke"],
        tags=["smoke", "vision", "persistence"],
        confidence=0.99,
        salience=0.5,
        evidence_refs=["artifact:smoke"],
    )


def build_vision_event_payload(events: list[VisionEventBundleItem]) -> VisionEventPayload:
    return VisionEventPayload(events=events)


def build_intake_envelope(
    correlation_id: UUID,
    payload: VisionEventPayload,
) -> BaseEnvelope:
    return BaseEnvelope(
        kind=KIND_VISION_EVENT_BUNDLE,
        source=SMOKE_SERVICE,
        correlation_id=correlation_id,
        payload=payload.model_dump(mode="json"),
    )


def expected_sql_row_fields(
    item: VisionEventBundleItem,
    correlation_id: UUID,
) -> dict[str, Any]:
    return {
        "event_id": item.event_id,
        "event_type": item.event_type,
        "narrative": item.narrative,
        "entities": item.entities,
        "tags": item.tags,
        "confidence": item.confidence,
        "salience": item.salience,
        "evidence_refs": item.evidence_refs,
        "correlation_id": str(correlation_id),
    }


def rdf_ntriple_markers(
    event_id: str,
    narrative: str,
    event_type: str,
    entities: list[str],
) -> list[str]:
    uri = f"http://conjourney.net/event/{event_id}"
    markers = [uri, narrative, event_type]
    markers.extend(entities)
    return markers


def build_rdf_write_request(event_id: str, nt_content: str) -> RdfWriteRequest:
    return RdfWriteRequest(
        id=event_id,
        source="vision-scribe",
        graph="orion:vision",
        triples=nt_content,
    )


def _purge_app_modules() -> None:
    for mod_name in list(sys.modules):
        if mod_name == "app" or mod_name.startswith("app."):
            sys.modules.pop(mod_name, None)


def coerce_sql_row(
    item: VisionEventBundleItem,
    correlation_id: UUID,
):
    """Contract-mode SQL row coercion using the real sql-writer model."""
    _purge_app_modules()
    sql_writer_root = os.path.join(REPO_ROOT, "services", "orion-sql-writer")
    saved_path = sys.path[:]
    try:
        if sql_writer_root not in sys.path:
            sys.path.insert(0, sql_writer_root)
        from app.models.vision_event import VisionEventSQL  # noqa: E402

        data = item.model_dump()
        data["correlation_id"] = str(correlation_id)
        return VisionEventSQL(**data)
    finally:
        sys.path[:] = saved_path
        _purge_app_modules()


def _fail(
    stage: str,
    *,
    event_id: str,
    correlation_id: str,
    channel: str = "",
    last_error: str = "",
) -> int:
    print(f"failed_stage={stage}")
    print(f"event_id={event_id}")
    print(f"correlation_id={correlation_id}")
    if channel:
        print(f"channel={channel}")
    if last_error:
        print(f"last_error={last_error}")
    return 1


def _pass(
    *,
    event_id: str,
    scribe_ack: bool,
    sql_row: bool,
    rdf_confirm: bool,
    rdf_enqueue_observed: bool,
) -> int:
    print("VISION PERSISTENCE SMOKE PASS")
    print(f"scribe_ack={str(scribe_ack).lower()}")
    print(f"sql_row={str(sql_row).lower()}")
    if rdf_confirm:
        print("rdf_confirm=true")
    elif rdf_enqueue_observed:
        print("rdf_enqueue_observed=true")
        print("rdf_confirm=not_available")
    else:
        print("rdf_confirm=false")
        print("rdf_enqueue_observed=false")
    print(f"event_id={event_id}")
    return 0


def run_contract_mode() -> int:
    event_id, correlation_id, narrative = new_smoke_ids()
    item = build_smoke_bundle_item(event_id, narrative)
    payload = build_vision_event_payload([item])
    envelope = build_intake_envelope(correlation_id, payload)

    assert envelope.kind == KIND_VISION_EVENT_BUNDLE
    assert envelope.payload["events"][0]["event_id"] == event_id

    expected = expected_sql_row_fields(item, correlation_id)

    saved_path = sys.path[:]
    scribe_root = os.path.join(REPO_ROOT, "services", "orion-vision-scribe")
    try:
        if scribe_root not in sys.path:
            sys.path.insert(0, scribe_root)
        from app.main import _build_event_triples  # noqa: E402

        nt = _build_event_triples(item)
    finally:
        sys.path[:] = saved_path
        _purge_app_modules()

    for marker in rdf_ntriple_markers(
        item.event_id, item.narrative, item.event_type, item.entities
    ):
        assert marker in nt, f"RDF N-Triples missing marker: {marker!r}"

    rdf_req = build_rdf_write_request(item.event_id, nt)
    assert rdf_req.id == item.event_id
    assert isinstance(rdf_req.triples, str)

    row = coerce_sql_row(item, correlation_id)
    for key, want in expected.items():
        got = getattr(row, key)
        assert got == want, f"sql row field {key}: got {got!r}, want {want!r}"

    print("VISION PERSISTENCE SMOKE PASS (contract mode)")
    print("mode=contract")
    print(f"event_id={event_id}")
    print(f"correlation_id={correlation_id}")
    print(f"intake_channel={CHANNEL_VISION_EVENTS}")
    print(f"intake_kind={KIND_VISION_EVENT_BUNDLE}")
    print(f"sql_write_channel={CHANNEL_SQL_WRITE}")
    print(f"sql_write_kind={KIND_VISION_EVENT_V1}")
    print(f"rdf_enqueue_channel={CHANNEL_RDF_ENQUEUE}")
    print(f"rdf_enqueue_kind={KIND_RDF_WRITE_REQUEST}")
    print(f"scribe_ack_channel={CHANNEL_SCRIBE_PUB}")
    print(f"scribe_ack_kind={KIND_VISION_SCRIBE_ACK}")
    return 0


async def _collect_envelopes(
    bus: OrionBusAsync,
    channel: str,
    queue: asyncio.Queue,
) -> None:
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg.get("data")
            decoded = bus.codec.decode(data)
            if decoded.ok and decoded.envelope is not None:
                await queue.put(decoded.envelope)


def _payload_dict(env: BaseEnvelope) -> dict[str, Any]:
    payload = env.payload
    return payload if isinstance(payload, dict) else {}


def _matches_correlation(env: BaseEnvelope, correlation_id: UUID) -> bool:
    return str(env.correlation_id) == str(correlation_id)


async def _wait_for_scribe_ack(
    queue: asyncio.Queue,
    correlation_id: UUID,
    timeout_sec: float,
) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            env = await asyncio.wait_for(queue.get(), timeout=min(remaining, 0.5))
        except asyncio.TimeoutError:
            continue
        if env.kind != KIND_VISION_SCRIBE_ACK:
            continue
        if not _matches_correlation(env, correlation_id):
            continue
        try:
            ack = VisionScribeAckPayload.model_validate(_payload_dict(env))
        except Exception as exc:
            return False, f"invalid ack payload: {exc}"
        if not ack.ok:
            return False, ack.error or ack.message or "ack.ok=false"
        return True, ""
    return False, "timeout waiting for vision.scribe.ack"


async def _wait_for_rdf_signals(
    enqueue_q: asyncio.Queue,
    confirm_q: asyncio.Queue,
    *,
    event_id: str,
    correlation_id: UUID,
    timeout_sec: float,
) -> tuple[bool, bool, str]:
    deadline = time.monotonic() + timeout_sec
    enqueue_seen = False
    confirm_seen = False
    while time.monotonic() < deadline and not (enqueue_seen and confirm_seen):
        remaining = deadline - time.monotonic()
        try:
            env = await asyncio.wait_for(enqueue_q.get(), timeout=min(remaining, 0.25))
            if _matches_correlation(env, correlation_id) and env.kind == KIND_RDF_WRITE_REQUEST:
                payload = _payload_dict(env)
                if str(payload.get("id")) == event_id:
                    enqueue_seen = True
        except asyncio.TimeoutError:
            pass
        try:
            env = await asyncio.wait_for(confirm_q.get(), timeout=min(remaining, 0.25))
            if env.kind in ("rdf.write.confirm", "rdf.write.result"):
                payload = _payload_dict(env)
                if str(payload.get("id")) == event_id and payload.get("ok") is True:
                    confirm_seen = True
        except asyncio.TimeoutError:
            pass
    err = ""
    if not enqueue_seen and not confirm_seen:
        err = "timeout waiting for rdf enqueue or confirm"
    return enqueue_seen, confirm_seen, err


def _query_vision_event_row(
    db_url: str,
    event_id: str,
    expected: dict[str, Any],
    *,
    timeout_sec: float,
) -> tuple[bool, str]:
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url, pool_pre_ping=True)
    deadline = time.monotonic() + timeout_sec
    last_err = "no row"
    sql = text(
        """
        SELECT event_id, event_type, narrative, entities, tags,
               confidence, salience, evidence_refs, correlation_id
        FROM vision_events
        WHERE event_id = :event_id
        LIMIT 1
        """
    )
    while time.monotonic() < deadline:
        try:
            with engine.connect() as conn:
                row = conn.execute(sql, {"event_id": event_id}).mappings().first()
            if row is None:
                last_err = "row not found yet"
                time.sleep(0.5)
                continue
            for key, want in expected.items():
                got = row.get(key)
                if key in ("entities", "tags", "evidence_refs"):
                    if list(got or []) != list(want):
                        last_err = f"field {key}: got {got!r}, want {want!r}"
                        break
                elif got != want:
                    last_err = f"field {key}: got {got!r}, want {want!r}"
                    break
            else:
                return True, ""
            time.sleep(0.5)
        except Exception as exc:
            last_err = str(exc)
            time.sleep(0.5)
    return False, last_err


async def run_live_mode(
    *,
    bus_url: str,
    db_url: str,
    timeout_sec: float,
) -> int:
    event_id, correlation_id, narrative = new_smoke_ids()
    item = build_smoke_bundle_item(event_id, narrative)
    payload = build_vision_event_payload([item])
    envelope = build_intake_envelope(correlation_id, payload)
    expected = expected_sql_row_fields(item, correlation_id)

    bus = OrionBusAsync(url=bus_url)
    await bus.connect()

    scribe_q: asyncio.Queue = asyncio.Queue()
    rdf_enqueue_q: asyncio.Queue = asyncio.Queue()
    rdf_confirm_q: asyncio.Queue = asyncio.Queue()
    collectors = [
        asyncio.create_task(_collect_envelopes(bus, CHANNEL_SCRIBE_PUB, scribe_q)),
        asyncio.create_task(_collect_envelopes(bus, CHANNEL_RDF_ENQUEUE, rdf_enqueue_q)),
        asyncio.create_task(_collect_envelopes(bus, CHANNEL_RDF_CONFIRM, rdf_confirm_q)),
    ]
    await asyncio.sleep(0.5)

    try:
        await bus.publish(CHANNEL_VISION_EVENTS, envelope)

        scribe_ok, scribe_err = await _wait_for_scribe_ack(
            scribe_q, correlation_id, timeout_sec
        )
        if not scribe_ok:
            return _fail(
                "scribe_ack",
                event_id=event_id,
                correlation_id=str(correlation_id),
                channel=CHANNEL_SCRIBE_PUB,
                last_error=scribe_err,
            )

        rdf_enqueue, rdf_confirm, rdf_err = await _wait_for_rdf_signals(
            rdf_enqueue_q,
            rdf_confirm_q,
            event_id=event_id,
            correlation_id=correlation_id,
            timeout_sec=timeout_sec,
        )
        if not rdf_confirm and not rdf_enqueue:
            return _fail(
                "rdf_enqueue",
                event_id=event_id,
                correlation_id=str(correlation_id),
                channel=CHANNEL_RDF_ENQUEUE,
                last_error=rdf_err,
            )

        sql_ok, sql_err = await asyncio.to_thread(
            _query_vision_event_row,
            db_url,
            event_id,
            expected,
            timeout_sec=timeout_sec,
        )
        if not sql_ok:
            return _fail(
                "sql_row",
                event_id=event_id,
                correlation_id=str(correlation_id),
                channel=CHANNEL_SQL_WRITE,
                last_error=sql_err,
            )

        return _pass(
            event_id=event_id,
            scribe_ack=True,
            sql_row=True,
            rdf_confirm=rdf_confirm,
            rdf_enqueue_observed=rdf_enqueue,
        )
    finally:
        for task in collectors:
            task.cancel()
        await asyncio.gather(*collectors, return_exceptions=True)
        await bus.close()


def _resolve_mode(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    bus_url = os.getenv("ORION_BUS_URL", "").strip()
    db_url = os.getenv("DATABASE_URL", "").strip() or os.getenv("POSTGRES_URI", "").strip()
    if bus_url and db_url:
        return "live"
    return ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vision persistence smoke")
    parser.add_argument(
        "--mode",
        choices=("live", "contract"),
        default=None,
        help="Smoke mode (default: live when ORION_BUS_URL and DATABASE_URL are set)",
    )
    args = parser.parse_args(argv)

    mode = _resolve_mode(args.mode or os.getenv("VISION_PERSISTENCE_SMOKE_MODE", "").strip())
    if not mode:
        print(
            "ERROR: select mode explicitly. Examples:\n"
            "  VISION_PERSISTENCE_SMOKE_MODE=contract bash scripts/smoke_vision_persistence_live.sh\n"
            "  ORION_BUS_URL=redis://... DATABASE_URL=postgresql+psycopg2://... "
            "VISION_PERSISTENCE_SMOKE_MODE=live bash scripts/smoke_vision_persistence_live.sh",
            file=sys.stderr,
        )
        return 2

    if mode == "contract":
        return run_contract_mode()

    bus_url = os.getenv("ORION_BUS_URL", "").strip()
    db_url = os.getenv("DATABASE_URL", "").strip() or os.getenv("POSTGRES_URI", "").strip()
    if not bus_url or not db_url:
        print(
            "ERROR: live mode requires ORION_BUS_URL and DATABASE_URL (or POSTGRES_URI).",
            file=sys.stderr,
        )
        return 2

    timeout_sec = float(os.getenv("VISION_PERSISTENCE_SMOKE_TIMEOUT_SEC", "30"))
    return asyncio.run(
        run_live_mode(bus_url=bus_url, db_url=db_url, timeout_sec=timeout_sec)
    )


if __name__ == "__main__":
    raise SystemExit(main())
