"""Exercises HeartbeatService._run()'s actual decode/dispatch loop with a
mocked bus (no real Redis) -- flagged as a gap by review: prior tests only
ever called _handle_grammar_message() directly, bypassing _run()'s own
decode/try-except entirely, so the "no single malformed event can crash the
subscriber loop" claim in service.py's docstring was true by inspection but
not demonstrated by any test.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from app.service import HeartbeatService


def _grammar_envelope_bytes(svc: HeartbeatService, *, payload: dict) -> bytes:
    env = BaseEnvelope(
        kind="grammar.event.v1",
        source=ServiceRef(name="orion-hub", version="0.1.0", node="test"),
        payload=payload,
    )
    return svc.codec.encode(env)


def _atom_emitted_payload(*, source_service: str = "orion-hub", atom_type: str = "observation") -> dict:
    return {
        "event_kind": "atom_emitted",
        "provenance": {"source_service": source_service},
        "atom": {
            "atom_type": atom_type,
            "confidence": 0.8,
            "salience": 0.5,
            "uncertainty": 0.2,
        },
    }


def _fake_bus_run(svc: HeartbeatService, raw_messages: list[bytes | None], monkeypatch) -> None:
    """Patches svc.bus.subscribe/iter_messages so _run() processes exactly
    the given list of raw message payloads (None = garbage/undecodable),
    then sets svc._stop so the loop exits cleanly afterward.
    """

    @asynccontextmanager
    async def fake_subscribe(*_channels, **_kwargs):
        yield object()  # _run() never inspects the pubsub object itself

    async def fake_iter_messages(_pubsub):
        for raw in raw_messages:
            data = raw if raw is not None else b"not-valid-json-or-envelope-bytes"
            yield {"channel": b"orion:grammar:event", "data": data}
        svc._stop.set()

    monkeypatch.setattr(svc.bus, "subscribe", fake_subscribe)
    monkeypatch.setattr(svc.bus, "iter_messages", fake_iter_messages)


@pytest.mark.asyncio
async def test_run_processes_valid_grammar_event_end_to_end(monkeypatch) -> None:
    svc = HeartbeatService()
    raw = _grammar_envelope_bytes(svc, payload=_atom_emitted_payload())
    _fake_bus_run(svc, [raw], monkeypatch)

    await svc._run()

    assert svc.events_seen == 1
    assert svc.events_absorbed == 1
    assert svc.substrate.tick_count == 1


@pytest.mark.asyncio
async def test_run_survives_undecodable_message(monkeypatch) -> None:
    svc = HeartbeatService()
    _fake_bus_run(svc, [None], monkeypatch)

    await svc._run()  # must not raise

    assert svc.events_seen == 0  # decode failure -> continue, before events_seen increments


@pytest.mark.asyncio
async def test_run_ignores_non_grammar_event_kind(monkeypatch) -> None:
    svc = HeartbeatService()
    env = BaseEnvelope(
        kind="system.health.v1",
        source=ServiceRef(name="orion-hub", version="0.1.0", node="test"),
        payload={"anything": True},
    )
    raw = svc.codec.encode(env)
    _fake_bus_run(svc, [raw], monkeypatch)

    await svc._run()

    assert svc.events_seen == 0  # env.kind filter runs before _handle_grammar_message


@pytest.mark.asyncio
async def test_run_survives_mixed_batch_good_bad_and_out_of_scope(monkeypatch) -> None:
    svc = HeartbeatService()
    good = _grammar_envelope_bytes(svc, payload=_atom_emitted_payload(source_service="orion-biometrics"))
    out_of_scope = _grammar_envelope_bytes(
        svc, payload=_atom_emitted_payload(source_service="orion-vision-retina")
    )
    good_2 = _grammar_envelope_bytes(svc, payload=_atom_emitted_payload(source_service="orion-cortex-orch"))
    _fake_bus_run(svc, [good, None, out_of_scope, good_2], monkeypatch)

    await svc._run()  # must not raise despite the undecodable entry in the middle

    assert svc.events_seen == 3  # the undecodable one never reaches events_seen
    assert svc.events_absorbed == 2
    assert svc.events_skipped_organ == 1
    assert svc.substrate.tick_count == 2


@pytest.mark.asyncio
async def test_run_survives_malformed_atom_fields(monkeypatch) -> None:
    svc = HeartbeatService()
    payload = _atom_emitted_payload()
    payload["atom"]["confidence"] = "not-a-number"
    raw = _grammar_envelope_bytes(svc, payload=payload)
    _fake_bus_run(svc, [raw], monkeypatch)

    await svc._run()  # must not raise

    assert svc.events_skipped_malformed == 1
    assert svc.events_absorbed == 0
