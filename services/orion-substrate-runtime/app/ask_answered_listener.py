"""Consume ``orion:ask:answered``: a name Juniper gave becomes a substrate entity.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 3.
The Hub publishes ``OrionAskAnsweredV1`` when Juniper answers an ask. For
``source_kind='vision_individual'`` answers, this writes one ``EntityNodeV1``
(``orion/substrate/adapters/vision_individual.py``) into the same substrate
graph store the worker already uses, through the same
``SubstrateGraphMaterializer`` path topic-foundry ingestion uses.

A ``vision_same_individual`` ("same person?") answer that carries ``names``
(2026-10-15: Juniper typed who one or both faces are) writes the same entity
for each named individual, exactly as a "who is this?" answer for it would.

A ``vision_person_confirm`` ("Is this <name>?", 2026-10-20) answered ``yes``
writes the same entity for ``<individual_id>`` named ``<name>`` (source_ref
"<individual_id>|<name>"); ``no`` names nobody.

Not handled here, on purpose:
- ``vision_individual.label`` in Postgres is set by ``orion-sql-writer`` from
  the ``orion_ask`` row on its own clock (the row is the truth, not this event).
- Dismissed asks and other ``source_kind`` values are ignored.

Fail-open: a bad message, missing store or DB error is logged and dropped; it
never takes the listener down.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Callable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.resilience import run_subscription
from orion.schemas.ask import OrionAskAnsweredV1

from .settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.runtime.ask_answered")

_ASK_ANSWERED_KIND = "orion.ask.answered.v1"
_VISION_INDIVIDUAL = "vision_individual"
_VISION_SAME_INDIVIDUAL = "vision_same_individual"
_VISION_PERSON_CONFIRM = "vision_person_confirm"
_MAX_ALIASES = 20

StoreGetter = Callable[[], Any]
EngineGetter = Callable[[], Any]


def lookup_individual(engine: Any, individual_id: str) -> tuple[Optional[str], Optional[str]]:
    """Return (kind, stream_id) from ``vision_individual``, or (None, None).

    Best-effort: the table may not exist yet (migration not applied) or the
    row may have been pruned; the entity is still written, typed "unknown".
    """
    if engine is None:
        return None, None
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT kind, stream_id FROM vision_individual WHERE individual_id = :id"),
                {"id": individual_id},
            ).first()
        if row is None:
            return None, None
        return (row[0], row[1])
    except Exception as exc:
        logger.warning("ask_answered individual_lookup_failed individual_id=%s error=%s", individual_id, exc)
        return None, None


def apply_answer_to_substrate(
    event: OrionAskAnsweredV1,
    *,
    get_store: StoreGetter,
    get_engine: EngineGetter,
) -> Optional[dict[str, Any]]:
    """Write/update the entity for one answered vision_individual ask, or one
    entity per name typed on an answered vision_same_individual ask.

    Returns a small result dict for logs/tests, or None when the event is not
    ours or could not be written. Synchronous: call via ``asyncio.to_thread``.
    """
    if event.status != "answered":
        return None
    if event.source_kind == _VISION_SAME_INDIVIDUAL:
        pair = [i for i in (event.source_ref or "").split("|") if i]
        written = []
        for individual_id, name in sorted((event.names or {}).items()):
            if individual_id not in pair or not (name or "").strip():
                logger.warning("ask_answered pair_name_skipped ask_id=%s individual_id=%s", event.ask_id, individual_id)
                continue
            result = _write_individual_entity(event, individual_id, name, get_store=get_store, get_engine=get_engine)
            if result is not None:
                written.append({"individual_id": individual_id, **result})
        return {"outcome": "pair_names", "entities": written} if written else None
    if event.source_kind == _VISION_PERSON_CONFIRM:
        individual_id, _, name = (event.source_ref or "").partition("|")
        if (event.answer or "").strip().lower() != "yes" or not individual_id or not name.strip():
            return None
        return _write_individual_entity(event, individual_id, name, get_store=get_store, get_engine=get_engine)
    if event.source_kind != _VISION_INDIVIDUAL or not (event.answer or "").strip():
        return None
    return _write_individual_entity(event, event.source_ref, event.answer or "", get_store=get_store, get_engine=get_engine)


def _write_individual_entity(
    event: OrionAskAnsweredV1,
    individual_id: str,
    label: str,
    *,
    get_store: StoreGetter,
    get_engine: EngineGetter,
) -> Optional[dict[str, Any]]:
    """One user-asserted entity for ``individual_id`` named ``label`` by ``event``."""
    from orion.core.schemas.cognitive_substrate import EntityNodeV1
    from orion.substrate.adapters.vision_individual import map_vision_individual_label_to_substrate
    from orion.substrate.falkor_codec import EXTERNALLY_OWNED_METADATA_KEYS
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.reconcile import SubstrateIdentityResolver

    kind, stream_id = lookup_individual(get_engine(), individual_id)
    record = map_vision_individual_label_to_substrate(
        individual_id=individual_id,
        label=label,
        kind=kind,
        ask_id=event.ask_id,
        answered_at=event.answered_at,
        stream_id=stream_id,
    )
    if record is None:
        return None
    store = get_store()
    if store is None:
        logger.warning("ask_answered skipped_no_store ask_id=%s individual_id=%s", event.ask_id, individual_id)
        return None

    node = record.nodes[0]
    existing = store.get_node_by_id(node.node_id)
    if isinstance(existing, EntityNodeV1):
        # Explicit update, not the materializer: merge_node() keeps the
        # existing label and lets existing metadata win, so a rename (even a
        # capitalization-only one) and the newest label_ask_id would both be
        # silently dropped. The old name is kept as an alias so recall by it
        # still works.
        renamed = existing.label != node.label
        aliases: list[str] = []
        for alias in [*existing.aliases, existing.label]:
            if alias and alias != node.label and alias not in aliases:
                aliases.append(alias)
        aliases = aliases[-_MAX_ALIASES:]
        metadata = {**existing.metadata}
        for key, value in node.metadata.items():
            if value is not None:
                metadata[key] = value
        updated = existing.model_copy(
            update={
                "label": node.label,
                "aliases": aliases,
                "entity_type": node.entity_type if node.entity_type != "unknown" else existing.entity_type,
                "provenance": node.provenance.model_copy(
                    update={"evidence_refs": sorted({*existing.provenance.evidence_refs, *node.provenance.evidence_refs})}
                ),
                "temporal": existing.temporal.model_copy(
                    update={"observed_at": max(existing.temporal.observed_at, node.temporal.observed_at)}
                ),
                "signals": existing.signals.model_copy(
                    update={"confidence": max(existing.signals.confidence, node.signals.confidence)}
                ),
                "metadata": metadata,
            }
        )
        identity_key = SubstrateIdentityResolver().canonical_node_key(updated)
        store.upsert_node(identity_key=identity_key, node=updated, skip_metadata_keys=EXTERNALLY_OWNED_METADATA_KEYS)
        outcome = "relabelled" if renamed else "updated"
    else:
        materializer = SubstrateGraphMaterializer(
            store=store,
            identity_resolver=SubstrateIdentityResolver(store=store),
        )
        result = materializer.apply_record(record)
        outcome = "created" if result.nodes_created else "merged"

    logger.info(
        "ask_answered entity_%s ask_id=%s individual_id=%s node_id=%s entity_type=%s",
        outcome,
        event.ask_id,
        individual_id,
        node.node_id,
        node.entity_type,
    )
    return {"outcome": outcome, "node_id": node.node_id, "entity_type": node.entity_type, "label": node.label}


async def _handle_bus_message(
    bus: OrionBusAsync,
    raw_msg: dict[str, Any],
    *,
    get_store: StoreGetter,
    get_engine: EngineGetter,
) -> Optional[dict[str, Any]]:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("ask_answered decode failed: %s", decoded.error)
        return None
    env = decoded.envelope
    if (env.kind or "") != _ASK_ANSWERED_KIND:
        logger.warning("ask_answered unsupported kind=%s", env.kind)
        return None
    try:
        event = OrionAskAnsweredV1.model_validate(env.payload or {})
    except ValueError as exc:
        logger.error("ask_answered invalid payload err=%s", exc)
        return None
    return await asyncio.to_thread(apply_answer_to_substrate, event, get_store=get_store, get_engine=get_engine)


async def run_ask_answered_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event | None = None,
    *,
    get_store: StoreGetter,
    get_engine: EngineGetter,
    settings: Settings | None = None,
) -> None:
    s = settings or get_settings()
    if not s.orion_bus_enabled:
        logger.info("Bus disabled; ask_answered listener not started")
        return
    if not s.enable_ask_answered_listener:
        logger.info("ask_answered listener disabled by config")
        return
    await run_subscription(
        bus,
        s.channel_ask_answered,
        lambda msg: _handle_bus_message(bus, msg, get_store=get_store, get_engine=get_engine),
        stop_event=stop_event,
        name="ask_answered listener",
    )


async def start_ask_answered_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event,
    *,
    get_store: StoreGetter,
    get_engine: EngineGetter,
    settings: Settings | None = None,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        run_ask_answered_listener(bus, stop_event, get_store=get_store, get_engine=get_engine, settings=settings),
        name="substrate-ask-answered-listener",
    )


async def stop_ask_answered_listener(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
