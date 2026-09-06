"""Publish the chat-turn attention frame as an AttentionSchemaV1 row.

Cortex is where a real chat turn kicks off, and `chat_stance.py` already
builds an `AttentionFrameV1` on every real turn (`build_attention_frame`,
gated by ORION_CURIOSITY_FRAME_ENABLED). That frame carries a selection
(`selected_action`), a possible top-down override, and suppressions -- a
complete what/why/confidence for the turn. This module projects it onto the
shared attention surface so cortex is a *producer* on that surface, not a
bystander to it (docs/superpowers/specs/2026-09-04-attention-schema-surface-
design.md, "Cortex is the kickoff").

Same module-bound-bus pattern as `current_turn_llm_signals.py`, same
fail-open contract: never raises, never blocks the turn. Returns True only
when a publish actually went out.
"""

from __future__ import annotations

import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.resilience import publish_with_reconnect
from orion.schemas.attention_frame import AttentionFrameV1
from orion.schemas.attention_schema import ATTENTION_SCHEMA_CHANNEL, ATTENTION_SCHEMA_KIND
from orion.substrate.attention_frame import to_attention_schema

from app.settings import settings

logger = logging.getLogger("cortex-exec.attention_schema_publish")

_BUS: OrionBusAsync | None = None


def bind_attention_schema_bus(bus: OrionBusAsync | None) -> None:
    global _BUS
    _BUS = bus


def reset_attention_schema_bus_for_tests() -> None:
    bind_attention_schema_bus(None)


def _source() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


async def publish_attention_schema(frame: AttentionFrameV1) -> bool:
    bus = _BUS
    if bus is None:
        logger.warning("attention_schema_publish_skipped reason=bus_unbound")
        return False
    try:
        row = to_attention_schema(frame)
        await publish_with_reconnect(
            bus,
            ATTENTION_SCHEMA_CHANNEL,
            BaseEnvelope(kind=ATTENTION_SCHEMA_KIND, source=_source(), payload=row.model_dump(mode="json")),
            log_label="attention_schema_publish",
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        logger.warning("attention_schema_publish_failed error=%s", exc)
        return False
