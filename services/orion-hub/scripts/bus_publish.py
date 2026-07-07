"""Thin bus-publish helpers for the pending-attention surface."""

from __future__ import annotations

import logging
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

logger = logging.getLogger("orion-hub.bus_publish")

CHANNEL_ATTENTION_LOOP_OUTCOME = "orion:attention:loop_outcome"


def _source_ref() -> ServiceRef:
    """Build the producer identity from hub settings, with a safe fallback.

    Hub settings require the full operator env (many CHANNEL_* keys). When that
    env is absent (e.g. importing this module in a bare test process), fall back
    to the settings defaults so envelope construction stays pure/import-safe.
    """
    try:
        from scripts.settings import settings

        return ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            node=settings.NODE_NAME,
        )
    except Exception:  # settings env not loaded
        return ServiceRef(name="hub", version="0.3.0", node="athena")


def build_loop_outcome_envelope(outcome: AttentionLoopOutcomeV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind="attention.loop.outcome.v1",
        source=_source_ref(),
        correlation_id=str(uuid4()),
        payload=outcome.model_dump(mode="json"),
    )


def publish_attention_loop_outcome(outcome: AttentionLoopOutcomeV1) -> None:
    """Publish the label event. Best-effort; caller swallows failures.

    Low-frequency human action, so a short-lived connection is acceptable.
    Reads ORION_BUS_URL from hub settings (never hardcode a bus URL).
    """
    import anyio

    from orion.core.bus.async_service import OrionBusAsync
    from scripts.settings import settings

    async def _run() -> None:
        bus = OrionBusAsync(str(settings.ORION_BUS_URL))
        await bus.connect()
        try:
            await bus.publish(CHANNEL_ATTENTION_LOOP_OUTCOME, build_loop_outcome_envelope(outcome))
        finally:
            await bus.close()

    anyio.run(_run)
