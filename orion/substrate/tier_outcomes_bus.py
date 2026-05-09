"""Synchronous Redis publish for substrate tier telemetry (cortex-exec chat_stance is sync)."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

import redis

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1

logger = logging.getLogger("orion.substrate.tier_outcomes_bus")

CHANNEL_SUBSTRATE_TIER_OUTCOMES = "orion:substrate:tier_outcomes"

_redis_client: redis.Redis | None = None


def _sync_redis() -> redis.Redis | None:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = str(os.getenv("ORION_BUS_URL", "") or "").strip()
    if not url:
        return None
    try:
        client = redis.Redis.from_url(url, decode_responses=False)
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:
        logger.debug("substrate_tier_outcomes_redis_connect_failed error=%s", exc)
        return None


def _reset_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        with suppress(Exception):
            _redis_client.close()
    _redis_client = None


def _correlation_uuid(ctx: dict[str, Any] | None) -> UUID:
    ctx = ctx if isinstance(ctx, dict) else {}
    for key in ("correlation_id", "trace_id"):
        raw = ctx.get(key)
        if raw is None:
            continue
        try:
            return UUID(str(raw))
        except (ValueError, TypeError):
            continue
    return uuid4()


def publish_substrate_tier_outcomes_sync(
    *,
    generated_at: str,
    cold_anchors: list[str],
    tier_outcomes: dict[str, list[str]],
    degraded_producers: list[str],
    ctx: dict[str, Any] | None = None,
) -> None:
    """Publish tier outcome telemetry when cold-path fan-out ran. Best-effort; never raises."""
    if not cold_anchors:
        return
    if str(os.getenv("ORION_BUS_ENABLED", "true")).strip().lower() in {"0", "false", "no"}:
        return

    payload = SubstrateTierOutcomesPayloadV1(
        generated_at=generated_at,
        cold_anchors=list(cold_anchors),
        tier_outcomes=dict(tier_outcomes),
        degraded_producers=sorted(set(degraded_producers)),
    )
    envelope = BaseEnvelope(
        kind="substrate.tier_outcomes.v1",
        source=ServiceRef(
            name="orion-cortex-exec",
            node=str(os.getenv("NODE_NAME") or os.getenv("SERVICE_NAME") or "").strip() or None,
        ),
        correlation_id=_correlation_uuid(ctx),
        payload=payload.model_dump(mode="json"),
    )
    client = _sync_redis()
    if client is None:
        logger.debug("substrate_tier_outcomes_skip no_redis")
        return
    try:
        data = OrionCodec().encode(envelope)
        client.publish(CHANNEL_SUBSTRATE_TIER_OUTCOMES, data)
    except Exception as exc:
        logger.debug("substrate_tier_outcomes_publish_failed error=%s", exc)
        _reset_redis()
