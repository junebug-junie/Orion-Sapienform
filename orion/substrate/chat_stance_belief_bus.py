"""Synchronous Redis publish for chat_stance's real per-turn belief
computation (cortex-exec chat_stance.py is sync) -- same shape as
orion/substrate/tier_outcomes_bus.py, its direct precedent.

Part of the Orion self-model rebuild arc (self_study Layer 1 broadening):
chat_stance.py computes a real UnifiedRelationalBeliefSetV1 every turn and,
until this module existed, discarded it after the turn (only a 4-key
compact summary survived 30 minutes in an in-process cache,
executor.py's _PRIOR_STANCE_CACHE). This gives it a durable home,
consumed by self_study.py's _behavioral_items().
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import suppress
from typing import Any, Sequence
from uuid import UUID, uuid4

import redis

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.chat_stance_belief import ChatStanceBeliefLogV1

logger = logging.getLogger("orion.substrate.chat_stance_belief_bus")

CHANNEL_CHAT_STANCE_BELIEF_WRITE = "orion:chat_stance:belief:write"

_redis_client: redis.Redis | None = None

# Real content, not a redaction stub (Juniper, 2026-09-05: she is the sole
# user/companion), but still length-capped -- storage/log sanity, not
# privacy.
_MAX_ANCHOR_SUMMARY_CHARS = 2000
_MAX_LINEAGE_SUMMARY_CHARS = 2000

_VALID_SHIFT_KINDS = {"NONE", "TOPIC", "STANCE", "REPAIR"}

# Review finding (2026-09-05): unlike tier_outcomes_bus.py's caller (which
# only publishes on the rare cold-anchor path), this module is called on
# every real chat turn -- a down/unreachable Redis would otherwise attempt a
# fresh blocking connect+ping every single turn with no memory of the prior
# failure. This cooldown makes a down Redis cost one slow attempt, then
# fast-fail for the rest of the window, instead of hammering it every turn.
_CONNECT_FAILURE_COOLDOWN_SEC = 30.0
_last_connect_failure_monotonic: float = 0.0

# Bounds the worst case even off the cooldown path -- a hung connect() still
# costs a bounded amount of thread time, not an indefinite one, now that the
# caller offloads this via asyncio.to_thread() rather than calling inline.
_SOCKET_CONNECT_TIMEOUT_SEC = 2.0
_SOCKET_TIMEOUT_SEC = 2.0


def _normalize_shift_kind(shift_kind: str | None) -> str | None:
    """Review finding (2026-09-05): every current producer of this value
    (orion/memory/consolidation_gate.py, recall_skip_gate.py,
    retrieval_intent.py) uppercases it before use -- this schema's Literal
    does not tolerate any other casing, and a mismatch here would otherwise
    fail Pydantic validation deep inside publish_chat_stance_belief_log_sync,
    silently dropping the whole log row via the outer best-effort except.
    Normalize defensively at this module's own boundary instead of trusting
    every caller to remember to uppercase first."""
    if not shift_kind:
        return None
    normalized = str(shift_kind).strip().upper()
    return normalized if normalized in _VALID_SHIFT_KINDS else None


def _sync_redis() -> redis.Redis | None:
    global _redis_client, _last_connect_failure_monotonic
    if _redis_client is not None:
        return _redis_client
    now = time.monotonic()
    if now - _last_connect_failure_monotonic < _CONNECT_FAILURE_COOLDOWN_SEC:
        return None
    url = str(os.getenv("ORION_BUS_URL", "") or "").strip()
    if not url:
        return None
    try:
        client = redis.Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT_SEC,
            socket_timeout=_SOCKET_TIMEOUT_SEC,
        )
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:
        logger.debug("chat_stance_belief_redis_connect_failed error=%s", exc)
        _last_connect_failure_monotonic = now
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


def build_anchor_summary(anchors: dict[str, Any] | None) -> str | None:
    """Real, plain-text description of which relational anchors were live
    this turn -- not a raw dump of every field, but not stripped to a count
    either. `anchors` is UnifiedRelationalBeliefSetV1.anchors (anchor name ->
    slice); each slice is whatever the unification layer's own producers
    return, so this stays defensive about shape rather than assuming a
    schema this module doesn't own."""
    if not isinstance(anchors, dict) or not anchors:
        return None
    parts = []
    for name, slice_ in sorted(anchors.items()):
        degraded = bool(getattr(slice_, "degraded", False)) if not isinstance(slice_, dict) else bool(slice_.get("degraded"))
        parts.append(f"{name}{'(degraded)' if degraded else ''}")
    text = "anchors this turn: " + ", ".join(parts)
    return text[:_MAX_ANCHOR_SUMMARY_CHARS]


def publish_chat_stance_belief_log_sync(
    *,
    anchors: dict[str, Any] | None,
    degraded_producers: Sequence[str] | None,
    lineage: Any,
    shift_kind: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> None:
    """Publish one real ChatStanceBeliefLogV1 row for this turn. Best-effort;
    never raises -- same discipline as publish_substrate_tier_outcomes_sync,
    its direct precedent. Fine-grained (every turn with real beliefs), per
    the self-model design doc's own resolved append-only-granularity
    question -- appending is cheap, complexity belongs at query time."""
    if str(os.getenv("ORION_BUS_ENABLED", "true")).strip().lower() in {"0", "false", "no"}:
        return
    ctx = ctx if isinstance(ctx, dict) else {}

    lineage_summary = None
    if lineage:
        try:
            lineage_summary = json.dumps(lineage, sort_keys=True, default=str)[:_MAX_LINEAGE_SUMMARY_CHARS]
        except Exception:
            lineage_summary = str(lineage)[:_MAX_LINEAGE_SUMMARY_CHARS]

    payload = ChatStanceBeliefLogV1(
        correlation_id=str(ctx.get("correlation_id") or "") or None,
        session_id=str(ctx.get("session_id") or "") or None,
        shift_kind=_normalize_shift_kind(shift_kind),
        anchor_summary=build_anchor_summary(anchors),
        degraded_producers=sorted(set(degraded_producers or [])),
        lineage_summary=lineage_summary,
    )
    envelope = BaseEnvelope(
        kind="chat_stance.belief.write.v1",
        source=ServiceRef(
            name="orion-cortex-exec",
            node=str(os.getenv("NODE_NAME") or os.getenv("SERVICE_NAME") or "").strip() or None,
        ),
        correlation_id=_correlation_uuid(ctx),
        payload=payload.model_dump(mode="json"),
    )
    client = _sync_redis()
    if client is None:
        logger.debug("chat_stance_belief_publish_skip no_redis")
        return
    try:
        data = OrionCodec().encode(envelope)
        client.publish(CHANNEL_CHAT_STANCE_BELIEF_WRITE, data)
    except Exception as exc:
        logger.debug("chat_stance_belief_publish_failed error=%s", exc)
        _reset_redis()
