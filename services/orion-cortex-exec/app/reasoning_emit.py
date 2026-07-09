"""Per-call reasoning telemetry emit (flag-gated, default OFF, never raises).

Builds a metadata-only `ReasoningCallV1` from the run's verb/mode/node/tokens
plus the `_extract_final_text` diagnostics dict and publishes it on the bus.

Hard rules:
- Metadata ONLY. The reasoning/thinking trace text is NEVER carried;
  `reasoning_trace_present` is a bool derived from diagnostics.
- Never raises. Bad input -> minimal valid ReasoningCallV1 (reasoning_present=False).
  Publish failure -> logged warning, swallowed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.reasoning import ReasoningCallV1

logger = logging.getLogger("orion.cortex.exec.reasoning_emit")

# Envelope kind must match the declared contract kind (channels.yaml /
# SCHEMA_REGISTRY) so the event is inspectable/routable by its true kind.
REASONING_CALL_KIND = "cognition.reasoning_call.v1"


def _coerce_correlation_uuid(value: Any) -> UUID:
    """Coerce any correlation id into a UUID (BaseEnvelope requires one).

    Mirrors main._uuid_from_correlation_id: a UUID-form string passes through;
    anything else is hashed via uuid5 so telemetry is never silently dropped by
    envelope validation. Never raises.
    """
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return uuid5(NAMESPACE_URL, str(value or "unknown"))


def _coerce_token(value: Any) -> int | None:
    """Best-effort non-negative int, else None. Never raises."""
    if value is None or isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced >= 0 else None


def _coerce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
    except Exception:
        return default
    return text or default


def build_reasoning_call(
    *,
    correlation_id: str,
    verb: str,
    mode: str,
    node_id: str,
    turn_id: str | None,
    thinking_enabled: bool,
    diagnostics: dict[str, Any],
    completion_tokens: Any,
    prompt_tokens: Any,
    thinking_tokens: Any = None,
) -> ReasoningCallV1:
    """Map run diagnostics + tokens -> ReasoningCallV1. Never raises.

    reasoning_present is true when the provider surfaced reasoning content, the
    provider advertised reasoning availability, or inline think tags were seen.
    reasoning_trace_present reflects only the presence of a structured reasoning
    trace (bool) — never the trace text itself.
    """
    try:
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        reasoning_present = bool(
            diag.get("provider_has_reasoning_content")
            or diag.get("provider_reasoning_available")
            or diag.get("think_tags_detected")
        )
        reasoning_trace_present = bool(diag.get("provider_has_reasoning_trace"))
        return ReasoningCallV1(
            correlation_id=_coerce_str(correlation_id, "unknown"),
            turn_id=turn_id if isinstance(turn_id, str) and turn_id.strip() else None,
            verb=_coerce_str(verb, "unknown"),
            mode=_coerce_str(mode, "unknown"),
            node_id=_coerce_str(node_id, "unknown"),
            reasoning_present=reasoning_present,
            thinking_enabled=bool(thinking_enabled),
            reasoning_trace_present=reasoning_trace_present,
            completion_tokens=_coerce_token(completion_tokens),
            prompt_tokens=_coerce_token(prompt_tokens),
            thinking_tokens=_coerce_token(thinking_tokens),
            emitted_at=datetime.now(timezone.utc),
        )
    except Exception:
        # Degrade to a minimal valid payload rather than ever propagating.
        logger.warning("reasoning_call_build_failed corr_id=%s", correlation_id, exc_info=True)
        return ReasoningCallV1(
            correlation_id=_coerce_str(correlation_id, "unknown"),
            emitted_at=datetime.now(timezone.utc),
        )


async def publish_reasoning_call(
    bus: Any,
    *,
    source: ServiceRef,
    channel: str,
    call: ReasoningCallV1,
) -> None:
    """Publish exactly one ReasoningCallV1 envelope. Never raises."""
    try:
        env = BaseEnvelope(
            kind=REASONING_CALL_KIND,
            source=source,
            correlation_id=_coerce_correlation_uuid(call.correlation_id),
            payload=call.model_dump(mode="json"),
        )
        await bus.publish(channel, env)
    except Exception as exc:
        logger.warning(
            "reasoning_call_publish_failed corr_id=%s error=%s",
            getattr(call, "correlation_id", "unknown"),
            exc,
        )
