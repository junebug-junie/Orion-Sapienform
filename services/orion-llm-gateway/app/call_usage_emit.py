"""The gateway's half of the per-call token ledger (contract: orion/schemas/llm_call_usage.py).

One ``llm.call.usage.v1`` per call that reached a model -- bus RPC (``main.handle_chat``) and
the HTTP passthroughs (``passthrough_proxy._CallReport``) -- recorded at the same point as the
grammar windows' per-call ``record``. A call the pool never granted and that reported no tokens
(a refusal, a queued client that left) is not a model call and is not recorded here; the
grammar windows still count it.

``offer`` puts the event in a bounded in-memory sink; ``main.main`` drains it onto
``orion:llm:call:usage``. Nothing here can raise into, or wait inside, the serving path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from orion.core.bus.bus_schemas import ServiceRef
from orion.llm.call_usage import UsageSink, new_call_id, token_counts
from orion.schemas.llm_call_usage import LlmCallUsageV1

from .settings import settings

logger = logging.getLogger("orion-llm-gateway.call_usage")

_sink = UsageSink()


def get_sink() -> UsageSink:
    return _sink


def reset_sink_for_tests() -> UsageSink:
    global _sink
    _sink = UsageSink()
    return _sink


def source_ref() -> ServiceRef:
    return ServiceRef(name=settings.service_name, node=getattr(settings, "node_name", None),
                      version=settings.service_version)


def _clip(value: Any, limit: int) -> Optional[str]:
    text = str(value or "").strip()
    return text[:limit] or None


def record_call(
    *,
    path: str,
    outcome: str,
    clock: Any,
    payloads: tuple = (),
    correlation_id: Optional[str] = None,
    route: Optional[str] = None,
    model: Optional[str] = None,
    served_by: Optional[str] = None,
    holder: Optional[str] = None,
    verb: Optional[str] = None,
    stream: bool = False,
) -> Optional[LlmCallUsageV1]:
    """Build and queue one event. Returns it (tests), or None when skipped or on any error."""
    if not settings.llm_gateway_call_usage_enabled:
        return None
    try:
        counts = token_counts(*payloads)
        reached_model = getattr(clock, "model_ms", None) is not None
        has_tokens = counts.prompt_tokens is not None or counts.completion_tokens is not None
        if not reached_model and not has_tokens:
            return None
        reply_model = None
        for payload in payloads:
            if isinstance(payload, dict) and payload.get("model"):
                reply_model = payload.get("model")
                break
        event = LlmCallUsageV1(
            call_id=new_call_id(),
            recorded_at=datetime.now(timezone.utc),
            recorder="gateway",
            path=path,
            correlation_id=_clip(correlation_id, 256),
            route=_clip(route, 128),
            model=_clip(reply_model or model, 256),
            served_by=_clip(served_by, 256),
            pool_role=_clip(getattr(clock, "role", None), 64),
            holder=_clip(holder, 256),
            verb=_clip(verb, 128),
            node=_clip(getattr(settings, "node_name", None), 128),
            outcome=_clip(outcome, 64) or "unknown",
            stream=bool(stream),
            prompt_tokens=counts.prompt_tokens,
            cached_prompt_tokens=counts.cached_prompt_tokens,
            completion_tokens=counts.completion_tokens,
            token_source=counts.token_source,
            duration_ms=getattr(clock, "model_ms", None),
            wait_ms=getattr(clock, "wait_ms", None),
            decode_tps=counts.decode_tps,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break a reply
        logger.warning("llm_call_usage_build_failed path=%s corr=%s", path, correlation_id, exc_info=True)
        return None
    _sink.offer(event)
    return event


def record_bus_call(result: Any, *, outcome: str, clock: Any, correlation_id: Optional[str],
                    route: Optional[str], holder: Optional[str], verb: Optional[str],
                    requested_model: Optional[str]) -> Optional[LlmCallUsageV1]:
    """Bus RPC: the reply dict's ``raw`` is llama.cpp's own response body."""
    result = result if isinstance(result, dict) else {}
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    return record_call(
        path="bus_rpc", outcome=outcome, clock=clock, payloads=(raw,) if raw else (),
        correlation_id=correlation_id, route=result.get("route") or route,
        model=result.get("model") or requested_model, served_by=result.get("served_by"),
        holder=holder, verb=verb, stream=False,
    )


__all__ = ["get_sink", "record_bus_call", "record_call", "reset_sink_for_tests", "source_ref"]
