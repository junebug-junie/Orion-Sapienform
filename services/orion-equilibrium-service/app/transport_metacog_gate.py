from __future__ import annotations

import logging
from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

logger = logging.getLogger("orion.equilibrium.transport_metacog_gate")


def build_transport_metacog_trigger_from_snapshot(
    payload: dict[str, Any],
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    latency_p95_threshold_ms: float,
) -> MetacogTriggerV1 | None:
    """Option A: a real RpcHealthSnapshotV1 window from orion:rpc_health:snapshot
    (docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md, PR #1313/
    #1315, live-verified real success/timeout/latency data).

    Two real, independently-grounded gate conditions, no invented thresholds beyond
    the one unavoidable latency default (same calibration caveat as
    telemetry_anomaly's threshold_multiplier -- ships with a starting value, needs
    real data to validate):
    - timeout_count > 0: unambiguous evidence real RPC calls failed this window. No
      threshold needed -- any real timeout is real evidence.
    - success_latency_ms_p95 above a configured ceiling: a real number already
      computed by RpcHealthAggregator, not derived/guessed here.
    An empty window (no calls at all) does not fire -- absence of traffic is not
    evidence of transport trouble, same "healthy-by-absence" rule the rpc_health
    organ adapter already applies (orion/signals/adapters/rpc_health.py).
    """
    service = str(payload.get("service") or "unknown")
    success_count = int(payload.get("success_count") or 0)
    timeout_count = int(payload.get("timeout_count") or 0)
    p95 = payload.get("success_latency_ms_p95")

    fired_conditions: list[str] = []
    if timeout_count > 0:
        fired_conditions.append(f"timeout_count={timeout_count}")
    if isinstance(p95, (int, float)) and float(p95) >= latency_p95_threshold_ms:
        fired_conditions.append(f"success_latency_ms_p95={float(p95):.1f}")

    if not fired_conditions:
        return None

    reason = f"transport:{service}:{'+'.join(fired_conditions)}"

    return MetacogTriggerV1(
        trigger_kind="transport",
        reason=reason[:500],
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[service] if service else [],
        upstream={
            "evidence_source": "rpc_health_snapshot",
            "fired_conditions": fired_conditions,
            "service": service,
            "success_count": success_count,
            "timeout_count": timeout_count,
            "success_latency_ms_p50": payload.get("success_latency_ms_p50"),
            "success_latency_ms_p95": p95,
            "success_latency_ms_max": payload.get("success_latency_ms_max"),
            "timeout_elapsed_ms_max": payload.get("timeout_elapsed_ms_max"),
            "channel_counts": payload.get("channel_counts"),
            "window_start": payload.get("window_start"),
            "window_end": payload.get("window_end"),
            "truncated": payload.get("truncated"),
            "latency_p95_threshold_ms": latency_p95_threshold_ms,
        },
    )


def build_transport_metacog_trigger_from_grammar_atom(
    atom: dict[str, Any],
    *,
    correlation_id: str,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
) -> MetacogTriggerV1 | None:
    """Option C: a real per-call RPC timeout, emitted as a GrammarEventV1 atom by
    orion/core/bus/async_service.py's _emit_rpc_timeout_grammar() -- generalizes
    chat_turn's own exec_turn_timeout/stance_timeout markers (scoped to one
    harness/thought RPC each) to every rpc_request() timeout across all 37+ real
    call sites sharing that one shared client.

    Terminal by construction: a real RPC already timed out by the time this atom
    exists (RpcHealthAggregator.record_timeout() already ran, synchronously, in the
    same call). No threshold to evaluate -- this always fires, subject to this
    trigger kind's own cooldown lane.
    """
    if not isinstance(atom, dict):
        return None
    if atom.get("semantic_role") != "rpc_transport_timeout":
        return None

    request_channel = str(atom.get("text_value") or "")
    summary = str(atom.get("summary") or "")
    reason = f"transport:rpc_timeout:{request_channel}"[:500] if request_channel else "transport:rpc_timeout"

    return MetacogTriggerV1(
        trigger_kind="transport",
        reason=reason,
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=[correlation_id] if correlation_id else [],
        upstream={
            "evidence_source": "rpc_transport_timeout_grammar",
            "fired_conditions": ["rpc_timeout"],
            "request_channel": request_channel,
            "summary": summary,
            "correlation_id": correlation_id,
        },
    )
