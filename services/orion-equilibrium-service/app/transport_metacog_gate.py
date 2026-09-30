from __future__ import annotations

from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1


def build_transport_metacog_trigger_from_snapshot(
    payload: dict[str, Any],
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
) -> MetacogTriggerV1 | None:
    """Option A (legacy): a real RpcHealthSnapshotV1 window from
    orion:rpc_health:snapshot (docs/superpowers/specs/2026-07-24-transport-metacog-
    trigger-design.md, PR #1313/#1315).

    **Timeouts only.** Fires when timeout_count > 0: unambiguous evidence real RPC
    calls failed this window, no threshold needed.

    The pooled-p95 latency branch (``success_latency_ms_p95 >= 5000``) was killed
    outright on 2026-09-24 (spec 2026-09-24-metacog-capture-and-transport-ewma-
    baseline-design.md). Verified live: cortex-orch windows averaged 2.1 calls with
    0 timeouts, and the slow call was metacog's own background LLM draft -- so the
    branch fired transport, which dispatched a draft, which tripped the branch
    again (~2,000 junk rows/day). Per-hop latency now lives in
    ``app/transport_baseline_gate.py``. Do not re-add a pooled latency ceiling.

    This whole function is retired too once EQUILIBRIUM_TRANSPORT_BASELINE_EMIT is
    on: the baseline gate's timeout/zero_success episodes replace it, and the
    service stops calling this so the same timeout never fires twice.

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


# RETIRED 2026-09-30: the third transport source, a FalkorDB poll of
# node:substrate.bus_synaptic's prediction_error (fraction of bus-synaptic
# edges at |z| >= 3), is gone -- builder, poll loop, settings and env keys.
# Metric-quality-gate findings (docs/superpowers/pr-reports/
# 2026-09-30-retire-bus-synaptic-transport-trigger-pr.md): the edge z-scores are
# computed at orion-bus-mirror's *dequeue* time by a single consumer that Redis
# disconnects for output-buffer overflow ~every 20 min (424 disconnects), so the
# fraction carries the observer's own lag; its stated purpose (catching one
# bespoke organ) is below its own noise band by design; and it fired 50-155
# episodes/day evenly across the clock without tracking real RPC-timeout storms,
# each drafting a content-free reflection. Do not re-add a mesh-wide fraction
# threshold here. A per-organ or publish-timestamp-based signal would need its
# own metric gate first.
