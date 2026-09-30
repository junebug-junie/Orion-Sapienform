from __future__ import annotations

from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1


# The rpc_health "legacy" pooled-timeout builder (Option A,
# ``build_transport_metacog_trigger_from_snapshot``) was killed outright on
# 2026-09-29 (docs/superpowers/pr-reports/2026-09-29-transport-gate-dedupe-and-
# hourly-pr.md). Every timeout it counted is an ``rpc_request()`` timeout, and
# every one of those also emits the per-call ``rpc_transport_timeout`` atom below
# (orion/core/bus/rpc_health.py: pooled ``timeout_count`` is only bumped by
# ``record_timeout()``, which ``rpc_request()`` always pairs with the atom).
# Live, 48 h to 2026-09-29: 675 of 677 timeouts it counted had a matching atom in
# the same 30 s window, so it was a second, coarser copy of the atom (one pooled
# count per window, cortex-exec/cortex-orch only, no request channel) that fired
# a second transport row for the same LLM-gateway timeouts. The atom is the one
# owner while the baseline gate is log-only; when the gate emits, see
# app/transport_timeout_owner.py. Do not re-add a pooled-timeout trigger.


def build_transport_metacog_trigger_from_grammar_atom(
    atom: dict[str, Any],
    *,
    correlation_id: str,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
) -> MetacogTriggerV1 | None:
    """The single owner of RPC timeouts while the baseline gate is log-only: a
    real per-call RPC timeout, emitted as a GrammarEventV1 atom by
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
