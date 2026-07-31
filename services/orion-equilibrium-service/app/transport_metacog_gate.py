from __future__ import annotations

from typing import Any

from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1


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


def build_transport_metacog_trigger_from_bus_synaptic(
    error: float,
    *,
    zen_state: str,
    pressure: float,
    recall_enabled: bool,
    error_threshold: float,
    previously_above: bool = False,
    node_age_sec: float | None = None,
    max_node_age_sec: float | None = None,
    edge_count: int | None = None,
) -> MetacogTriggerV1 | None:
    """Third evidence source: node:substrate.bus_synaptic's prediction_error
    (bus_synaptic_prediction_error(), orion/substrate/prediction_error.py --
    PR #1377/#1380), read directly from FalkorDB, not a bus message like
    Options A/C above.

    Passively covers organs Options A/C structurally cannot see -- neither
    self-reported RpcHealthSnapshotV1 (Option A) nor OrionBusAsync.rpc_request()
    instrumentation (Option C) sees orion-harness-governor's bespoke long-poll
    RPC, but the bus synaptic graph's passive wiretap does (live-verified,
    docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md's
    2026-07-25 revisions).

    error_threshold default 1.0 is not a new arbitrary calibration -- it is
    bus_synaptic_prediction_error's own saturation ceiling, which by
    construction means the aggregated edges' mean |zscore| already reached
    3.0, the same anomaly bar services/orion-hub/scripts/
    bus_synaptic_graph_routes.py's debug routes already use for a human
    reading a table.
    """
    # --- Staleness guard (2026-07-30) -------------------------------------
    # The poll reads whatever number is currently sitting on the node, with no
    # notion of when it was written. Confirmed live: node:substrate.bus_synaptic
    # sat frozen at a stale 1.0 for hours while this loop kept firing off it
    # every 30s. A frozen value is not a reading -- refuse rather than report a
    # stale high-water mark as a present-tense anomaly.
    if (
        max_node_age_sec is not None
        and node_age_sec is not None
        and node_age_sec > max_node_age_sec
    ):
        return None

    if error < error_threshold:
        return None

    # --- Edge-triggered, not level-triggered (2026-07-30) ------------------
    # THE fix for this trigger polluting orion_metacog. This was a pure level
    # check evaluated every poll, so a single sustained condition re-drafted an
    # LLM reflection on every tick for as long as it lasted. With a 30s poll and
    # a 30s cooldown lane (i.e. no effective rate limit at all) that is ~2,880
    # near-identical entries per day; live, transport wrote 1,812 rows in 24h,
    # ~48% of them from this one branch.
    #
    # Metacognition is "something notable HAPPENED", not "something is STILL
    # the case". A state that persists is one event, not one event per tick.
    # Firing only on the rising edge -- the transition into anomaly -- is what
    # makes this an event source instead of a sampler, and it is also what makes
    # the error_threshold far less load-bearing: a mis-set threshold now costs
    # one spurious entry per episode instead of one every 30 seconds.
    #
    # State is in-process, so a restart mid-episode re-fires once. That is an
    # accepted, bounded cost (one entry per restart) rather than a reason to
    # reach for durable checkpointing here.
    if previously_above:
        return None

    reason = f"transport:bus_synaptic:episode_start:error={error:.3f}"[:500]

    return MetacogTriggerV1(
        trigger_kind="transport",
        reason=reason,
        zen_state=zen_state,
        pressure=pressure,
        recall_enabled=recall_enabled,
        signal_refs=["node:substrate.bus_synaptic"],
        upstream={
            "evidence_source": "bus_synaptic_prediction_error",
            "fired_conditions": [f"error>={error_threshold}", "rising_edge"],
            "error": error,
            "error_threshold": error_threshold,
            "edge_count": edge_count,
            # Named so a reader of the orion_metacog row can tell this is the
            # START of an anomaly episode, not a periodic restatement of an
            # ongoing one -- the distinction the pre-2026-07-30 rows lack.
            "transition": "below_to_above",
            "node_age_sec": node_age_sec,
        },
    )
