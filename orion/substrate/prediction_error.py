from __future__ import annotations

from typing import Any

from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.route_projection import RouteArbitrationProjectionV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.substrate.chat_loop.grammar_extract import compute_chat_pressure_hints

_THRESHOLD = 0.30


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _latest_run(runs) -> Any:
    """Return the run with the most recent ``last_updated_at`` in a mapping of runs,
    or ``None`` if the mapping is empty."""
    best = None
    for run in runs.values():
        if best is None or run.last_updated_at > best.last_updated_at:
            best = run
    return best


def execution_prediction_error(
    prev: ExecutionTrajectoryProjectionV1,
    curr: ExecutionTrajectoryProjectionV1,
) -> float:
    """0-1 surprise score: how much did execution pressure hints change this batch?

    Diffs a ``curr`` run against the ``prev`` run sharing its ``trace_id`` where that
    identity persists across polls (a run genuinely revised in place). Confirmed live
    2026-07-21 against 26 real ``execution_trajectory_reducer`` receipts: every one was
    ``operation: "create"`` with a unique ``target_id`` -- real cortex-exec runs observed
    here are single-shot (created once, never revised), so an exact ``trace_id`` match
    structurally never occurs for this workload shape. Falling back to "no matching prev
    run -> contributes nothing" (the original behavior) made this instrument permanently
    return ``0.0`` regardless of real execution volume -- not a data-scarcity gap, a wrong
    comparison key. When no trace_id match exists, diff against ``prev``'s most-recently-
    updated run instead (by ``last_updated_at``) -- the best available "what did we expect"
    reference, equivalent to comparing this tick's freshest execution snapshot against last
    tick's freshest one. A run that genuinely does get revised in place still prefers its
    own exact match, so this is additive, not a behavior change for that (currently
    unobserved, but schema-legal) case.
    """
    deltas: list[float] = []
    prev_fallback = _latest_run(prev.runs)
    for trace_id, curr_run in curr.runs.items():
        prev_run = prev.runs.get(trace_id)
        if prev_run is None:
            prev_run = prev_fallback
        if prev_run is None:
            continue
        for key in ("execution_load", "execution_friction", "failure_pressure", "reasoning_load"):
            pv = prev_run.pressure_hints.get(key, 0.0)
            cv = curr_run.pressure_hints.get(key, 0.0)
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def transport_prediction_error(
    prev: TransportBusProjectionV1,
    curr: TransportBusProjectionV1,
) -> float:
    """0-1 surprise score: how much did transport bus health change this batch?"""
    deltas: list[float] = []
    for bus_id, curr_bus in curr.buses.items():
        prev_bus = prev.buses.get(bus_id)
        if prev_bus is None:
            continue
        for field in ("stream_backlog_health", "delivery_confidence", "stream_backlog_pressure"):
            pv = getattr(prev_bus, field, 0.0)
            cv = getattr(curr_bus, field, 0.0)
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def biometrics_prediction_error(
    prev: NodeBiometricsProjectionV1,
    curr: NodeBiometricsProjectionV1,
) -> float:
    """0-1 surprise score: how much did node biometrics pressure hints change this batch?

    Unlike ``execution_prediction_error``'s fixed four-key set, biometrics
    ``pressure_hints`` keys are not enumerable in advance -- they are populated
    conditionally per node role by ``orion/substrate/biometrics_loop/
    grammar_extract.py::extract_node_state_from_events()`` (``strain`` always when a
    body_state atom carries salience, ``gpu`` only for ``local_llm_heavy`` nodes,
    ``memory_pressure``/``thermal_pressure``/``disk_pressure`` only when the
    matching pressure-signal atom is present). Confirmed live against real
    ``substrate_node_biometrics_projection`` data 2026-07-21: a GPU node (``atlas``)
    carries ``{"gpu", "strain"}`` while an orchestration node (``athena``) carries
    ``{"strain", "disk_pressure", "memory_pressure", "thermal_pressure"}`` -- no
    single fixed key list covers every node. So this diffs the union of keys
    present on either side of a given node, defaulting a missing key to 0.0 the
    same way ``execution_prediction_error`` defaults a missing fixed key.
    """
    deltas: list[float] = []
    for node_id, curr_node in curr.nodes.items():
        prev_node = prev.nodes.get(node_id)
        if prev_node is None:
            continue
        keys = set(prev_node.pressure_hints) | set(curr_node.pressure_hints)
        for key in keys:
            # pressure_hints is typed dict[str, Any] (unlike execution's pydantic-
            # enforced dict[str, float]), since node role gates which keys ever get
            # set -- coerce defensively rather than let a malformed/non-numeric
            # value raise out of a poll tick.
            try:
                pv = float(prev_node.pressure_hints.get(key, 0.0) or 0.0)
                cv = float(curr_node.pressure_hints.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def chat_prediction_error(
    prev: ChatSessionProjectionV1,
    curr: ChatSessionProjectionV1,
) -> float:
    """0-1 surprise score: how much did a chat turn's pressure hints change this batch?

    **Fallback added 2026-07-22, same defect and same fix as
    ``execution_prediction_error``/``route_prediction_error``.** The original docstring
    claimed ``ChatTurnStateV1`` is "revised in place per ``turn_id``," implying an exact
    ``turn_id`` match across successive projection snapshots would be meaningful the same
    way execution's per-``trace_id`` comparison is. That's true at the schema level
    (``reduce_chat_trace_events`` does overwrite ``updated.turns[turn_id]``), but it misses
    the actual live behavior: hub emits a turn's entire event burst in one shot
    (``build_chat_turn_grammar_events`` in ``services/orion-hub/scripts/grammar_emit.py``
    shares one ``trace_id`` across every layer -- trace_started, chat root, context,
    raw_input, repair_signal, stance_disposition, trace_ended), so a turn is created once
    and never revisited. Since ``prev``/``curr`` are loaded moments apart around one tick,
    a turn processed *this* tick is by definition new -- it cannot have a ``prev_turn``
    (an exact match structurally never occurs), and every already-existing turn is
    identical between ``prev``/``curr`` (delta 0 by construction). Live-confirmed
    2026-07-22: ``node:substrate.chat`` had never been written despite
    ``substrate_chat_session_projection`` holding 241 real turns accumulated since
    2026-06-19 -- not a data-scarcity gap, a structurally-skipped-new-content gap.

    Fix: when no exact ``turn_id`` match exists, diff against ``prev``'s most-recently-
    updated turn instead (by ``last_updated_at``, via the shared ``_latest_run`` helper) --
    "how does this turn compare to the last one we saw," not "how did this turn's own
    content evolve" (which cannot be answered for a turn that never gets revised). Exact
    matches still take priority, so a turn that genuinely were revised in place (schema-legal,
    just unobserved in practice) is unaffected.

    Unlike the other three instruments, the pressure hints themselves are not stored
    on the projection -- ``compute_chat_pressure_hints()``
    (``orion/substrate/chat_loop/grammar_extract.py:114``) is a pure function of a
    ``ChatTurnStateV1`` that only gets called transiently, at reduction time, to build
    a receipt's ``after`` payload. This function calls it directly on both the
    previous and current turn state for each shared ``turn_id`` rather than reading a
    persisted ``pressure_hints`` dict, since none exists on ``ChatTurnStateV1``.

    Known intra-instrument redundancy (CLAUDE.md metric-quality-gate step 2, re-checked
    against this instrument specifically, not skipped): ``compute_chat_pressure_hints()``
    defines ``topic_coherence = max(0.0, 1.0 - repair_pressure_level)``, an affine
    (monotonic) transform of the same ``repair_pressure_level`` that also drives
    ``repair_pressure`` directly. A change in ``repair_pressure_level`` therefore moves
    both ``repair_pressure`` and ``topic_coherence`` by the same magnitude, giving that
    one underlying signal roughly 2x the weight of ``conversation_load`` in the 3-key
    mean rather than an even 1x/1x/1x split. This is intentional, not an oversight: the
    three keys diffed here are exactly ``compute_chat_pressure_hints()``'s full, already-
    tested output contract (not a new subset invented for this instrument), and
    ``topic_coherence`` is kept rather than dropped so this function stays a literal diff
    of "the hints this reducer already reports," not a hand-curated reweighting of them --
    reintroducing the "hand-classified vocabulary" problem charter §6 item 3 was written
    to avoid, just one layer down. If this weighting becomes a real problem in practice
    (verified against live data, not asserted), the fix is upstream in
    ``compute_chat_pressure_hints()`` itself, not a silent key-drop here.
    """
    deltas: list[float] = []
    prev_fallback = _latest_run(prev.turns)
    for turn_id, curr_turn in curr.turns.items():
        prev_turn = prev.turns.get(turn_id)
        if prev_turn is None:
            prev_turn = prev_fallback
        if prev_turn is None:
            continue
        prev_hints = compute_chat_pressure_hints(prev_turn)
        curr_hints = compute_chat_pressure_hints(curr_turn)
        for key in ("conversation_load", "repair_pressure", "topic_coherence"):
            pv = prev_hints.get(key, 0.0)
            cv = curr_hints.get(key, 0.0)
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def route_prediction_error(
    prev: RouteArbitrationProjectionV1,
    curr: RouteArbitrationProjectionV1,
) -> float:
    """0-1 surprise score: how much did a route arbitration run's *decision* change
    this batch?

    **Deliberately not a continuous-magnitude diff like the other three instruments
    in this module.** ``RouteArbitrationRunStateV1``'s fields
    (``orion/schemas/route_projection.py``) are categorical/discrete -- ``lane``,
    ``lane_reason``, ``output_mode`` are strings, ``mind_requested`` is a bool -- there
    is no numeric magnitude to subtract. Applying ``execution_prediction_error``'s
    ``abs(cv - pv)`` shape here would be meaningless (strings don't subtract) or
    would require an arbitrary numeric encoding of categories, which is exactly the
    kind of hand-authored taxonomy-on-top-of-taxonomy this charter's item 3 was
    written to avoid (see charter §6 item 3: "not a port of ``tensions.py``'s
    hand-classified kind vocabulary onto field channels").

    Instead this computes a categorical mismatch rate: for each field compared, score
    ``1.0`` if the value differs between ``prev``/``curr``, else ``0.0``, then average
    across the compared fields and across matched runs (by ``trace_id``, mirroring
    ``reduce_route_trace_events``'s create/update-by-``trace_id`` semantics --
    ``orion/substrate/route_loop/reducer.py`` line ~146-147). The fields compared
    (``lane``, ``lane_reason``, ``output_mode``, ``mind_requested``) were chosen
    because together they represent the arbitration *decision* itself -- which lane a
    turn was routed to, why, what output mode it produced, and whether mind
    escalation was requested -- as opposed to bookkeeping fields
    (``correlation_id``, ``session_id``, ``turn_id``, ``evidence_event_ids``,
    ``last_updated_at``) that change on every revision by construction and would
    saturate the score at 1.0 for every batch, making it useless as a surprise
    signal. ``mind_skip_reason`` was left out: it is a free-text explanation that is
    non-null only when ``mind_requested`` is already false, so including it would
    double-count the same underlying decision already captured by ``mind_requested``.

    A mismatch rate averaged over four boolean-valued comparisons is already bounded
    to ``[0, 1]`` by construction (each per-field score is 0.0 or 1.0, so the mean of
    N such scores is bounded [0, 1] for any N > 0) -- unlike the other three
    instruments' unbounded absolute deltas, which need ``min(1.0, mean / _THRESHOLD)``
    to saturate into a [0, 1] surprise score.

    **Do not apply the module's ``_THRESHOLD = 0.30`` scaling here.** That scaling
    exists to convert an unbounded continuous magnitude into a saturating [0, 1]
    score; a categorical mismatch rate has no such unboundedness to correct for, and
    dividing an already-[0, 1] value by 0.30 would push most non-zero mismatches
    straight to the 1.0 ceiling, destroying the very distinction (one field flipped
    vs. all four flipped) that makes this signal informative. If a future patch is
    tempted to "fix" this into consistency with the other three functions by adding
    the ``_THRESHOLD`` scale here too, that is a regression, not a cleanup -- leave
    this deviation as-is unless the field types themselves change from categorical to
    continuous.

    **Trace_id-match fallback (added 2026-07-22):** same defect and same fix as
    ``execution_prediction_error`` -- real route-arbitration runs observed live are
    single-shot creates (confirmed for the one live sample checked; sparse total volume,
    9-10 receipts ever, limits sample size, but the reducer code path
    (``orion/substrate/route_loop/reducer.py``) is structurally identical to execution's
    create-once-per-turn shape). Without a fallback, a trace_id match would essentially
    never occur and this instrument would read ``0.0`` forever regardless of real
    arbitration volume. When no trace_id match exists, compare against ``prev``'s
    most-recently-updated run instead (by ``last_updated_at``).
    """
    fields = ("lane", "lane_reason", "output_mode", "mind_requested")
    run_scores: list[float] = []
    prev_fallback = _latest_run(prev.runs)
    for trace_id, curr_run in curr.runs.items():
        prev_run = prev.runs.get(trace_id)
        if prev_run is None:
            prev_run = prev_fallback
        if prev_run is None:
            continue
        field_scores = [
            1.0 if getattr(prev_run, field) != getattr(curr_run, field) else 0.0
            for field in fields
        ]
        run_scores.append(_mean(field_scores))
    return _mean(run_scores) if run_scores else 0.0
