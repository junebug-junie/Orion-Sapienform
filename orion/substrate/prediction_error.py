from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from orion.bus.ewma import compute_ewma_update
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.route_projection import RouteArbitrationProjectionV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.structural_mass.git_delta import GitChurnDelta
from orion.substrate.chat_loop.grammar_extract import compute_chat_pressure_hints

_THRESHOLD = 0.30

# Z-score saturation for bus_synaptic_prediction_error, distinct from _THRESHOLD
# above (calibrated for 0-1-scale pressure-hint deltas, not z-score units).
# Reuses the zscore_threshold=3.0 convention already live in
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies() route.
_BUS_SYNAPTIC_ZSCORE_SATURATION = 3.0

# 2026-07-26: floor-bias fix. mean(|z|) over a batch of ~N(0,1)-distributed
# z-scores does NOT rest at 0 when traffic is genuinely calm -- E[|Z|] for a
# standard normal is sqrt(2/pi), not 0, because averaging absolute values of a
# zero-mean distribution can't average to zero. Confirmed live (2026-07-26,
# 2h real window, 3527 ticks, values recovered from the *unfixed* aggregate):
# median mean(|z|) was 0.7995, essentially exactly this constant, and the
# aggregate NEVER read below ~0.51 raw (0.17 post-saturation) across the whole
# window. Subtracted out below so this domain can genuinely rest near 0 like
# its four siblings (execution/biometrics/chat/route), instead of permanently
# reporting "moderately surprised" regardless of real mesh conditions.
_BUS_SYNAPTIC_CALM_FLOOR = math.sqrt(2.0 / math.pi)

# 2026-07-28: execution_prediction_error's own EWMA baseline calibration. alpha=0.2
# reuses services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update's own
# alpha (via the shared orion/bus/ewma.py copy) rather than inventing a new number --
# both update per real observation (a real edge/a real execution-tick batch), not on
# a fixed wall-clock cadence like orion-field-digester's alpha=0.02 (which runs once
# per RECEIPT_POLL_INTERVAL_SEC tick regardless of traffic). Saturation reuses the
# same z>=3.0 "anomalous" convention as _BUS_SYNAPTIC_ZSCORE_SATURATION above and
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies() route.
_EXECUTION_PREDICTION_ERROR_EWMA_ALPHA = 0.2
_EXECUTION_PREDICTION_ERROR_ZSCORE_SATURATION = 3.0

# 2026-07-28: domain-specific variance floor for compute_ewma_update, replacing
# that function's own default (_MIN_VARIANCE=1e-6, calibrated for orion-bus-
# mirror's real-time-gap-in-seconds domain). Live-confirmed via
# orion/bus/ewma.py's own unit test data reproduced here: replaying this
# domain's real historical raw-delta sequence (120 real `substrate.execution_
# trajectory` receipts, 2026-07-28) through the shared default floor left the
# EWMA's real per-tick variance settling around 4.2e-11 once warmed up -- five
# orders of magnitude below 1e-6, so the shared floor dominated every real
# z-score computed (max ever: 0.045, saturating error at only 0.015 across the
# full history) instead of this domain's real spread. Set one order of
# magnitude below that smallest real warmed-up variance -- low enough that
# genuine signal drives the z-score rather than the floor, still nonzero to
# guard the first non-cold-start tick's div-by-zero (prev_variance exactly
# 0.0). Replaying the same 120-receipt history through this floor instead
# gives a healthy, non-degenerate spread (max error 0.97, mean 0.12, no
# saturation) -- not re-tuned per this domain's future drift, so revisit if
# execution's real delta scale ever shifts by orders of magnitude.
_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE = 1e-10


# 2026-07-30: codebase_prediction_error's EWMA baseline calibration (Phase 1 skeleton,
# docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md -- no bus wiring, no
# NODE_CHANNELS registration yet, see that spec's "Recommended next patch"). alpha=0.2
# reuses execution_prediction_error's own alpha rather than inventing a new number for
# a domain that, like execution, updates per real observation (a real git-churn tick,
# not a fixed wall-clock cadence). min_variance calibrated against this repo's own real,
# if shallow-clone-limited, commit history via
# scripts/analysis/measure_codebase_prediction_error.py -- see that constant's own
# comment for the live numbers.
_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA = 0.2
_CODEBASE_PREDICTION_ERROR_ZSCORE_SATURATION = 3.0

# 2026-07-30: live-confirmed via scripts/analysis/measure_codebase_prediction_error.py
# against this repo's own real (shallow-clone-limited to 49 ticks/50 commits locally --
# a real limitation, not a full backfill; see the design spec's "Backfill" section for
# why a deeper replay needs scripts/analysis/backfill_structural_mass.py, not this
# script) recent commit history, one tick per real commit: raw per-tick line-churn
# magnitude (lines_changed) ranged 0-15,285 (mean 2,611, stdev 4,314; only 2/48 ticks
# were true zero-churn no-ops), and the EWMA's own accumulated variance warmed up into
# the 1.5M-60M range within a handful of ticks -- squared-magnitude units on a
# thousands-scale raw signal are naturally large, several orders of magnitude *above*
# the shared orion/bus/ewma.py default (1e-6), the mirror-image relationship to
# execution_prediction_error's real variance (which sat far *below* that default).
# Because the EWMA's own accumulated variance dominates this floor immediately once
# any real deviation has been folded in, this constant's only actual effect across the
# whole replayed window was guarding the single first real z-score computation (the
# second observed tick, where prev_variance is still exactly 0.0 from the cold-start
# branch) against amplifying a small real deviation into a false-large z purely from
# dividing by a near-zero floor -- every later tick's z-score was already governed by
# the domain's own real accumulated variance regardless of this constant's exact value.
# Set low relative to this domain's observed real variance scale (not tuned to match
# it -- there is nothing here for it to match, it is intentionally never the binding
# term past the first tick) while staying safely nonzero; revisit only if a much
# larger/deeper replay (post-backfill) shows a materially different cold-start regime.
_CODEBASE_PREDICTION_ERROR_MIN_VARIANCE = 1.0


@dataclass(frozen=True)
class CodebaseMassBaseline:
    """EWMA baseline state for ``codebase_prediction_error``, threaded explicitly by
    the caller rather than read off a persisted projection -- unlike the other five
    domains in this module, no ``structural_mass`` projection schema or bus channel
    exists yet (Phase 1 skeleton only; see this module's codebase_prediction_error
    docstring)."""

    ewma: float = 0.0
    variance: float = 0.0
    n: int = 0


@dataclass(frozen=True)
class CodebasePredictionErrorResult:
    score: float
    baseline: CodebaseMassBaseline


def codebase_prediction_error(
    delta: GitChurnDelta,
    baseline: CodebaseMassBaseline,
) -> CodebasePredictionErrorResult:
    """0-1 surprise score: how much did this tick's git churn deviate from this
    domain's own recent normal.

    **Phase 1 skeleton** (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
    design.md) -- not wired into any tick, no ``NODE_CHANNELS`` registration, no bus
    channel. Exists so the replay script
    (``scripts/analysis/measure_codebase_prediction_error.py``) has a real function to
    call before any of that is built, per this program's "measure before minting"
    discipline (SSP §7) -- same order every other domain in this module has followed.

    Unlike the other five domains, there is no ``prev``/``curr`` projection pair to
    diff here -- ``GitChurnDelta`` (``orion/structural_mass/git_delta.py``) is
    already a diff (of two git SHAs), so this function's job is scoring *that*
    delta's magnitude against this domain's own EWMA baseline, the same shape
    ``execution_prediction_error`` uses once it already has its raw per-tick delta
    in hand. Raw magnitude is ``delta.lines_changed`` (net lines added + removed) --
    the single continuous scale that best reflects "how much of the codebase moved,"
    with commit/file counts available on ``delta`` itself for callers that want them
    as separate descriptive fields (e.g. a future concept-classification consumer)
    without folding them into this one scalar and reintroducing an arbitrary
    cross-scale weighting (commits vs. files vs. lines) this function deliberately
    avoids, the same reasoning ``route_prediction_error``'s docstring gives for not
    hand-encoding categorical fields into one number.

    No baseline state is read from or written to any persisted store -- the caller
    owns threading ``baseline`` from one call to the next (mirrors
    ``compute_ewma_update``'s own explicit-state-in, explicit-state-out shape,
    rather than the other domains' "mutate the projection in place" pattern, since
    there is no projection object to mutate here yet).

    Returns ``score=0.0`` on the very first observation (``baseline.n == 0``) --
    ``compute_ewma_update`` returns no z-score until a real baseline exists, and
    reporting a nonzero value here would misrepresent "no baseline yet" as
    "measured, not anomalous" (this repo's "no empty-shell cognition" rule). A
    below-baseline tick (this commit churned *less* than usual) is clamped to 0.0,
    not reported as negative surprise -- "surprising" here means "more churn than
    usual," matching ``execution_prediction_error``'s same clamp for the same
    reason.
    """
    raw_magnitude = float(delta.lines_changed)
    update = compute_ewma_update(
        prev_ewma=baseline.ewma,
        prev_variance=baseline.variance,
        prev_count=baseline.n,
        value=raw_magnitude,
        alpha=_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=_CODEBASE_PREDICTION_ERROR_MIN_VARIANCE,
    )
    new_baseline = CodebaseMassBaseline(
        ewma=update.ewma, variance=update.variance, n=baseline.n + 1
    )
    if update.zscore is None:
        return CodebasePredictionErrorResult(score=0.0, baseline=new_baseline)
    score = min(
        1.0, max(0.0, update.zscore) / _CODEBASE_PREDICTION_ERROR_ZSCORE_SATURATION
    )
    return CodebasePredictionErrorResult(score=score, baseline=new_baseline)


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

    **2026-07-28 baseline fix.** Same disease as ``recent_perturbations``' old
    fixed ``/20.0`` cap and ``bus_synaptic_prediction_error``'s pre-2026-07-26 calm
    floor: this used to saturate via a fixed ``_mean(deltas) / _THRESHOLD`` (0.30)
    divisor. Live-confirmed 2026-07-28 against 118 real ``substrate.execution_
    trajectory`` receipts: mean error 0.0001, max ever observed 0.0009 -- real
    cortex-exec pressure-hint deltas run about three orders of magnitude below
    ``_THRESHOLD``, so this instrument reads ~0 essentially always regardless of
    real execution turbulence. Not a calibration nit -- it is structurally
    incapable of ever reading "surprised," the mirror-image failure of bus_
    synaptic's old floor bug (that one couldn't read "calm"; this one can't read
    "surprised"). Root cause is the same in both cases: a hand-picked constant
    standing in for a self-calibrating baseline.

    Fixed the same way ``recent_perturbations`` was: track this domain's own
    EWMA mean/variance of the raw per-tick ``_mean(deltas)`` (persisted on the
    projection itself -- ``prediction_error_baseline_ewma``/``_var``/``_n``, see
    ``ExecutionTrajectoryProjectionV1``'s docstring) via ``orion.bus.ewma.
    compute_ewma_update``, then score *this* tick's raw delta as a z-score against
    that live baseline instead of dividing by a fixed constant. Passes its own
    ``_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE`` rather than that function's
    default -- see that constant's own comment: this domain's real variance is
    five orders of magnitude below the shared default, which would otherwise
    silently reintroduce a milder version of this same bug one layer down.
    Unlike bus_
    synaptic_prediction_error, no calm-floor subtraction is needed here: bus_
    synaptic's floor exists because it averages ``|z|`` (already-signed z-scores
    folded to their absolute value) across many edges every tick, and
    ``mean(|Z|)`` for a calm, zero-mean population has a strictly positive
    expected value (``sqrt(2/pi)``) by construction. This function instead
    produces exactly one raw z-score per tick from ``compute_ewma_update`` --
    a single signed value, not a mean of absolutes -- so it can genuinely rest
    at (or below) zero when calm with no bias to correct for. A below-baseline
    tick (execution unusually *calmer* than its own recent normal) is clamped to
    0.0 rather than reported as negative surprise, since "surprising" here means
    "more turbulent than usual," not "different from usual" in either direction.

    Returns 0.0 (not the old raw/``_THRESHOLD`` score) on the first tick with any
    real deltas -- ``compute_ewma_update`` returns no z-score until a baseline
    exists (``prev_count == 0``), and reporting a value here would misrepresent
    "no baseline yet" as "measured, not anomalous" (this repo's "no empty-shell
    cognition" rule). The baseline absorbs that tick's value regardless, so the
    very next tick already has one observation to compare against.
    """
    deltas: list[float] = []
    prev_fallback = _latest_run(prev.runs)
    for trace_id, curr_run in curr.runs.items():
        prev_run = prev.runs.get(trace_id)
        if prev_run is None:
            prev_run = prev_fallback
        if prev_run is None:
            continue
        for key in ("cortex_exec_step_load", "execution_friction", "failure_pressure", "reasoning_load"):
            pv = prev_run.pressure_hints.get(key, 0.0)
            cv = curr_run.pressure_hints.get(key, 0.0)
            deltas.append(abs(cv - pv))
    if not deltas:
        return 0.0

    raw_mean_delta = _mean(deltas)
    update = compute_ewma_update(
        prev_ewma=prev.prediction_error_baseline_ewma,
        prev_variance=prev.prediction_error_baseline_ewma_var,
        prev_count=prev.prediction_error_baseline_ewma_n,
        value=raw_mean_delta,
        alpha=_EXECUTION_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE,
    )
    curr.prediction_error_baseline_ewma = update.ewma
    curr.prediction_error_baseline_ewma_var = update.variance
    curr.prediction_error_baseline_ewma_n = prev.prediction_error_baseline_ewma_n + 1
    if update.zscore is None:
        return 0.0
    return min(1.0, max(0.0, update.zscore) / _EXECUTION_PREDICTION_ERROR_ZSCORE_SATURATION)


def transport_prediction_error(
    prev: TransportBusProjectionV1,
    curr: TransportBusProjectionV1,
) -> float:
    """0-1 surprise score: how much did transport bus health change this batch?

    **Retired from live use 2026-07-26** (docs/superpowers/specs/2026-07-26-
    transport-domain-retirement-bus-synaptic-successor-design.md). This diffs
    stream_backlog_health/delivery_confidence/stream_backlog_pressure -- fields
    fed by BUS_OBSERVER_STREAMS' narrow 2-Redis-Stream "world_pulse" census, not
    real inter-service bus traffic. Already excluded from attention_self_model
    .py's ACTIVE_INFERENCE_DOMAINS as known-dead; `services/orion-substrate-
    runtime/app/worker.py::_transport_tick()` no longer calls this function
    live -- bus_synaptic_prediction_error() (below) is the real, mesh-wide
    successor, already flowing into every consumer this fed. Kept here (not
    deleted) though it now has zero callers anywhere in the repo, live or
    offline -- checked: measure_transport_bus_signal_history.py and measure_
    transport_biometrics_prediction_error_correlation.py only mention this
    function's name in prose docstrings, they read persisted values directly
    from Postgres, no import. Do not wire this back into a live tick.
    """
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


def bus_synaptic_prediction_error(edge_zscores: list[float]) -> float:
    """0-1 surprise score: how anomalous is live bus traffic right now, aggregated
    across the bus synaptic graph's (``orion_bus_synapse``) real-time EWMA/z-score
    edges.

    **Deliberately not a prev/curr diff, unlike the other continuous-magnitude
    instruments in this module.** The bus synaptic graph
    (``services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update``)
    already maintains its own rolling EWMA baseline per edge and computes each
    edge's z-score against that baseline continuously as new traffic arrives --
    the "prev expectation" is already baked into the z-score itself, so there is
    no separate prev/curr snapshot for this function to diff. The caller queries
    current ``|zscore|`` values from FalkorDB (see
    ``services/orion-substrate-runtime/app/worker.py::_bus_synaptic_tick``) and
    is responsible for filtering to edges above the graph's own documented
    cold-start reliability floor (``services/orion-bus-mirror/README.md``:
    ``count < ~5`` is unreliable) -- this function does no filtering itself, it
    only aggregates whatever list it is given.

    Saturates via ``_BUS_SYNAPTIC_ZSCORE_SATURATION`` (3.0), not this module's
    ``_THRESHOLD`` (0.30) -- that constant is calibrated for the other domains'
    0-1-scale pressure-hint deltas, not z-score units. 3.0 reuses the
    ``zscore_threshold=3.0`` convention already live in
    ``services/orion-hub/scripts/bus_synaptic_graph_routes.py``'s ``anomalies()``
    route rather than inventing a new calibration.

    **2026-07-26: subtracts ``_BUS_SYNAPTIC_CALM_FLOOR`` before saturating.**
    ``mean(|z|)`` does not rest at 0 for genuinely calm traffic -- see that
    constant's docstring. Without this correction the function returned
    values with a live-confirmed floor around 0.27 no matter how quiet the
    mesh actually was, which silently and permanently biased
    ``prediction_error_confidence`` downward once this domain was wired into
    ``attention_self_model.ACTIVE_INFERENCE_DOMAINS`` (caught before any live
    consumer existed). Now returns ``max(0, mean(|z|) - floor) / (saturation -
    floor)``, clamped to [0, 1] -- still saturates to 1.0 at zscore 3.0
    (unchanged ceiling), but now genuinely rests near 0 like the other four
    domains' raw deltas.
    """
    if not edge_zscores:
        return 0.0
    # **Saturate PER EDGE, then average -- not average, then saturate.**
    #
    # 2026-07-30: this function was averaging UNBOUNDED per-edge z-scores and
    # only clamping the aggregate at the end, which let a single pathological
    # edge dictate the whole reading. Live-confirmed against the real
    # `orion_bus_synapse` graph, using the caller's own recency-filtered edge
    # set (244 live edges):
    #
    #     median(|z|)                 0.504   <- 237 of 244 edges are calm
    #     mean(|z|)  [old]            3.982   -> prediction_error 1.0000
    #     mean(min(|z|, 3.0)) [new]   0.687   -> prediction_error 0.0000
    #
    # Only 7 of 244 edges (2.9%) exceeded 3.0 sigma, and one stale
    # cortex-orch->Channel edge carried |z| = 7087.8 by itself. Inter-arrival
    # gap z-scores are heavy-tailed by construction, so the arithmetic mean of
    # unbounded values is not a robust estimator of "how anomalous is the mesh
    # right now" -- it is closer to a max. The result was a permanently
    # saturated 1.0 that produced real, continuous false "Bus Anomaly Detected"
    # alerts (orion-equilibrium-service) and pinned this domain's contribution
    # to AST/HOT's prediction_error_confidence.
    #
    # Clamping each edge at the saturation point first preserves the metric's
    # meaning ("average anomaly level across the mesh") while bounding any one
    # edge's contribution to "fully anomalous" instead of "arbitrarily large".
    # The calm-floor calibration below is unaffected: for a genuinely calm
    # population mean(|z|) ~= sqrt(2/pi) ~= 0.798, far under the clamp, so no
    # calm value is altered by it.
    #
    # **Known, deliberate consequence**: a single catastrophically anomalous
    # edge no longer moves this signal on its own (one edge at |z|=7087 now
    # contributes 3.0/N, not 7087/N). That is correct for a mesh-wide average,
    # but it means single-edge anomaly detection is NOT covered by this metric
    # and must not be assumed from it -- if that is wanted it should be its own
    # named signal (worst-edge / count-above-threshold), not folded back in
    # here where it would re-create exactly this bug.
    clamped = [
        min(abs(z), _BUS_SYNAPTIC_ZSCORE_SATURATION) for z in edge_zscores
    ]
    mean_abs_z = _mean(clamped)
    excess = max(0.0, mean_abs_z - _BUS_SYNAPTIC_CALM_FLOOR)
    return min(1.0, excess / (_BUS_SYNAPTIC_ZSCORE_SATURATION - _BUS_SYNAPTIC_CALM_FLOOR))
