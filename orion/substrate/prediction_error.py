from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orion.bus.ewma import compute_ewma_update
from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.chat_projection import ChatSessionProjectionV1
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.route_projection import RouteArbitrationProjectionV1
from orion.schemas.transport_projection import TransportBusProjectionV1
from orion.structural_mass.git_delta import GitChurnDelta
from orion.structural_mass.graph_delta import GraphStructuralDelta
from orion.structural_mass.pr_lifecycle import PrLifecycleDelta
from orion.substrate.chat_loop.grammar_extract import compute_chat_pressure_hints

_THRESHOLD = 0.30

# Z-score saturation for bus_synaptic_prediction_error, distinct from _THRESHOLD
# above (calibrated for 0-1-scale pressure-hint deltas, not z-score units).
# Reuses the zscore_threshold=3.0 convention already live in
# services/orion-hub/scripts/bus_synaptic_graph_routes.py's anomalies() route.
_BUS_SYNAPTIC_ZSCORE_SATURATION = 3.0

# _BUS_SYNAPTIC_CALM_FLOOR (= sqrt(2/pi)) was REMOVED 2026-07-30 along with the
# mean-based formula it corrected. It was a real fix for a real bias in that
# formula (2026-07-26: mean(|z|) over ~N(0,1) z-scores rests at E[|Z|] =
# sqrt(2/pi), not 0, so the domain permanently reported "moderately surprised"),
# but bus_synaptic_prediction_error no longer takes a mean, so there is no bias
# left for it to correct -- and keeping it would have actively broken the
# replacement (see that function's docstring, point 2: against the real
# population, narrower than unit normal, this floor over-subtracted and pinned
# the metric at exactly 0.0). Deleted rather than left in place unused, per
# CLAUDE.md's "kill means kill": a retired constant that still exists is one a
# future patch reintroduces by accident.

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


# 2026-07-30: codebase_prediction_error's EWMA baseline calibration (Phase 1 contract
# patch, docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md). alpha=0.2
# reuses execution_prediction_error's own alpha for all three sub-domains rather than
# inventing three new numbers -- each sub-domain, like execution, updates per real
# observation (a real producer tick), not a fixed wall-clock cadence.
_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA = 0.2
_CODEBASE_PREDICTION_ERROR_ZSCORE_SATURATION = 3.0

# 2026-07-30: live-confirmed via scripts/analysis/measure_codebase_prediction_error.py
# against this repo's own real git history (1395 real ticks, one per commit, this
# repo's local clone deepened past its earlier ~50-commit shallow boundary during
# this session): warmed-up EWMA variance (n>=5) ranged 43,876.72 to
# 1,393,837,889,266.80, mean ~1.69e10 -- several orders of magnitude above any
# floor near 1.0, so this floor is never the binding term past the very first
# real tick, same conclusion (different exact numbers) as the Phase-1 skeleton
# patch's single-domain version of this same constant. Kept as its own floor, not
# shared with _PR_MIN_VARIANCE/_GRAPH_MIN_VARIANCE below, because -- as the next
# constant's own comment shows -- assuming "large raw magnitude implies large
# variance too" across sub-domains is exactly the mistake this split exists to
# avoid; each floor is set from that domain's own measured numbers, not inferred
# from another's.
_GIT_MIN_VARIANCE = 1.0

# 2026-07-30: live-confirmed via the same replay script against this repo's real
# GitHub PR history (1395 real ticks): warmed-up EWMA variance (n>=5) ranged
# 0.0000 to 1.5058, mean 0.1499 -- **this is the domain the borrowed-git-floor
# mistake would actually have broken.** The Phase-1-skeleton reasoning ("PR raw
# magnitude is single-digit scale, several orders of magnitude below git's, so a
# shared floor would be wrong") was correct about raw magnitude but did not
# check *variance* specifically before this contract patch's first draft reused
# git's 1.0 floor here anyway -- variance of small-integer counts near their own
# mean is frequently sub-1 (confirmed: real mean 0.1499, real min exactly 0.0),
# so a 1.0 floor would have been the *binding* term on most real PR ticks,
# silently flattening this domain's real z-scores toward the floor's own scale
# instead of its actual spread -- the identical failure shape
# execution_prediction_error's own fix already exists to prevent, just
# reintroduced one domain over by assuming a plausible-sounding number instead
# of measuring it. Set two orders of magnitude below the real mean (0.1499) --
# small enough to stay out of the way of real variance on almost every tick,
# nonzero to guard the true zero-variance ticks (real min was exactly 0.0)
# against a division by zero.
_PR_MIN_VARIANCE = 0.001

# 2026-07-30: live-confirmed via the same replay script against this repo's real,
# sparse graphify-out/graph.json history: only 6 real diffed ticks exist at all
# (graph is by far this composite's sparsest domain), and only 2 of those are
# "warmed up" (n>=5, i.e. the 5th and 6th real observations) -- not a large
# sample, an honest one. (An earlier version of this replay script's own
# instrumentation bug inflated this to a reported "n=240" by re-appending the
# same frozen baseline.variance on every one of the ~240 intervening non-firing
# ticks between real graph observations -- caught and fixed by code review
# 2026-07-30, see scripts/analysis/measure_codebase_prediction_error.py's own
# comment at the fix site. The *values* were already correct even under the
# bug, since duplicating real numbers doesn't change their min/max/mean -- but
# the sample-count evidence itself was fabricated, worth naming plainly rather
# than quietly correcting.) Those 2 real values ranged 2,616,891,929.63 to
# 3,243,755,333.97 -- several orders of magnitude above any floor near 1.0,
# same conclusion as git for the same underlying reason: this domain's raw
# magnitude is itself in the hundreds-to-hundred-thousands range from the
# 2026-07-14 destructive-update incident alone (see
# orion/structural_mass/graph_delta.py's own docstring for that incident's
# exact numbers), so its squared-deviation variance is naturally huge too.
_GRAPH_MIN_VARIANCE = 1.0


@dataclass(frozen=True)
class _DomainEwmaBaseline:
    ewma: float = 0.0
    variance: float = 0.0
    n: int = 0


@dataclass(frozen=True)
class CodebaseMassBaseline:
    """EWMA baseline state for ``codebase_prediction_error``, threaded explicitly by
    the caller rather than read off a persisted projection -- unlike the other five
    domains in this module, no ``structural_mass`` projection schema exists to mutate
    in place (this domain is bus-event-driven, not reducer-projection-driven; see the
    design spec's "Dedicated service" section). One independent sub-baseline per
    producer domain (``git``/``pr``/``graph``), since each producer runs on its own
    interval and a given tick may carry only one domain's real delta (the other two
    ``None`` -- see ``codebase_prediction_error``'s docstring)."""

    git: _DomainEwmaBaseline = field(default_factory=_DomainEwmaBaseline)
    pr: _DomainEwmaBaseline = field(default_factory=_DomainEwmaBaseline)
    graph: _DomainEwmaBaseline = field(default_factory=_DomainEwmaBaseline)

    def to_json_dict(self) -> dict:
        """Wire format for the consumer patch's ``substrate_codebase_mass_baseline``
        table (docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md,
        "Producer + consumer patch design") -- same
        ``dataclasses.dataclass`` -> ``dict`` -> ``jsonb`` convention
        ``orion/structural_mass/snapshot_history.py::GraphSnapshotStats`` already
        uses, not a new pattern invented here."""
        return {
            "git": {"ewma": self.git.ewma, "variance": self.git.variance, "n": self.git.n},
            "pr": {"ewma": self.pr.ewma, "variance": self.pr.variance, "n": self.pr.n},
            "graph": {"ewma": self.graph.ewma, "variance": self.graph.variance, "n": self.graph.n},
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "CodebaseMassBaseline":
        def _sub(key: str) -> _DomainEwmaBaseline:
            raw = data.get(key) or {}
            return _DomainEwmaBaseline(
                ewma=float(raw.get("ewma", 0.0)),
                variance=float(raw.get("variance", 0.0)),
                n=int(raw.get("n", 0)),
            )

        return cls(git=_sub("git"), pr=_sub("pr"), graph=_sub("graph"))


@dataclass(frozen=True)
class CodebasePredictionErrorResult:
    score: float
    baseline: CodebaseMassBaseline


def _domain_zscore(
    magnitude: float | None,
    baseline: _DomainEwmaBaseline,
    *,
    min_variance: float,
) -> tuple[float | None, _DomainEwmaBaseline]:
    """One sub-domain's EWMA update: absent magnitude (this tick carried no real
    delta for this domain) leaves the baseline untouched and contributes no
    z-score -- not a 0.0 "calm" reading, an honest "no observation this tick"
    (this repo's "no empty-shell cognition" rule, applied per sub-domain)."""
    if magnitude is None:
        return None, baseline
    update = compute_ewma_update(
        prev_ewma=baseline.ewma,
        prev_variance=baseline.variance,
        prev_count=baseline.n,
        value=magnitude,
        alpha=_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=min_variance,
    )
    new_baseline = _DomainEwmaBaseline(
        ewma=update.ewma, variance=update.variance, n=baseline.n + 1
    )
    if update.zscore is None:
        return None, new_baseline
    return max(0.0, update.zscore), new_baseline


def codebase_prediction_error(
    *,
    git_delta: GitChurnDelta | None,
    pr_delta: PrLifecycleDelta | None,
    graph_delta: GraphStructuralDelta | None,
    baseline: CodebaseMassBaseline,
) -> CodebasePredictionErrorResult:
    """0-1 composite surprise score across all three ``structural_mass`` producer
    domains -- how much did this tick's git churn, GitHub PR activity, and/or
    graphify structural change deviate from each domain's own recent normal.

    **Contract patch** (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
    design.md) -- the scoring half of Phase 1's "measure first, register once MET"
    order; bus channel/schema registration and service/consumer wiring are separate,
    later patches (this function has no bus dependency itself, so it can be tested
    and calibrated before either exists).

    **Each of the three producers runs on its own independent interval** (design
    spec's "Dedicated service" section -- git: cheap/frequent, PR lifecycle:
    coarser/rate-limit-aware, graph: event-triggered off graphify updates), so a
    given tick's inputs may populate only one, two, or all three of
    ``git_delta``/``pr_delta``/``graph_delta``; the other(s) are ``None``, meaning
    "that producer didn't fire this tick," not "that producer observed zero
    change." Each populated domain is scored independently against its *own* EWMA
    baseline (mirrors ``execution_prediction_error``'s single-domain z-score, just
    three parallel instances instead of one) rather than folding raw magnitudes
    from different unit scales (lines of code vs. PR counts vs. graph node/edge
    counts) into one combined raw number first -- that would reintroduce exactly
    the arbitrary cross-scale weighting problem the individual domains' own
    docstrings (``git_delta.py``, ``pr_lifecycle.py``) already avoid at one level
    down. Normalizing each domain to a z-score *before* combining means the
    composite is a mean of already-unit-less, already-self-calibrating values, the
    same principle ``bus_synaptic_prediction_error`` uses when it aggregates many
    already-normalized edge z-scores into one score.

    The final score is the mean of whichever sub-domain z-scores are actually
    available this tick (at least one -- if all three are ``None``, i.e. no
    producer fired, this returns ``score=0.0`` without touching the baseline at
    all, a real "nothing happened," not a computed reading), saturating at
    ``_CODEBASE_PREDICTION_ERROR_ZSCORE_SATURATION`` (3.0) same as every other
    z-score-based domain in this module. A below-baseline sub-domain tick is
    clamped to 0.0 before averaging (mirrors every other domain's "surprising
    means more than usual, not merely different" clamp), so a quiet git day
    alongside a genuinely surprising PR spike still surfaces the PR spike rather
    than being diluted toward 0 by an unclamped negative git contribution.
    """
    git_magnitude = None if git_delta is None else float(git_delta.lines_changed)
    pr_magnitude = (
        None
        if pr_delta is None
        else float(pr_delta.submitted_count + pr_delta.merged_count + pr_delta.closed_without_merge_count)
    )
    graph_magnitude = (
        None
        if graph_delta is None
        else float(abs(graph_delta.node_count_delta) + abs(graph_delta.edge_count_delta))
    )

    git_zscore, new_git = _domain_zscore(git_magnitude, baseline.git, min_variance=_GIT_MIN_VARIANCE)
    pr_zscore, new_pr = _domain_zscore(pr_magnitude, baseline.pr, min_variance=_PR_MIN_VARIANCE)
    graph_zscore, new_graph = _domain_zscore(graph_magnitude, baseline.graph, min_variance=_GRAPH_MIN_VARIANCE)

    new_baseline = CodebaseMassBaseline(git=new_git, pr=new_pr, graph=new_graph)
    zscores = [z for z in (git_zscore, pr_zscore, graph_zscore) if z is not None]
    if not zscores:
        return CodebasePredictionErrorResult(score=0.0, baseline=new_baseline)
    mean_zscore = sum(zscores) / len(zscores)
    score = min(1.0, mean_zscore / _CODEBASE_PREDICTION_ERROR_ZSCORE_SATURATION)
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
    """0-1 score: what FRACTION of live bus-synaptic edges are currently
    anomalous (``|zscore| >= _BUS_SYNAPTIC_ZSCORE_SATURATION``), across the
    bus synaptic graph's (``orion_bus_synapse``) real-time EWMA/z-score edges.

    **Deliberately not a prev/curr diff, unlike the other continuous-magnitude
    instruments in this module.** The bus synaptic graph
    (``services/orion-bus-mirror/app/graph_writer.py::compute_ewma_update``)
    already maintains its own rolling EWMA baseline per edge and computes each
    edge's z-score against that baseline continuously as new traffic arrives --
    the "prev expectation" is already baked into the z-score itself. The caller
    queries current ``|zscore|`` values from FalkorDB (see
    ``services/orion-substrate-runtime/app/worker.py::_bus_synaptic_tick``) and
    is responsible for filtering to edges above the graph's own documented
    cold-start reliability floor and recency window; this function does no
    filtering itself, it only aggregates whatever list it is given.

    **2026-07-30: this is a COUNTING metric now, not a magnitude one.**
    It previously computed ``max(0, mean(|z|) - CALM_FLOOR) / (SATURATION -
    CALM_FLOOR)``. That is retired, and the calm floor is gone with it. The
    reasoning is kept because two *different* wrong versions preceded this one:

    1. ``mean(|z|)`` over an unbounded, heavy-tailed population is a disguised
       ``max()``. Live-measured on the real graph: median |z| 0.399, p90
       1.123, but mean 29.278 -- of which 28.6 came from a SINGLE stale edge
       carrying |z| = 7087.8. Inter-arrival gap z-scores are heavy-tailed by
       construction, so one pathological edge dictated the whole reading. The
       node sat pinned at 1.0, driving continuous false "Bus Anomaly Detected"
       alerts through ``orion-equilibrium-service``'s transport gate.
    2. Clamping each edge before averaging fixed the tail but broke the floor,
       and was caught in review before it merged. ``_BUS_SYNAPTIC_CALM_FLOOR``
       was ``sqrt(2/pi)``, the theoretical ``E|Z|`` for a *standard normal*
       population. The real population is narrower than unit normal -- its
       z-scores are computed against an EWMA variance that the same outliers
       inflate -- so the clamped live mean was 0.5575 against a floor of
       0.7979. Negative headroom: pinned at exactly 0.0, needing 19 of 222
       edges at 3-sigma simultaneously just to leave zero and all 222 to reach
       the consumer's alert threshold. A permanently-silent detector instead
       of a permanently-firing one -- CLAUDE.md's metric quality gate step 4
       ("a metric reading a suspiciously clean 0.0 is not automatically
       'confirmed calm' either") verbatim.

    Counting the anomalous fraction is immune to both failures *by
    construction* rather than by calibration, which is why it was chosen over
    a third attempt at tuning a magnitude:

    - Bounded [0, 1] with no clamp needed -- it is a proportion.
    - Robust to any tail: one edge at |z| = 7087 counts exactly the same as
      one edge at |z| = 3.01, namely 1/N. The original bug cannot recur.
    - Its rest point is an interpretable, theory-anchored quantity rather than
      a fitted constant: for a genuinely calm standard-normal edge population
      it is ``P(|Z| >= 3) = 0.0027``. Live-measured baseline on the real mesh
      (60 samples over 10 minutes): median 0.026, mean 0.027, p95 0.072, max
      0.094, across 24 distinct values -- roughly 10x the normal-theory value,
      consistent with the known heavier-than-normal tail, and non-degenerate in
      both directions (it never read 0.0 and never read 1.0).

      A first 2-minute sample suggested a max of 0.043; the 10-minute sample
      found 0.094. Recorded because the short window would have supported a
      materially wrong threshold claim -- the same "window too thin to rank"
      trap the precision-weighted-attention spec already hit once.

    **Deliberate, disclosed scope limit**: this is a MESH-WIDE detector and
    structurally cannot resolve a single-organ failure. Live per-organ edge
    counts put the busiest organ (``orion-social-memory``, 12 of ~235 edges)
    at 0.051 if every one of its edges went anomalous at once -- below the
    observed baseline max of 0.094. Even the three busiest organs failing
    together reads only 0.136, a mere 1.45x that baseline max, so this metric
    cannot cleanly separate a few-organ event from noise at ANY threshold.
    What it does resolve reliably is a broad mesh event (>=15-20% of edges,
    several times the baseline). Single-organ detection needs a
    per-organ signal, not a lower threshold on this one; ``services/orion-hub/
    scripts/bus_synaptic_graph_routes.py``'s ``/propagate`` route already
    walks per-organ blast radius and is the right place to build it. Do not
    "fix" this by lowering the consumer's threshold into the noise band.
    """
    if not edge_zscores:
        return 0.0
    anomalous = sum(1 for z in edge_zscores if abs(z) >= _BUS_SYNAPTIC_ZSCORE_SATURATION)
    return anomalous / len(edge_zscores)
