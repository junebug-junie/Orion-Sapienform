from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

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

# 2026-08-19: same fix, same reasoning, for chat_prediction_error -- found via a
# live self-model audit (predicted_shift's per-domain argmax, see that function's
# own docstring in attention_self_model.py): chat had won 0/19,426 ticks over a
# real 7-day window despite having genuine, non-flat prediction-error deltas. Root
# cause confirmed live: chat_prediction_error still used the module's fixed
# `_THRESHOLD = 0.30` divisor -- the exact defect execution_prediction_error was
# fixed for on 2026-07-28 -- and this instrument's own prior docstring already
# grouped "execution/chat/route" together as reading near-zero for that reason.
# Derived chat's real raw-delta scale from live `prediction_error_by_domain.chat`
# (already-scaled output, nowhere near the 1.0 saturation ceiling so
# raw = output * 0.30 recovers it exactly): 19,425 real ticks, stddev 0.0024 ->
# derived raw-delta stddev ~7.24e-4, derived raw variance ~5.24e-7. One order of
# magnitude below that (same convention as _EXECUTION_PREDICTION_ERROR_MIN_VARIANCE
# above) is ~5.2e-8; using the round 5e-8 same as that convention's precedent.
# Same alpha/saturation as execution -- no domain-specific reason found yet to
# diverge from that already-established real-observation-cadence alpha and the
# z>=3.0 "anomalous" convention shared by every other z-score domain in this module.
_CHAT_PREDICTION_ERROR_EWMA_ALPHA = 0.2
_CHAT_PREDICTION_ERROR_ZSCORE_SATURATION = 3.0
_CHAT_PREDICTION_ERROR_MIN_VARIANCE = 5e-8


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
    alpha: float,
    min_variance: float,
) -> tuple[float | None, _DomainEwmaBaseline]:
    """One sub-domain's EWMA update: absent magnitude (this tick carried no real
    delta for this domain) leaves the baseline untouched and contributes no
    z-score -- not a 0.0 "calm" reading, an honest "no observation this tick"
    (this repo's "no empty-shell cognition" rule, applied per sub-domain).

    ``alpha`` is caller-supplied, not hardcoded to any one domain's constant
    (review finding, 2026-08-19, while wiring this same helper into
    ``perception_prediction_error``'s z-score migration below: this
    function used to hardcode ``_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA``
    directly, which happened to numerically match every other domain's
    alpha=0.2 convention so far, but silently coupled every future caller
    to codebase's own constant specifically -- a caller wanting or needing
    a different alpha would have silently gotten codebase's instead, with
    nothing to signal it). ``codebase_prediction_error`` below now passes
    its own constant explicitly; behavior for that domain is unchanged.
    """
    if magnitude is None:
        return None, baseline
    update = compute_ewma_update(
        prev_ewma=baseline.ewma,
        prev_variance=baseline.variance,
        prev_count=baseline.n,
        value=magnitude,
        alpha=alpha,
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

    git_zscore, new_git = _domain_zscore(
        git_magnitude, baseline.git, alpha=_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA, min_variance=_GIT_MIN_VARIANCE
    )
    pr_zscore, new_pr = _domain_zscore(
        pr_magnitude, baseline.pr, alpha=_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA, min_variance=_PR_MIN_VARIANCE
    )
    graph_zscore, new_graph = _domain_zscore(
        graph_magnitude, baseline.graph, alpha=_CODEBASE_PREDICTION_ERROR_EWMA_ALPHA, min_variance=_GRAPH_MIN_VARIANCE
    )

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


def _score_and_update_ewma_prediction_error(
    prev: Any,
    curr: Any,
    *,
    raw_mean_delta: float,
    alpha: float,
    min_variance: float,
    zscore_saturation: float,
) -> float:
    """Shared EWMA-baseline scoring + update, factored out 2026-08-19 (code
    review, rule-of-three): `execution_prediction_error`'s 2026-07-28 fix and
    `chat_prediction_error`'s 2026-08-19 fix had each independently hand-copied
    this exact "score this tick's raw delta as a z-score against a live EWMA
    baseline, then update that baseline" sequence, with no shared helper.
    `biometrics_prediction_error` still uses the old fixed-`_THRESHOLD` divisor
    and is the next domain that would need this same migration -- not done
    here, out of scope for this patch (see that function's docstring: its
    real dynamic range doesn't show the same near-zero-forever symptom chat
    and execution did, so it wasn't independently confirmed broken).

    `prev`/`curr` are duck-typed: any object exposing
    `prediction_error_baseline_ewma`/`_var`/`_n` float/int attributes (every
    projection using this pattern shares those exact field names -- see
    `ExecutionTrajectoryProjectionV1`/`ChatSessionProjectionV1`'s matching
    docstrings). Mutates `curr`'s three baseline fields in place -- same
    contract every caller of this pattern already had before this extraction;
    **callers remain responsible for persisting `curr` afterward** (this
    function has no I/O of its own, matching this whole module's "pure
    function" design) -- see each tick's own `worker.py` wiring for the
    corresponding save call.
    """
    update = compute_ewma_update(
        prev_ewma=prev.prediction_error_baseline_ewma,
        prev_variance=prev.prediction_error_baseline_ewma_var,
        prev_count=prev.prediction_error_baseline_ewma_n,
        value=raw_mean_delta,
        alpha=alpha,
        min_variance=min_variance,
    )
    curr.prediction_error_baseline_ewma = update.ewma
    curr.prediction_error_baseline_ewma_var = update.variance
    curr.prediction_error_baseline_ewma_n = prev.prediction_error_baseline_ewma_n + 1
    if update.zscore is None:
        return 0.0
    return min(1.0, max(0.0, update.zscore) / zscore_saturation)


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

    return _score_and_update_ewma_prediction_error(
        prev,
        curr,
        raw_mean_delta=_mean(deltas),
        alpha=_EXECUTION_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE,
        zscore_saturation=_EXECUTION_PREDICTION_ERROR_ZSCORE_SATURATION,
    )


# transport_prediction_error() DELETED 2026-07-31.
#
# Retired from live use 2026-07-26 (docs/superpowers/specs/2026-07-26-transport-
# domain-retirement-bus-synaptic-successor-design.md) but kept in the file on
# the reasoning that "deleting it buys nothing and risks a future script wanting
# it for genuine historical replay." That reasoning was wrong twice over:
#
#   1. It buys the thing CLAUDE.md section 0A actually asks for. A retired instrument
#      that still exists is one a future patch wires back in -- and the entry
#      that kept it alive downstream (orion-substrate-runtime's
#      _PREDICTION_ERROR_DOMAIN_NODE_IDS) survived FIVE DAYS past the write being
#      killed for exactly that reason: nothing failed when it stayed.
#   2. The replay case is not lost. It diffed TransportBusProjectionV1's
#      stream_backlog_health / delivery_confidence / stream_backlog_pressure --
#      fields still persisted, still readable, and recoverable in ~10 lines by
#      any script that genuinely needs them. Keeping a live-importable symbol is
#      not the cheap way to preserve that; git history is.
#
# The successor is bus_synaptic_prediction_error() above: mesh-wide real
# inter-service publish traffic via orion_bus_synapse's per-edge EWMA z-scores,
# rather than a 2-Redis-Stream "world_pulse" census that was never bus health.
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

    **2026-08-19 baseline fix, same disease and same fix as
    ``execution_prediction_error``.** Live-confirmed via a real self-model audit:
    ``predicted_shift``'s per-domain argmax (`attention_self_model.py`) had never
    once named ``chat`` across 19,426 real ticks over a 7-day window, despite chat
    having genuine, non-flat deltas -- this function's own fixed ``_THRESHOLD =
    0.30`` divisor (the exact defect fixed for execution on 2026-07-28) scaled
    them down to a range execution/biometrics/bus_synaptic's much larger real
    deltas could never lose to. Fixed the same way: track this domain's own EWMA
    mean/variance of the raw per-tick ``_mean(deltas)`` (persisted on the
    projection -- ``prediction_error_baseline_ewma``/``_var``/``_n``, see
    ``ChatSessionProjectionV1``'s docstring) and score this tick's raw delta as a
    z-score against that live baseline instead of the fixed divisor. See
    ``_CHAT_PREDICTION_ERROR_MIN_VARIANCE``'s own comment for the live-data
    derivation behind its value. Returns 0.0 on the first tick with any real
    deltas, same reasoning as execution's identical cold-start case.
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
    if not deltas:
        return 0.0

    return _score_and_update_ewma_prediction_error(
        prev,
        curr,
        raw_mean_delta=_mean(deltas),
        alpha=_CHAT_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=_CHAT_PREDICTION_ERROR_MIN_VARIANCE,
        zscore_saturation=_CHAT_PREDICTION_ERROR_ZSCORE_SATURATION,
    )


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


# ---------------------------------------------------------------------------
# capability:vision -- perceptual availability
#
# 2026-08-13: an earlier draft of this node derived vision health from the bus
# synaptic graph's `gap_zscore` over the orion:vision:* channels, reusing
# bus_synaptic_prediction_error's counting rule. That was **deleted rather than
# tuned**, because it measured the wrong layer, in two ways that no threshold
# fixes:
#
#   1. It z-scores message INTER-ARRIVAL TIME, and the vision pipeline's
#      cadence is set by a fixed scheduler (config/vision_frame_router.yaml's
#      `min_seconds_between_tasks_per_camera: 5`). Z-scoring a metronome
#      measures scheduler jitter, not perception.
#   2. It is structurally blind to a blinded camera. Measured live by posting
#      synthetic frames to the vision host: a pure-black frame and a flat-grey
#      frame each returned `ok=True` with **0 objects**, while the same
#      detector on the real frame returned 6 (max score 0.72). A capped lens, a
#      dark room, or a frozen stream all produce perfectly regular, perfectly
#      successful bus traffic carrying no information whatsoever. Cadence
#      cannot see that. Yield can, trivially.
#
# The replacement reads the eye's own output instead of the bus's rhythm, and
# uses no EWMA anywhere -- there is no baseline to smooth, only a clock and a
# count.
# ---------------------------------------------------------------------------

# Deadband and saturation for perceptual availability, anchored to the measured
# live cadence of orion:vision:artifacts (2026-08-13, 60s pubsub census: 12
# messages, one every 5.0s), which is the channel carrying real detector
# output. The deadband is 3x that healthy interval -- wide enough that ordinary
# scheduler jitter reads exactly 0.0, narrow enough that a stopped eye is
# unambiguous well inside a minute.
_VISION_STALENESS_GRACE_SEC = 15.0
_VISION_STALENESS_SATURATION_SEC = 60.0


def vision_channel_staleness_pressure(
    age_seconds: float,
    *,
    grace_seconds: float = _VISION_STALENESS_GRACE_SEC,
    saturation_seconds: float = _VISION_STALENESS_SATURATION_SEC,
) -> float:
    """0-1 availability pressure from the age of the newest vision artifact.

    Answers "is the eye producing at all". The caller computes ``age_seconds``
    against a clock on a fixed tick, so this rises during silence -- the
    property the deleted EWMA approach structurally could not have, since an
    event-triggered statistic is never recomputed when the events stop.

    This is the "explicit freshness channel" the perception design doc requires,
    and the reason that doc also insists staleness must never be modelled by
    decay: a decaying value converges toward calm, which is backwards. Silence
    has to converge toward alarm. Concretely, this is the shape of the
    ``node:substrate.route`` incident CLAUDE.md §0A records, where a value that
    had merely stopped being refreshed was indistinguishable from a genuine
    calm reading.

    Rest point is exactly 0.0 and genuinely reachable: at health the newest
    artifact is ~5s old, inside the deadband. Saturates at 1.0, so it is
    bounded without clamping the input.
    """
    # +inf means "no artifact ever seen", which is the alarm end, not the calm
    # end -- mapping it to 0.0 would contradict this function's own invariant
    # that a longer silence never reads calmer. NaN is a malformed clock
    # reading rather than a claim about the eye, so it stays fail-open at 0.0.
    if math.isnan(age_seconds):
        return 0.0
    if age_seconds == math.inf:
        return 1.0
    if age_seconds <= grace_seconds:
        return 0.0
    span = saturation_seconds - grace_seconds
    if span <= 0:
        return 1.0
    return min(1.0, (age_seconds - grace_seconds) / span)


def perceptual_yield(object_counts: list[int]) -> float:
    """Mean detected objects per artifact over a recent window.

    A raw observable, not a pressure: no normalisation, no baseline, no
    smoothing. Recorded on ``node:substrate.vision`` so a blinded eye is
    *visible*, and deliberately NOT mapped to ``capability:vision`` pressure
    yet -- see ``perceptual_blindness_pressure`` for why that step needs
    evidence this patch does not have.

    Measured live 2026-08-13 on the real cam0 stream: ~6-8 objects per frame
    against a black or flat-grey probe frame's 0.
    """
    if not object_counts:
        return 0.0
    usable = [max(0, int(c)) for c in object_counts]
    return sum(usable) / len(usable)


def perceptual_blindness_pressure(
    object_counts: list[int],
    *,
    min_samples: int = 12,
) -> float:
    """0-1 pressure for "artifacts are arriving but carry nothing".

    The failure the availability channel cannot see: the pipeline is healthy,
    tasks return ``ok=True``, messages arrive on schedule, and every frame is
    empty. Verified reachable by probe -- a black frame returns 0 objects
    through the real detector while the same detector returns 6 on the live
    scene.

    **Deliberately not wired to pressure in the patch that introduces it**, and
    the reason is a genuine ambiguity rather than caution theatre: a sustained
    zero is equally consistent with a blinded eye and with *an empty dark
    room at 3am*. Distinguishing those needs a per-stream temporal prior --
    the "day-shape" of Movement II in the perception design doc, which knows
    that this room is normally empty at 3am and normally is not at 6pm. Until
    that exists, a zero-yield alarm would fire every night on a working camera,
    which is the false-positive twin of the fabricated ``confidence=1.0`` this
    whole node exists to delete. Recorded and observed first; promoted only
    once there is something to interpret it against.

    ``min_samples`` guards the cold-start case: a single empty frame is a
    blink, not blindness.
    """
    if len(object_counts) < min_samples:
        return 0.0
    return 1.0 if perceptual_yield(object_counts) <= 0.0 else 0.0


# ---------------------------------------------------------------------------
# node:substrate.perception -- P2, perceptual prediction error
#
# 2026-08-19 (docs/superpowers/specs/2026-08-12-perception-frontier-design.md,
# "P2 -- Perceptual prediction error"). Proposal-mode record (CLAUDE.md
# section 0A -- memory/identity/cognition-loop changes need it): this patch
# was authored in the same design-conversation-then-explicit-go-ahead pattern
# as commit ac633a411 and the Movement III PR, not a separate proposal doc.
#
# **Theory anchor** (metric quality gate step 3): predictive-coding / free-
# energy-style surprise -- deviation of a new observation from a running
# expectation of "what this stream normally looks like". This is the same
# family the design doc's own Thesis/Movement sections cite, not a
# post-hoc label; `bus_synaptic_prediction_error` and every domain above use
# the identical shape (an EWMA baseline, a real-time deviation from it), just
# over a scalar rather than a vector.
#
# **Independence** (metric quality gate step 2, checked explicitly against
# every existing perceptual signal, not assumed): distinct causal chain from
# each of the other three signals a naive reader might conflate this with.
#   - `node:substrate.vision`'s `perception_staleness`
#     (``vision_channel_staleness_pressure``) measures ARRIVAL TIMING -- the
#     age of the newest artifact against a wall clock. It cannot see a
#     healthy-cadence, content-frozen or content-drifting stream at all.
#   - `node:substrate.vision`'s `perception_yield`
#     (``perceptual_yield``) measures the DETECTOR'S OBJECT COUNT -- a
#     completely different model's discrete label output, unrelated to the
#     embedding model's continuous feature space this function reads.
#   - `bus_synaptic_prediction_error` measures MESH-WIDE PUBLISH-LATENCY
#     anomalies across ~all bus edges -- transport timing, not any single
#     stream's visual content.
#   This function is the only one of the four whose input is the embedding
#   MODEL'S OWN CONTENT ENCODING of a frame. A camera that is perfectly on
#   schedule (staleness=0), still returning detections (yield>0), on a mesh
#   with zero anomalous edges (bus_synaptic=0) can still have its visual
#   content silently change (someone walks into frame, the lighting flips) --
#   exactly the case none of the other three can see, and the reason this is
#   additive signal, not a redundant re-derivation of one already in place.
#
# **Existing-mechanism check** (metric quality gate step 5): searched the
# repo for a vector-valued EWMA utility before writing one.
# ``orion/bus/ewma.py::compute_ewma_update`` and every ``_DomainEwmaBaseline``
# above are scalar-only (one float mean/variance/z-score per domain); no
# vector-EWMA exists anywhere in this repo as of this patch. The EWMA below
# is therefore a fresh, minimal, componentwise mean update over a list of
# floats -- no numpy dependency added (neither this module nor
# ``services/orion-substrate-runtime`` had one in ``requirements.txt`` before
# this patch; a 1152-dim Python list comprehension at this tick's real
# cadence, one real embedding message every several seconds at most, costs
# nothing worth a new dependency for).
#
# **Correction, 2026-08-19 (same day):** this paragraph originally ended
# here claiming "no z-score/variance tracking is needed" for this domain,
# unlike the six scalar domains above. That was wrong, not just an initial
# simplification -- see ``_PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION``'s
# own comment for the live numbers that disproved it. The vector-EWMA
# reasoning above (no numpy dependency, componentwise mean) stands
# unchanged; a second, ordinary scalar ``_DomainEwmaBaseline`` was added
# alongside it to z-score the raw cosine-distance magnitude the vector EWMA
# produces, reusing this module's existing scalar mechanism rather than
# inventing a second one.
#
# **Reversibility** (metric quality gate step 6): shipped shadow-only,
# default-off (``SUBSTRATE_PERCEPTION_PREDICTION_ERROR_TICK_ENABLED=false``),
# wired to no consumer. Trivially reversible -- flip the flag back off, or
# delete the isolated ``substrate_perception_embedding_baseline`` table this
# patch adds (single writer, see that table's own migration comment).
#
# **Live-data sanity check** (metric quality gate step 4): VERIFIED
# 2026-08-19 for the raw magnitude, the same day this comment was written --
# (a) calm rest point confirmed real, not decayed-to-zero: 57 calm-state
# ticks (one camera stream, ~30 min) hand-pulled from
# ``substrate_reduction_receipts``, mean 0.0055, variance 2.10e-5, no exact
# zeros, no flat successive-value ratio; (b) fired on a real transition (a
# room-occupancy change, ~0.03) and, separately, on a real induced stimulus
# (the camera physically knocked over mid-session, 0.1219, ~25-sigma against
# the calm baseline above) and recovered to calm within 3 ticks (90s), not
# stuck high; (c) ``embedding_staleness`` decay-to-zero ruled out by the same
# hand-pulled history -- see this domain's live review note in
# ``orion/substrate/endogenous_curiosity.py`` for the full numbers and the
# separate finding this check's own result led to (the raw magnitude was
# real and non-degenerate, but numerically incomparable to every other
# domain's z-scored ``prediction_error`` -- the reason the z-score migration
# a few lines below exists).
#
# **This check itself needs re-running post-migration, on the z-scored
# score, before trusting it for any consumer decision** -- same "shadow,
# wait, verify by hand" two-phase pattern this comment already followed
# once. The numbers above prove the raw magnitude was well-behaved; they do
# not by themselves prove ``_PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION``
# (3.0, copied from every other domain's convention, not independently
# derived from this domain's own distribution shape) is well-calibrated --
# a tight, near-zero-mean calm distribution can saturate a 3-sigma bar on
# ordinary variation more easily than a domain with a wider natural spread.
# Required before trusting the migrated score for any consumer decision:
#   (a) pull real z-scored ticks by hand once enough have accumulated
#       post-deploy and confirm calm ticks clamp to ~0.0 (not a nonzero
#       floor) and stay there across ordinary room activity, not just the
#       single knocked-over-camera event already observed;
#   (b) confirm the raw-magnitude findings above still hold for the new
#       formula's own inputs -- no decay-to-zero, no permanent stick-high;
#   (c) if ordinary room activity (not a genuinely dramatic event) is
#       already crossing ``endogenous_curiosity.py``'s ``min_error=0.55``
#       regularly, the saturation constant is miscalibrated for this
#       domain's real skew and needs its own value, not the borrowed 3.0.
# ---------------------------------------------------------------------------

# Same alpha convention as execution_prediction_error/codebase_prediction_error
# above (0.2, one real observation per real embedding-bearing artifact, not a
# fixed wall-clock cadence).
_PERCEPTION_PREDICTION_ERROR_EWMA_ALPHA = 0.2

# **Correction, 2026-08-19, same day as the live-data check above:** the
# removed comment that used to sit here claimed "there is no variance floor
# to calibrate here at all" and shipped `perception_prediction_error()`
# returning the raw `1 - cos(...)` magnitude directly -- explicitly flagged
# in that function's own docstring as "the honest crude first version,
# explicitly deferred... later." That claim was wrong, not just premature:
# a bounded [0,1] magnitude has exactly as much real variance as any other
# domain's raw delta: this domain's own live-data check (2026-08-19, 60
# real ticks/30 min, one camera stream) measured it directly --
# calm-state mean 0.0055, variance 2.10e-5 (n=57, spike ticks excluded) --
# and a real stimulus (the camera physically knocked over mid-session)
# produced 0.1219, a genuine ~25-sigma event against that calm baseline.
# That is exactly the kind of magnitude execution_prediction_error/
# codebase_prediction_error already z-score before comparing against
# min_error, and perception was the one domain in this module skipping
# that step -- confirmed live (not assumed) to be why it could never
# cross endogenous_curiosity.py's shared min_error=0.55 threshold: raw
# cosine-distance surprise and z-scored deltas from other domains were
# never on the same numeric footing, so the "generic, domain-agnostic"
# scan (see that module's own docstring) was accidentally domain-blind
# for exactly this one input scale. Migrated below to the same
# raw-magnitude -> z-score -> saturate pattern `codebase_prediction_error`
# uses via `_domain_zscore`, so `min_error` means the same thing here as
# it does for every other migrated domain.
#
# Floor set three orders of magnitude below the measured real variance
# (2.10e-5) above, same convention _CHAT_PREDICTION_ERROR_MIN_VARIANCE
# and _EXECUTION_PREDICTION_ERROR_MIN_VARIANCE use against their own
# measured floors -- guards early-tick/degenerate division without ever
# dominating real variance once warmed up.
_PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION = 3.0
_PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE = 1e-8


@dataclass(frozen=True)
class PerceptionEmbeddingBaseline:
    """Per-stream running EWMA of the embedding vector itself (not a scalar
    mean/variance like every other domain above) -- ``embedding_ewma`` is
    componentwise ``alpha * new + (1 - alpha) * prev`` over the whole vector.
    ``n`` counts real observations folded in, mirroring every other
    domain's cold-start bookkeeping (``n == 0`` means no baseline yet).
    """

    embedding_ewma: tuple[float, ...] = ()
    n: int = 0

    # Second, independent scalar EWMA -- of the raw `1 - cos(...)` surprise
    # *magnitude* itself, not the embedding vector above. Deliberately its
    # own counter (`surprise_n`, not `n`): `n` advances on every real
    # non-degenerate observation including the very first cold-start seed
    # (which produces no raw surprise value at all, nothing to fold in
    # here), so `surprise_n` always lags `n` by exactly one real comparison.
    # Reuses `_DomainEwmaBaseline`/`_domain_zscore` (this module's existing,
    # validated per-domain z-score mechanism -- see `codebase_prediction_
    # error` above) rather than inventing a second normalization scheme.
    surprise_ewma: float = 0.0
    surprise_variance: float = 0.0
    surprise_n: int = 0

    def to_json_dict(self) -> dict:
        return {
            "embedding_ewma": list(self.embedding_ewma),
            "n": self.n,
            "surprise_ewma": self.surprise_ewma,
            "surprise_variance": self.surprise_variance,
            "surprise_n": self.surprise_n,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "PerceptionEmbeddingBaseline":
        raw = data.get("embedding_ewma") if isinstance(data, dict) else None
        vec: tuple[float, ...] = ()
        if isinstance(raw, list):
            try:
                vec = tuple(float(x) for x in raw)
            except (TypeError, ValueError):
                vec = ()
        n_raw = data.get("n", 0) if isinstance(data, dict) else 0
        try:
            n = int(n_raw)
        except (TypeError, ValueError):
            n = 0
        # Missing keys (rows persisted before this migration) default to a
        # cold scalar baseline -- same "no history yet is a real, honest
        # state" convention as the rest of this class, not an error.
        try:
            surprise_ewma = float(data.get("surprise_ewma", 0.0)) if isinstance(data, dict) else 0.0
        except (TypeError, ValueError):
            surprise_ewma = 0.0
        try:
            surprise_variance = float(data.get("surprise_variance", 0.0)) if isinstance(data, dict) else 0.0
        except (TypeError, ValueError):
            surprise_variance = 0.0
        try:
            surprise_n = int(data.get("surprise_n", 0)) if isinstance(data, dict) else 0
        except (TypeError, ValueError):
            surprise_n = 0
        return cls(
            embedding_ewma=vec,
            n=max(0, n),
            surprise_ewma=surprise_ewma,
            surprise_variance=surprise_variance,
            surprise_n=max(0, surprise_n),
        )


@dataclass(frozen=True)
class PerceptionPredictionErrorResult:
    """``score`` is ``None`` on cold start (no prior baseline for this
    stream, or a dimension mismatch against a prior one -- e.g. the
    embedding model changed) -- there is no expectation yet to be surprised
    against, and reporting 0.0 there would misrepresent "no baseline yet" as
    "measured, not anomalous" (this repo's "no empty-shell cognition" rule,
    same convention every EWMA domain above follows via
    ``compute_ewma_update``'s own ``zscore=None`` first-observation case).
    """

    score: float | None
    baseline: PerceptionEmbeddingBaseline


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float | None:
    """``None`` on a degenerate comparison (mismatched length or a zero-norm
    vector) rather than raising or returning a fabricated 0.0 -- a zero-norm
    embedding is itself a malformed-input case, not a legitimate "maximally
    different" reading.
    """
    if len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    return dot / (norm_a * norm_b)


def perception_prediction_error(
    embedding: Sequence[float],
    baseline: PerceptionEmbeddingBaseline,
    *,
    alpha: float = _PERCEPTION_PREDICTION_ERROR_EWMA_ALPHA,
) -> PerceptionPredictionErrorResult:
    """0-1 surprise score for one real embedding-bearing vision artifact,
    per this stream's own recent normal.

    Two stages, both against this stream's own running state: (1) raw
    magnitude ``1 - cos(embedding, running_EWMA_embedding)`` -- the design
    doc's original P2 formula, kept unchanged as the input signal; (2) that
    magnitude z-scored against a second EWMA baseline of the magnitude
    itself, saturating at ``_PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION``,
    same ``_domain_zscore`` mechanism ``codebase_prediction_error`` uses.
    Stage (1) alone shipped 2026-08-19 as "the honest crude first version" and
    was the actual returned score for one day; migrated to include stage (2)
    the same day once live data showed the raw magnitude never approached
    other domains' numeric scale (see ``_PERCEPTION_PREDICTION_ERROR_
    ZSCORE_SATURATION``'s own comment for the real numbers).

    Cold start (``baseline.n == 0``) and a dimension mismatch against the
    prior baseline (e.g. the embedding model was swapped) are treated the
    same way: seed a fresh baseline from this observation, report no score.
    A dimension change is not a real optimizer for "how did this frame
    differ" -- there is no honest way to cosine-compare vectors of different
    length, so re-seeding (not raising, not silently truncating) is the only
    behavior that doesn't fabricate a reading. This stream's very first real
    comparison (embedding baseline warm, scalar surprise baseline still
    cold) also reports no score, for the identical reason one level up --
    see ``_domain_zscore``.

    **A zero-norm (all-zero) observation is rejected outright, before the
    cold-start/reseed check runs, not just on the comparison path** (review
    finding): seeding a fresh baseline from a zero vector would make every
    subsequent real observation permanently degenerate too, since
    ``_cosine_similarity`` already refuses to compare against a zero-norm
    baseline -- one bad reading at cold start would otherwise silently stall
    the whole stream (score always ``None``, baseline never updates again)
    instead of just skipping that one reading and staying genuinely cold
    until a real observation arrives.

    Cosine similarity of two real (non-degenerate) vectors ranges [-1, 1], so
    the raw ``1 - cos`` magnitude ranges [0, 2] in principle; clamped to
    [0, 1] before it ever reaches the z-score stage (SigLIP-style embeddings
    are L2-normalized and in practice cluster in a positive cone for real
    photographic frames, so this clamp is a safety bound, not the expected
    operating point).

    Both EWMA baselines are updated on every real, non-degenerate
    observation -- including cold-start and dimension-mismatch reseeds --
    the same "the baseline absorbs this tick's value regardless of whether a
    score was reported" behavior ``execution_prediction_error`` documents
    for its own first-tick case.
    """
    vec = [float(x) for x in embedding]
    if not vec:
        return PerceptionPredictionErrorResult(score=None, baseline=baseline)

    vec_norm = math.sqrt(sum(x * x for x in vec))
    if vec_norm <= 0.0:
        # Degenerate on arrival -- do not seed/reseed the baseline from bad
        # data, and do not report a score for it. Baseline (including a
        # genuinely cold ``n == 0`` one) is left untouched.
        return PerceptionPredictionErrorResult(score=None, baseline=baseline)

    if baseline.n == 0 or len(baseline.embedding_ewma) != len(vec):
        new_baseline = PerceptionEmbeddingBaseline(embedding_ewma=tuple(vec), n=1)
        return PerceptionPredictionErrorResult(score=None, baseline=new_baseline)

    cos = _cosine_similarity(vec, baseline.embedding_ewma)
    if cos is None:
        # Degenerate (zero-norm) input -- do not update the baseline on bad
        # data, and do not report a score for it either.
        return PerceptionPredictionErrorResult(score=None, baseline=baseline)

    raw_surprise = max(0.0, min(1.0, 1.0 - cos))
    new_ewma = tuple(
        alpha * e + (1.0 - alpha) * b for e, b in zip(vec, baseline.embedding_ewma)
    )

    # z-score the raw cosine-distance magnitude against this stream's own
    # recent normal, same pattern codebase_prediction_error uses per
    # sub-domain -- see _PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION's own
    # comment for why the raw magnitude alone was not comparable across
    # domains. `None` on this stream's first real comparison (surprise_n
    # still 0 -- no baseline yet to be surprised against), same "no
    # empty-shell cognition" convention as the cold-start case above.
    surprise_zscore, new_surprise_baseline = _domain_zscore(
        raw_surprise,
        _DomainEwmaBaseline(
            ewma=baseline.surprise_ewma,
            variance=baseline.surprise_variance,
            n=baseline.surprise_n,
        ),
        alpha=alpha,
        min_variance=_PERCEPTION_PREDICTION_ERROR_MIN_VARIANCE,
    )
    new_baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=new_ewma,
        n=baseline.n + 1,
        surprise_ewma=new_surprise_baseline.ewma,
        surprise_variance=new_surprise_baseline.variance,
        surprise_n=new_surprise_baseline.n,
    )
    if surprise_zscore is None:
        return PerceptionPredictionErrorResult(score=None, baseline=new_baseline)
    score = min(1.0, surprise_zscore / _PERCEPTION_PREDICTION_ERROR_ZSCORE_SATURATION)
    return PerceptionPredictionErrorResult(score=score, baseline=new_baseline)
