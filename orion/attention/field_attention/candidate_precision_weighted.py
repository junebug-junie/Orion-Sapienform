"""Candidate A -- precision-weighted prediction-error salience.

**Shadow, read-only candidate instrument. Not wired into any live consumer.**

Sentience Striving Program (`orion/sentience_striving_program/README.md`) §7 ("measure
before minting", "multi-theory, not single-theory") and
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-
plan.md`'s "Candidate A" section. This is one of two parallel candidate replacements for
the un-calibrated `compute_salience()` formula in
`orion/attention/field_attention/scoring.py` -- Candidate B (Global Workspace/
Society-of-Mind rank-aggregation) is being built separately against the same real data
window. **Do not modify `scoring.py` or `selectors.py` here** -- this module is a parallel
measurement, not a live change, per the plan doc's non-goals ("Not choosing between
Candidate A and B here... Both get built as real, replayed, comparable instruments;
integration is decided from data, later.").

## Theory anchor

Feldman & Friston 2010, "Attention, Uncertainty, and Free-Energy" (confirmed against the
primary Wikipedia summary of the Free Energy Principle, "Perceptual precision, attention
and salience" section, 2026-07-21): attention is precision-optimization on prediction
error. Precision is the inverse variance ("temperature") of a signal's own historical
fluctuation -- a large prediction error is more salient when it occurs against a
historically *quiet* (low-variance, high-precision) backdrop than when the same magnitude
error occurs against a historically noisy one.

    salience(target) = precision(target) x |prediction_error(target)|
    precision(target) = 1 / variance(target's own real historical prediction-error series)

## Cross-target comparability (added in review, 2026-07-22)

`precision_weighted_salience()`'s raw `salience` is **not directly comparable across
targets** and is **not a valid drop-in for `FieldAttentionTargetV1.salience_score`**
(schema-bound to `[0, 1]`) -- it is unbounded (precision saturates at
`1 / PRECISION_VARIANCE_FLOOR = 1e6`; the real biometrics replay produced raw salience
ranging 0.00-57,100.00). This was missed in the original build and caught only in a
later compliance review, not by the original live-data check, which only confirmed
non-degenerate variance *within one target's own time series* and never checked whether
the resulting scale was usable for the cross-target *competition* this candidate exists
to run. Any real comparison across targets/domains must go through
`normalize_across_targets()` (below) first -- comparing two raw `.salience` values from
different targets directly is exactly the mistake that function exists to prevent.

## Real data source and scoping (2026-07-21, live-verified at build time)

`prediction_error(target)`'s *current* value comes from the five already-shipped
instruments in `orion/substrate/prediction_error.py` (`execution_prediction_error`,
`transport_prediction_error`, `biometrics_prediction_error`, `chat_prediction_error`,
`route_prediction_error`). `precision(target)` needs that same target's own *historical*
series -- FalkorDB substrate-graph nodes (`node:substrate.*`) hold only a single current
value with no history (confirmed live via `redis-cli GRAPH.QUERY` against `n.node_id`,
2026-07-21). The real historical series lives in Postgres
`substrate_reduction_receipts`, filtered by `reducer_name` (the receipt's
`state_deltas[0].reducer_id`, written by `_prediction_error_receipt()` in
`services/orion-substrate-runtime/app/worker.py`), ordered by `created_at`, with the
error value at `receipt_json->'state_deltas'->0->'after'->'pressure_hints'->>
'prediction_error'`.

**Honest scoping requirement, re-checked live at build time (not inherited from the
morning's design-doc numbers, which were already hours stale by the time this was
built):**

```
psql -h localhost -p 55432 -U postgres -d conjourney -c "
  SELECT reducer_name, count(*) FROM substrate_reduction_receipts
  WHERE reducer_name IN (
    'substrate.node_biometrics', 'substrate.execution_trajectory',
    'substrate.transport_bus', 'substrate.chat_session', 'substrate.route_arbitration'
  ) GROUP BY reducer_name;"
```

returned exactly one row: `substrate.node_biometrics` (168 receipts, real spread
0.0000-0.1656, non-degenerate). The other four reducer names -- despite their underlying
grammar reducers (`execution_trajectory`, `transport_bus`, `chat_grammar`,
`route_grammar`) all being `enabled=true` in the live `.env` -- had **zero** rows: their
prediction-error writers (`_tick`'s `if error > 0.0` gate at each of the four call sites
in `worker.py`) produced no qualifying delta within the live retention window at check
time. A second, load-bearing finding not in the earlier design doc: `substrate_
reduction_receipts` retains **success** receipts for only
`ORION_RECEIPT_RETENTION_SUCCESS_MINUTES=30` minutes (confirmed in
`services/orion-substrate-runtime/.env` and `app/settings.py`, and empirically: 6,353
total live rows spanning 2026-07-03 -> now, but all `expires_at` within 30 minutes of
their own `created_at`, so a background pruner is actively deleting older rows). This
means the "real historical series" available to this module's `precision_weighted_
salience()` is *always* bounded to a rolling ~30-minute window, for every target, not
just for whichever targets happen to be fresh today -- a structural property of the data
source, not a today-only staleness artifact.

**Scope for this patch: `substrate.node_biometrics` only.** Per the task's explicit
instruction, this module's precision computation is not silently extended to
execution/transport/chat/route -- they simply have no real, non-degenerate receipt
history to compute a meaningful variance from at build time. If/when a future live run of
one of those four lanes produces enough qualifying receipts, the replay script
(`scripts/analysis/measure_precision_weighted_salience_probe.py`) will report that
honestly rather than silently including a reducer this module was never checked against.

## 2026-07-30 update: live-wired, and the rolling-window recompute was the live bug

Both paragraphs above are historical (2026-07-21 build-time) record, kept as-is rather
than rewritten -- but two of their claims are no longer accurate as of 2026-07-30 and a
reader relying on them would be misled:

- **No longer "shadow, read-only."** `selectors.py::select_node_targets` imports and
  calls `precision_weighted_salience()`/`normalize_across_targets()` live, on every real
  `orion-attention-runtime` tick (see that module's own 2026-07-30 docstring section).
  Do not treat this module as inert.
- **No longer scoped to `node_biometrics` only.** All five `PREDICTION_ERROR_NATIVE_TARGETS`
  (`selectors.py`) are live-scored, `bus_synaptic` included (its own retirement of
  `transport` as predecessor is documented there).

**The live incident this update fixes** (found in a same-day Sentience Striving Program
officer review, 2026-07-30 -- see `orion/sentience_striving_program/README.md` §12 for
the full record): `precision_weighted_salience()` itself is fine and unchanged below --
the bug was upstream, in what fed it. The original live wiring called this function with
a freshly re-queried window from `substrate_reduction_receipts` on *every single tick*
(`AttentionRuntimeStore.load_prediction_error_history`), and that table only retains
success receipts for `ORION_RECEIPT_RETENTION_SUCCESS_MINUTES` (30 min live) -- a rolling
window, not a cumulative history. Live-confirmed 2026-07-30 (~23:30-23:42 UTC):
`node:substrate.chat` was the *only* one of the five domains with any qualifying receipts
in that window at all (the other four sat at `n_samples=0`), with exactly `n=2` real
samples, `variance` barely above `PRECISION_VARIANCE_FLOOR` (~1.56e-6),
`precision=640000`, and `confidence_score=0.1` (`n_samples / QUALIFYING_MIN_ROWS`, an
honest, correctly-computed low number). Because it was the sole real competitor,
`normalize_across_targets()`'s own documented single-target edge case (below) correctly
and unavoidably pinned its `salience_score` to `1.0` -- mathematically correct given the
edge case's own contract, but the *input* feeding it (`n_samples=2`, i.e. two receipts
that happened to survive a 30-minute prune window) was not a real basis for that
confidence. This held for a 280+-tick live streak and repeatedly won real
`FieldGoalProvenanceV1` publishes via `goal_provenance.py`'s dominance-streak producer,
each one biasing real chat-level attention scoring through
`orion/substrate/attention/goal_context.py::set_active_goal()`.

**Fix**: `n_samples`/`variance` are no longer recomputed from a pruning-window-bounded
raw list on every tick. `PrecisionEwmaBaseline` (below) is a small, persisted,
incrementally-updated running baseline -- one row per target, advanced by exactly one
real new `substrate_reduction_receipts` row at a time via `orion.bus.ewma.
compute_ewma_update` (`AttentionRuntimeStore.advance_node_prediction_error_baseline`,
`services/orion-attention-runtime/app/store.py`) -- so `observation_count` is a true,
monotonically-increasing count of every real receipt this target has *ever* incorporated,
never reset by the retention pruner. `precision_weighted_salience_from_baseline()` scores
against that persisted baseline instead of a freshly recomputed population variance.
`precision_weighted_salience(error_history: list[float])` (below) is kept exactly as it
was -- it remains the right tool for offline/replay analysis over a full historical
export where a persisted incremental baseline doesn't apply
(`scripts/analysis/measure_precision_weighted_salience_probe.py`,
`measure_candidate_a_vs_b_head_to_head.py`) -- it is simply no longer what the live tick
path calls. As defense in depth beyond this fix, `goal_provenance.py::
top_node_substrate_target` also now requires a minimum `confidence_score` before a target
is eligible to win a goal-provenance publish at all, so a future single-competitor edge
case (from a new target, or a cold-started baseline after a service/table reset) cannot
reproduce the same failure shape even if this specific baseline fix were somehow bypassed.
"""

from __future__ import annotations

from dataclasses import dataclass

from orion.bus.ewma import compute_ewma_update

# Precision = 1 / variance diverges as variance -> 0 (a target whose recent error has
# been almost perfectly constant). This is the "variance-near-zero instability risk"
# named in the tentative-plan doc's Candidate A section. Floored here at a concrete
# epsilon so precision saturates at a finite ceiling (1 / PRECISION_VARIANCE_FLOOR)
# instead of diverging to +inf or raising a ZeroDivisionError.
#
# Only used by `precision_weighted_salience()` (the raw-list, offline/replay path) --
# the live EWMA-baseline path below uses `NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE`
# instead (see that constant's own comment for why it is not simply reused here).
PRECISION_VARIANCE_FLOOR: float = 1e-6

# 2026-07-30 EWMA-baseline fix (see module docstring's "live-wired" update above).
# `alpha` reuses `execution_prediction_error`/`codebase_prediction_error`'s own 0.2
# (`orion/substrate/prediction_error.py`) rather than inventing a new number: like
# those domains, this baseline advances once per real observation (one real
# `substrate_reduction_receipts` row), not on a fixed wall-clock cadence, so the same
# per-observation smoothing constant applies for the same reason.
NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA: float = 0.2

# Domain-specific `min_variance` for `compute_ewma_update`, replacing that function's
# own default (`_MIN_VARIANCE=1e-6`, calibrated for orion-bus-mirror's real-time-gap-
# in-seconds domain) -- reused blindly once already in this exact codebase
# (`execution_prediction_error`'s first draft) and caught only after live data showed
# the borrowed constant was five orders of magnitude off that domain's real scale.
# Checked live here too rather than assumed clean by analogy (2026-07-30, `psql`
# against real `substrate_reduction_receipts`, all five `PREDICTION_ERROR_NATIVE_
# TARGETS` reducers):
#
#   reducer                        n    min     max   mean      variance
#   substrate.route_arbitration    2    0.0004  0.0004 0.0004   0.0 (exact tie)
#   substrate.chat_session         2    0.0028  0.0053 0.00405  3.12e-06
#   substrate.node_biometrics    129    0.0003  0.2618 0.04967  2.50e-03
#   substrate.bus_synaptic        45    0.0003  1.0    0.09944  2.92e-02
#   substrate.execution_trajectory 91   0.0041  1.0    0.31321  1.05e-01
#
# Unlike `codebase_prediction_error`'s git/pr/graph sub-domains (raw magnitudes on
# wildly different unit scales -- lines changed vs. PR counts vs. graph node/edge
# deltas -- each needing its own floor), all five targets here are the *output* of
# `orion/substrate/prediction_error.py`'s own instruments, which already saturate
# every one of them to the same normalized `[0, 1]` "surprise score" scale. A single
# shared floor is therefore the right shape (not five separate ones), but the module's
# existing `PRECISION_VARIANCE_FLOOR` (1e-6) was never itself checked against this
# specific value family before being reused live -- checking it now: the smallest real
# non-degenerate variance measured (chat, 3.12e-06) sits only ~5x above 1e-6, meaning
# that floor was providing almost no real headroom against exactly the kind of
# coincidentally-tiny difference between two low-volume samples this whole incident is
# about (chat's real n=2 pair differed by only ~0.0025). Raised one order of magnitude
# to 1e-5: still four-plus orders of magnitude below every domain's real non-degenerate
# variance observed above (biometrics 2.50e-03 is the next-smallest), so it never binds
# for a domain with genuine spread, while giving a low-n near-tied reading a slightly
# less extreme precision ceiling (1e5 instead of 1e6). This is a minor, disclosed
# tightening, not the load-bearing fix -- the load-bearing fix is
# `observation_count` becoming a real cumulative count (via `PrecisionEwmaBaseline`
# below) plus `goal_provenance.py`'s confidence-gated selection; a target can still
# legitimately reach `variance_floored=True` here once it has genuinely accumulated
# many real observations that happen to be near-constant, which is real information
# (Feldman & Friston 2010's "high precision" case), not a bug.
NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE: float = 1e-5


@dataclass(frozen=True)
class PrecisionWeightedSalienceResult:
    """Everything needed to inspect *why* a salience score came out the way it did --
    not just the final scalar. `variance_floored` in particular is the concrete,
    inspectable signal for the instability risk named in the design doc: a report can
    tally how often it fires rather than trusting an unbounded score blindly."""

    salience: float
    precision: float
    variance: float
    current_error: float
    n_samples: int
    variance_floored: bool


_EMPTY_RESULT = PrecisionWeightedSalienceResult(
    salience=0.0,
    precision=0.0,
    variance=0.0,
    current_error=0.0,
    n_samples=0,
    variance_floored=False,
)


def _population_variance(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def precision_weighted_salience(
    error_history: list[float],
) -> PrecisionWeightedSalienceResult:
    """Candidate A salience for one target: precision x |current prediction error|.

    ``error_history``: a target's own real historical prediction-error magnitudes,
    oldest-first (e.g. a window of real rows from `substrate_reduction_receipts` for one
    `reducer_name`, in `created_at ASC` order -- see module docstring for the exact
    query shape and the current real scope, `substrate.node_biometrics` only). The last
    element is treated as the "current" error being weighted; the *entire* list
    (including that current point) is used to estimate the process's own variance, per
    Feldman & Friston 2010's framing of precision as a property of the signal's own
    recent fluctuation, not a leave-one-out estimate of "everything before now."

    Pure function. No I/O. Never raises.

    Edge cases:
    - Empty history -> zero salience/precision, `n_samples=0`. There is no "current"
      error to weight in the first place.
    - Single-sample history -> population variance of one point is exactly 0.0 by
      definition (no fluctuation observed yet). This is NOT treated as a separate
      "undefined" case from near-zero variance -- one real observation is real data at
      the smallest possible sample size, not missing data -- so it floors the same way
      any other near-zero-variance history does (see below), with `n_samples=1`
      reported so a caller/report can weight its confidence in the result accordingly.
    - Near-zero variance (a target whose error has been almost constant across the
      window) -> precision would otherwise diverge; floored at
      ``1 / PRECISION_VARIANCE_FLOOR`` (large but finite), and `variance_floored=True`
      is reported explicitly so a caller can flag the instability risk rather than
      silently trust an extreme score.
    """
    n = len(error_history)
    if n == 0:
        return _EMPTY_RESULT

    values = [float(v) for v in error_history]
    current_error = values[-1]
    variance = _population_variance(values)
    variance_floored = variance < PRECISION_VARIANCE_FLOOR
    effective_variance = max(variance, PRECISION_VARIANCE_FLOOR)
    precision = 1.0 / effective_variance
    salience = precision * abs(current_error)

    return PrecisionWeightedSalienceResult(
        salience=salience,
        precision=precision,
        variance=variance,
        current_error=current_error,
        n_samples=n,
        variance_floored=variance_floored,
    )


def normalize_across_targets(raw_scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize raw ``precision_weighted_salience().salience`` values across a
    set of *competing* targets into ``[0, 1]``.

    **Why this exists (found in review, 2026-07-22, not part of the original build):**
    ``precision_weighted_salience()``'s raw output is unbounded (precision saturates at
    ``1 / PRECISION_VARIANCE_FLOOR = 1e6``, so raw salience can reach the tens of
    thousands -- confirmed live: the real biometrics replay produced a range of
    0.00-57,100.00). The existing system this candidate is meant to replace,
    `FieldAttentionTargetV1.salience_score`, is schema-constrained to ``[0, 1]``
    (`orion/schemas/field_attention_frame.py`) -- the raw, unnormalized output is not a
    valid drop-in for that field. Worse than a schema mismatch: without normalization,
    cross-target comparison (the entire point of a *competition* score) is dominated by
    each target's own historical variance scale, not by how surprising its current error
    actually is -- a target that has always been quiet will structurally out-rank a
    noisier-but-currently-alarming one purely from scale, independent of real relative
    importance.

    **Min-max, not softmax, and why**: a softmax-style normalization would need a
    temperature/scale hyperparameter to decide how sharply it compresses the
    distribution -- exactly the kind of hand-guessed, uncalibrated weight this whole
    program exists to eliminate (see charter §7 / the tentative-plan doc's "Finding: the
    disease is systemic, not one bug"). Min-max has no free parameter: it maps the
    lowest-scoring real competitor to 0.0 and the highest to 1.0, preserving relative
    rank exactly, with nothing to calibrate.

    **This is a per-tick, per-competing-set operation, not part of
    `precision_weighted_salience()` itself** -- that function has no knowledge of what
    else is competing this tick (it only sees one target's own history), so it cannot
    self-normalize. Callers must compute raw scores for every real target competing in
    the same tick, then pass the whole set here before comparing across targets.
    Comparing two *raw* `precision_weighted_salience().salience` values from different
    targets without this step is the exact mistake this function exists to prevent.

    Edge cases:
    - Empty input -> ``{}``.
    - All-equal raw scores (including a single-target input) -> every target gets
      ``1.0``, not an arbitrary ``0.0`` floor. There is no real basis to differentiate
      "tied for most salient" from "least salient" when every real competitor scored
      identically -- flooring to 0.0 would misrepresent a tie as "nothing here matters,"
      which is not what the data says.
    """
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    lo, hi = min(values), max(values)
    if (hi - lo) < 1e-12:
        return {target_id: 1.0 for target_id in raw_scores}
    span = hi - lo
    return {target_id: (v - lo) / span for target_id, v in raw_scores.items()}


# -- Persisted EWMA baseline (2026-07-30 fix) ----------------------------------
# See module docstring's "live-wired" update section for the incident this section
# exists to fix. `precision_weighted_salience()`/`normalize_across_targets()` above
# are UNCHANGED -- this is a new, parallel input path for the live tick, not a
# rewrite of the existing pure functions.


@dataclass(frozen=True)
class PrecisionEwmaBaseline:
    """One `node:substrate.*` target's persisted, incrementally-updated running
    prediction-error baseline -- the live-tick replacement for re-querying a raw
    history list from `substrate_reduction_receipts` every tick (see module
    docstring). Threaded explicitly by the caller and read from / written back to a
    real persisted row (`AttentionRuntimeStore.advance_node_prediction_error_baseline`,
    `substrate_node_prediction_error_baseline` table) -- the same explicit-baseline-
    threading shape `orion/substrate/prediction_error.py`'s `_DomainEwmaBaseline`/
    `CodebaseMassBaseline` use, applied here to a Postgres-row-backed target instead
    of a projection-field-backed one (no `FieldStateV1`-adjacent projection schema
    exists for this data; a small dedicated table was the minimal seam, see the
    manual-migration file this table lives in for the exact DDL).

    `observation_count` is the field that actually fixes the live incident: unlike
    the old `n_samples = len(raw_history_list)`, this is a true monotonically
    increasing count of every real receipt this target has EVER incorporated,
    surviving `substrate_reduction_receipts`' ~30-minute retention prune indefinitely
    (the prune only ever deletes the raw receipt row after this baseline has already
    absorbed its value -- deleting the source data does not undo an update already
    folded into `ewma`/`variance`/`observation_count`).

    `last_value` is the most recently incorporated real observation -- what
    `precision_weighted_salience_from_baseline()` treats as "the current error being
    weighted," the same role `error_history[-1]` played in the raw-list path. `None`
    only for the true cold-start case (`observation_count == 0`, no real receipt has
    ever been incorporated for this target)."""

    ewma: float = 0.0
    variance: float = 0.0
    observation_count: int = 0
    last_value: float | None = None


def advance_precision_baseline(
    baseline: PrecisionEwmaBaseline,
    new_values: list[float],
    *,
    alpha: float,
    min_variance: float,
) -> PrecisionEwmaBaseline:
    """Fold zero or more real NEW observations (oldest-first -- e.g. every real
    `substrate_reduction_receipts` row that landed since this baseline was last
    advanced) into a persisted EWMA baseline, one at a time, via
    `orion.bus.ewma.compute_ewma_update`. Pure function, no I/O -- the caller
    (`AttentionRuntimeStore.advance_node_prediction_error_baseline`) is responsible
    for fetching `new_values` and persisting the returned baseline.

    Returns ``baseline`` unchanged (the identical object, not merely an equal one)
    when ``new_values`` is empty -- "nothing new landed at this exact poll instant"
    must never be treated as "this target has no real history," which is the exact
    failure this whole fix exists to correct. A caller can safely skip persisting
    when the return value ``is`` the input (no real advance happened this tick).
    """
    if not new_values:
        return baseline
    result = baseline
    for value in new_values:
        update = compute_ewma_update(
            prev_ewma=result.ewma,
            prev_variance=result.variance,
            prev_count=result.observation_count,
            value=value,
            alpha=alpha,
            min_variance=min_variance,
        )
        result = PrecisionEwmaBaseline(
            ewma=update.ewma,
            variance=update.variance,
            observation_count=result.observation_count + 1,
            last_value=value,
        )
    return result


def precision_weighted_salience_from_baseline(
    baseline: PrecisionEwmaBaseline,
    *,
    min_variance: float,
) -> PrecisionWeightedSalienceResult:
    """Candidate A salience computed from a persisted, incrementally-updated EWMA
    baseline (`advance_precision_baseline`, above) instead of a freshly recomputed
    population variance over a rolling-window history list
    (`precision_weighted_salience`, above -- kept for offline/replay analysis, see
    module docstring). This is the function the live tick path
    (`selectors.py::select_node_targets`) actually calls as of 2026-07-30.

    ``variance``/``precision``/``salience`` in the returned result use the EWMA's own
    running variance estimate, not a population variance recomputed from scratch.
    ``n_samples`` is ``baseline.observation_count`` -- a real cumulative count that
    survives `substrate_reduction_receipts` retention pruning indefinitely, unlike the
    raw-list path's ``len(error_history)`` (bounded to whatever currently fits inside
    the ~30-minute retention window). ``current_error`` is ``baseline.last_value``.

    Edge case: ``observation_count == 0`` (no real receipt has ever been incorporated
    for this target -- true cold start, not merely "nothing new this tick") returns
    the same zero-everything result as ``precision_weighted_salience([])`` -- "no
    data" and "confidently calm" remain different claims, per that function's own
    documented contract.
    """
    if baseline.observation_count == 0 or baseline.last_value is None:
        return _EMPTY_RESULT

    variance_floored = baseline.variance < min_variance
    effective_variance = max(baseline.variance, min_variance)
    precision = 1.0 / effective_variance
    current_error = baseline.last_value
    salience = precision * abs(current_error)

    return PrecisionWeightedSalienceResult(
        salience=salience,
        precision=precision,
        variance=baseline.variance,
        current_error=current_error,
        n_samples=baseline.observation_count,
        variance_floored=variance_floored,
    )


def cross_domain_variance_floor(
    baselines: dict[str, PrecisionEwmaBaseline],
    target_id: str,
    *,
    min_variance: float,
) -> float:
    """Per-tick, per-target variance floor derived from this tick's OTHER real
    competing domains, replacing a single hand-picked absolute constant applied
    uniformly to every target regardless of that target's own natural error scale.

    **The live incident this fixes (found 2026-08-20, Sentience Striving Program
    item 4 investigation):** ``node:substrate.route``'s prediction-error signal
    (``route_prediction_error()``, a categorical decision-mismatch rate) is
    genuinely, organically near-constant -- confirmed live: the 30 most recent
    real receipts at investigation time were all exactly ``0.0003``, and the
    persisted EWMA baseline's ``variance`` had underflowed to ``~2.9e-39``. That is
    real data, not a decay artifact (CLAUDE.md's metric-quality-gate #4
    distinction). But feeding it through the single global
    ``NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE = 1e-5`` floor gave route a
    precision of ``1/1e-5 = 100,000`` -- roughly 1,270x-19,300x every other native
    domain's own *organic* variance-derived precision (confirmed live same day:
    biometrics/bus_synaptic/execution/chat variances ranged 0.0127-0.193,
    precision ~5-79). Since ``normalize_across_targets()`` is min-max
    (rank-preserving, not magnitude-correcting -- see that function's own
    docstring), route's raw score was the tick's maximum essentially every tick,
    so it won goal-provenance dominance by construction -- not because its
    current error was genuinely more surprising, but because its variance floor
    was orders of magnitude looser relative to its own error scale than everyone
    else's. This is the confirmed root cause of the ``voluntary_override`` branch
    structurally never firing (Sentience Striving Program README, item 4).

    **Fix, and why this shape specifically:** a single global floor cannot be
    "right" for every domain at once when domains have genuinely different
    organic noise scales by design (a categorical decision-mismatch rate vs. a
    continuous physiological/execution magnitude). Rather than hand-pick a new
    per-domain constant -- which would repeat the same mistake with more
    constants instead of one, calibrated numbers that do not transfer across
    domains -- this floor is read live from the OTHER real domains competing in
    the *same tick*: the median real ``variance`` among every other target with
    ``observation_count > 0`` this tick. A target cannot claim more precision
    than the median of what its real, currently-live siblings are actually
    showing. No new tunable is introduced (matching ``normalize_across_
    targets()``'s own "no free parameter" discipline, cited above) -- this is
    min-max's same principle applied one step earlier, to the floor instead of
    the final score.

    **Disclosed trade-off (code review, 2026-08-20):** this compresses genuine
    precision differentiation between domains, not just the pathological
    outlier -- any target whose real variance sits below the sibling median
    (in the live snapshot above, ``bus_synaptic``/``biometrics``) gets floored
    up too, losing some of the precision advantage it legitimately earned by
    being quieter. That is an accepted, intended consequence of "no domain can
    claim more precision than its real siblings currently show," not a bug.

    ``target_id`` is excluded from its own floor computation -- a domain's floor
    must come from its *competitors*, not partly from itself, or a target could
    (marginally, but non-zero) suppress its own floor by being unusually quiet.

    **Correlated-degeneracy guard (code review, 2026-08-20):** a plain median
    over the other siblings degrades back to the exact pathology this function
    exists to fix if a *majority* of those siblings are themselves at/below
    ``min_variance`` at the same tick (e.g. 3 of 4 siblings underflow
    simultaneously -- a real possibility, not hypothetical: the same
    near-constant-signal mechanism that flattened ``route`` can flatten any
    domain during a correlated quiet period). In that case the median itself
    would be degenerate, and ``max(min_variance, degenerate_median)`` would
    just reproduce the original ``100,000``-precision bug for every affected
    domain at once. Guarded explicitly: real (non-degenerate) siblings must
    form a strict majority of the real competitor set, or this falls back to
    ``min_variance`` instead of trusting a degenerate-majority median.

    Falls back to ``min_variance`` (the original global constant, unchanged and
    still the ultimate backstop) when fewer than one other real competitor
    exists this tick (e.g. cold start, every other target has zero
    observations) or when the majority-non-degenerate guard above fails. That
    constant is not being removed, only demoted from "the only floor" to "the
    floor of last resort when there is no trustworthy live sibling data to
    derive a better one from."
    """
    other_variances = sorted(
        baseline.variance
        for other_id, baseline in baselines.items()
        if other_id != target_id and baseline.observation_count > 0
    )
    if not other_variances:
        return min_variance
    non_degenerate_count = sum(1 for v in other_variances if v > min_variance)
    degenerate_count = len(other_variances) - non_degenerate_count
    if non_degenerate_count <= degenerate_count:
        # Real siblings are not a trustworthy majority this tick -- see
        # "Correlated-degeneracy guard" above. Do not derive a floor from
        # mostly-degenerate data; fall back to the global constant.
        return min_variance
    n = len(other_variances)
    mid = n // 2
    median = (
        other_variances[mid]
        if n % 2 == 1
        else (other_variances[mid - 1] + other_variances[mid]) / 2.0
    )
    return max(min_variance, median)
