"""Candidate B: rank-aggregated independent scorers (Global Workspace Theory /
Society-of-Mind -- Baars 1988 "A Cognitive Theory of Consciousness", Dehaene
2014 "Consciousness and the Brain", Minsky 1986 "The Society of Mind").

**Shadow candidate. Read-only. Not wired to any live consumer.** Per
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-
tentative-plan.md`'s "Candidate B" section: this is one of two parallel,
structurally different replacement candidates for
`orion/attention/field_attention/scoring.py::compute_salience()`'s hand-typed
linear weighted sum (`pressure*0.45 + novelty*0.20 + urgency*0.25 +
confidence*0.10`, zero citation, zero calibration -- see that doc's "Finding:
the disease is systemic, not one bug"). Candidate A (Friston-style
precision-weighted prediction error) is a separate, parallel effort; this
module does not build or depend on it.

**Structural claim**: instead of one formula with N weights to hand-tune,
three independently-sourced real scorers each rank targets on their own
terms, and a citable social-choice rank-aggregation method (Borda count,
de Borda 1770) combines the rankings. No cross-scorer weight is ever
guessed or calibrated -- each scorer only has to be internally sound, not
commensurable with the others on a shared numeric scale.

The three scorers are exactly the "smallest probe" triad named in
`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-
design.md`'s blue-sky extension 2 ("Society-of-Mind competition"):
magnitude, novelty, unresolved-duration (dwell). All three reuse existing,
already-shipped, already-live real signal sources -- this module adds no new
producer, only a new *combination* mechanism over data that already exists:

1. `magnitude_scorer` -- passthrough/validation over real prediction-error
   magnitudes already computed by `orion.substrate.prediction_error`'s five
   `*_prediction_error()` functions (execution/transport/biometrics/chat/
   route), each already a real [0,1] surprise score.
2. `novelty_scorer` -- thin wrapper around the already-live, already-real
   `orion.attention.field_attention.scoring.novelty_for_target()`, reused
   here in a genuinely different structural role: an independent
   rank-aggregation voter, not a weighted-summed term inside
   `compute_salience()`.
3. `dwell_scorer` -- real duration-of-unresolved-coalition signal from
   `AttentionBroadcastProjectionV1.dwell_ticks`/`attended_node_ids`
   (`orion/substrate/attention_broadcast.py`, already live, already fixed of
   a prior loop-scoped-vs-global-scalar bug per program history).

**Real live-data finding, load-bearing (2026-07-21)**: as of this build, the magnitude scorer's real target universe
(`node:substrate.{biometrics,execution,transport,chat,route}`, written by
`services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node`)
and the novelty scorer's real target universe (`node:athena`/`node:atlas`/
`node:circe` plus `capability:*`/`field:recent_perturbations`, drawn from
`FieldStateV1.node_vectors` as ingested by `orion-field-digester`) do not
currently overlap at all in live data -- confirmed by direct query, not
assumed. This does not make the combiner below wrong; it means real
three-way competition on the *same* target requires either (a) waiting for
target-universe overlap to occur naturally (e.g. a `substrate.*` domain node
entering the dwell coalition, which has been observed, rarely) or (b) a
follow-up patch teaching `orion-field-digester` to ingest the
prediction-error domain nodes into `FieldStateV1.node_vectors` too -- out of
scope for this shadow-only patch, named as a real follow-up.

Non-goals of this module:
- Does not modify `orion/attention/field_attention/scoring.py` or
  `selectors.py` (the live formula).
- Does not wire into `compute_salience()`, `select_system_targets`,
  `orion/self_state/builder.py`, `LinearSalienceCombiner`/`seed-v1`, or any
  live consumer.
- Does not invent a fourth scorer or a hand-tuned aggregation formula -- the
  whole point of this candidate is to have zero cross-factor weights.

## Live-data sanity check status (CLAUDE.md §0A metric-quality-gate step 4) -- disclosed honestly, 2026-07-22

An earlier version of this docstring claimed a "companion replay script" existed for
this module. It did not -- that was a false, unverified citation, caught in review, not
a stale reference to something that once existed. Corrected here rather than silently
dropped, so the gap is visible instead of implied-covered.

**What real data has actually been run through this module, as of this patch**:
`magnitude_scorer()` only, via `scripts/analysis/measure_society_of_mind_magnitude_probe.py`
(new, this patch) -- real `substrate_reduction_receipts` history, reusing Candidate A's
already-proven Postgres-loading code rather than duplicating it. See that script's own
report for real numbers.

**What has NOT been run against real data**: `novelty_scorer()` (needs real
`substrate_attention_frames` history), `dwell_scorer()` (needs real
`substrate_coalition_dwell_log` history), and `aggregate_borda()`'s full three-scorer
combination -- i.e. the plan doc's own Candidate B acceptance check ("do the three
scorers ever disagree, and is that disagreement itself informative -- report real
examples, not a summary claim") is **not yet met**. Only synthetic-fixture unit tests
exist for those three functions. This is a real, disclosed gap, not a completed
capability -- treat any claim that this candidate is "live-data validated" as false
until a real three-scorer replay exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from orion.attention.field_attention.scoring import clamp01, novelty_for_target
from orion.schemas.field_attention_frame import FieldAttentionFrameV1

# services/orion-substrate-runtime/app/worker.py::_write_prediction_error_node
# writes real FalkorDB nodes at this exact id convention for all five
# producer domains. Exposed here so the replay script and any future caller
# share one source of truth for the convention instead of re-deriving it.
PREDICTION_ERROR_TARGET_PREFIX = "node:substrate."

# Dwell-ticks normalization cap for `dwell_scorer`'s [0,1] output. 30s tick
# cadence (`orion-substrate-runtime`'s broadcast tick, confirmed live via
# `substrate_coalition_dwell_log` row spacing 2026-07-21) * 120 ticks = ~1h of
# continuous unresolved dwell before this scorer saturates to 1.0. Chosen as
# a round, documented number rather than fit to any particular observed
# distribution -- real dwell_ticks values observed live on 2026-07-21 ranged
# from single digits up to 4900+ (a multi-day-old un-reset coalition), so
# *some* cap is required for this scorer's output to stay informative rather
# than dominated by however long the process has been running since its last
# restart. Callers needing raw duration for their own reporting should read
# `dwell_ticks` directly rather than back-deriving it from this score.
DWELL_TICKS_SATURATION = 120

_NEG_INF = float("-inf")


def magnitude_scorer(prediction_errors: dict[str, float]) -> dict[str, float]:
    """Real prediction-error magnitude per target -- passthrough, not a
    re-derivation.

    Metric-quality-gate note (CLAUDE.md Sec 0A): `orion.substrate.
    prediction_error`'s five `*_prediction_error()` functions already
    produce a real [0,1] surprise score per producer domain (four via
    `min(1.0, mean(|delta|) / 0.30)`, `route_prediction_error` via a
    categorical mismatch rate -- see that module's own docstrings for why
    the two shapes differ). This scorer performs no additional weighting or
    transform of its own -- it *is* the magnitude vote, taken as-is. The
    only work done here is defensive: clamp to [0,1] and drop non-finite
    values, since the caller may be assembling this dict from several
    real-but-independently-sourced upstream calls (one per producer domain)
    and a single malformed value must not poison the whole tick's ranking.

    Callers build `prediction_errors` themselves, one key per producer
    domain, e.g. `{"node:substrate.biometrics": biometrics_prediction_error
    (prev, curr), ...}` -- see `PREDICTION_ERROR_TARGET_PREFIX` and the
    companion replay script for the real historical data source (this
    module does no I/O; `substrate_reduction_receipts` is the only place
    real history for these values exists, per Candidate A's own established
    finding, reused here rather than re-derived).
    """
    out: dict[str, float] = {}
    for target_id, value in prediction_errors.items():
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if fv != fv or fv in (float("inf"), float("-inf")):  # NaN/inf guard
            continue
        out[target_id] = clamp01(fv)
    return out


def novelty_scorer(
    target_ids: list[str],
    current_salience: dict[str, float],
    previous_frame: FieldAttentionFrameV1 | None,
) -> dict[str, float]:
    """Real novelty vote per target -- thin wrapper around the already-live
    `novelty_for_target()`.

    Metric-quality-gate note: this is the *same* real signal RPT/Lamme
    already contributes to the live (broken) `compute_salience()` weighted
    sum -- reused here in a genuinely different structural role (an
    independent rank-aggregation voter competing on its own terms, not a
    0.20-weighted addend inside one formula). Per this candidate's own
    design brief: "a genuinely different structural role for the same real
    signal." Requires `current_salience[target_id]` -- the target's
    *pre-novelty* salience for the current tick (mirrors
    `selectors.py::_build_target`'s own two-pass compute: pressure/urgency/
    confidence first, then novelty diffs that pre-novelty salience against
    the same target's *prior* frame -- computing novelty from a
    salience that already includes this tick's own novelty contribution
    would be circular). A `target_id` missing from `current_salience`
    defaults to 0.0 (no observed pressure this tick), the same convention
    `measure_emergent_clustering_probe.py::extract_target_salience_map`
    already uses for tick-level absence.
    """
    return {
        target_id: novelty_for_target(
            target_id, current_salience.get(target_id, 0.0), previous_frame
        )
        for target_id in target_ids
    }


def dwell_scorer(
    attended_node_ids: list[str],
    dwell_ticks: int,
    *,
    saturation_ticks: int = DWELL_TICKS_SATURATION,
) -> dict[str, float]:
    """Real unresolved-duration vote: every target currently inside the
    active broadcast coalition gets the same normalized `dwell_ticks` score;
    everything else gets no vote at all (empty dict entry, not 0.0 --
    important for the Borda combiner's partial-ballot handling below: a
    target this scorer has no opinion about must be treated as "unranked",
    not "actively scored lowest for a real reason").

    Metric-quality-gate note, load-bearing (2026-07-21 live query against
    `substrate_coalition_dwell_log`): `attended_node_ids` was the empty list
    in 2837 of 2840 real rows over the most recent 24h window (99.9%) --
    the coalition being "dwelt on" is, almost always, the empty coalition
    (no `selected_action`/`open_loop_id` resolved that tick), not a real
    target. `dwell_ticks` itself is real and non-degenerate (a genuine,
    live, monotonically-increasing-with-resets counter, observed ranging
    from single digits to 4900+ across the same window), but the *target
    attribution* dimension is almost entirely absent in current live data.
    This is disclosed here plainly rather than smoothed over: this scorer
    is real and correctly wired, but in today's live substrate it will
    almost never cast a real per-target vote. The 2837/2840 figure above is
    from a one-off live query against `substrate_coalition_dwell_log`, not
    from a replay script -- no script currently measures this dwell-emptiness
    fraction (unlike `magnitude_scorer()`, which
    `scripts/analysis/measure_society_of_mind_magnitude_probe.py` does cover
    against real data; see that scorer's own docstring above and the module's
    top-level "Live-data sanity check status" section).

    `attended_node_ids` is de-duplicated (a target should not get inflated
    influence just because `AttentionBroadcastProjectionV1.attended_node_ids`
    happens to list it more than once) while preserving first-seen order,
    though the resulting score is uniform across all attended targets
    regardless of order -- there is exactly one coalition-wide dwell
    duration, not a per-target one, per the real schema
    (`AttentionBroadcastProjectionV1.dwell_ticks` is a single int, not a
    per-node mapping).
    """
    if dwell_ticks <= 0 or not attended_node_ids:
        return {}
    cap = max(1, int(saturation_ticks))
    score = clamp01(float(dwell_ticks) / float(cap))
    deduped = list(dict.fromkeys(attended_node_ids))
    return {target_id: score for target_id in deduped}


@dataclass(frozen=True)
class BordaResult:
    """Output of one tick's rank-aggregation over `universe`."""

    universe: tuple[str, ...]
    totals: dict[str, float]
    ranking: tuple[str, ...]
    winner: str | None
    per_scorer_top1: dict[str, str | None]
    disagreement: bool


def scorer_top1(scores: dict[str, float]) -> str | None:
    """The single highest-scored target for one scorer's own ballot.
    Deterministic tie-break: highest score, then alphabetical target_id --
    same convention `measure_emergent_clustering_probe.py::top1_winner`
    already uses, reused here rather than re-invented."""
    if not scores:
        return None
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _borda_points_for_scorer(scores: dict[str, float], universe: list[str]) -> dict[str, float]:
    """Points one scorer's ballot assigns to every target in `universe`.

    Classic Borda count (de Borda 1770): for N candidates, the worst-ranked
    gets 0 points and the best-ranked gets N-1, with every intervening rank
    getting one more point than the rank below it.

    Two explicit, documented deviations from the textbook version, both
    real and necessary here, neither hand-tuned:

    1. **Tied real scores share the average of their tied positions'
       points**, rather than an arbitrary tiebreak silently favoring one of
       two identically-scored targets. Standard Borda tie-handling
       (sometimes called "average rank" or "fractional Borda"), not
       invented for this module.
    2. **A target absent from `scores` (this scorer has no real evidence
       about it) is treated as tied for this scorer's own last place** --
       ranked at or below every target the scorer actually scored, never
       above. This is the standard treatment for partial ballots in
       rank-aggregation (a voter who does not rank a candidate is not
       silently excluded from affecting that candidate's total, nor is the
       candidate given a charitable average-of-everyone-else score) --
       chosen specifically because "no evidence this target matters" should
       never accidentally out-rank "measured, real evidence this target
       matters less than something else." Implemented by sorting absent
       targets to `-inf`, which then naturally tie-groups with each other
       (not with any real-scored target) and shares the lowest available
       points via deviation 1 above -- no separate code path needed.
    """
    n = len(universe)
    if n == 0:
        return {}
    if n == 1:
        return {universe[0]: 0.0}
    keyed = [(t, scores[t] if t in scores else _NEG_INF) for t in universe]
    ordered = sorted(keyed, key=lambda kv: (kv[1], kv[0]))
    points: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg_points = sum(range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            points[ordered[k][0]] = avg_points
        i = j + 1
    return points


def aggregate_borda(
    scorer_scores: dict[str, dict[str, float]],
    universe: list[str] | None = None,
) -> BordaResult:
    """Combine N independent scorers' rankings into one Borda-count total.

    Why Borda count and not a weighted sum: a weighted sum requires
    guessing/calibrating a cross-scorer exchange rate (exactly the disease
    named in this module's own top docstring -- `compute_salience()`'s
    uncalibrated `0.45/0.20/0.25/0.10` split). Borda count needs no such
    rate -- each scorer only orders targets on its own internal scale, and
    the combination step operates purely on rank position, which is
    commensurable across scorers by construction (rank 1 always means
    "this scorer's own best pick," regardless of what numeric scale
    produced it). Why not a Condorcet method (e.g. pairwise majority):
    Condorcet methods can produce cycles (no single winner) with only 3
    voters and can leave ties unresolved without an additional tiebreak
    rule of their own; Borda always produces one complete, total-ordered
    ranking from any input, which is what a per-tick "who wins this
    competition" question needs. Both are real, standard, citable social-
    choice methods -- Borda is chosen for its guaranteed-total-order
    property, not because it was the only option considered.

    `universe` defaults to the union of every target any scorer actually
    scored this tick. Passing it explicitly lets a caller widen the
    universe to include targets not present in any scorer for this specific
    tick (e.g. to compare against a fixed real target list across many
    ticks) -- any such target gets `_NEG_INF`-tied-last treatment from every
    scorer, i.e. total score 0.0, same as any other fully-unscored target.

    `disagreement` is true iff two or more scorers' own top-1 picks (see
    `scorer_top1`) differ from each other. A scorer that scored nothing this
    tick (empty dict, `scorer_top1` returns None) does not count toward
    disagreement either way -- silence is not itself a vote.
    """
    if universe is None:
        resolved_universe = sorted({t for s in scorer_scores.values() for t in s})
    else:
        resolved_universe = sorted(set(universe))

    totals: dict[str, float] = {t: 0.0 for t in resolved_universe}
    per_scorer_top1: dict[str, str | None] = {}
    for name, scores in scorer_scores.items():
        per_scorer_top1[name] = scorer_top1(scores)
        for t, p in _borda_points_for_scorer(scores, resolved_universe).items():
            totals[t] += p

    ranking = tuple(sorted(resolved_universe, key=lambda t: (-totals[t], t)))
    winner = ranking[0] if ranking else None
    real_top1s = {v for v in per_scorer_top1.values() if v is not None}
    disagreement = len(real_top1s) > 1

    return BordaResult(
        universe=tuple(resolved_universe),
        totals=totals,
        ranking=ranking,
        winner=winner,
        per_scorer_top1=per_scorer_top1,
        disagreement=disagreement,
    )
