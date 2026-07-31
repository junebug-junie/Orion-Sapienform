"""Generic Borda-count rank-aggregation (de Borda 1770) -- combine N
independently-sourced real scorers' rankings into one total ordering,
without ever guessing/calibrating a cross-scorer exchange rate.

Extracted 2026-07-31 from `orion.attention.field_attention.
candidate_society_of_mind` (Candidate B, PR #1484/#1488), where this
machinery was first proven for Layer 5 field attention's Global Workspace
Theory / Society-of-Mind rank-aggregation (Baars 1988, Dehaene 2014, Minsky
1986). That module still owns the field-attention-specific scorers
(magnitude/novelty/dwell) and imports the pure aggregation primitives below
rather than redefining them.

Second real consumer as of this extraction:
`orion.substrate.attention.salience` (chat-level/open-loop attention,
`orion/substrate/attention/`), which reuses this exact aggregation --
same theory anchor, different subsystem -- to combine `evidence_strength`/
`evidence_breadth` into a coalition-strength score, replacing the killed
hand-picked `SEED_WEIGHTS` linear blend. See
`orion/sentience_striving_program/README.md`'s 2026-07-31 entry for the
full kill/replace rationale.

This module has no domain-specific dependencies (no `FieldAttentionFrameV1`,
no `orion.substrate` import) by design -- it is pure data-in/data-out so any
future third rank-aggregation consumer can reuse it without inheriting
either subsystem's own import weight.
"""

from __future__ import annotations

from dataclasses import dataclass

_NEG_INF = float("-inf")


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
    named in this module's own history -- see the top docstring). Borda
    count needs no such rate -- each scorer only orders targets on its own
    internal scale, and the combination step operates purely on rank
    position, which is commensurable across scorers by construction (rank 1
    always means "this scorer's own best pick," regardless of what numeric
    scale produced it). Why not a Condorcet method (e.g. pairwise
    majority): Condorcet methods can produce cycles (no single winner) with
    only 3 voters and can leave ties unresolved without an additional
    tiebreak rule of their own; Borda always produces one complete,
    total-ordered ranking from any input, which is what a per-tick "who
    wins this competition" question needs. Both are real, standard, citable
    social-choice methods -- Borda is chosen for its guaranteed-total-order
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
