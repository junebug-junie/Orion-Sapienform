"""Shared evidence-derived salience -- the single source of salience truth.

Pure and import-light (schemas + common + orion.attention.rank_aggregation,
which is itself dependency-free -- no `orion.substrate`, no `requests`) so
the thin `orion-thought` service could import it without dragging the graph
engine (it deliberately does not -- see `services/orion-thought/app/
reverie.py`'s own docstring for why it reimplements small helpers locally
instead). Consumed by coalition selection (`scoring.score_loop`) AND
reverie (`derive_salience`, indirectly via `loop.salience`).

**2026-07-31: killed and replaced (CLAUDE.md Sec 0A, "kill means kill, no
fallback to the thing being killed").** This module used to be a
deterministic 7-feature vector scored by a hand-seeded linear combiner
(`SEED_WEIGHTS`: `evidence_strength 0.30, novelty_vs_known 0.20, recency
0.13, recurrence 0.15, evidence_breadth 0.12, dwell 0.10, habituation
-0.35`), documented in its own prior version as intended to become
"learned" via `scripts/refit_salience_weights.py` -- a pass that never ran
(`WEIGHTS_VERSION` stayed `"seed-v1"` its entire life). This is the same
disease `orion/attention/field_attention/scoring.py::compute_salience()`
was killed for in PR #1484 (an un-calibrated weight blend standing in for
theory), in a different subsystem (chat-level/open-loop attention instead
of Layer 5 field attention).

**What replaced it.** Two of the seven terms were already real, not
hand-picked: `evidence_strength` (the strongest single detector's own real
activation) and `evidence_breadth` (how many independent
detectors/evidence_refs corroborate this loop). These map directly onto
Global Workspace Theory / Society-of-Mind coalition formation (Baars 1988,
Dehaene 2014's "ignition" model) -- the exact theory anchor already live
for Layer 5's Candidate B (`orion.attention.field_attention.
candidate_society_of_mind`). `borda_coalition_salience()` below combines
them via Borda rank-aggregation (de Borda 1770) across the real competing
set of loops scored together in one tick (see `scoring.build_open_loops`),
reusing that module's already-proven `orion.attention.rank_aggregation`
machinery rather than a new hand-tuned weight.

**What was NOT replaced, on purpose.** `recency`, `recurrence`, `dwell`,
`novelty_vs_known`, `habituation` had no comparable real theory anchor
(recency's 6h half-life and dwell/habituation's blend weights were picked,
not measured; novelty_vs_known collapsed to a flat 0.15 for any known
loop). Per the Metric Quality Gate's step 3 ("if there is no real theory,
do not build a detector for it yet -- say so and stop"), these are killed
with nothing put back -- not reinvented, not kept as always-zero fields.

**Named, disclosed gap.** `habituation` was, as far as this investigation
found, the only automatic repeat-suppression mechanism in this scoring
path. `substrate_reverie_refractory` (the Resolve/Dismiss flow) only
suppresses a loop *after* a human explicitly acts on it -- it does not run
automatically. A loop that keeps generating strong evidence but that nobody
has ever explicitly resolved/dismissed can now re-win coalition attention
indefinitely, with nothing damping it. See
`orion/sentience_striving_program/README.md`'s 2026-07-31 entry.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from orion.attention.rank_aggregation import aggregate_borda
from orion.schemas.attention_frame import AttentionSignalV1, SalienceFeaturesV1
from orion.substrate.attention.common import bounded

logger = logging.getLogger("orion.substrate.attention.salience")

_TRUTHY = {"1", "true", "yes", "on"}

# Real, still-live env key. As of this patch it no longer selects between
# two salience formulas -- there is only one. Its one remaining real
# dependent is services/orion-thought/app/reverie.py's trace-publish gate
# (`settings.attention_salience_v2_enabled`, a SEPARATELY defined pydantic
# field -- that thin service deliberately never imports this module, see
# reverie.py's own docstring). `salience_v2_enabled()` itself has no call
# site left anywhere in `orion.substrate` after this patch (score_loop() and
# _apply_voluntary_attention()'s checks were both removed -- see their own
# comments for why each check's stated rationale evaporated once there was
# only one formula). Kept defined here, unused internally, ONLY so
# `scripts/analysis/measure_chat_attention_ground_truth_gap.py`'s already-
# published historical report keeps importing successfully (it reads this
# exact flag/truthy-set to describe what was live on 2026-07-30).
SALIENCE_V2_FLAG = "ORION_ATTENTION_SALIENCE_V2_ENABLED"

# Provenance string for AttentionSalienceTraceV1/PendingAttentionCardV1/
# AttentionLoopOutcomeV1's `weights_version` field. Replaces killed
# `"seed-v1"` -- there is no weight dict to version anymore (Borda has no
# free cross-scorer parameters), but the field stays a real "which scoring
# regime produced this row" marker.
WEIGHTS_VERSION = "gwt-coalition-v1"

# Distinct detectors/refs to saturate breadth (unchanged, real -- see
# evidence_breadth_for()).
BREADTH_NORM = 4.0


def salience_v2_enabled() -> bool:
    return str(os.getenv(SALIENCE_V2_FLAG, "false")).strip().lower() in _TRUTHY


def evidence_strength_for(signals: Sequence[AttentionSignalV1]) -> float:
    """The strongest single detector's own real activation for this loop."""
    strengths = [float(s.salience) * float(s.confidence) for s in signals]
    return bounded(max(strengths)) if strengths else 0.0


def evidence_breadth_for(signals: Sequence[AttentionSignalV1]) -> float:
    """How many independent real sources/evidence_refs corroborate this
    loop, normalized by BREADTH_NORM."""
    distinct: set[str] = set()
    for s in signals:
        distinct.add(str(s.source))
        for ref in (s.evidence_refs or []):
            distinct.add(str(ref))
    return bounded(len(distinct) / BREADTH_NORM)


def compute_evidence_features(signals: Sequence[AttentionSignalV1]) -> SalienceFeaturesV1:
    """The two real, per-loop evidence magnitudes -- everything
    `borda_coalition_salience()` needs to vote with. Pure function of this
    loop's own deduped signal(s); no history/clock dependency (recency,
    dwell, habituation, recurrence were killed -- see module docstring)."""
    return SalienceFeaturesV1(
        evidence_strength=evidence_strength_for(signals),
        evidence_breadth=evidence_breadth_for(signals),
    )


def borda_coalition_salience(
    features_by_loop_id: dict[str, SalienceFeaturesV1],
) -> dict[str, float]:
    """Rank-aggregate evidence_strength/evidence_breadth across every loop
    competing in one tick -- the real "competing set of loops scored
    together" (see `scoring.build_open_loops`, the real call site this
    module's design brief asked to be traced) -- using the same Borda-count
    machinery already proven for Layer 5's Candidate B
    (`orion.attention.field_attention.candidate_society_of_mind`).

    Returns a [0,1] score per loop_id:

    - n == 0: `{}`.
    - n == 1: nothing to rank against -- the same "nothing to aggregate"
      degenerate case Candidate B's own docstring names for a single real
      *scorer* (here it's a single *candidate*, not a single scorer, but
      the same logic applies: ranking one thing against itself is
      undefined). Falls back to `mean(evidence_strength, evidence_breadth)`
      -- this loop's own raw evidence, unmodified by competition. This is a
      named, disclosed edge-case fallback, not the general formula: it only
      ever applies when exactly one loop is being scored this tick, and the
      0.5/0.5 mean here is not a tuned cross-scorer weight, it is "there is
      nothing else to compute" for a set of size 1.
    - n > 1: each of the two voters' Borda points (range `[0, n-1]`) is
      normalized to `[0, 1]` by dividing by `(n-1)` -- a per-scorer
      "normalized rank" -- then the two are averaged. The 0.5/0.5 split
      here is a structural symmetry of having exactly two equally-weighted
      voters (Borda count gives every voter's ballot equal say by
      construction), not a hand-picked cross-scorer exchange rate like
      `SEED_WEIGHTS` was.
    """
    loop_ids = list(features_by_loop_id)
    n = len(loop_ids)
    if n == 0:
        return {}
    if n == 1:
        only = loop_ids[0]
        feats = features_by_loop_id[only]
        return {only: bounded((feats.evidence_strength + feats.evidence_breadth) / 2.0)}

    strength_scores = {lid: f.evidence_strength for lid, f in features_by_loop_id.items()}
    breadth_scores = {lid: f.evidence_breadth for lid, f in features_by_loop_id.items()}
    result = aggregate_borda(
        {"evidence_strength": strength_scores, "evidence_breadth": breadth_scores},
        universe=loop_ids,
    )
    denom = float(n - 1)
    return {
        lid: bounded((result.totals.get(lid, 0.0) / denom) / 2.0)
        for lid in loop_ids
    }
