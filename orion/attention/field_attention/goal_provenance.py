"""Field-native goal-provenance candidate selection (Sentience Striving Program §6
Objective 3).

Pure, DB/bus-free logic for orion-attention-runtime's goal-provenance producer -- see
docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-observability-
design.md for the full design. Scoped to that doc's "node-target domain lanes only"
first slice: Candidate A's real node:substrate.* prediction-error domains
(PREDICTION_ERROR_NATIVE_TARGETS), not the frame's full merged dominant_targets list --
host/capability targets (Candidate B, live since 2026-07-30) are real but out of scope
for this first producer per that doc's Recommended next patch.
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1

# 2026-07-30 fix (Sentience Striving Program officer review -- see
# `orion/sentience_striving_program/README.md` §12 for the full incident record).
# Defense in depth, added ON TOP OF the real fix (`candidate_precision_weighted.py`'s
# `PrecisionEwmaBaseline` replacing the rolling-window recompute) -- not a substitute
# for it. Live-confirmed 2026-07-30: a target with `confidence_score=0.1`
# (`node:substrate.chat`, `observation_count`/`n_samples=2`, far below
# `QUALIFYING_MIN_ROWS=20`) won a sustained 280+-tick goal-provenance dominance
# streak purely because it was the tick's *sole* real competitor --
# `normalize_across_targets()`'s own documented single-target edge case correctly
# and unavoidably assigns `salience_score=1.0` in that situation (there is no other
# real competitor to rank against), so `top_node_substrate_target()`'s plain
# `max(candidates, key=salience_score)` had no way to distinguish "genuinely
# dominant" from "alone in the room with barely any real data."
#
# `confidence_score` (`selectors.py::select_node_targets`) is
# `clamp01(observation_count / QUALIFYING_MIN_ROWS)` -- `1.0` means the target has
# accumulated at least `QUALIFYING_MIN_ROWS` real observations. Requiring the full
# `1.0` here (not some fraction of it) reuses the exact same "qualifying" bar this
# codebase already treats as the real trust threshold for a variance estimate
# (`QUALIFYING_MIN_ROWS`'s own name and `scripts/analysis/
# measure_precision_weighted_salience_probe.py`'s identical constant) rather than
# picking a new, separately-calibrated fraction -- one real bar, reused, not two.
# This gate is intentionally independent of the EWMA-baseline fix: even after that
# fix, `observation_count` still starts at 0 for a target with no persisted baseline
# row yet (a brand-new target, or one whose baseline table was reset) -- this floor
# ensures such a target cannot win a real goal-provenance publish while genuinely
# thin, regardless of what future bug might reintroduce a rolling-window-style
# under-count elsewhere.
MIN_CONFIDENCE_FOR_GOAL_PROVENANCE: float = 1.0


def qualified_node_targets(frame: FieldAttentionFrameV1) -> list[FieldAttentionTargetV1]:
    """The candidates a goal may name: Candidate A's real ``node:substrate.*``
    domains with enough observations to be trusted (see
    ``MIN_CONFIDENCE_FOR_GOAL_PROVENANCE``). Exposed so the producer can skip
    the competition read when fewer than two qualify -- with 0 or 1 candidate
    no competition set can change the answer."""
    return [
        t
        for t in frame.node_targets
        if t.target_id in PREDICTION_ERROR_NATIVE_TARGETS
        and t.confidence_score >= MIN_CONFIDENCE_FOR_GOAL_PROVENANCE
    ]


def top_node_substrate_target(
    frame: FieldAttentionFrameV1,
    *,
    competing: Collection[str] | None = None,
    current: str | None = None,
) -> FieldAttentionTargetV1 | None:
    """The highest-salience target among ``frame.node_targets`` that is one of
    Candidate A's real ``node:substrate.*`` domains AND has accumulated enough real
    observations to be trusted with a goal-provenance win
    (``confidence_score >= MIN_CONFIDENCE_FOR_GOAL_PROVENANCE``, see that constant's
    own comment for the live incident this guards against).

    Deliberately NOT ``frame.dominant_targets[0]`` (the frame's global top-1 winner):
    since the 2026-07-30 Candidate B patch, that slot is frequently a physical host or
    capability target instead (real novelty-scored competition, not degenerate) --
    reading it directly would make this producer rarely or never fire, which is exactly
    the "never-fires" degenerate failure mode CLAUDE.md's metric-quality-gate warns
    against. This is a real sub-competition winner within the node-target subset, not
    a proxy for the whole field.

    A tick where every real ``node:substrate.*`` competitor is still below the
    confidence floor returns ``None`` -- the same honest "nothing real to report yet"
    this producer already gives a tick with zero qualifying candidates, not a forced
    pick of the least-thin option.
    """
    candidates = qualified_node_targets(frame)
    if not candidates:
        return None
    # Precedence, top to bottom (the one bridge, 2026-09-06 -- design doc
    # "The read side"): a qualified candidate the substrate competition is
    # holding; else the streak's current target while it is still qualified
    # (hysteresis: an unknown/empty read must not flap the emission debounce,
    # pinned by test_bridge_hysteresis_keeps_the_current_target...); else the
    # raw field top-1. Not a reconciler: one id set, read once per tick.
    seen = [t for t in candidates if competing and t.target_id in competing]
    held = [t for t in candidates if current is not None and t.target_id == current]
    pool = seen or held or candidates
    return max(pool, key=lambda t: t.salience_score)


@dataclass
class DominanceStreak:
    target_id: str | None = None
    count: int = 0


def update_dominance_streak(
    streak: DominanceStreak,
    target_id: str | None,
    *,
    min_streak: int,
) -> tuple[DominanceStreak, bool]:
    """Advance a real-tick dominance streak; returns ``(new_streak, should_emit)``.

    ``should_emit`` is True once the SAME target has been the node-target subset's
    real top-1 winner for >= ``min_streak`` consecutive real field ticks -- a debounce
    against momentary flips (the same delta-gating discipline
    ``orion/sentience_striving_program/README.md`` §8 names as carried forward from
    O2/O3), not a new calibrated metric: ``min_streak`` is a control-flow gate on an
    already-real, already-live signal (``salience_score``), not a new instrument
    subject to CLAUDE.md's full metric-quality-gate.

    Emits on every qualifying tick once the streak has reached ``min_streak``, not
    only the tick that first crosses it -- matches
    ``orion/substrate/attention/goal_context.py``'s own "latest wins, replace on
    injection" semantics: a target that is still genuinely dominant should keep
    refreshing its own goal record's ``received_at``, not go stale under Part B's
    staleness dead-man's-switch just because the streak that produced it happened to
    start hours ago.
    """
    if target_id is None:
        return DominanceStreak(target_id=None, count=0), False
    if target_id != streak.target_id:
        return DominanceStreak(target_id=target_id, count=1), False
    new_streak = DominanceStreak(target_id=target_id, count=streak.count + 1)
    return new_streak, new_streak.count >= min_streak
