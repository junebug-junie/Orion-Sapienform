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

from dataclasses import dataclass

from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1


def top_node_substrate_target(frame: FieldAttentionFrameV1) -> FieldAttentionTargetV1 | None:
    """The highest-salience target among ``frame.node_targets`` that is one of
    Candidate A's real ``node:substrate.*`` domains.

    Deliberately NOT ``frame.dominant_targets[0]`` (the frame's global top-1 winner):
    since the 2026-07-30 Candidate B patch, that slot is frequently a physical host or
    capability target instead (real novelty-scored competition, not degenerate) --
    reading it directly would make this producer rarely or never fire, which is exactly
    the "never-fires" degenerate failure mode CLAUDE.md's metric-quality-gate warns
    against. This is a real sub-competition winner within the node-target subset, not
    a proxy for the whole field.
    """
    candidates = [t for t in frame.node_targets if t.target_id in PREDICTION_ERROR_NATIVE_TARGETS]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.salience_score)


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
