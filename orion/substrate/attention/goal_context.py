"""Goal → attention wire (spec 2026-07-07 Step 2, the scope-doubler).

The substrate attention path has no goal awareness by design gap. This module is
the minimal contract that lets an active ``FieldGoalProvenanceV1`` reach the
top-down bias combiner: a small module-level store holding the current active
goal as a ``GoalContext``, mirroring the existing module-level state pattern in
``attention_broadcast`` (``_recent_selected_counts``).

Populated by a consumer of the goal channel (``set_active_goal``); read by
``build_substrate_attention_frame`` only when ``ORION_ATTENTION_TOPDOWN_ENABLED``.
Never raises; when empty, attention is pure bottom-up (current behavior).

2026-07-30: repointed from ``GoalProposalV1`` (drives-era schema, permanently
producer-less since the drives deletion) to ``FieldGoalProvenanceV1`` (field-native
goal-provenance, produced by orion-attention-runtime) -- see
``docs/superpowers/specs/2026-07-30-goal-provenance-and-decision-lattice-
observability-design.md``. ``update_from_goal``'s contract (latest-active-wins,
terminal status clears) is unchanged; only the accepted schema type changed.

MVP proxy: the store keeps the most-recent *active* goal (latest wins), carrying
its priority for the bias magnitude. A full "highest-priority among all live
goals" query needs a goal set with expiry — a documented follow-on.

Staleness dead-man's-switch (2026-07-30, Part B of the design above): ``current()``
clears a held goal past ``_MAX_GOAL_AGE_SEC`` regardless of status. This is
deliberately NOT continuous decay -- see the design doc for why a leaky-integrator
decay here would risk the same saturation/floor bugs CLAUDE.md's metric-quality-gate
already names twice. It exists only to stop a *stalled producer* from biasing
attention toward a goal from hours ago forever; a live producer's own
"latest wins" replacement already handles the normal case.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from orion.schemas.field_goal import FieldGoalProvenanceV1
from orion.substrate.attention.top_down import GoalContext

# proposal_status values (ProposalStatus literal) that count as an active want
# worth biasing attention toward. Terminal/inactive states (superseded, archived,
# completed, failed) are ignored.
_ACTIVE_STATES = {"proposed", "active", "planned", "executing"}

# Real default TBD by measurement once the producer has live cadence data (see
# design doc Part B, Acceptance check 7) -- 4h is a conservative placeholder,
# long enough that no plausible real producer cadence trips it accidentally.
_DEFAULT_MAX_GOAL_AGE_SEC = 4 * 60 * 60


def _max_goal_age_sec() -> float:
    raw = os.getenv("ORION_ATTENTION_GOAL_CONTEXT_MAX_AGE_SEC")
    if raw is None:
        return float(_DEFAULT_MAX_GOAL_AGE_SEC)
    try:
        return float(raw)
    except ValueError:
        return float(_DEFAULT_MAX_GOAL_AGE_SEC)


class GoalContextStore:
    def __init__(self) -> None:
        self._current: Optional[GoalContext] = None

    def update_from_goal(self, goal: FieldGoalProvenanceV1) -> None:
        """Adopt this goal as the current active context (latest active wins).

        A terminal/inactive status for the *currently held* goal CLEARS the store
        (so a completed/failed goal stops steering attention); other non-active
        statuses are ignored. Malformed input is a no-op."""
        try:
            status = str(getattr(goal, "proposal_status", "proposed"))
            artifact_id = getattr(goal, "artifact_id", None)
            if status not in _ACTIVE_STATES:
                # If the goal that just went terminal is the one we're holding,
                # clear it — otherwise it would bias attention forever.
                if self._current is not None and self._current.goal_artifact_id == artifact_id:
                    self._current = None
                return
            self._current = GoalContext(
                priority=max(0.0, min(1.0, float(getattr(goal, "priority", 0.0) or 0.0))),
                goal_artifact_id=artifact_id,
            )
        except Exception:
            return

    def current(self) -> Optional[GoalContext]:
        """Return the held goal, or None if absent/stale.

        Staleness check is read-time only (no background sweep, no new tick) --
        see module docstring. A stalled producer is the only real risk this
        guards against; a live producer's own replace-on-injection semantics
        already keep this fresh under normal operation.
        """
        if self._current is not None:
            age_sec = (datetime.now(timezone.utc) - self._current.received_at).total_seconds()
            if age_sec > _max_goal_age_sec():
                self._current = None
        return self._current

    def clear(self) -> None:
        self._current = None


# Module-level default store (see module docstring).
_default_store = GoalContextStore()


def set_active_goal(goal: FieldGoalProvenanceV1) -> None:
    _default_store.update_from_goal(goal)


def get_active_goal() -> Optional[GoalContext]:
    return _default_store.current()


def clear_active_goal() -> None:
    _default_store.clear()
