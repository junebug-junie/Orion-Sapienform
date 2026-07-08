"""Goal → attention wire (spec 2026-07-07 Step 2, the scope-doubler).

The substrate attention path has no goal awareness by design gap. This module is
the minimal contract that lets an active ``GoalProposalV1`` reach the top-down
bias combiner: a small module-level store holding the current active goal as a
``GoalContext``, mirroring the existing module-level state pattern in
``attention_broadcast`` (``_recent_selected_counts``).

Populated by a consumer of the goal channel (``set_active_goal``); read by
``build_substrate_attention_frame`` only when ``ORION_ATTENTION_TOPDOWN_ENABLED``.
Never raises; when empty, attention is pure bottom-up (current behavior).

MVP proxy: the store keeps the most-recent *active* goal (latest wins), carrying
its priority for the bias magnitude. A full "highest-priority among all live
goals" query needs a goal set with expiry — a documented follow-on.
"""
from __future__ import annotations

from typing import Optional

from orion.core.schemas.drives import GoalProposalV1
from orion.substrate.attention.top_down import GoalContext

# proposal_status values (ProposalStatus literal) that count as an active want
# worth biasing attention toward. Terminal/inactive states (superseded, archived,
# completed, failed) are ignored.
_ACTIVE_STATES = {"proposed", "active", "planned", "executing"}


class GoalContextStore:
    def __init__(self) -> None:
        self._current: Optional[GoalContext] = None

    def update_from_goal(self, goal: GoalProposalV1) -> None:
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
            drive_origin = getattr(goal, "drive_origin", None)
            if not drive_origin:
                return
            self._current = GoalContext(
                drive_origin=str(drive_origin),
                priority=max(0.0, min(1.0, float(getattr(goal, "priority", 0.0) or 0.0))),
                goal_artifact_id=artifact_id,
            )
        except Exception:
            return

    def current(self) -> Optional[GoalContext]:
        return self._current

    def clear(self) -> None:
        self._current = None


# Module-level default store (see module docstring).
_default_store = GoalContextStore()


def set_active_goal(goal: GoalProposalV1) -> None:
    _default_store.update_from_goal(goal)


def get_active_goal() -> Optional[GoalContext]:
    return _default_store.current()


def clear_active_goal() -> None:
    _default_store.clear()
