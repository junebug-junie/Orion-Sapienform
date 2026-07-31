"""Cross-process-local cache of the current real, field-native active goal, for
autonomy capability gating (Sentience Striving Program §6 Objective 6).

Mirrors ``orion/substrate/attention/goal_context.py``'s ``GoalContextStore`` pattern
exactly (same active/terminal semantics, same read-time staleness dead-man's-switch),
but keeps the full ``FieldGoalProvenanceV1`` artifact rather than a reduced
``GoalContext`` dataclass -- ``orion/autonomy/capability_policy.py::evaluate_capability()``
reads ``.proposal_status`` directly, a field attention's reduced shape deliberately drops.

This module has no persistence and no cross-process visibility of its own -- each
service that needs real goal-gated capability checks runs its own bus listener
(``goal_state_listener.py``) and gets its own in-process copy. That is the same
per-process tradeoff ``goal_context.py`` already accepted for the attention consumer;
see that module's docstring for why (no Postgres-backed cross-process store exists
for this yet -- a documented follow-on, not solved by this module).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from orion.schemas.field_goal import FieldGoalProvenanceV1

# proposal_status values (ProposalStatus literal) that count as an active want
# worth gating a capability on. Terminal/inactive states (superseded, archived,
# completed, failed) are ignored -- mirrors goal_context.py's _ACTIVE_STATES.
_ACTIVE_STATES = {"proposed", "active", "planned", "executing"}

# Same conservative placeholder as goal_context.py's _DEFAULT_MAX_GOAL_AGE_SEC --
# real default TBD by measurement once both real consumers have live cadence data.
_DEFAULT_MAX_GOAL_AGE_SEC = 4 * 60 * 60


def _max_goal_age_sec() -> float:
    raw = os.getenv("ORION_AUTONOMY_GOAL_STATE_MAX_AGE_SEC")
    if raw is None:
        return float(_DEFAULT_MAX_GOAL_AGE_SEC)
    try:
        return float(raw)
    except ValueError:
        return float(_DEFAULT_MAX_GOAL_AGE_SEC)


class ActiveGoalStore:
    def __init__(self) -> None:
        self._current: Optional[FieldGoalProvenanceV1] = None

    def update_from_goal(self, goal: FieldGoalProvenanceV1) -> None:
        """Adopt this goal as the current active goal (latest active wins).

        A terminal/inactive status for the *currently held* goal CLEARS the store
        (so a completed/failed goal stops gating capabilities); other non-active
        statuses are ignored. Malformed input is a no-op."""
        try:
            status = str(getattr(goal, "proposal_status", "proposed"))
            artifact_id = getattr(goal, "artifact_id", None)
            if status not in _ACTIVE_STATES:
                if self._current is not None and self._current.artifact_id == artifact_id:
                    self._current = None
                return
            self._current = goal
        except Exception:
            return

    def current(self) -> Optional[FieldGoalProvenanceV1]:
        """Return the held goal, or None if absent/stale.

        Staleness check is read-time only (no background sweep, no new tick) --
        matches goal_context.py::GoalContextStore.current(). Guards only against a
        stalled producer; a live producer's own replace-on-injection semantics keep
        this fresh under normal operation.
        """
        if self._current is not None:
            age_sec = (datetime.now(timezone.utc) - self._current.ts).total_seconds()
            if age_sec > _max_goal_age_sec():
                self._current = None
        return self._current

    def clear(self) -> None:
        self._current = None


# Module-level default store (see module docstring: per-process, not shared).
_default_store = ActiveGoalStore()


def set_active_goal(goal: FieldGoalProvenanceV1) -> None:
    _default_store.update_from_goal(goal)


def get_active_goal() -> Optional[FieldGoalProvenanceV1]:
    return _default_store.current()


def clear_active_goal() -> None:
    _default_store.clear()
