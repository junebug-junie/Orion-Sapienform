"""The curiosity replica's own state (Redis ``orion:curiosity:replica:state``) and its plan act.

One key per loop instance; the replica never reads or writes Prime's state (design section 7). Pending plan
requests are NOT persisted: each has a 90 s deadline, so a restart drops them and Hub falls back on its own.

The plan act (design "Act ledger"): actor ``curiosity.replica``, verb ``infer``, signal ``artifact_produced``
(no new signal). It opens when the model was reached and closes on the run the plan ordered
(orion/curiosity/replica/track_record.py ``settle_plan``): at once when the plan could not be used, else
when the spend log has scored the run. ``goal_ref = curiosity_run:<run_id>`` joins it to the run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.grammar.acts import build_act
from orion.mindloop.backoff import TryKind
from orion.schemas.curiosity_replica import REPLICA_ACTOR
from orion.schemas.grammar import ActV1

STATE_KEY = "orion:curiosity:replica:state"
SOURCE_SERVICE = "orion-mind"
PLAN_COMPONENT = "curiosity_replica.plan"
PLAN_REF_PREFIX = "curiosity_replica_plan:"
# The plan act is graded on its run. A run can wait in durable admission for hours (heat, a busy lane);
# past this the ledger expires the act (not counted as an outcome) and the replica lets it go.
PLAN_ACT_HORIZON_SEC = 24 * 3600.0
PLAN_BASIS = ("the curiosity run this plan ordered moved at least one belief (curiosity_run_outcomes "
              "realized_nats > 0 on a completed turn), the same number the three-arm test scores")
MAX_NOTES = 6
MAX_OPEN_PLANS = 20


class OpenPlanV1(BaseModel):
    """A used plan whose act waits for its run's score."""

    model_config = ConfigDict(extra="forbid")

    act_id: str
    thought_id: str
    run_id: str
    opened_at: AwareDatetime
    picks: list[str] = Field(default_factory=list)


class CuriosityReplicaStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.replica.state.v1"] = "curiosity.replica.state.v1"
    open_plans: list[OpenPlanV1] = Field(default_factory=list)
    # Lines for the next frame that are not reasons to think (a gate refusal, a request given up on).
    notes: list[str] = Field(default_factory=list)
    plans_total: int = 0
    # The loop library's fields (orion/mindloop/runtime.py LOOP_STATE_FIELDS).
    last_thought_at: Optional[AwareDatetime] = None
    no_model_streak: int = 0
    no_model_tries_total: int = 0
    last_try_kind: Optional[TryKind] = None
    retry_not_before: Optional[AwareDatetime] = None


def add_note(state: CuriosityReplicaStateV1, text: str) -> CuriosityReplicaStateV1:
    return state.model_copy(update={"notes": [*state.notes, text][-MAX_NOTES:]})


def plan_act(thought_id: str, run_id: str) -> ActV1:
    """Deterministic in its arguments, so the later close names the same act."""
    return build_act(
        actor=REPLICA_ACTOR, verb="infer", key=thought_id, signal_id="artifact_produced",
        horizon_sec=PLAN_ACT_HORIZON_SEC, basis=PLAN_BASIS, object_ref=f"{PLAN_REF_PREFIX}{thought_id}",
        goal_ref=f"curiosity_run:{run_id}",
    )


def hold_plan(state: CuriosityReplicaStateV1, plan: OpenPlanV1) -> CuriosityReplicaStateV1:
    keep = [p for p in state.open_plans if p.run_id != plan.run_id][-(MAX_OPEN_PLANS - 1):]
    return state.model_copy(update={"open_plans": [*keep, plan], "plans_total": state.plans_total + 1})


def opened_before(plan: OpenPlanV1, now: datetime, seconds: float) -> bool:
    return (now - plan.opened_at).total_seconds() > seconds


__all__ = [
    "CuriosityReplicaStateV1",
    "OpenPlanV1",
    "PLAN_ACT_HORIZON_SEC",
    "PLAN_COMPONENT",
    "PLAN_REF_PREFIX",
    "SOURCE_SERVICE",
    "STATE_KEY",
    "add_note",
    "hold_plan",
    "opened_before",
    "plan_act",
]
