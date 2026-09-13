"""History-derived scheduling authorization; never an information/reward signal."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from hashlib import sha256
import yaml
from pydantic import BaseModel, ConfigDict, Field
from orion.schemas.reverie_visual import VisualBaselineEligibilityV1, VisualActivityV1

class VisualBaselinePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    policy_id: str = "visual_baseline.v1"
    interval_sec: float = Field(default=5400, gt=0)
    retry_sec: float = Field(default=600, gt=0)
    freshness_sec: float = Field(default=60, gt=0)
    timeout_sec: float = Field(default=2, gt=0, le=10)
    target_id: str = "host:circe_gpu"
    thought_url: str = "http://orion-athena-thought:7155"

def load_baseline_policy() -> VisualBaselinePolicy:
    path = Path(__file__).resolve().parents[2] / "config/proposals/visual_baseline.v1.yaml"
    return VisualBaselinePolicy.model_validate(yaml.safe_load(path.read_text()) if path.exists() else {})

def validate_eligibility(value, *, now=None, policy=None, target_id=None, template=None, proposal_kind=None):
    policy = policy or load_baseline_policy()
    now = now or datetime.now(timezone.utc)
    if not policy.enabled:
        return "visual_baseline_disabled"
    try:
        value = VisualBaselineEligibilityV1.model_validate(value)
        if value.policy_id != policy.policy_id:
            return "visual_baseline_policy_mismatch"
        age = (now - value.observed_at).total_seconds()
        if age < 0 or age > policy.freshness_sec:
            return "visual_baseline_stale"
        if value.due_at > now:
            return "visual_baseline_not_due"
        if value.last_success_at is not None:
            if not value.last_success_chain_id or not value.last_success_sha256:
                return "visual_baseline_missing_provenance"
            if value.due_at != value.last_success_at + timedelta(seconds=policy.interval_sec):
                return "visual_baseline_due_mismatch"
        elif value.last_success_chain_id or value.last_success_sha256:
            return "visual_baseline_provenance_mismatch"
        if ((target_id is not None and target_id != policy.target_id)
            or (template is not None and template != "render_scene")
            or (proposal_kind is not None and proposal_kind != "express")):
            return "visual_baseline_route_mismatch"
    except (ValueError, TypeError):
        return "visual_baseline_invalid"
    return None

def schedule(activity: VisualActivityV1, checkpoint: dict, *, now: datetime, policy: VisualBaselinePolicy):
    """Pure checkpoint reducer. One pending need and bounded retries across restarts."""
    state = dict(checkpoint)
    if not policy.enabled:
        return None, state, "visual_baseline_disabled"
    age = (now - activity.observed_at).total_seconds()
    if activity.history_status != "ok" or age < 0 or age > policy.freshness_sec:
        return None, state, "visual_activity_unavailable_or_stale"
    state.setdefault("activation_at", now.isoformat())
    due = (activity.last_success_at + timedelta(seconds=policy.interval_sec)
           if activity.last_success_at else datetime.fromisoformat(state["activation_at"]))
    success = activity.last_success_chain_id
    if state.get("last_success_chain_id") != success:
        for key in ("need_id", "next_attempt_not_before", "last_attempt_id"):
            state.pop(key, None)
    state["last_success_chain_id"] = success
    if due > now:
        return None, state, "visual_baseline_not_due"
    state.setdefault("need_id", "visual-need:" + sha256((state["activation_at"] + ":" + str(success)).encode()).hexdigest()[:24])
    if activity.active_attempt_id:
        return None, state, "visual_baseline_active"
    if state.get("next_attempt_not_before") and now < datetime.fromisoformat(state["next_attempt_not_before"]):
        return None, state, "visual_baseline_cooldown"
    eligibility = VisualBaselineEligibilityV1(need_id=state["need_id"], observed_at=activity.observed_at,
        due_at=due, last_success_at=activity.last_success_at, last_success_chain_id=success,
        last_success_sha256=activity.last_success_sha256, policy_id=policy.policy_id)
    reason = validate_eligibility(eligibility, now=now, policy=policy)
    if reason:
        return None, state, reason
    state["next_attempt_not_before"] = (now + timedelta(seconds=policy.retry_sec)).isoformat()
    state["last_attempt_id"] = f"{eligibility.need_id}:{now.isoformat()}"
    return eligibility, state, "visual_baseline_due"
