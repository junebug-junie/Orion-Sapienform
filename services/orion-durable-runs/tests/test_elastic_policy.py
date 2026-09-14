from orion.durable_admission.elastic import activation_decision
from orion.durable_admission.policy import decide_lane
import pytest


def decision(**changes):
    args=dict(requirement={"preferred_lane":"agent","alternatives":["agent-burst"],"allow_elastic_activation":True},
        meta={"configured":True,"backend_key":"http://burst","activatable":True,"compatible_with":["agent"],"activation_capabilities":{"context_tokens":131072}},
        waited=1200,threshold=1200,widening=True,preferred_start=3600,queued_ahead=0,
        budget={"drain":300,"transition":60,"cold":600},hysteresis=120,environment={"eligible":True})
    args.update(changes)
    return activation_decision(**args)

@pytest.mark.parametrize("changes,reason",[
    ({"waited":1199},"wait_threshold"),
    ({"environment":{"eligible":False,"reason":"thermal_hot"}},"thermal_hot"),
    ({"preferred_start":1000},"cold_start_hysteresis"),
    ({"queued_ahead":3000},"cold_start_hysteresis"),
    ({"widening":False},"widening_disabled"),
    ({"requirement":{"allow_elastic_activation":True,"operator_override":"agent"}},"operator_pin"),
    ({"requirement":{"preferred_lane":"agent"}},"run_elastic_disabled"),
    ({"meta":{"configured":True,"backend_key":"http://burst","activatable":True}},"compatibility_not_declared"),
    ({"requirement":{"preferred_lane":"agent","alternatives":["agent-burst"],"allow_elastic_activation":True,"requirements":{"minimum_context_tokens":200000}}},"hard_requirements_incompatible"),
])
def test_activation_suppressions(changes,reason):
    assert decision(**changes)["suppression_reason"] == reason

def test_cold_budget_is_declaration_not_instant_capacity():
    result=decision()
    assert result["suppression_reason"] is None
    assert result["predicted_burst_start"] == 960
    assert result["budget_kind"] == "conservative_declaration"

def test_resident_burst_still_needs_run_permission():
    result=decide_lane({"preferred_lane":"agent","alternatives":["agent-burst"]},
        {"agent-burst":{"configured":True,"healthy":True,"backend_key":"burst","compatible_with":["agent"]}},
        waited_seconds=2000,active_remaining={},queued_ahead={},lease_seconds=90,widening_enabled=True)
    assert result.assigned_lane is None
    assert result.suppressed["agent-burst"] == "run_elastic_disabled"

def test_retained_burst_can_reactivate_without_migrating():
    assert decision(retained_burst=True,preferred_start=None)["suppression_reason"] is None
