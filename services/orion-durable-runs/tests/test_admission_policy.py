from orion.durable_admission.policy import decide_lane, satisfies


def lanes():
    return {"agent": {"backend_key": "a", "configured": True, "healthy": True,
                      "capabilities": {"structured_output": True, "context_tokens": 32768}},
            "metacog": {"backend_key": "b", "configured": True, "healthy": True,
                        "compatible_with": ["agent"], "capabilities": {"structured_output": True, "context_tokens": 32768}}}


def decide(**kwargs):
    inputs = dict(requirement={"preferred_lane": "agent", "alternatives": ["metacog"],
                              "requirements": {"structured_output": True, "minimum_context_tokens": 32768}},
                  lanes=lanes(), waited_seconds=1200, active_remaining={"a": 1800},
                  queued_ahead={}, lease_seconds=90, widening_enabled=True)
    inputs.update(kwargs)
    return decide_lane(**inputs)


def test_twenty_minutes_widens_without_removing_preferred_and_uses_work_budget():
    assert decide(waited_seconds=1199).eligible_lanes == ["agent"]
    result = decide()
    assert result.eligible_lanes == ["agent", "metacog"]
    assert result.assigned_lane == "metacog" and result.estimates == {"agent": 1800, "metacog": 0}


def test_preferred_wins_if_it_becomes_free_at_widening_boundary():
    result = decide(active_remaining={})
    assert result.assigned_lane == "agent"


def test_hysteresis_blocks_pointless_change():
    result = decide(active_remaining={"a": 120})
    assert result.assigned_lane is None and result.suppressed["metacog"] == "hysteresis"


def test_operator_override_and_experiment_pin_disable_widening():
    for key in ("operator_override", "pinned_lane"):
        req = {"preferred_lane": "agent", key: "agent", "alternatives": ["metacog"]}
        result = decide(requirement=req)
        assert result.eligible_lanes == ["agent"] and result.assigned_lane is None
    result = decide(requirement={"preferred_lane": "agent", "operator_override": "metacog", "pinned_lane": "agent"})
    assert result.assigned_lane == "metacog"


def test_hard_requirements_still_apply_to_override():
    result = decide(requirement={"preferred_lane": "agent", "operator_override": "metacog",
                               "requirements": {"minimum_context_tokens": 65536}})
    assert result.assigned_lane is None
    assert result.suppressed["metacog"] == "hard_requirements_incompatible"


def test_alternative_needs_explicit_compatibility_and_actual_route():
    catalog = lanes()
    catalog["metacog"].pop("compatible_with")
    assert decide(lanes=catalog).suppressed["metacog"] == "compatibility_not_declared"
    catalog["metacog"]["compatible_with"] = ["agent"]
    catalog["metacog"]["configured"] = False
    assert decide(lanes=catalog).assigned_lane is None


def test_observed_busy_and_unknown_backend_are_never_assigned():
    for busy in (True, None):
        catalog = lanes()
        catalog["metacog"]["external_busy"] = busy
        assert decide(lanes=catalog).assigned_lane is None


def test_aliases_share_work_ahead_and_quality_drop_is_bounded():
    catalog = lanes()
    catalog["metacog"]["backend_key"] = "a"
    assert decide(lanes=catalog).assigned_lane is None
    catalog = lanes()
    catalog["metacog"]["quality_drop"] = 2
    assert decide(lanes=catalog).suppressed["metacog"] == "quality_drop"
    assert not satisfies({"context_tokens": True}, {"minimum_context_tokens": 1})


def test_native_work_ahead_prevents_widened_work_overtaking():
    assert decide(queued_ahead={"b": 400}).assigned_lane is None
