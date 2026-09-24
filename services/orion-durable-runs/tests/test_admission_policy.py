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


def test_chat_burst_widens_without_elastic_permission_but_only_while_gate_open():
    # `chat-burst` (2026-09-21) is Juniper's chat worker lent to this queue by a Hub gate.
    # Unlike `agent-burst` it needs no run-level `allow_elastic_activation`: nothing is
    # physically borrowed. The gate shows up here as health -- the gateway reports the route
    # `operator_closed` while shut, which refresh_lanes maps to healthy=False.
    lanes_ = {"agent": {"backend_key": "a", "configured": True, "healthy": True, "capabilities": {}},
              "chat-burst": {"backend_key": "c", "configured": True, "healthy": True,
                             "compatible_with": ["agent"], "capabilities": {}, "quality_drop": 0}}
    req = {"preferred_lane": "agent", "alternatives": ["chat-burst"]}
    open_gate = decide(requirement=req, lanes=lanes_)
    assert open_gate.eligible_lanes == ["agent", "chat-burst"]
    assert open_gate.assigned_lane == "chat-burst"
    lanes_["chat-burst"]["healthy"] = False  # status == operator_closed
    closed = decide(requirement=req, lanes=lanes_)
    assert closed.assigned_lane is None
    assert closed.suppressed["chat-burst"] == "health_unknown_or_unavailable"


def test_widen_alternatives_is_policy_additive_and_never_removes():
    from orion.durable_admission.policy import widen_alternatives
    lanes_ = {"agent": {}, "agent-burst": {"compatible_with": ["agent"]},
              "chat-burst": {"compatible_with": ["agent"]}, "metacog": {"compatible_with": ["quick"]}}
    # Frozen before chat-burst existed: it is added, agent-burst kept first, metacog (not
    # compatible with agent) left out, and the stored requirement object is not mutated.
    req = {"preferred_lane": "agent", "alternatives": ["agent-burst"]}
    out = widen_alternatives(req, lanes_)
    assert out["alternatives"] == ["agent-burst", "chat-burst"]
    assert req["alternatives"] == ["agent-burst"]
    # An explicit (non-derived) list is widened the same way; nothing is ever dropped.
    assert widen_alternatives({"preferred_lane": "agent", "alternatives": ["metacog"]}, lanes_)["alternatives"] == ["metacog", "agent-burst", "chat-burst"]
    # Already complete -> unchanged; unknown preferred -> unchanged.
    assert widen_alternatives(out, lanes_)["alternatives"] == out["alternatives"]
    assert widen_alternatives({"preferred_lane": "quick"}, lanes_)["alternatives"] == ["metacog"]


def test_widened_lane_becomes_eligible_after_the_wait_threshold():
    from orion.durable_admission.policy import widen_alternatives
    lanes_ = {"agent": {"backend_key": "a", "configured": True, "healthy": True, "capabilities": {}},
              "chat-burst": {"backend_key": "c", "configured": True, "healthy": True,
                             "compatible_with": ["agent"], "capabilities": {}, "quality_drop": 0}}
    frozen = {"preferred_lane": "agent", "alternatives": []}
    assert decide(requirement=frozen, lanes=lanes_).eligible_lanes == ["agent"]
    result = decide(requirement=widen_alternatives(frozen, lanes_), lanes=lanes_)
    assert result.eligible_lanes == ["agent", "chat-burst"] and result.assigned_lane == "chat-burst"
