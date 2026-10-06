import pytest

from orion.memory.recall_skip_gate import RecallSkipGateResult
from orion.memory.retrieval_intent import derive_retrieval_intent


@pytest.mark.parametrize(
    "stance,attention,appraisal,expected_intent,expected_rule",
    [
        ({"task_mode": "reflective_dialogue"}, {}, {"shift_kind": "NONE"}, "relational", "relational_mode"),
        ({"task_mode": "instrumental"}, {}, {"shift_kind": "TOPIC", "novelty_score": 0.7}, "semantic", "topic_shift"),
        ({}, {"open_loops": [{"id": "gpu"}]}, {}, "open_loop", "open_loops_present"),
    ],
)
def test_derive_retrieval_intent_rules(stance, attention, appraisal, expected_intent, expected_rule):
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief=stance,
        attention_frame=attention,
        appraisal=appraisal,
        hub_chat_lane=None,
        user_message="test",
        shift_novelty_floor=0.35,
    )
    assert intent == expected_intent
    assert rule_id == expected_rule


def test_derive_retrieval_intent_phase0_skip():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=True, reasons=["low_info_social"]),
        stance_brief={"task_mode": "reflective_dialogue"},
        attention_frame={"open_loops": [{"id": "gpu"}]},
        appraisal={"shift_kind": "TOPIC", "novelty_score": 0.9},
        hub_chat_lane=None,
        user_message="hey",
        shift_novelty_floor=0.35,
    )
    assert intent == "none"
    assert rule_id == "phase0_skip"


def test_derive_retrieval_intent_continuity_only():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response", "interaction_regime": "instrumental"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane=None,
        user_message="ok sounds good",
        shift_novelty_floor=0.35,
    )
    assert intent == "continuity"
    assert rule_id == "continuity_only"


def test_derive_retrieval_intent_entity_query():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane=None,
        user_message="What did we decide about GPU migration?",
        shift_novelty_floor=0.35,
    )
    assert intent == "semantic"
    assert rule_id == "entity_query"


def test_brain_lane_belief_default_when_eligible_beliefs_exist():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane="brain",
        user_message="ok",
        eligible_belief_count=3,
        brain_belief_default_enabled=True,
    )
    assert intent == "semantic"
    assert rule_id == "brain_lane_belief_default"


def test_derive_retrieval_intent_contradiction_seed():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={"contradiction_refs": ["crys_abc"]},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane=None,
        user_message="ok",
        shift_novelty_floor=0.35,
        seed_crystallization_id="crys_xyz",
    )
    assert intent == "contradiction"
    assert rule_id == "contradiction_seed"
