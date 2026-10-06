import pytest

from orion.memory.recall_skip_gate import RecallSkipGateResult
from orion.memory.retrieval_intent import derive_retrieval_intent


@pytest.mark.parametrize(
    "stance,attention,appraisal,expected_intent,expected_rule",
    [
        ({"task_mode": "reflective_dialogue"}, {}, {"shift_kind": "NONE"}, "relational", "relational_mode"),
        ({"task_mode": "instrumental"}, {}, {"shift_kind": "TOPIC", "novelty_score": 0.7}, "semantic", "topic_shift"),
        ({}, {}, {"shift_kind": "REPAIR", "novelty_score": 0.6}, "open_loop", "repair_shift"),
        ({"task_mode": "technical_collaboration"}, {}, {}, "procedural", "procedural_mode"),
        ({"conversation_frame": "planning"}, {}, {}, "procedural", "procedural_mode"),
    ],
)
def test_derive_retrieval_intent_rules(stance, attention, appraisal, expected_intent, expected_rule):
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief=stance,
        attention_frame=attention,
        appraisal=appraisal,
        hub_chat_lane=None,
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
        shift_novelty_floor=0.35,
    )
    assert intent == "continuity"
    assert rule_id == "continuity_only"


@pytest.mark.parametrize("signals, intent, rule_id", [
    ([{"phrase": "Vincent", "type": "person"}], "relational", "turn_names_person"),
    ([{"phrase": "the rack move", "type": "plan"}], "procedural", "turn_names_plan"),
    ([{"phrase": "Hecate", "type": "concept"}], "semantic", "turn_names_topic"),
    ([{"phrase": "Chicago", "type": "place"}, {"phrase": "Vincent", "type": "person"}], "relational", "turn_names_person"),
    ([{"phrase": "", "type": "person"}], "continuity", "continuity_only"),
    ([], "continuity", "continuity_only"),
    (None, "continuity", "continuity_only"),
])
def test_turn_signal_referent_rule(signals, intent, rule_id):
    """The same-turn LLM's typed reading of the message picks the intent (no word list)."""
    assert derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane=None,
        turn_signals=signals,
    ) == (intent, rule_id)


# The shape of a live chat attention frame on 2026-10-06: several substrate
# prediction-error loops, none of them a conversational thread.
LIVE_FRAME = {"open_loops": [
    {"id": "open-loop-1", "target_type": "concept", "description": "Biometrics prediction error"},
    {"id": "open-loop-2", "target_type": "concept", "description": "Transport prediction error"},
]}


@pytest.mark.parametrize("stance, signals, expected", [
    ({"task_mode": "reflective_dialogue"}, [], "relational"),
    ({"task_mode": "direct_response"}, [{"phrase": "Vincent", "type": "person"}], "relational"),
    ({"task_mode": "direct_response"}, [{"phrase": "Hecate", "type": "concept"}], "semantic"),
    ({"task_mode": "technical_collaboration"}, [], "procedural"),
    ({"task_mode": "direct_response"}, [], "continuity"),
])
def test_open_loops_in_the_frame_no_longer_force_open_loop(stance, signals, expected):
    """Regression: 630/630 live purposeful recalls were open_loop because any frame loop won."""
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False), stance_brief=stance, attention_frame=LIVE_FRAME,
        appraisal={}, hub_chat_lane=None, turn_signals=signals)
    assert intent == expected and rule_id != "open_loops_present"


def test_relational_and_topic_rules_precede_repair_and_contradiction():
    common = dict(skip_gate=RecallSkipGateResult(skip=False), attention_frame={"contradiction_refs": ["c"]},
                  hub_chat_lane=None)
    assert derive_retrieval_intent(stance_brief={"task_mode": "playful_exchange"},
        appraisal={"shift_kind": "REPAIR", "novelty_score": 0.9}, **common) == ("relational", "relational_mode")
    assert derive_retrieval_intent(stance_brief={},
        appraisal={"shift_kind": "TOPIC", "novelty_score": 0.9}, **common) == ("semantic", "topic_shift")
    assert derive_retrieval_intent(stance_brief={},
        appraisal={"shift_kind": "TOPIC", "novelty_score": 0.2}, **common) == ("contradiction", "contradiction_seed")


def test_brain_lane_belief_default_when_eligible_beliefs_exist():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane="brain",
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
        shift_novelty_floor=0.35,
        seed_crystallization_id="crys_xyz",
    )
    assert intent == "contradiction"
    assert rule_id == "contradiction_seed"
