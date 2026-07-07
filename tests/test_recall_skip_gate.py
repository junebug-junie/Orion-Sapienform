from orion.memory.recall_skip_gate import recall_skip_gate


def test_skips_greeting_with_low_novelty():
    result = recall_skip_gate(
        user_message="hey Orion",
        appraisal={"novelty_score": 0.1, "shift_kind": "NONE"},
        has_repair_grammar_signal=False,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is True
    assert "low_info_social" in result.reasons


def test_no_skip_when_repair_signal():
    result = recall_skip_gate(
        user_message="hey",
        appraisal={"novelty_score": 0.1, "shift_kind": "NONE"},
        has_repair_grammar_signal=True,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is False


def test_malformed_novelty_treated_as_missing():
    result = recall_skip_gate(
        user_message="hey",
        appraisal={"novelty_score": "0.1"},
        has_repair_grammar_signal=False,
    )
    assert result.skip is True
    assert "novelty_below_floor" in result.reasons


def test_no_skip_substantive_topic():
    result = recall_skip_gate(
        user_message="still drowning in move logistics alone",
        appraisal={"novelty_score": 0.72, "shift_kind": "TOPIC"},
        has_repair_grammar_signal=False,
        max_novelty=0.25,
        shift_novelty_floor=0.35,
    )
    assert result.skip is False
