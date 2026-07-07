from orion.memory.consolidation_gate import consolidation_memory_gate


def _turn(prompt, response, novelty=0.1, shift="NONE", significance=0.2):
    return {
        "prompt": prompt,
        "response": response,
        "spark_meta": {
            "turn_change_appraisal": {
                "novelty_score": novelty,
                "shift_kind": shift,
                "turn_change_status": "ok",
            },
            "memory_significance_score": significance,
        },
    }


def test_gate_skips_greeting_window():
    turns = [
        _turn("hey", "hi there"),
        _turn("thanks", "anytime"),
    ]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "skip"
    assert "low_info_social" in result.reasons


def test_gate_proposes_topic_shift():
    turns = [_turn("move logistics alone", "that sounds heavy", novelty=0.72, shift="TOPIC", significance=0.55)]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"
    assert result.dominant_shift == "TOPIC"


def test_gate_proposes_on_repair_signal():
    turns = [_turn("hey", "sorry about earlier")]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=True,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"


def test_gate_proposes_substantive_text_below_floors():
    turns = [_turn("still drowning in move logistics alone", "ok", novelty=0.1, significance=0.2)]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"
    assert "substantive_text" in result.reasons
