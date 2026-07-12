import pytest
from unittest.mock import AsyncMock

from orion.memory.consolidation_gate import consolidation_memory_gate
from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window


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


def test_gate_skips_low_info_window_with_high_novelty():
    # Regression: a bare high novelty score on a low-info ("hi") turn used to
    # sail through as "novelty_above_floor" with no relation to the text itself.
    turns = [_turn("hi", "hey there", novelty=0.9, shift="NONE")]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "skip"
    assert "low_info_social" in result.reasons


def test_gate_skips_low_info_window_with_high_significance():
    # Regression: same bug, but via the significance_above_floor branch.
    turns = [_turn("hi", "hey there", novelty=0.1, significance=0.9)]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "skip"
    assert "low_info_social" in result.reasons


def test_gate_proposes_novelty_above_floor_with_substantive_text():
    turns = [_turn("move logistics alone", "that sounds heavy", novelty=0.72, shift="NONE", significance=0.55)]
    result = consolidation_memory_gate(
        turns=turns,
        grammar_repair_signal=False,
        min_novelty=0.35,
        min_significance=0.40,
    )
    assert result.action == "propose"
    assert "novelty_above_floor" in result.reasons


@pytest.mark.asyncio
async def test_fetch_collects_repair_signal_event_ids():
    pool = AsyncMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "event_id": "evt-1",
                "event_json": {"atom": {"semantic_role": "repair_signal"}},
            }
        ]
    )
    turns = [{"correlation_id": "corr-1"}]
    repair, event_ids = await fetch_grammar_evidence_for_window(
        pool,
        turns=turns,
        node_id="athena",
        enabled=True,
    )
    assert repair is True
    assert event_ids == ["evt-1"]
    pool.fetch.assert_awaited_once_with(
        """
            SELECT event_id, event_json
            FROM grammar_events
            WHERE trace_id = $1
            ORDER BY created_at ASC
            """,
        "hub.chat:athena:corr-1",
    )
