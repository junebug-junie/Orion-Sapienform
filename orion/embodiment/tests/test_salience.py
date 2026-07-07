from __future__ import annotations

from orion.embodiment.salience import SalienceState, evaluate_salience


def test_completed_conversation_is_salient():
    state = SalienceState()
    ev = evaluate_salience(
        {"type": "conversation_completed", "with": "Juniper", "utterances": 4}, state)
    assert ev.salient is True
    assert ev.summary


def test_first_encounter_is_salient_once():
    state = SalienceState()
    e1 = evaluate_salience({"type": "encounter", "player_id": "j"}, state)
    e2 = evaluate_salience({"type": "encounter", "player_id": "j"}, state)
    assert e1.salient is True
    assert e2.salient is False  # deduped


def test_bare_proximity_not_salient():
    ev = evaluate_salience({"type": "proximity", "player_id": "j"}, SalienceState())
    assert ev.salient is False


def test_zero_utterance_conversation_not_salient():
    ev = evaluate_salience(
        {"type": "conversation_completed", "with": "Juniper", "utterances": 0},
        SalienceState())
    assert ev.salient is False


def test_encounter_missing_player_id_not_salient():
    ev = evaluate_salience({"type": "encounter"}, SalienceState())
    assert ev.salient is False
