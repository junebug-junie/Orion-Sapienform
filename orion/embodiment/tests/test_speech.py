from __future__ import annotations

from orion.embodiment.speech import (
    build_speech_prompt,
    is_injectable,
    is_repeated_utterance,
    should_speak,
)
from orion.schemas.embodiment import WorldPerceptionV1


def _perception_in_convo() -> WorldPerceptionV1:
    return WorldPerceptionV1(
        player_id="orion",
        position={"x": 0.0, "y": 0.0},
        nearby_players=[{"player_id": "p9", "name": "Juniper", "position": {"x": 1.0, "y": 0.0}, "distance": 1.0}],
        active_conversation={
            "conversation_id": "conv1",
            "participants": ["orion", "p9"],
            "messages": [
                {"author": "Juniper", "text": "hey Orion"},
                {"author": "Orion", "text": "hi there"},
                {"author": "Juniper", "text": "how are you?"},
            ],
        },
    )


def test_should_speak_true_when_own_in_active_conversation():
    assert should_speak(_perception_in_convo(), "orion") is True


def test_should_speak_false_when_not_participant():
    p = _perception_in_convo()
    assert should_speak(p, "somebody_else") is False


def test_should_speak_false_when_no_active_conversation():
    p = WorldPerceptionV1(player_id="orion", position={"x": 0.0, "y": 0.0})
    assert should_speak(p, "orion") is False


def test_build_speech_prompt_contains_interlocutor_and_recent_lines():
    prompt = build_speech_prompt(_perception_in_convo(), "orion")
    assert "Juniper" in prompt
    assert "how are you?" in prompt


def test_is_injectable_rejects_empty_and_whitespace():
    assert is_injectable("hello") is True
    assert is_injectable("") is False
    assert is_injectable("   \n  ") is False
    assert is_injectable(None) is False  # type: ignore[arg-type]


def test_repeated_utterance_normalizes_case_and_spacing():
    assert is_repeated_utterance("  I am with you.\n", "i am  with you.") is True
    assert is_repeated_utterance("I am with you", "I am with you.") is True
    assert is_repeated_utterance("I hear you.", "Tell me more.") is False
    assert is_repeated_utterance("", "i am with you.") is False
