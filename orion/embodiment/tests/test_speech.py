from __future__ import annotations

from orion.embodiment.speech import (
    build_speech_prompt,
    is_injectable,
    latest_partner_line,
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
    assert "Your task is to answer this latest line from Juniper:\nhow are you?" in prompt
    assert "responds to the latest line above" in prompt


def test_latest_partner_line_ignores_orion_last_line():
    p = _perception_in_convo()
    p.active_conversation["messages"].append({"author": "Orion", "text": "I am still here."})
    assert latest_partner_line(p, "orion") == "how are you?"


def test_build_speech_prompt_treats_goodbye_as_departure_to_answer():
    p = _perception_in_convo()
    p.active_conversation["messages"].append({"author": "Juniper", "text": "I gotta run, bye."})
    prompt = build_speech_prompt(p, "orion")
    assert "Your task is to answer this latest line from Juniper:\nI gotta run, bye." in prompt
    assert "acknowledge the departure naturally" in prompt


def test_is_injectable_rejects_empty_and_whitespace():
    assert is_injectable("hello") is True
    assert is_injectable("") is False
    assert is_injectable("   \n  ") is False
    assert is_injectable(None) is False  # type: ignore[arg-type]
