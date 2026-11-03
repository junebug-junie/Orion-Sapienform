"""Stance hears the street (walkway camera spec idea 5): the brief's street
line reaches `environmental_context`, and a quiet street changes nothing."""

from __future__ import annotations

from app.chat_stance import _environmental_context, _situation_summary_from_ctx


def _ctx(street: str | None) -> dict:
    perception = {"street_summary": street} if street is not None else {}
    return {
        "situation_brief": {
            "environment": {"current_weather": {"condition": "clear"}},
            "perception": perception,
        },
        "situation_prompt_fragment": {},
    }


def test_street_reaches_environmental_context() -> None:
    situation = _situation_summary_from_ctx(_ctx("Walkway rhythm: the black dog did not come (usually around 07:40)."))
    env = situation["environment"]
    assert env["street_summary"].startswith("Walkway rhythm")
    assert _environmental_context(env) == (
        "clear; Walkway rhythm: the black dog did not come (usually around 07:40)."
    )


def test_self_describing_street_line_reaches_stance_whole() -> None:
    """The brief's street line now opens with what the camera is; stance must
    carry that framing, not cut it off or re-label it "Street:"."""
    line = (
        "Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway,"
        " the street and our patio: On the walkway just now: a person walks a dog;"
        " a cyclist passes. Around the walkway now: an unfamiliar person (seen 3 min ago)."
    )
    env = _situation_summary_from_ctx(_ctx(line))["environment"]
    assert env["street_summary"] == line
    assert _environmental_context(env) == f"clear; {line}"


def test_quiet_street_leaves_weather_alone() -> None:
    env = _situation_summary_from_ctx(_ctx(None))["environment"]
    assert env["street_summary"] == ""
    assert _environmental_context(env) == "clear"
    assert _environmental_context({}) is None


def test_street_line_with_two_motion_stories_reaches_stance_whole() -> None:
    """2026-11-03: spot-checked motion stories ride on the street line; the
    stance clip keeps a realistic line with two of them whole."""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    from orion.situational.perception_reader import STREET_STORY_CHARS, street_story_line

    now = datetime(2026, 11, 3, 1, 30, tzinfo=timezone.utc)
    long_story = "A person walked left to right and away from the house on the walkway, " + "x" * 200
    story = street_story_line(
        [{"story_id": "a", "started_at": now - timedelta(minutes=3), "sentence": long_story},
         {"story_id": "b", "started_at": now - timedelta(minutes=90), "sentence": long_story}],
        camera="walkway", tz=ZoneInfo("America/Denver"),
    )
    assert story and len(story) < 2 * (STREET_STORY_CHARS + 12) + 60
    line = (
        "Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway,"
        " the street and our patio: On the walkway just now: A vehicle is observed in the frame;"
        " A vehicle is observed in the walkway area. On the walkway in the last 15 minutes: 5 vehicles"
        " I have each seen on at least 2 days but have no name for. " + story +
        " Walkway rhythm: vehicle #d5ac3b usually comes around 20:04; vehicle #303e6a did not come"
        " (usually around 19:18)."
    )
    env = _situation_summary_from_ctx(_ctx(line))["environment"]
    assert env["street_summary"] == line
    assert _environmental_context(env) == f"clear; {line}"
