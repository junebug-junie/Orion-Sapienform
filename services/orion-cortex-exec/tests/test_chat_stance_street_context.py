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
        "clear; Street: Walkway rhythm: the black dog did not come (usually around 07:40)."
    )


def test_quiet_street_leaves_weather_alone() -> None:
    env = _situation_summary_from_ctx(_ctx(None))["environment"]
    assert env["street_summary"] == ""
    assert _environmental_context(env) == "clear"
    assert _environmental_context({}) is None
