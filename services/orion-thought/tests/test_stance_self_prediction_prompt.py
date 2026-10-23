"""Self-prediction A1: the stance prompt asks for the bet only when the flag is on."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from app import bus_listener
from app.bus_listener import build_stance_react_context
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "stance_react.j2"


def _render(**ctx) -> str:
    base = {
        "user_message": "hi",
        "stance_inputs": {"user_message": "hi"},
        "association": {},
        "repair_bundle": None,
        "coalition_projection": None,
    }
    base.update(ctx)
    return Environment().from_string(_PROMPT.read_text()).render(**base)


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        user_message="howdy",
        association=HubAssociationBundleV1(
            correlation_id="corr-1", broadcast=None, broadcast_stale=True, read_source="hub_sql_fallback"
        ),
        repair_bundle=None,
        stance_inputs={"user_message": "howdy"},
    )


def test_block_absent_when_flag_off() -> None:
    assert "self_prediction" not in _render()


def test_block_present_and_after_the_slice_when_flag_on() -> None:
    out = _render(self_prediction_enabled=True)
    assert "- self_prediction:" in out
    for key in ("asks_question_p", "fetches_outside_p", "steps_lo, steps_hi", "reply_chars_lo, reply_chars_hi"):
        assert key in out
    # A top-level field, not wedged inside the stance_harness_slice list.
    assert out.index("companion_closing_move:") < out.index("- self_prediction:")
    assert "never sees it" in out


def test_context_sets_flag_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(bus_listener.settings, "stance_self_prediction_enabled", True)
    assert build_stance_react_context(_request())["self_prediction_enabled"] is True


def test_context_omits_flag_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(bus_listener.settings, "stance_self_prediction_enabled", False)
    assert "self_prediction_enabled" not in build_stance_react_context(_request())


def test_setting_ships_on() -> None:
    from app.settings import ThoughtSettings

    assert ThoughtSettings.model_fields["stance_self_prediction_enabled"].default is True
