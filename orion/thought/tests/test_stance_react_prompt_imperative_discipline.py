from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stance_react_prompt_imperative_discipline() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    assert "IMPERATIVE DISCIPLINE" in text
    assert "efference copy" in text.lower() or "what Orion must DO" in text
    assert "world-contact" in text.lower() or "world contact" in text.lower()
    assert "if user says" not in text.lower()


def test_stance_react_prompt_defines_imperative_as_approach_not_extra_work() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    assert "what Orion must DO" not in text
    assert "not new work the message did not ask for" in text
    assert "may shape how, never add what" in text


def test_stance_react_self_signal_line_renders_without_prior_self_signal_marker() -> None:
    from jinja2 import Environment

    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    rendered = Environment().from_string(text).render(
        user_message="hi",
        stance_inputs={"user_message": "hi"},
        association={},
        repair_bundle=None,
        coalition_projection=None,
    )
    self_signal = rendered.index(
        "The advisory self-signal blocks (Oríon's Mind coloring; autonomy drives, "
        "tensions and recent actions) may shape how, never add what."
    )
    attention = rendered.index("A selected ATTENTION FRAME ask is the one exception")
    assert self_signal < attention
    assert "PRIOR SELF-SIGNAL" not in rendered
    assert "Mind coloring and autonomy recent_actions blocks" not in rendered


def test_stance_react_prompt_does_not_ask_model_for_record_identity() -> None:
    text = (REPO_ROOT / "orion/cognition/prompts/stance_react.j2").read_text(encoding="utf-8")
    for field in ("event_id", "session_id", "created_at"):
        assert field not in text, field
