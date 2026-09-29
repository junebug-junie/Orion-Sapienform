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
