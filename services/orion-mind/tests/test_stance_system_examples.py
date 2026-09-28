"""The stance system prompt's examples are one-shot anchors for the metacog model.

corr beab81a3: with only the operational smoketest example, Juniper's work-travel
news came back as "Acknowledge the update without over-interpreting." / "Stay
concise and operational." / "Operational update on work travel..." -- the
example's phrasing, reused. These tests keep a contrasting example present and
keep every example valid, so the prompt cannot teach labels the handoff rejects.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from orion.schemas.chat_stance import ChatStanceBrief

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


@pytest.fixture(autouse=True)
def _prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy_examples", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


def _examples(prompt: str) -> list[dict]:
    return [json.loads(line) for line in prompt.splitlines() if re.match(r"^\{.*\}$", line.strip())]


def test_prompt_carries_an_operational_and_a_personal_example() -> None:
    from app.stance_handoff import _stance_system_prompt

    examples = _examples(_stance_system_prompt(None))
    assert len(examples) == 2
    modes = {ex["task_mode"] for ex in examples}
    assert "direct_response" in modes
    assert modes - {"direct_response"}, "a non-operational example must contrast the smoketest one"


def test_examples_do_not_share_relevance_phrasing() -> None:
    from app.stance_handoff import _stance_system_prompt

    operational, personal = _examples(_stance_system_prompt(None))
    for key in ("self_relevance", "juniper_relevance", "stance_summary"):
        assert operational[key] != personal[key]
    assert "operational" not in personal["juniper_relevance"].lower()


@pytest.mark.parametrize("origin", [None, "orion"])
def test_every_example_passes_coercion_unchanged_and_validates(origin) -> None:
    from app.stance_handoff import _stance_system_prompt, try_coerce_stance_payload

    for example in _examples(_stance_system_prompt(origin)):
        coerced, changed = try_coerce_stance_payload(example)
        assert not changed, example
        ChatStanceBrief.model_validate(coerced)


def test_prompt_defines_the_relevance_fields() -> None:
    from app.stance_handoff import _stance_system_prompt

    prompt = _stance_system_prompt(None)
    assert "self_relevance: what this turn means to Orion" in prompt
    assert "juniper_relevance: what this turn says about Juniper's own life" in prompt
    assert "not phrasing to reuse" in prompt
