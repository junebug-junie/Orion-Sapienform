from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "stance_react.j2"


def _render(**ctx) -> str:
    template = Environment().from_string(_PROMPT.read_text())
    base = {
        "user_message": "hi",
        "stance_inputs": {"user_message": "hi"},
        "association": {},
        "repair_bundle": None,
        "coalition_projection": None,
    }
    base.update(ctx)
    return template.render(**base)


def test_block_absent_without_coloring() -> None:
    out = _render()
    assert "PRIOR SELF-SIGNAL" not in out


def test_block_present_with_coloring() -> None:
    coloring = {
        "attention_frontier": [{"label": "continuity", "summary": "our last thread", "score": 0.9}],
        "reflective_themes": ["continuity"],
        "curiosity_threads": ["what changed"],
        "self_relevance": "touches my continuity",
        "juniper_relevance": "Juniper is checking in",
        "identity_salience": "high",
    }
    out = _render(mind_coloring=coloring)
    assert "PRIOR SELF-SIGNAL" in out
    assert "reconcile, do not obey" in out
    assert "those WIN" in out
    assert "continuity" in out
    assert "Juniper is checking in" in out


def test_block_does_not_introduce_output_keys() -> None:
    # The advisory block must not tell the model to emit new top-level JSON keys.
    out = _render(mind_coloring={"attention_frontier": [], "reflective_themes": ["x"]})
    assert "do not invent extra top-level keys" in out or "do not invent" in out.lower()
