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


def test_block_renders_new_coloring_keys_when_present() -> None:
    """select_mind_coloring now forwards user_intent, uncertainty, and Orion
    work-shape labels; the stance prompt must actually print them."""
    coloring = {
        "attention_frontier": [{"label": "continuity", "summary": "our last thread", "score": 0.9}],
        "reflective_themes": ["continuity"],
        "curiosity_threads": ["what changed"],
        "self_relevance": "touches my continuity",
        "juniper_relevance": "Juniper is checking in",
        "identity_salience": "high",
        "user_intent": "Look into substrate.route edges",
        "uncertainty_summary": "substrate.route:0.31",
        "expected_depth": "deep",
        "cross_cutting": "yes",
        "foresight_note": "This may touch ACL grants",
    }
    out = _render(mind_coloring=coloring)
    assert "Look into substrate.route edges" in out
    assert "substrate.route:0.31" in out
    assert "expected_depth:" in out and "deep" in out
    assert "cross_cutting:" in out and "yes" in out
    assert "This may touch ACL grants" in out


def test_block_omits_new_coloring_keys_when_absent() -> None:
    """Do not invent labels the selector never produced."""
    out = _render(
        mind_coloring={
            "attention_frontier": [],
            "reflective_themes": ["x"],
            "curiosity_threads": [],
            "self_relevance": None,
            "juniper_relevance": None,
            "identity_salience": None,
        }
    )
    assert "user_intent:" not in out
    assert "uncertainty_summary:" not in out
    assert "expected_depth:" not in out
    assert "cross_cutting:" not in out
    assert "foresight_note:" not in out
