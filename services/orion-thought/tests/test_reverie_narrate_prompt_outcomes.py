"""Golden-prompt coverage for verdict-aware narration (fix/reverie-verdict-
aware-narration). Renders the real reverie_narrate.j2 through build_reverie_context
so the actual prompt text the LLM sees is exercised, not just the context dict
shape -- the point of the fix is what gets said to the model, not just what data
reaches it.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1

_PROMPT = (
    Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
)


def _render(**ctx) -> str:
    template = Environment().from_string(_PROMPT.read_text())
    return template.render(**ctx)


def _broadcast(loops=(("ol-1", {}),)):
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id=oid, description="d", **s) for oid, s in loops]),
        attended_node_ids=["n-1"],
        selected_open_loop_id=loops[0][0] if loops else None,
    )


def test_no_outcome_block_when_no_loop_has_a_verdict():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    out = _render(**ctx)
    assert "ALREADY-SETTLED LOOPS" not in out


def test_outcome_block_present_and_instructs_settled_framing():
    from app import reverie

    ctx = reverie.build_reverie_context(
        _broadcast(),
        loop_outcomes={"ol-1": {"verdict": "resolved", "note": "fixed the flakiness", "age_days": 4}},
    )
    out = _render(**ctx)
    assert "ALREADY-SETTLED LOOPS" in out
    assert "settled" in out
    assert "never as an open struggle" in out
    # the actual verdict data must reach the model, not just the instruction text
    assert "resolved" in out
    assert "fixed the flakiness" in out


def test_outcome_block_absent_when_open_loops_is_empty():
    """Regression: the settled-loops instruction must not render when there is
    nothing in open_loops to apply it to (it used to render unconditionally)."""
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast(loops=()))
    out = _render(**ctx)
    assert "ALREADY-SETTLED LOOPS" not in out
    assert "open_loops:" not in out
