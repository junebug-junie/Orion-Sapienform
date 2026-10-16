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
    assert "RECENT VERDICTS ON THESE LOOPS" not in out


def test_outcome_block_present_and_instructs_settled_framing():
    from app import reverie

    ctx = reverie.build_reverie_context(
        _broadcast(),
        loop_outcomes={"ol-1": {"verdict": "resolved", "note": "fixed the flakiness", "actor": "juniper",
                                "age_sec": 3 * 3600}},
    )
    out = _render(**ctx)
    assert "RECENT VERDICTS ON THESE LOOPS" in out
    assert "never as an open struggle" in out
    # the actual verdict, its actor and its age reach the model, not just the instruction text
    assert 'verdict: Juniper resolved this 3h ago (note: "fixed the flakiness")' in out


def test_system_verdict_renders_as_automatic_not_human():
    """Regression (2026-10-14 audit): the old block said every verdict "was
    recorded by a human closing this loop -- it is ground truth"."""
    from app import reverie

    ctx = reverie.build_reverie_context(
        _broadcast(),
        loop_outcomes={"ol-1": {"verdict": "decayed_unattended", "note": "", "actor": "system:implicit_decay",
                                "age_sec": 2 * 3600}},
    )
    out = _render(**ctx)
    assert "verdict: the system's automatic decay (nobody engaged with it) marked this decayed unattended 2h ago" in out
    assert "recorded by a human" not in out
    assert "it is\n  ground truth" not in out
    assert "not a human judgement" in out


def test_stale_verdict_never_reaches_the_prompt():
    from app import reverie

    ctx = reverie.build_reverie_context(
        _broadcast(),
        loop_outcomes={"ol-1": {"verdict": "decayed_unattended", "note": "", "actor": "system:implicit_decay",
                                "age_sec": 22 * 86400}},
    )
    out = _render(**ctx)
    assert "RECENT VERDICTS" not in out
    assert "verdict:" not in out
    assert "22" not in out


def test_world_first_signal_replaces_boilerplate_why_it_matters():
    from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

    from app import reverie

    loop = OpenLoopV1(
        id="ol-1", description="Bus synaptic prediction error",
        why_it_matters="novel or unresolved current-turn target with substrate pressure",
        magnitude=PredictionErrorMagnitudeV1(value=0.06213, age_sec=21.0, percentile_now=0.900138,
                                             band="high", trend="flat"),
        provenance={"world_first_reason": "body unusual for itself (high)"},
    )
    b = AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[loop]), attended_node_ids=["node:substrate.bus_synaptic"],
        selected_open_loop_id="ol-1",
    )
    out = _render(**reverie.build_reverie_context(b))
    assert ("ol-1 | Bus synaptic prediction error | band high; higher than 90% of its last 7 days; "
            "trend flat; value 0.0621; why selected: body unusual for itself (high)") in out
    assert "novel or unresolved current-turn target" not in out


def test_reverie_branch_does_not_call_one_node_a_coalition():
    from app import reverie

    out = _render(**reverie.build_reverie_context(_broadcast()))
    reverie_part = out.split("FAIL-CLOSED REMINDER")[0]
    # Only the explicit instruction NOT to open with "The coalition" may mention it.
    assert reverie_part.lower().count("coalition") == 1
    assert 'Do not open with "The\n  coalition"' in reverie_part


def test_chain_step_block_carries_prior_thought_and_hint():
    from app import reverie

    cont = reverie.ChainContinuity(index=1, is_final=False, prior_thoughts=["first thought about ol-1"],
                                   next_focus_hint="why it recurs at night")
    out = _render(**reverie.build_reverie_context(_broadcast(), continuity=cont))
    assert "TRAIN OF THOUGHT (step 2 of this chain)" in out
    assert '- "first thought about ol-1"' in out
    assert "Last step you said you wanted to turn next to: why it recurs at night" in out
    assert "- next_focus:" in out


def test_final_chain_step_does_not_ask_for_next_focus():
    from app import reverie

    cont = reverie.ChainContinuity(index=3, is_final=True, prior_thoughts=["a", "b", "c"])
    ctx = reverie.build_reverie_context(_broadcast(), continuity=cont)
    assert ctx["chain_step"]["prior_thoughts"] == ["b", "c"]  # last 2 only
    out = _render(**ctx)
    assert "- next_focus:" not in out


def test_standalone_tick_has_no_chain_block():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    assert "chain_step" not in ctx
    out = _render(**ctx)
    assert "TRAIN OF THOUGHT" not in out
    assert "- next_focus:" not in out


def test_outcome_block_absent_when_open_loops_is_empty():
    """Regression: the settled-loops instruction must not render when there is
    nothing in open_loops to apply it to (it used to render unconditionally)."""
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast(loops=()))
    out = _render(**ctx)
    assert "RECENT VERDICTS ON THESE LOOPS" not in out
    assert "open_loops:" not in out
