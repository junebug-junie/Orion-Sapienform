"""Reverie chain that develops (2026-10-16, spec 2026-07-14 items #2/#3).

Chain steps feed forward (earlier thoughts + the previous step's next_focus),
the prompt asks for next_focus in chain mode only, and a step that returns no
next_focus ends the chain (thought_complete).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

NOW = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)
_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
GROUNDED = "The execution signal ol-1 is far above its own usual week and nothing has explained it yet."


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="Execution prediction error",
                                                      salience=0.8)]),
        attended_node_ids=["node:substrate.execution"],
        selected_open_loop_id="ol-1",
        selected_description="Execution prediction error",
    )


def _thought(idx, next_focus):
    return SpontaneousThoughtV1(
        thought_id=f"th-{idx}", correlation_id="c",
        coalition=CoalitionSnapshotV1(attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
                                      open_loop_ids=["ol-1"], generated_at=NOW),
        interpretation=f"step {idx}: {GROUNDED}", salience=0.8, evidence_refs=["ol-1"],
        next_focus=next_focus,
    )


# --- feed-forward and early stop ----------------------------------------------

@pytest.mark.asyncio
async def test_step_n_receives_previous_text_and_next_focus():
    from app import chain

    calls = []

    async def step(chain_id, index, *, prior_thoughts, next_focus_hint, is_last):
        calls.append((index, prior_thoughts, next_focus_hint, is_last))
        return _thought(index, f"focus-{index}")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=3, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "max_steps"
    assert calls[0] == (0, (), None, False)
    assert calls[1] == (1, (f"step 0: {GROUNDED}",), "focus-0", False)
    assert calls[2] == (2, (f"step 0: {GROUNDED}", f"step 1: {GROUNDED}"), "focus-1", True)


@pytest.mark.asyncio
async def test_chain_stops_when_a_step_returns_no_next_focus():
    from app import chain

    store = chain.InMemoryRefractoryStore()

    async def step(chain_id, index, **_kw):
        return _thought(index, "keep looking" if index == 0 else None)

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=store,
        broadcast_reader=_broadcast, max_steps=4, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "thought_complete"
    assert c.thought_ids == ["th-0", "th-1"]
    assert store.is_suppressed("ol-1", NOW) is True  # a finished thought habituates


@pytest.mark.asyncio
async def test_last_step_without_next_focus_is_max_steps_not_complete():
    from app import chain

    async def step(chain_id, index, *, is_last, **_kw):
        return _thought(index, None if is_last else "more")

    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=_broadcast, max_steps=2, publish=False, now_fn=lambda: NOW,
    )
    assert c.terminal_reason == "max_steps"
    assert len(c.thought_ids) == 2


def test_thought_complete_chain_queues_a_compaction_request():
    from app import chain
    from orion.schemas.reverie import ReverieChainV1

    req = chain.build_compaction_request(
        ReverieChainV1(chain_id="ch", theme_key="ol-1", thought_ids=["t"], terminal_reason="thought_complete")
    )
    assert req is not None and req.reason == "reverie_chain_thought_complete"


# --- chain-mode prompt block only in chain mode -------------------------------

def test_chain_block_renders_only_in_chain_mode():
    from app import reverie

    standalone = _render(**reverie.build_reverie_context(_broadcast()))
    assert "TRAIN OF THOUGHT" not in standalone
    assert "next_focus:" not in standalone

    step = reverie.ChainStep("ch", 1, ("I noticed execution spiking.",), "what ran just before")
    chained = _render(**reverie.build_reverie_context(_broadcast(), chain_step=step))
    assert "TRAIN OF THOUGHT (this is step 2" in chained
    assert '"I noticed execution spiking."' in chained
    assert "Take it one step further — do not restate it." in chained
    assert "look next at: what ran just before" in chained
    assert "- next_focus:" in chained

    last = reverie.ChainStep("ch", 3, ("x" * 50,), None, True)
    final = _render(**reverie.build_reverie_context(_broadcast(), chain_step=last))
    assert "TRAIN OF THOUGHT" in final and "- next_focus:" not in final


def test_chain_carry_is_bounded():
    from app import reverie

    step = reverie.ChainStep("ch", 5, tuple(f"t{i} " + "y" * 1000 for i in range(6)), "z" * 900)
    ctx = reverie.chain_context_for_prompt(step)
    assert len(ctx["chain_prior_thoughts"]) == reverie.MAX_CHAIN_PRIOR_THOUGHTS
    assert ctx["chain_prior_thoughts"][0].startswith("t3 ")  # the most recent ones
    assert all(len(t) <= reverie.MAX_CHAIN_PRIOR_THOUGHT_CHARS for t in ctx["chain_prior_thoughts"])
    assert len(ctx["chain_next_focus_hint"]) == reverie.MAX_NEXT_FOCUS_CHARS


def test_first_chain_step_asks_for_next_focus_without_prior_block():
    from app import reverie

    out = _render(**reverie.build_reverie_context(_broadcast(), chain_step=reverie.ChainStep("ch", 0)))
    assert "TRAIN OF THOUGHT (this is step 1" in out
    assert "Earlier in this train of thought" not in out
    assert "- next_focus:" in out


# --- the tick threads chain context and parses next_focus ---------------------

@pytest.mark.asyncio
async def test_tick_threads_chain_step_and_parses_next_focus(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie, "load_recent_loop_outcomes", lambda ids: {})
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps({
        "interpretation": GROUNDED, "evidence_refs": ["ol-1"], "next_focus": "  what ran before it  ",
    })})
    step = reverie.ChainStep("ch-1", 1, ("earlier words about ol-1",), "the deploy")
    t = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast,
                                       cortex_client=cortex, chain_context=step)
    assert t is not None
    assert (t.chain_id, t.thought_index, t.next_focus) == ("ch-1", 1, "what ran before it")
    ctx = cortex.execute_plan.call_args.kwargs["req"].context
    assert ctx["chain_prior_thoughts"] == ["earlier words about ol-1"]
    assert ctx["chain_next_focus_hint"] == "the deploy"


@pytest.mark.asyncio
async def test_standalone_tick_never_keeps_a_next_focus(monkeypatch):
    from app import reverie

    monkeypatch.setattr(reverie, "load_recent_loop_outcomes", lambda ids: {})
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={"final_text": json.dumps({
        "interpretation": GROUNDED, "evidence_refs": ["ol-1"], "next_focus": "stray",
    })})
    t = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert t is not None and t.next_focus is None
    assert "chain_mode" not in cortex.execute_plan.call_args.kwargs["req"].context


@pytest.mark.parametrize("raw", [None, "", "   ", "null", "None", 7])
def test_empty_next_focus_parses_to_none(raw):
    from app import reverie

    t = reverie.parse_reverie_payload(
        {"interpretation": GROUNDED, "evidence_refs": ["ol-1"], "next_focus": raw},
        coalition=CoalitionSnapshotV1(attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
                                      open_loop_ids=["ol-1"], generated_at=NOW),
        correlation_id="c", broadcast=_broadcast(),
    )
    assert t.next_focus is None


def test_legacy_tuple_chain_context_still_reads():
    from app import reverie

    step = reverie._as_chain_step(("ch", 2))
    assert step == reverie.ChainStep("ch", 2) and step[0] == "ch" and step[1] == 2


@pytest.mark.asyncio
async def test_calm_tick_still_runs_expectation_scoring(monkeypatch):
    """A calm tick skips the chain but must not starve expectation scoring,
    which otherwise only runs inside run_reverie_once."""
    from app import chain

    scorer = AsyncMock()
    monkeypatch.setattr(chain, "_maybe_score_pending_expectation", scorer)
    calm = AttentionBroadcastProjectionV1(frame=AttentionFrameV1(open_loops=[]),
                                          attended_node_ids=[], selected_open_loop_id=None)
    step = AsyncMock()
    c = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=lambda: calm, publish=False, now_fn=lambda: NOW,
    )
    assert c is None
    step.assert_not_awaited()
    scorer.assert_awaited_once()


def test_concern_card_branch_renders_chain_block_and_asks_next_focus():
    """Semantic-lift chains must also be asked for next_focus, or every lift
    chain would stop after step 1 as thought_complete."""
    from app import reverie
    from orion.schemas.reverie import ConcernCardV1

    card = ConcernCardV1(card_id="c1", coalition_ref="harness_closure:x", human_text="t" * 40)
    step = reverie.ChainStep("ch", 1, ("an earlier concern thought",), "the deploy")
    out = _render(**reverie.build_reverie_context(_broadcast(), concern_cards=[card], chain_step=step))
    assert "concern_cards" in out or "PRIMARY INPUT" in out
    assert "TRAIN OF THOUGHT (this is step 2" in out
    assert '"an earlier concern thought"' in out
    assert "- next_focus:" in out
