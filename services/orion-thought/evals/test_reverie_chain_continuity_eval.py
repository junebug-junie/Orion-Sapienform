"""Replay eval: a reverie chain is a train of thought, not one prompt sampled N times.

Replays recorded thoughts from real 2026-10-09..16 chains (fixture provenance in
the JSON) through the real `run_reverie_chain` and renders the real
`reverie_narrate.j2` prompt for every step. Before this change every step of a
chain rendered the identical prompt (no prior thought fed forward,
`next_focus` populated 0/647). Asserts:

- step k's prompt carries step k-1's thought verbatim;
- no two consecutive prompts in a chain are identical;
- a recorded chain whose step 1 restated step 0 now stops at step 1 ("settled")
  instead of running all 4 steps;
- the attended loop's real world-first numbers reach the prompt and the
  boilerplate why_it_matters does not;
- the reverie prompt never calls the single attended node a "coalition".
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
    PredictionErrorMagnitudeV1,
)
from orion.schemas.reverie import SpontaneousThoughtV1
from orion.schemas.thought import CoalitionSnapshotV1

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "reverie_chains_2026-10-16.json").read_text())
_PROMPT = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
_TEMPLATE = Environment().from_string(_PROMPT.read_text())


def _broadcast() -> AttentionBroadcastProjectionV1:
    raw = _FIXTURE["broadcast_loop"]
    loop = OpenLoopV1(
        id=raw["id"], description=raw["description"], target_type=raw["target_type"],
        why_it_matters=raw["why_it_matters"], salience=raw["salience"],
        magnitude=PredictionErrorMagnitudeV1(**raw["magnitude"]),
        provenance=raw["provenance"], source_refs=raw["source_refs"],
    )
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[loop]),
        attended_node_ids=raw["attended_node_ids"],
        selected_open_loop_id=raw["id"],
        dwell_ticks=raw["dwell_ticks"],
    )


async def _replay(thoughts: list[str], max_steps: int = 4):
    """Run the real chain; each step renders the real prompt, then 'answers' with the recorded thought."""
    from app import chain, reverie

    broadcast = _broadcast()
    prompts: list[str] = []

    async def step(chain_id, index, continuity):
        ctx = reverie.build_reverie_context(broadcast, continuity=continuity)
        prompts.append(_TEMPLATE.render(**ctx))
        return SpontaneousThoughtV1(
            thought_id=f"{chain_id}-{index}", correlation_id="eval",
            coalition=CoalitionSnapshotV1(attended_node_ids=list(broadcast.attended_node_ids),
                                         selected_open_loop_id=broadcast.selected_open_loop_id,
                                         open_loop_ids=[broadcast.selected_open_loop_id],
                                         generated_at=broadcast.generated_at),
            interpretation=thoughts[index], salience=0.8,
            evidence_refs=[broadcast.selected_open_loop_id],
            next_focus=None if continuity.is_final else "what keeps it elevated",
        )

    result = await chain.run_reverie_chain(
        AsyncMock(), step_fn=step, refractory_store=chain.InMemoryRefractoryStore(),
        broadcast_reader=lambda: broadcast, max_steps=max_steps, publish=False,
    )
    return result, prompts


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", _FIXTURE["recorded_chains"], ids=lambda r: r["chain_id"][:8])
async def test_each_step_prompt_carries_the_previous_thought(recorded):
    thoughts = recorded["thoughts"]
    result, prompts = await _replay(thoughts)
    assert result.terminal_reason == "max_steps"
    assert len(prompts) == len(thoughts) == 4
    for k in range(1, len(prompts)):
        assert thoughts[k - 1] in prompts[k], f"step {k} prompt lacks step {k - 1}'s thought"
        assert prompts[k] != prompts[k - 1], f"steps {k - 1} and {k} rendered the identical prompt"
    assert "TRAIN OF THOUGHT" in prompts[0] and "What you already thought" not in prompts[0]


@pytest.mark.asyncio
async def test_recorded_restatement_now_settles_instead_of_running_four_steps():
    rec = _FIXTURE["repeat_chain"]
    result, prompts = await _replay(rec["thoughts"])
    assert result.terminal_reason == "settled"
    assert len(result.thought_ids) == 2  # was 4 in the recorded chain


@pytest.mark.asyncio
async def test_prompt_carries_real_signal_and_no_coalition_label():
    _, prompts = await _replay(_FIXTURE["recorded_chains"][0]["thoughts"], max_steps=1)
    p = prompts[0].split("FAIL-CLOSED REMINDER")[0]
    assert "band high; higher than 90% of its last 7 days" in p
    assert "why selected: body unusual for itself (high)" in p
    assert "novel or unresolved current-turn target" not in p
    assert p.lower().count("coalition") == 1  # only the "do not open with 'The coalition'" rule
