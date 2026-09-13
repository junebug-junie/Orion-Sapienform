from __future__ import annotations

import pytest

from orion.harness.finalize import (
    needs_response_repair,
    response_repair_reason_for,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)


def test_needs_repair_false_when_aligned() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="aligned")) is False
    assert response_repair_reason_for(make_reflection()) is None


def test_needs_repair_true_for_misaligned_uncertain_strain() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="misaligned")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="misaligned")) == "misaligned"
    assert needs_response_repair(make_reflection(alignment_verdict="uncertain")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="uncertain")) == "uncertain"
    assert (
        needs_response_repair(
            make_reflection(alignment_verdict="aligned", strain_unresolved=True)
        )
        is True
    )
    assert (
        response_repair_reason_for(
            make_reflection(alignment_verdict="aligned", strain_unresolved=True)
        )
        == "strain_unresolved"
    )


@pytest.mark.asyncio
async def test_aligned_chain_passthrough_skips_repair_llm() -> None:
    thought = make_thought()
    draft_text = "hey. i'm here. what's on your mind?"
    molecule = build_draft_molecule(
        correlation_id="c-yo",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    cortex_calls: list[object] = []

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.25)

    async def cortex_client(req: object):
        cortex_calls.append(req)
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t-5b"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_response_repair_text",
            lambda _result: "assumption: checking in. next concrete move: story or quiet.",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-yo",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(
                mode="concrete_bias", rule_lines=["show assumptions"]
            ),
            user_message="yo",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
        )

    assert chain.final_text == draft_text
    assert chain.finalize_changed is False
    assert chain.response_repair_ran is False
    assert chain.response_repair_reason is None
    # Overlay must not force repair: only 5b (one cortex call), never a rewrite LLM.
    assert len(cortex_calls) == 1


@pytest.mark.asyncio
async def test_misaligned_chain_invokes_repair() -> None:
    thought = make_thought()
    draft_text = "motor draft that misses the frame"
    molecule = build_draft_molecule(
        correlation_id="c-mis",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="misaligned", strain_unresolved=False)

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(_req: object):
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_response_repair_text",
            lambda _result: "repaired text",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-mis",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(),
            user_message="hello",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
        )

    assert chain.final_text == "repaired text"
    assert chain.response_repair_ran is True
    assert chain.response_repair_reason == "misaligned"
    assert chain.finalize_changed is True
