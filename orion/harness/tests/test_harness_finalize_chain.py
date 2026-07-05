from __future__ import annotations

import pytest

from orion.harness.finalize import emit_post_turn_closure, run_harness_finalize_chain
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.schemas.harness_finalize import GrammarReceiptV1


@pytest.mark.asyncio
async def test_run_harness_finalize_chain_orchestrates_5a_through_6b() -> None:
    thought = make_thought()
    coalition = build_coalition_snapshot(thought)
    receipts = [GrammarReceiptV1(step_index=0, summary="step", grammar_event_id="g-1")]
    draft_text = "internal draft"
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=receipts,
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    appraisal = make_appraisal()
    reflection = make_reflection()
    verdict_calls: list[str] = []
    outcome_calls: list[str] = []

    async def substrate_client(_mol: object):
        return appraisal

    async def cortex_client(_req: object):
        return {
            "final_text": reflection.model_dump(mode="json"),
            "trace_id": "trace-1",
        }

    async def verdict_publish_fn(*_: object, **__: object) -> None:
        verdict_calls.append("verdict")

    async def outcome_publish_fn(*_: object, **__: object) -> None:
        outcome_calls.append("outcome")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_voice_finalize_text",
            lambda _result: "final for juniper",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-1",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=receipts,
            repair_overlay=make_repair_overlay(),
            user_message="hello",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
            verdict_publish_fn=verdict_publish_fn,
            outcome_publish_fn=outcome_publish_fn,
        )

    assert chain.final_text == "final for juniper"
    assert chain.substrate_appraisal is appraisal
    assert verdict_calls == ["verdict"]
    assert outcome_calls == ["outcome"]


@pytest.mark.asyncio
async def test_emit_post_turn_closure_publishes() -> None:
    thought = make_thought()
    appraisal = make_appraisal()
    reflection = make_reflection()
    from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

    verdict = await emit_verdict_molecule(correlation_id="c-1", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-1",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="draft",
        final_text="final",
        finalize_changed=False,
    )
    published: list[object] = []

    async def capture(closure: object, **_: object) -> None:
        published.append(closure)

    closure = await emit_post_turn_closure(
        correlation_id="c-1",
        outcome_molecule=outcome,
        verdict_molecule_id="verdict-1",
        publish_fn=capture,
    )
    assert published == [closure]
    assert closure.surprise_unresolved is False
