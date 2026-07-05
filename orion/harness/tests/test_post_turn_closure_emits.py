from __future__ import annotations

import pytest

from orion.harness.finalize import emit_post_turn_closure, emit_turn_outcome_molecule, emit_verdict_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought
from orion.schemas.harness_finalize import HarnessPostTurnClosureV1


@pytest.mark.asyncio
async def test_post_turn_closure_emits_on_outcome() -> None:
    thought = make_thought()
    appraisal = make_appraisal()
    reflection = make_reflection()
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
    published: list[HarnessPostTurnClosureV1] = []

    async def capture(closure: HarnessPostTurnClosureV1, **_: object) -> None:
        published.append(closure)

    closure = await emit_post_turn_closure(
        correlation_id="c-1",
        outcome_molecule=outcome,
        verdict_molecule_id=verdict.reflection.substrate_appraisal_id,
        publish_fn=capture,
    )

    assert published == [closure]
    assert closure.correlation_id == "c-1"
    assert closure.closure_source == "harness_post_turn_appraisal"
    assert closure.surprise_unresolved is False


@pytest.mark.asyncio
async def test_post_turn_closure_emits_prediction_error_signal() -> None:
    """High surprise unresolved → closure.surprise_unresolved for N+1 strain reducers."""
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.9)
    reflection = make_reflection(
        alignment_verdict="misaligned",
        strain_unresolved=True,
    )
    verdict = await emit_verdict_molecule(correlation_id="c-2", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-2",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="draft",
        final_text="revised final",
        finalize_changed=True,
    )
    closure = await emit_post_turn_closure(
        correlation_id="c-2",
        outcome_molecule=outcome,
        verdict_molecule_id="verdict-2",
    )
    assert closure.surprise_unresolved is True
    assert closure.outcome_molecule_id
    assert closure.verdict_molecule_id == "verdict-2"
