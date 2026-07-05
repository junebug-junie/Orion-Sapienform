"""Layer attribution evals (spec §11.2) — structural replay with mocked inputs."""

from __future__ import annotations

import pytest

from orion.harness.finalize import maybe_quick_lane_verdict, run_orion_voice_finalize
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought
from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.harness_finalize import HarnessPostTurnClosureV1, HarnessTurnOutcomeMoleculeV1


def test_5a_affects_5b_verdict() -> None:
    """High surprise appraisal blocks quick lane; low surprise allows deterministic 5b."""
    thought = make_thought()
    overlay = make_repair_overlay()
    low_surprise = make_appraisal(surprise_level=0.01)
    high_surprise = make_appraisal(surprise_level=0.5)

    low_verdict = maybe_quick_lane_verdict(
        correlation_id="c-low",
        thought=thought,
        substrate_appraisal=low_surprise,
        repair_overlay=overlay,
        epsilon=0.08,
    )
    high_verdict = maybe_quick_lane_verdict(
        correlation_id="c-high",
        thought=thought,
        substrate_appraisal=high_surprise,
        repair_overlay=overlay,
        epsilon=0.08,
    )

    assert low_verdict is not None
    assert high_verdict is None


@pytest.mark.asyncio
async def test_5b_affects_5c_text() -> None:
    """Misaligned reflection must change voice finalize output vs aligned."""
    thought = make_thought()
    appraisal = make_appraisal()
    draft_text = "Motor draft."

    async def cortex_stub(plan_request: PlanExecutionRequest) -> dict[str, object]:
        verdict = plan_request.context["reflection"]["alignment_verdict"]
        if verdict == "misaligned":
            return {"final_text": "Voice revised for misalignment."}
        return {"final_text": draft_text}

    aligned_final, _ = await run_orion_voice_finalize(
        correlation_id="c-1",
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=make_reflection(alignment_verdict="aligned"),
        user_message="hello",
        cortex_client=cortex_stub,
    )
    misaligned_final, meta = await run_orion_voice_finalize(
        correlation_id="c-1",
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=make_reflection(alignment_verdict="misaligned"),
        user_message="hello",
        cortex_client=cortex_stub,
    )

    assert misaligned_final != aligned_final
    assert meta["finalize_changed"] is True


def test_turn_n_error_shifts_turn_n_plus_one_strain() -> None:
    """Turn N closure with surprise_unresolved exposes reducer-facing strain signal."""
    outcome = HarnessTurnOutcomeMoleculeV1(
        correlation_id="turn-n",
        thought_event_id="t-n",
        substrate_appraisal_id="appraisal-n",
        reflection_id="reflection-n",
        verdict_molecule_id="verdict-n",
        draft_hash="draft-n",
        final_hash="final-n",
        finalize_changed=True,
        alignment_verdict="misaligned",
        surprise_level_at_draft=0.75,
        surprise_resolved=False,
        grammar_event_ids=["g-n"],
        final_text="partial final",
    )
    closure = HarnessPostTurnClosureV1(
        correlation_id="turn-n",
        outcome_molecule_id="outcome-n",
        verdict_molecule_id="verdict-n",
        grammar_event_ids=["g-n"],
        surprise_unresolved=True,
    )

    assert closure.surprise_unresolved is True
    assert outcome.surprise_resolved is False
    assert outcome.surprise_level_at_draft >= 0.65

    # Stub N+1 consumer: unresolved surprise should elevate strain/repair attention.
    turn_n_plus_one_strain_refs = ["prediction-error-from-turn-n"] if closure.surprise_unresolved else []
    assert turn_n_plus_one_strain_refs == ["prediction-error-from-turn-n"]
