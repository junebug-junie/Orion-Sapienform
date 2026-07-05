from __future__ import annotations

import pytest

from orion.harness.finalize import run_orion_voice_finalize
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_thought
from orion.schemas.cortex.schemas import PlanExecutionRequest


@pytest.mark.asyncio
async def test_voice_changes_on_misaligned_verdict() -> None:
    thought = make_thought()
    appraisal = make_appraisal()
    draft_text = "Motor draft that misses the relational frame."

    async def cortex_stub(plan_request: PlanExecutionRequest) -> dict[str, object]:
        reflection = plan_request.context["reflection"]
        verdict = reflection["alignment_verdict"]
        if verdict == "misaligned":
            return {"final_text": "Revised voice that meets the relational frame."}
        return {"final_text": draft_text}

    aligned_final, aligned_meta = await run_orion_voice_finalize(
        correlation_id="c-1",
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=make_reflection(alignment_verdict="aligned"),
        user_message="just be here with me",
        cortex_client=cortex_stub,
    )
    misaligned_final, misaligned_meta = await run_orion_voice_finalize(
        correlation_id="c-1",
        draft_text=draft_text,
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=make_reflection(alignment_verdict="misaligned"),
        user_message="just be here with me",
        cortex_client=cortex_stub,
    )

    assert misaligned_final != aligned_final
    assert misaligned_meta["finalize_changed"] is True
