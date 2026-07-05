from __future__ import annotations

import pytest

from orion.harness.finalize import run_finalize_reflection
from orion.harness.tests.fixtures import make_appraisal, make_thought
from orion.schemas.cortex.schemas import PlanExecutionRequest


@pytest.mark.asyncio
async def test_reflect_consumes_substrate_appraisal() -> None:
    thought = make_thought()

    async def _unexpected_cortex(_: PlanExecutionRequest) -> dict[str, object]:
        raise AssertionError("cortex should not run without substrate appraisal")

    with pytest.raises(ValueError, match="substrate_appraisal"):
        await run_finalize_reflection(
            correlation_id="c-1",
            draft_text="draft answer",
            thought=thought,
            substrate_appraisal=None,
            cortex_client=_unexpected_cortex,
        )


@pytest.mark.asyncio
async def test_reflect_quick_lane_skips_cortex_when_eligible() -> None:
    thought = make_thought()
    appraisal = make_appraisal()
    cortex_called = False

    async def _cortex(_: PlanExecutionRequest) -> dict[str, object]:
        nonlocal cortex_called
        cortex_called = True
        return {}

    reflection, quick_lane_skipped_5b, cortex_trace_id = await run_finalize_reflection(
        correlation_id="c-1",
        draft_text="draft answer",
        thought=thought,
        substrate_appraisal=appraisal,
        cortex_client=_cortex,
    )

    assert quick_lane_skipped_5b is True
    assert cortex_called is False
    assert cortex_trace_id is None
    assert reflection.reflection_source == "deterministic_quick_gate"
    assert reflection.quick_lane_skipped_llm is True
    assert reflection.substrate_appraisal_id == appraisal.molecule_id
