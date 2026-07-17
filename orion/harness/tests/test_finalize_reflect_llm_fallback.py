from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.harness.finalize import run_finalize_reflection
from orion.harness.tests.fixtures import make_appraisal, make_repair_overlay, make_thought


@pytest.mark.asyncio
async def test_reflect_llm_failure_uses_degraded_reflection_when_quick_lane_blocked() -> None:
    """Degraded fallback must fail closed: reflect LLM failure is not proof the
    draft is aligned, so the fallback verdict is "misaligned" (not "aligned").
    This forces the downstream voice-finalize pass (5c) to materially revise
    the draft instead of shipping it through unreviewed."""
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    async def _boom(_req: object) -> dict[str, object]:
        raise RuntimeError("llamacpp 400")

    reflection, quick_skipped, _trace = await run_finalize_reflection(
        correlation_id="c-fallback",
        draft_text="partial draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        cortex_client=_boom,
    )

    assert quick_skipped is False
    assert reflection.reflection_source == "degraded_llm_failure_fallback"
    assert reflection.alignment_verdict == "misaligned"
