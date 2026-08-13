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


@pytest.mark.asyncio
async def test_reflect_llm_malformed_payload_uses_degraded_reflection() -> None:
    """When the LLM's payload extracts cleanly (a real `final_text` field) but
    then fails FinalizeReflectionV1's own Pydantic validation, the error must
    fall through to the degraded reflection path instead of escaping
    run_finalize_reflection uncaught.

    The payload here matters: it must survive extract_finalize_reflection_payload
    (so it doesn't raise from a DIFFERENT place -- missing/unrecognized text
    fields raise inside extract_cortex_payload_text, which is always inside
    the try and therefore insensitive to this bug either way) and only fail
    at FinalizeReflectionV1.model_validate(...) itself, which is the call this
    fix actually moved inside the try. Confirmed by hand: this exact payload
    (`{"final_text": "{}"}`) raises pydantic.ValidationError uncaught on the
    pre-fix code (parse call outside try) and degrades gracefully on the
    fixed code -- a payload that fails at extraction instead (e.g. missing
    `final_text`/`text`/`content`) passes on BOTH pre- and post-fix code, so
    it would not actually have caught this regression."""
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    async def _malformed(_req: object) -> dict[str, object]:
        return {"final_text": "{}"}

    reflection, quick_skipped, _trace = await run_finalize_reflection(
        correlation_id="c-malformed",
        draft_text="partial draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        cortex_client=_malformed,
    )

    assert quick_skipped is False
    assert reflection.reflection_source == "degraded_llm_failure_fallback"
    assert reflection.alignment_verdict == "misaligned"
