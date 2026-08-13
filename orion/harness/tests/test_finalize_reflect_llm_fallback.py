from __future__ import annotations

import json
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


@pytest.mark.asyncio
async def test_reflect_llm_payload_missing_id_fields_backfilled_not_degraded() -> None:
    """The enrichment defaults (correlation_id/thought_event_id/
    substrate_appraisal_id/draft_hash) exist to backstop an LLM payload that
    omits just those bookkeeping fields while still supplying real reflection
    content (imperative/tone/strain_refs/alignment_verdict/alignment_notes/
    strain_unresolved) -- per the prompt template
    (orion/cognition/prompts/harness_finalize_reflect.j2), the LLM is asked
    to echo those IDs itself, and a truncation/omission failure mode that
    drops just one is realistic.

    Confirmed by hand this previously did NOT work even after moving the
    parse call inside the try: extract_finalize_reflection_payload only ever
    returns a `str` (never the `dict` its enrichment's `isinstance(...,
    dict)` guard checked for), so the setdefault block was silently dead --
    this payload used to force an unnecessary degrade instead of validating.
    Now the defaults are merged inside parse_finalize_reflection_payload
    itself, after the str->dict conversion, so they actually apply."""
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    async def _missing_ids_only(_req: object) -> dict[str, object]:
        payload = {
            "imperative": "respond helpfully",
            "tone": "neutral",
            "strain_refs": [],
            "alignment_verdict": "aligned",
            "alignment_notes": [],
            "strain_unresolved": False,
        }
        return {"final_text": json.dumps(payload)}

    reflection, quick_skipped, _trace = await run_finalize_reflection(
        correlation_id="c-backfill",
        draft_text="partial draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        cortex_client=_missing_ids_only,
    )

    assert quick_skipped is False
    assert reflection.reflection_source == "substrate_informed_pass"
    assert reflection.alignment_verdict == "aligned"
    assert reflection.correlation_id == "c-backfill"
    assert reflection.thought_event_id == thought.event_id
    assert reflection.substrate_appraisal_id == appraisal.molecule_id
    assert reflection.draft_hash == appraisal.draft_hash
