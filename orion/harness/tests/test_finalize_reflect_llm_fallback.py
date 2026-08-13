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
async def test_reflect_text_payload_gets_server_owned_ids_injected() -> None:
    """Regression gate for the 2026-08-13 turn crash: 4 validation errors on
    FinalizeReflectionV1 (correlation_id, thought_event_id,
    substrate_appraisal_id, draft_hash all "Field required").

    cortex-exec returns the reflection as model TEXT, not a dict, so the
    identity-field injection -- which was guarded by `isinstance(raw_payload,
    dict)` -- never ran on a real turn. The harness owns these four fields
    and overwrites whatever the model sends (see the sibling test below).

    This asserts the SUCCESS path, not a fallback: a well-formed reflection
    must validate AND carry the server-injected ids. Asserting only "no
    crash" would pass against a fix that silently degrades every reflection
    to misaligned, which is the wrong repair.
    """
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    # Exactly what the model emits: the judgement fields, none of the ids.
    llm_text = json.dumps(
        {
            "schema_version": "finalize.reflection.v1",
            "imperative": "Confirm system state",
            "tone": "steady",
            "strain_refs": [],
            "alignment_verdict": "aligned",
            "alignment_notes": ["draft matches imperative"],
            "strain_unresolved": False,
        }
    )

    async def _text_payload(_req: object) -> dict[str, object]:
        return {"final_text": llm_text, "trace_id": "t-abc"}

    reflection, quick_skipped, trace = await run_finalize_reflection(
        correlation_id="c-text",
        draft_text="partial draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        cortex_client=_text_payload,
    )

    assert quick_skipped is False
    assert trace == "t-abc"
    # The real fix: parsed as a genuine reflection, NOT degraded.
    assert reflection.reflection_source == "substrate_informed_pass"
    assert reflection.alignment_verdict == "aligned"
    # The four fields that were missing in the crash.
    assert reflection.correlation_id == "c-text"
    assert reflection.thought_event_id == thought.event_id
    assert reflection.substrate_appraisal_id == appraisal.molecule_id
    assert reflection.draft_hash == appraisal.draft_hash


@pytest.mark.asyncio
async def test_reflect_server_owned_ids_overwrite_model_supplied_values() -> None:
    """The harness owns the four identity fields; a model-supplied value must
    lose. This was `setdefault` (model wins) while the injection was dead code,
    so it had never been exercised. It matters because _reflection_id() and
    _verdict_molecule_id() hash the reflection's copies while the sibling
    fields on the same outcome molecule come from the real appraisal object --
    one transcribed-wrong draft_hash and those keys silently decorrelate.
    """
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    llm_text = json.dumps(
        {
            "schema_version": "finalize.reflection.v1",
            "correlation_id": "WRONG-corr",
            "thought_event_id": "WRONG-thought",
            "substrate_appraisal_id": "WRONG-appraisal",
            "draft_hash": "WRONG-hash",
            "imperative": "Confirm system state",
            "tone": "steady",
            "strain_refs": [],
            "alignment_verdict": "aligned",
            "alignment_notes": [],
            "strain_unresolved": False,
        }
    )

    async def _lying_payload(_req: object) -> dict[str, object]:
        return {"final_text": llm_text}

    reflection, _quick_skipped, _trace = await run_finalize_reflection(
        correlation_id="c-authoritative",
        draft_text="partial draft",
        thought=thought,
        substrate_appraisal=appraisal,
        repair_overlay=make_repair_overlay(),
        cortex_client=_lying_payload,
    )

    assert reflection.correlation_id == "c-authoritative"
    assert reflection.thought_event_id == thought.event_id
    assert reflection.substrate_appraisal_id == appraisal.molecule_id
    assert reflection.draft_hash == appraisal.draft_hash


@pytest.mark.asyncio
async def test_reflect_malformed_payload_uses_degraded_reflection() -> None:
    """A payload that is not parseable JSON must fail closed to the degraded
    reflection rather than raising out of the turn. Complements the test
    above: that one proves good payloads still succeed, this one proves the
    parse itself is inside the guarded block."""
    thought = make_thought(repair_pressure_level=0.9)
    appraisal = make_appraisal(surprise_level=0.5)

    async def _malformed(_req: object) -> dict[str, object]:
        return {"final_text": "not valid JSON at all", "trace_id": "t-bad"}

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
