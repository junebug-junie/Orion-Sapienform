"""A repair asked for by an unsupported perception claim is checked once more (2026-11-04).

Eval 2026-11-03 (orion/harness/evals/perception_claims_replay_eval.py): 5b caught
an invented name story 9/9, but the first repair kept it in 6/9, and nothing
looked at the repair again. Now: re-check once; if the claim is still there,
repair once more from the re-check's notes. Bounded, and fails open.
"""
from __future__ import annotations

import pytest

from orion.harness.finalize import (
    UNSUPPORTED_CLAIM_REASON,
    UNSUPPORTED_CLAIM_TWICE_REASON,
    has_unsupported_claim,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_appraisal, make_reflection, make_repair_overlay, make_thought

NOTES = "Situation:\n- Walkway rhythm: no match for Rowan (a person, name from Juniper, 10-25) yet today."
CLAIM = "unsupported claim: 'I started calling that slot Rowan'; notes say: 'name from Juniper'"


def _bad(note=CLAIM):
    return make_reflection(alignment_verdict="misaligned", alignment_notes=[note])


async def _run(reflections, repairs, *, notes=NOTES, fail_recheck=False):
    thought = make_thought()
    draft = "Rowan came right on her usual 11:39; I started calling that slot Rowan."
    molecule = build_draft_molecule(
        correlation_id="c-recheck", thought=thought, draft_text=draft, grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=make_repair_overlay(),
    )
    calls: list[str] = []
    reflections, repairs = list(reflections), list(repairs)

    async def substrate_client(_m):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(req):
        if "substrate_appraisal" in req.context:
            calls.append(f"reflect:{req.context['draft_text']}")
            if fail_recheck and len([c for c in calls if c.startswith("reflect")]) > 1:
                raise RuntimeError("cortex down")
            return {"reflection": reflections.pop(0)}
        calls.append(f"repair:{req.context['draft_text']}")
        return {"text": repairs.pop(0)}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("orion.harness.finalize.extract_finalize_reflection_payload",
                   lambda r: r["reflection"].model_dump(mode="json"))
        mp.setattr("orion.harness.finalize.extract_response_repair_text", lambda r: r["text"])
        chain = await run_harness_finalize_chain(
            correlation_id="c-recheck", draft_text=draft, draft_molecule=molecule, thought=thought,
            grammar_receipts=[], repair_overlay=make_repair_overlay(), user_message="where?",
            voice_contract=None, cortex_client=cortex_client, substrate_client=substrate_client,
            situation_notes=notes,
        )
    return chain, calls


def test_the_prefix_is_the_prompts_output_contract() -> None:
    assert has_unsupported_claim(_bad())
    assert has_unsupported_claim(_bad("  Unsupported claim: x; notes say: y"))
    assert not has_unsupported_claim(make_reflection(alignment_verdict="misaligned", alignment_notes=["tone off"]))


@pytest.mark.asyncio
async def test_clean_recheck_sends_the_first_repair() -> None:
    chain, calls = await _run([_bad(), make_reflection(alignment_verdict="aligned")], ["fixed once"])
    assert chain.final_text == "fixed once"
    assert chain.response_repair_reason == UNSUPPORTED_CLAIM_REASON
    assert calls == ["reflect:" + calls[0][8:], "repair:" + calls[0][8:], "reflect:fixed once"]


@pytest.mark.asyncio
async def test_claim_still_there_gets_one_more_repair_from_the_rechecks_notes() -> None:
    chain, calls = await _run([_bad(), _bad("unsupported claim: 'a vehicle slot'; notes say: 'a person'")],
                              ["fixed once", "fixed twice"])
    assert chain.final_text == "fixed twice"
    assert chain.response_repair_reason == UNSUPPORTED_CLAIM_TWICE_REASON
    assert calls[-2:] == ["reflect:fixed once", "repair:fixed once"]
    assert len(calls) == 4  # bounded: no third check


@pytest.mark.asyncio
async def test_no_recheck_for_other_misalignment_or_without_notes() -> None:
    other = make_reflection(alignment_verdict="misaligned", alignment_notes=["tone off"])
    chain, calls = await _run([other], ["fixed"])
    assert len(calls) == 2 and chain.response_repair_reason == "misaligned"
    chain, calls = await _run([_bad()], ["fixed"], notes=None)
    assert len(calls) == 2 and chain.response_repair_reason == "misaligned"


@pytest.mark.asyncio
async def test_a_failed_recheck_keeps_the_first_repair() -> None:
    chain, _calls = await _run([_bad()], ["fixed once"], fail_recheck=True)
    assert chain.final_text == "fixed once"
    assert chain.response_repair_reason == UNSUPPORTED_CLAIM_REASON
