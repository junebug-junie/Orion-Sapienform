from __future__ import annotations

import pytest

from orion.harness.finalize import (
    STRAIN_ONLY_NOT_REPAIRED_REASON,
    needs_response_repair,
    response_repair_reason_for,
    response_repair_skip_reason,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)


def test_needs_repair_false_when_aligned() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="aligned")) is False
    assert response_repair_reason_for(make_reflection()) is None


def test_needs_repair_true_for_misaligned_uncertain() -> None:
    assert needs_response_repair(make_reflection(alignment_verdict="misaligned")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="misaligned")) == "misaligned"
    assert needs_response_repair(make_reflection(alignment_verdict="uncertain")) is True
    assert response_repair_reason_for(make_reflection(alignment_verdict="uncertain")) == "uncertain"
    # Open strain does not change a rejected draft's reason.
    assert response_repair_reason_for(
        make_reflection(alignment_verdict="uncertain", strain_unresolved=True)) == "uncertain"


def test_open_strain_alone_does_not_repair_an_aligned_draft() -> None:
    """#3618: 56% of aligned + strain-only repairs returned the draft unchanged; a rewrite
    cannot close Orion's open loops. The skip is named, never silent."""
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=True)
    assert needs_response_repair(reflection) is False
    assert response_repair_reason_for(reflection) is None
    assert response_repair_skip_reason(reflection) == STRAIN_ONLY_NOT_REPAIRED_REASON
    # Aligned with no strain: no skip reason either.
    assert response_repair_skip_reason(make_reflection(alignment_verdict="aligned")) is None
    # A rejected draft is repaired, so it has no skip reason.
    assert response_repair_skip_reason(
        make_reflection(alignment_verdict="misaligned", strain_unresolved=True)) is None


@pytest.mark.parametrize("strain", [True, False])
def test_an_unsupported_claim_repairs_an_aligned_draft(strain: bool) -> None:
    """An unsupported-claim note names a defect in the reply itself. Before #3618 an
    aligned draft with one was repaired only when strain happened to be open too
    (live 30 days: 1 repaired, 2 shipped as written)."""
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=strain,
                                 alignment_notes=["unsupported claim: a car at 18:20; notes say none"])
    assert needs_response_repair(reflection) is True
    assert response_repair_reason_for(reflection) == "unsupported_claim"
    assert response_repair_skip_reason(reflection) is None


@pytest.mark.asyncio
async def test_aligned_chain_passthrough_skips_repair_llm() -> None:
    thought = make_thought()
    draft_text = "hey. i'm here. what's on your mind?"
    molecule = build_draft_molecule(
        correlation_id="c-yo",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    cortex_calls: list[object] = []

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.25)

    async def cortex_client(req: object):
        cortex_calls.append(req)
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t-5b"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_response_repair_text",
            lambda _result: "assumption: checking in. next concrete move: story or quiet.",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-yo",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(
                mode="concrete_bias", rule_lines=["show assumptions"]
            ),
            user_message="yo",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
        )

    assert chain.final_text == draft_text
    assert chain.finalize_changed is False
    assert chain.response_repair_ran is False
    assert chain.response_repair_reason is None
    # Overlay must not force repair: only 5b (one cortex call), never a rewrite LLM.
    assert len(cortex_calls) == 1


@pytest.mark.asyncio
async def test_misaligned_chain_invokes_repair() -> None:
    thought = make_thought()
    draft_text = "motor draft that misses the frame"
    molecule = build_draft_molecule(
        correlation_id="c-mis",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="misaligned", strain_unresolved=False)

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(_req: object):
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_response_repair_text",
            lambda _result: "repaired text",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-mis",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(),
            user_message="hello",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
        )

    assert chain.final_text == "repaired text"
    assert chain.response_repair_ran is True
    assert chain.response_repair_reason == "misaligned"
    assert chain.finalize_changed is True


@pytest.mark.asyncio
async def test_cut_short_chain_skips_repair_and_keeps_marked_findings() -> None:
    """A misaligned verdict would normally trigger a prose rewrite. For a cut-short
    findings draft that rewrite synthesizes evidence into a finished-sounding
    answer, so it is skipped and the outcome molecule carries the marked text."""
    from orion.harness.cut_short import CUT_SHORT_MARKER

    thought = make_thought()
    draft_text = f"{CUT_SHORT_MARKER}\nThis turn was stopped...\n- Read returned: real evidence"
    molecule = build_draft_molecule(
        correlation_id="c-cut",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="misaligned", strain_unresolved=False)

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(_req: object):
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda _result: reflection.model_dump(mode="json"),
        )
        mp.setattr(
            "orion.harness.finalize.extract_response_repair_text",
            lambda _result: "Here is the complete answer.",
        )
        chain = await run_harness_finalize_chain(
            correlation_id="c-cut",
            draft_text=draft_text,
            draft_molecule=molecule,
            thought=thought,
            grammar_receipts=[],
            repair_overlay=make_repair_overlay(),
            user_message="investigate",
            voice_contract=None,
            cortex_client=cortex_client,
            substrate_client=substrate_client,
            cut_short=True,
        )

    assert chain.final_text == draft_text
    assert chain.response_repair_ran is False
    assert chain.finalize_changed is False
    assert chain.outcome_molecule is not None
    assert chain.outcome_molecule.final_text == draft_text


def test_a_failed_judge_repairs_with_reason_reflection_failed_not_misaligned():
    # Unified-turn A2: the fallback "misaligned" from a failed 5b call still triggers repair, but the
    # reason readers see says the judge failed, not that the draft missed.
    from orion.schemas.harness_finalize import REFLECTION_SOURCE_JUDGE_FAILED

    failed = make_reflection(alignment_verdict="misaligned", reflection_source=REFLECTION_SOURCE_JUDGE_FAILED)
    assert needs_response_repair(failed)
    assert response_repair_reason_for(failed) == "reflection_failed"


@pytest.mark.asyncio
async def test_a_failed_required_repair_is_recorded_as_ran_and_failed() -> None:
    """Fail-closed acceptance (#3686 review): 5b rejected the draft, the repair call
    fails. Nothing is published as final text, and the outcome says the repair ran
    and failed, with the reason and the error."""
    from orion.harness.finalize import HarnessFinalizeFailedError

    thought = make_thought()
    draft_text = "motor draft that misses the frame"
    molecule = build_draft_molecule(
        correlation_id="c-fail", thought=thought, draft_text=draft_text, grammar_receipts=[],
        coalition_snapshot=build_coalition_snapshot(thought), repair_overlay=make_repair_overlay(),
    )
    reflection = make_reflection(alignment_verdict="misaligned")

    async def substrate_client(_mol: object):
        return make_appraisal(surprise_level=0.5)

    async def cortex_client(req: object):
        if req.plan.verb_name == "orion_response_repair":
            raise RuntimeError("gpu_pool_unavailable")
        return {"final_text": reflection.model_dump(mode="json"), "trace_id": "t"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("orion.harness.finalize.extract_finalize_reflection_payload",
                   lambda _result: reflection.model_dump(mode="json"))
        with pytest.raises(HarnessFinalizeFailedError) as info:
            await run_harness_finalize_chain(
                correlation_id="c-fail", draft_text=draft_text, draft_molecule=molecule,
                thought=thought, grammar_receipts=[], repair_overlay=make_repair_overlay(),
                user_message="hello", voice_contract=None,
                cortex_client=cortex_client, substrate_client=substrate_client,
            )
    outcome = info.value.partial.outcome_molecule
    assert outcome.finalize_failed is True
    assert "gpu_pool_unavailable" in (outcome.failure_reason or "")
    assert outcome.final_text == ""
    assert outcome.response_repair_ran is True
    assert outcome.response_repair_reason == "misaligned"
