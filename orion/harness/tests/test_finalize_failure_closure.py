from __future__ import annotations

import pytest

from orion.harness.finalize import (
    HarnessFinalizeFailedError,
    emit_finalize_failure_artifacts,
    emit_turn_outcome_molecule,
    emit_verdict_molecule,
    run_harness_finalize_chain,
)
from orion.harness.runner import build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.schemas.harness_finalize import GrammarReceiptV1, HarnessPostTurnClosureV1


@pytest.mark.asyncio
async def test_finalize_failed_outcome_forces_surprise_unresolved_on_quick_lane() -> None:
    """Aligned quick-lane would resolve surprise; finalize_failed must not."""
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.02)
    reflection = make_reflection(alignment_verdict="aligned", strain_unresolved=False)
    verdict = await emit_verdict_molecule(correlation_id="c-fail", reflection=reflection)
    outcome = await emit_turn_outcome_molecule(
        correlation_id="c-fail",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        draft_text="motor draft body",
        final_text="",
        finalize_changed=False,
        finalize_failed=True,
        failure_reason="RPC timeout",
    )
    assert outcome.finalize_failed is True
    assert outcome.surprise_resolved is False
    assert outcome.draft_text_excerpt == "motor draft body"


@pytest.mark.asyncio
async def test_emit_finalize_failure_artifacts_publishes_outcome_closure_and_system_error() -> None:
    thought = make_thought()
    appraisal = make_appraisal(surprise_level=0.02)
    reflection = make_reflection(alignment_verdict="aligned")
    verdict = await emit_verdict_molecule(correlation_id="c-artifacts", reflection=reflection)
    verdict_id = "verdict-artifacts"

    outcomes: list[object] = []
    closures: list[HarnessPostTurnClosureV1] = []
    errors: list[dict[str, object]] = []

    async def outcome_publish(molecule: object, **_: object) -> None:
        outcomes.append(molecule)

    async def closure_publish(closure: HarnessPostTurnClosureV1, **_: object) -> None:
        closures.append(closure)

    async def error_publish(payload: dict[str, object], **_: object) -> None:
        errors.append(payload)

    partial = await emit_finalize_failure_artifacts(
        correlation_id="c-artifacts",
        error="orion_voice_finalize exec failed: LLMGatewayService: RPC timeout",
        draft_text="motor draft before timeout",
        thought=thought,
        substrate_appraisal=appraisal,
        reflection=reflection,
        verdict_molecule=verdict,
        verdict_molecule_id=verdict_id,
        grammar_receipts=[],
        user_message="hello juniper",
        quick_lane_skipped_5b=True,
        outcome_publish_fn=outcome_publish,
        closure_publish_fn=closure_publish,
        system_error_publish_fn=error_publish,
    )

    assert len(outcomes) == 1
    assert outcomes[0].finalize_failed is True  # type: ignore[attr-defined]
    assert len(closures) == 1
    assert closures[0].surprise_unresolved is True
    assert closures[0].user_message_excerpt == "hello juniper"
    assert len(errors) == 1
    assert errors[0]["phase"] == "orion_voice_finalize"
    assert partial.verdict_molecule_id == verdict_id
    assert partial.quick_lane_skipped_5b is True


@pytest.mark.asyncio
async def test_run_harness_finalize_chain_voice_failure_raises_with_partial_state() -> None:
    thought = make_thought()
    coalition = build_coalition_snapshot(thought)
    receipts = [GrammarReceiptV1(step_index=0, summary="step", grammar_event_id="g-1")]
    draft_text = "internal draft"
    molecule = build_draft_molecule(
        correlation_id="c-voice-fail",
        thought=thought,
        draft_text=draft_text,
        grammar_receipts=receipts,
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    appraisal = make_appraisal(surprise_level=0.02)
    reflection = make_reflection(alignment_verdict="aligned")
    closures: list[HarnessPostTurnClosureV1] = []
    errors: list[dict[str, object]] = []
    cortex_calls: list[object] = []

    async def substrate_client(_mol: object):
        return appraisal

    async def cortex_client(req: object):
        cortex_calls.append(req)
        if len(cortex_calls) == 1:
            return {"final_text": reflection.model_dump(mode="json"), "trace_id": "trace-1"}
        raise ValueError("orion_voice_finalize exec failed: LLMGatewayService: RPC timeout")

    async def closure_publish(closure: HarnessPostTurnClosureV1, **_: object) -> None:
        closures.append(closure)

    async def error_publish(payload: dict[str, object], **_: object) -> None:
        errors.append(payload)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "orion.harness.finalize.extract_finalize_reflection_payload",
            lambda result: reflection.model_dump(mode="json"),
        )
        with pytest.raises(HarnessFinalizeFailedError) as exc_info:
            await run_harness_finalize_chain(
                correlation_id="c-voice-fail",
                draft_text=draft_text,
                draft_molecule=molecule,
                thought=thought,
                grammar_receipts=receipts,
                repair_overlay=make_repair_overlay(),
                user_message="hello",
                voice_contract=None,
                cortex_client=cortex_client,
                substrate_client=substrate_client,
                closure_publish_fn=closure_publish,
                system_error_publish_fn=error_publish,
            )

    partial = exc_info.value.partial
    assert partial.outcome_molecule.finalize_failed is True
    assert partial.outcome_molecule.surprise_resolved is False
    assert len(closures) == 1
    assert closures[0].surprise_unresolved is True
    assert len(errors) == 1
