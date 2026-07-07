from __future__ import annotations

from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.harness.finalize import HarnessFinalizeChainResult
from orion.harness.runner import HarnessMotorResult, build_draft_molecule, build_coalition_snapshot
from orion.harness.tests.fixtures import (
    make_appraisal,
    make_grounding_capsule,
    make_reflection,
    make_repair_overlay,
    make_thought,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1, HarnessRunV1


def _motor_result(thought) -> HarnessMotorResult:
    coalition = build_coalition_snapshot(thought)
    receipts = []
    molecule = build_draft_molecule(
        correlation_id="c-1",
        thought=thought,
        draft_text="internal draft",
        grammar_receipts=receipts,
        coalition_snapshot=coalition,
        repair_overlay=make_repair_overlay(),
    )
    return HarnessMotorResult(
        draft_text="internal draft",
        grammar_receipts=receipts,
        step_count=1,
        exit_code=0,
        draft_molecule=molecule,
    )


@pytest.mark.asyncio
async def test_harness_run_artifact_published() -> None:
    from app import bus_listener

    thought = make_thought()
    req = HarnessRunRequestV1(
        correlation_id="c-1",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    appraisal = make_appraisal()
    reflection = make_reflection()
    motor = _motor_result(thought)

    async def _fake_finalize_chain(**_: object) -> HarnessFinalizeChainResult:
        from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

        verdict = await emit_verdict_molecule(
            correlation_id="c-1",
            reflection=reflection,
            publish_fn=AsyncMock(),
        )
        outcome = await emit_turn_outcome_molecule(
            correlation_id="c-1",
            thought=thought,
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            draft_text="internal draft",
            final_text="final for juniper",
            finalize_changed=False,
            publish_fn=AsyncMock(),
        )
        return HarnessFinalizeChainResult(
            final_text="final for juniper",
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            outcome_molecule=outcome,
            finalize_changed=False,
            quick_lane_skipped_5b=True,
            verdict_molecule_id="verdict-1",
        )

    bus = AsyncMock()
    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=motor)),
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_finalize_chain), patch.object(
        bus_listener,
        "emit_post_turn_closure",
        AsyncMock(return_value=AsyncMock()),
    ):
        run = await bus_listener.handle_harness_run_request(
            bus,
            req,
            reply_to="orion:harness:run:result:c-1",
        )

    assert isinstance(run, HarnessRunV1)
    assert run.finalize_ran is True
    assert run.final_text == "final for juniper"
    assert run.draft_text == "internal draft"
    assert bus.publish.await_count >= 2
    channels = [call.args[0] for call in bus.publish.await_args_list]
    assert "orion:harness:run:result:c-1" in channels
    assert bus_listener.settings.channel_harness_run_artifact in channels


@pytest.mark.asyncio
async def test_harness_run_carries_recall_from_grounding_capsule() -> None:
    from app import bus_listener

    capsule = make_grounding_capsule()
    thought = make_thought().model_copy(update={"grounding_capsule": capsule})
    req = HarnessRunRequestV1(
        correlation_id="c-recall",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    appraisal = make_appraisal()
    reflection = make_reflection()
    motor = _motor_result(thought)

    async def _fake_finalize_chain(**_: object) -> HarnessFinalizeChainResult:
        from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

        verdict = await emit_verdict_molecule(
            correlation_id="c-recall",
            reflection=reflection,
            publish_fn=AsyncMock(),
        )
        outcome = await emit_turn_outcome_molecule(
            correlation_id="c-recall",
            thought=thought,
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            draft_text="internal draft",
            final_text="final for juniper",
            finalize_changed=False,
            publish_fn=AsyncMock(),
        )
        return HarnessFinalizeChainResult(
            final_text="final for juniper",
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            outcome_molecule=outcome,
            finalize_changed=False,
            quick_lane_skipped_5b=True,
            verdict_molecule_id="verdict-recall",
        )

    bus = AsyncMock()
    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=motor)),
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_finalize_chain), patch.object(
        bus_listener,
        "emit_post_turn_closure",
        AsyncMock(return_value=AsyncMock()),
    ):
        run = await bus_listener.handle_harness_run_request(
            bus,
            req,
            reply_to="orion:harness:run:result:c-recall",
        )

    expected_digest = capsule.memory_digest or capsule.continuity_digest
    assert run.memory_digest == expected_digest
    assert run.recall_debug is not None
    assert run.recall_debug["pcr_ran"] == bool(capsule.provenance.get("pcr_ran"))
    assert run.recall_debug["source"] == "pcr_phase3"


@pytest.mark.asyncio
async def test_harness_run_recall_empty_without_capsule() -> None:
    from app import bus_listener

    thought = make_thought()
    req = HarnessRunRequestV1(
        correlation_id="c-no-recall",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    appraisal = make_appraisal()
    reflection = make_reflection()
    motor = _motor_result(thought)

    async def _fake_finalize_chain(**_: object) -> HarnessFinalizeChainResult:
        from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

        verdict = await emit_verdict_molecule(
            correlation_id="c-no-recall",
            reflection=reflection,
            publish_fn=AsyncMock(),
        )
        outcome = await emit_turn_outcome_molecule(
            correlation_id="c-no-recall",
            thought=thought,
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            draft_text="internal draft",
            final_text="final for juniper",
            finalize_changed=False,
            publish_fn=AsyncMock(),
        )
        return HarnessFinalizeChainResult(
            final_text="final for juniper",
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            outcome_molecule=outcome,
            finalize_changed=False,
            quick_lane_skipped_5b=True,
            verdict_molecule_id="verdict-no-recall",
        )

    bus = AsyncMock()
    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=motor)),
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_finalize_chain), patch.object(
        bus_listener,
        "emit_post_turn_closure",
        AsyncMock(return_value=AsyncMock()),
    ):
        run = await bus_listener.handle_harness_run_request(
            bus,
            req,
            reply_to="orion:harness:run:result:c-no-recall",
        )

    assert run.recall_debug is None
    assert run.memory_digest is None


def _raw_bus_message(*, kind: str, reply_to: str, payload: dict) -> dict:
    """Build a codec-encoded raw pubsub message that _handle_bus_message can decode."""
    envelope = BaseEnvelope(
        kind=kind,
        source=ServiceRef(name="test-producer"),
        correlation_id=uuid4(),
        reply_to=reply_to,
        payload=payload,
    )
    return {"type": "message", "data": OrionCodec().encode(envelope)}


@pytest.mark.asyncio
async def test_handle_bus_message_error_path_carries_recall_from_capsule() -> None:
    from app import bus_listener

    capsule = make_grounding_capsule()
    thought = make_thought().model_copy(update={"grounding_capsule": capsule})
    req = HarnessRunRequestV1(
        correlation_id="c-err",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    raw_msg = _raw_bus_message(
        kind="harness.run.request.v1",
        reply_to="orion:harness:run:result:c-err",
        payload=req.model_dump(mode="json"),
    )

    bus = AsyncMock()
    bus.codec = OrionCodec()
    with patch.object(
        bus_listener,
        "handle_harness_run_request",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        await bus_listener._handle_bus_message(bus, raw_msg)

    assert bus.publish.await_count == 2
    err_payload = bus.publish.await_args_list[0].args[1].payload
    assert err_payload["compliance_verdict"] == "failed"
    assert err_payload["memory_digest"] == (capsule.memory_digest or capsule.continuity_digest)
    assert err_payload["recall_debug"] is not None
    assert err_payload["recall_debug"]["pcr_ran"] == bool(capsule.provenance.get("pcr_ran"))


@pytest.mark.asyncio
async def test_handle_bus_message_error_path_recall_none_when_request_unbound() -> None:
    from app import bus_listener

    # Payload fails HarnessRunRequestV1.model_validate, so `request` stays None.
    raw_msg = _raw_bus_message(
        kind="harness.run.request.v1",
        reply_to="orion:harness:run:result:c-bad",
        payload={"not": "a valid request"},
    )

    bus = AsyncMock()
    bus.codec = OrionCodec()
    await bus_listener._handle_bus_message(bus, raw_msg)

    assert bus.publish.await_count == 2
    err_payload = bus.publish.await_args_list[0].args[1].payload
    assert err_payload["compliance_verdict"] == "failed"
    assert err_payload["recall_debug"] is None
    assert err_payload["memory_digest"] is None


@pytest.mark.asyncio
async def test_harness_run_publishes_post_turn_closure_on_bus() -> None:
    from app import bus_listener

    thought = make_thought()
    req = HarnessRunRequestV1(
        correlation_id="c-closure",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    appraisal = make_appraisal()
    reflection = make_reflection()
    motor = _motor_result(thought)

    async def _fake_finalize_chain(**kwargs: object) -> HarnessFinalizeChainResult:
        from orion.harness.finalize import emit_turn_outcome_molecule, emit_verdict_molecule

        verdict = await emit_verdict_molecule(
            correlation_id="c-closure",
            reflection=reflection,
            bus=kwargs["bus"],
        )
        outcome = await emit_turn_outcome_molecule(
            correlation_id="c-closure",
            thought=thought,
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            draft_text="internal draft",
            final_text="final for juniper",
            finalize_changed=False,
            bus=kwargs["bus"],
        )
        return HarnessFinalizeChainResult(
            final_text="final for juniper",
            substrate_appraisal=appraisal,
            reflection=reflection,
            verdict_molecule=verdict,
            outcome_molecule=outcome,
            finalize_changed=False,
            quick_lane_skipped_5b=True,
            verdict_molecule_id="verdict-closure",
        )

    bus = AsyncMock()
    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=motor)),
    ), patch.object(bus_listener, "run_harness_finalize_chain", _fake_finalize_chain):
        run = await bus_listener.handle_harness_run_request(
            bus,
            req,
            reply_to="orion:harness:run:result:c-closure",
        )

    assert run.finalize_ran is True
    channels = [call.args[0] for call in bus.publish.await_args_list]
    assert bus_listener.settings.channel_post_turn_closure in channels
    assert bus_listener.settings.channel_harness_run_artifact in channels

    closure_calls = [
        call
        for call in bus.publish.await_args_list
        if call.args[0] == bus_listener.settings.channel_post_turn_closure
    ]
    assert len(closure_calls) == 1
    closure_envelope = closure_calls[0].args[1]
    assert closure_envelope.kind == "harness.post_turn.closure.v1"
    assert closure_envelope.payload["correlation_id"] == "c-closure"
    assert closure_envelope.payload["surprise_unresolved"] is False


@pytest.mark.asyncio
async def test_harness_run_refused_when_thought_deferred() -> None:
    from app import bus_listener

    capsule = make_grounding_capsule()
    thought = make_thought(disposition="defer").model_copy(
        update={"grounding_capsule": capsule}
    )
    req = HarnessRunRequestV1(
        correlation_id="c-defer",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    bus = AsyncMock()
    run = await bus_listener.handle_harness_run_request(
        bus,
        req,
        reply_to="orion:harness:run:result:c-defer",
    )
    assert run.compliance_verdict == "refused"
    assert run.finalize_ran is False
    assert bus.publish.await_count == 2
    # recall carries even when the motor never runs (refusal path)
    assert run.memory_digest == (capsule.memory_digest or capsule.continuity_digest)
    assert run.recall_debug is not None


@pytest.mark.asyncio
async def test_harness_run_carries_recall_on_motor_fail() -> None:
    from app import bus_listener

    capsule = make_grounding_capsule()
    thought = make_thought().model_copy(update={"grounding_capsule": capsule})
    req = HarnessRunRequestV1(
        correlation_id="c-motorfail",
        thought_event=thought,
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
    )
    empty_motor = HarnessMotorResult(
        draft_text="",
        grammar_receipts=[],
        step_count=0,
        exit_code=1,
        draft_molecule=None,
    )

    bus = AsyncMock()
    with patch.object(
        bus_listener,
        "HarnessRunner",
        return_value=AsyncMock(run=AsyncMock(return_value=empty_motor)),
    ):
        run = await bus_listener.handle_harness_run_request(
            bus,
            req,
            reply_to="orion:harness:run:result:c-motorfail",
        )

    assert run.finalize_ran is False
    assert run.memory_digest == (capsule.memory_digest or capsule.continuity_digest)
    assert run.recall_debug is not None
    assert run.recall_debug["source"] == "pcr_phase3"
