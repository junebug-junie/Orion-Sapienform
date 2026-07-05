from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.finalize import HarnessFinalizeChainResult
from orion.harness.runner import HarnessMotorResult, build_draft_molecule, build_coalition_snapshot
from orion.harness.tests.fixtures import (
    make_appraisal,
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
async def test_harness_run_refused_when_thought_deferred() -> None:
    from app import bus_listener

    thought = make_thought(disposition="defer")
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
