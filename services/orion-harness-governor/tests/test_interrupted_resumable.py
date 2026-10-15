"""An externally killed durable turn is reported failed (so it retries and resumes),
never finalized into an answer."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.runner import HarnessMotorResult
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


def _request(resume_key):
    return HarnessRunRequestV1(
        correlation_id="00000000-0000-4000-8000-000000000001",
        thought_event=make_thought(),
        user_message="study",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        resume_key=resume_key,
    )


def _motor(**kw):
    return HarnessMotorResult(draft_text="half a thought that stops mid", grammar_receipts=[], step_count=112, **kw)


async def _run(request, motor):
    from app import bus_listener

    chain = AsyncMock(side_effect=AssertionError("finalize must not run"))
    with patch.object(bus_listener, "HarnessRunner", return_value=AsyncMock(run=AsyncMock(return_value=motor))), \
            patch.object(bus_listener, "run_harness_finalize_chain", chain):
        return await bus_listener.handle_harness_run_request(AsyncMock(), request, reply_to="r")


@pytest.mark.asyncio
async def test_sigkilled_durable_turn_fails_without_finalizing_and_keeps_the_draft():
    run = await _run(_request("run-1"), _motor(exit_code=-9, grounding_status="fcc_nonzero_exit"))
    assert run.final_text is None and run.finalize_ran is False
    assert run.grounding_status == "interrupted_resumable"
    assert run.draft_text == "half a thought that stops mid"


def test_only_durable_externally_killed_turns_qualify():
    from app.bus_listener import interrupted_for_resume

    killed = _motor(exit_code=-9, grounding_status="fcc_nonzero_exit")
    assert interrupted_for_resume(_request("run-1"), killed) is True
    assert interrupted_for_resume(_request(None), killed) is False  # not a durable turn
    assert interrupted_for_resume(_request("run-1"), _motor(exit_code=0)) is False
    ceiling = _motor(exit_code=-9, grounding_status="fcc_context_ceiling_exceeded")
    assert interrupted_for_resume(_request("run-1"), ceiling) is False  # cut-short path owns it
