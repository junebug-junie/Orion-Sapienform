"""resume_key travels request -> HarnessRunner -> fcc runner, and only when set."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orion.harness.runner import HarnessRunner
from orion.harness.tests.fixtures import make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


def _request(**extra):
    return HarnessRunRequestV1(
        correlation_id="turn-1",
        thought_event=make_thought(),
        user_message="study",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        **extra,
    )


async def _run(request):
    captured = {}

    async def motor(**kwargs):
        captured.update(kwargs)
        yield {"type": "error", "error": "synthetic stop", "error_code": "test"}

    await HarnessRunner(AsyncMock(), fcc_runner=motor, fcc_timeout_sec=999).run(request)
    return captured


@pytest.mark.asyncio
async def test_motor_receives_the_resume_key():
    assert (await _run(_request(resume_key="run-abc")))["resume_key"] == "run-abc"


@pytest.mark.asyncio
async def test_a_request_without_a_key_does_not_pass_one():
    # Custom fcc_runner callables that predate the field keep working.
    assert "resume_key" not in await _run(_request())


def test_old_wire_shape_round_trips_with_no_key():
    request = _request()
    assert request.resume_key is None
    assert HarnessRunRequestV1.model_validate(request.model_dump(mode="json")).resume_key is None
