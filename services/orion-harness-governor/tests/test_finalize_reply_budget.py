"""The governor replies with the draft before the caller stops waiting (run a153451fe423).

Held curiosity turns carry ``reply_budget_sec``: past it, Hub has stopped listening and
durable-runs has released the run's GPU hold, so a finalize still running then is wasted and
its draft lost. Finalize is bounded to what is left; overrun replies on the failed-finalize
path with ``draft_text`` intact.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from orion.harness.runner import HarnessMotorResult, build_coalition_snapshot, build_draft_molecule
from orion.harness.tests.fixtures import make_repair_overlay, make_thought
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecPermissionV1
from orion.schemas.harness_finalize import HarnessRunRequestV1


def _request(reply_budget_sec: float | None) -> HarnessRunRequestV1:
    return HarnessRunRequestV1(
        correlation_id="c-budget",
        thought_event=make_thought(),
        user_message="hello",
        permissions=ContextExecPermissionV1(),
        answer_contract=AnswerContract(),
        reply_budget_sec=reply_budget_sec,
    )


def _motor(thought) -> HarnessMotorResult:
    molecule = build_draft_molecule(
        correlation_id="c-budget", thought=thought, draft_text="Orion's draft verdict",
        grammar_receipts=[], coalition_snapshot=build_coalition_snapshot(thought),
        repair_overlay=make_repair_overlay(),
    )
    return HarnessMotorResult(draft_text="Orion's draft verdict", grammar_receipts=[], step_count=107,
                              exit_code=None, draft_molecule=molecule)


async def _run_with_slow_finalize(req: HarnessRunRequestV1, finalize_sleep: float):
    from app import bus_listener

    async def slow_chain(**_kw):
        await asyncio.sleep(finalize_sleep)
        raise AssertionError("finalize should have been cut")

    with patch.object(bus_listener, "HarnessRunner",
                      return_value=AsyncMock(run=AsyncMock(return_value=_motor(req.thought_event)))), \
            patch.object(bus_listener, "run_harness_finalize_chain", slow_chain):
        return await bus_listener.handle_harness_run_request(AsyncMock(), req, reply_to="orion:harness:run:result:c")


@pytest.mark.asyncio
async def test_finalize_overrunning_the_reply_budget_replies_with_the_draft() -> None:
    started = time.monotonic()
    run = await _run_with_slow_finalize(_request(0.2), finalize_sleep=30.0)
    assert time.monotonic() - started < 5.0
    assert run.finalize_ran is False and run.final_text is None
    assert run.draft_text == "Orion's draft verdict"
    assert run.grounding_status.startswith("finalize_reply_deadline")


@pytest.mark.asyncio
async def test_no_time_left_after_the_motor_skips_finalize() -> None:
    from app import bus_listener

    req = _request(0.01)
    called = AsyncMock()

    async def slow_motor(*_a, **_kw):
        await asyncio.sleep(0.05)
        return _motor(req.thought_event)

    with patch.object(bus_listener, "HarnessRunner", return_value=AsyncMock(run=slow_motor)), \
            patch.object(bus_listener, "run_harness_finalize_chain", called):
        run = await bus_listener.handle_harness_run_request(AsyncMock(), req, reply_to="r")
    called.assert_not_awaited()
    assert run.draft_text == "Orion's draft verdict"
    assert run.grounding_status.startswith("finalize_reply_deadline")


@pytest.mark.asyncio
async def test_an_rpc_timeout_inside_finalize_is_not_renamed() -> None:
    from app import bus_listener

    async def chain():
        raise TimeoutError("cortex rpc timed out")

    with pytest.raises(TimeoutError, match="cortex rpc"):
        await bus_listener.run_bounded_finalize(chain(), 60.0)


@pytest.mark.asyncio
async def test_no_budget_means_unbounded() -> None:
    from app import bus_listener

    async def chain():
        return "done"

    assert bus_listener.finalize_seconds_left(_request(None), time.monotonic()) is None
    assert await bus_listener.run_bounded_finalize(chain(), None) == "done"
