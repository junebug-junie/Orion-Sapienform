"""harness.echo_smoke through the REAL AdmissionRuntime: real Postgres checkpoints and run registry,
the real GPU pool runtime in process (tests/pool_fixture.py). Only the Hub turn RPC is faked.

Proves the wiring the graph-level tests cannot: the runtime picks the smoke's graph, the step runs
under the pool's hold, the step result lands in ``GET /runs/<id>`` (``status()``) and in the terminal
detail, an empty turn is retried (never completed), and the hold is released.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI does).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from test_admission_runtime_postgres import DSN, Runner, runtime, with_database  # noqa: E402

from orion.schemas.durable_run import CuriosityTurnResultV1, DurableRunRequestV1  # noqa: E402
from orion.schemas.harness_step_run import HARNESS_ECHO_SMOKE_SESSION_ID, HARNESS_ECHO_SMOKE_TAG  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

REPLY = "harness echo smoke ok 2026-10-29T01:00:00Z"


def smoke_request(run_id: str) -> DurableRunRequestV1:
    return DurableRunRequestV1.model_validate({
        "run_id": run_id, "workflow": "harness.echo_smoke", "correlation_id": "trace-" + run_id,
        "brief": {"smoke": "harness.echo_smoke", "prompt": "Run date -u, reply one line.", "timeout_sec": 60},
        "admission": {"resource": "llm.route.agent", "preferred_lane": "agent"},
    })


def with_turn(rt, replies):
    """Give the fake runner the Hub turn RPC the smoke graph late-binds (``runner._run_turn``)."""
    calls = []

    async def run_turn(req):
        calls.append(req)
        text = replies.pop(0) if replies else REPLY
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=text, ok=bool(text))

    rt.runner._run_turn = run_turn
    return calls


async def drive_until_terminal(rt, store, run_id, *, rounds=40):
    for _ in range(rounds):
        row = await store.get_run(run_id)
        if row.get("terminal"):
            return row
        await rt._drive(row)
        await asyncio.sleep(0.06)   # past DURABLE_RUNS_RETRY_BASE_SEC=0.05 backoff
    return await store.get_run(run_id)


def test_the_smoke_runs_one_harness_step_under_the_pool_hold_and_exposes_its_result():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        calls = with_turn(rt, [REPLY])
        req = smoke_request("harness-echo-smoke-pg-001")
        assert (await rt.submit(req))["workflow_kind"] == "harness.echo_smoke"
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed"
        [turn] = calls
        [hold] = gpu.leases(holder="durable-runs:" + req.run_id, kind="hold")
        assert turn.gpu_lease is not None and turn.gpu_lease.lease_id == hold["lease_id"]
        assert turn.session_id == HARNESS_ECHO_SMOKE_SESSION_ID and turn.source_tag == HARNESS_ECHO_SMOKE_TAG
        assert (await gpu.lease(hold["lease_id"]))["status"] == "released"
        status = await rt.status(req.run_id)
        step = status["harness_steps"]["echo"]
        assert step["text"] == REPLY and step["turn_correlation_id"] == turn.correlation_id
        events = {e["event"]: e for e in await store.history(req.run_id)}
        assert events["run.started"]["detail"]["step_correlation_id"] == turn.correlation_id
        assert events["run.completed"]["detail"]["turn_correlation_id"] == turn.correlation_id
        assert events["run.completed"]["detail"]["steps"]["echo"]["text"] == REPLY
        await rt.close()
    asyncio.run(with_database(scenario))


def test_an_empty_turn_is_retried_on_a_fresh_identity_never_completed_empty():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        calls = with_turn(rt, ["", REPLY])
        req = smoke_request("harness-echo-smoke-pg-002")
        await rt.submit(req)
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed" and len(calls) == 2
        assert calls[0].correlation_id != calls[1].correlation_id and calls[1].attempt == 2
        status = await rt.status(req.run_id)
        assert status["harness_steps"]["echo"]["attempts"] == 2
        assert status["harness_steps"]["echo"]["turn_correlation_id"] == calls[1].correlation_id
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_turn_that_never_answers_fails_the_run_with_its_error_and_names_the_turn():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, DURABLE_RUNS_RETRY_MAX_ATTEMPTS=2)
        calls = with_turn(rt, ["", "", "", ""])
        req = smoke_request("harness-echo-smoke-pg-003")
        await rt.submit(req)
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "failed" and len(calls) == 2
        events = {e["event"]: e for e in await store.history(req.run_id)}
        failed = events["run.failed"]["detail"]
        assert "harness_step_failed:echo:empty_generation" in failed["error"]
        assert failed["turn_correlation_id"] == calls[-1].correlation_id
        status = await rt.status(req.run_id)
        assert not status["harness_steps"]
        await rt.close()
    asyncio.run(with_database(scenario))
