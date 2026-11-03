"""The real AdmissionRuntime waits out a deploy for a reading turn published to nobody.

Live 2026-10-31: 16 reading runs failed ``turn_exception:RpcNoListener`` in four bursts, each a
service redeploy. Here the same reading run hits ``RpcPublishedToNobody`` on its first turn and
must stay unfinished, ask the pool nothing until the wait is over, then take a fresh hold and
complete -- without spending an attempt. A run that keeps hitting it past
DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES fails as it did before.

Real Postgres, real in-process pool, real client + codec (tests/pool_fixture.py).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion.core.bus.async_service import RpcPublishedToNobody  # noqa: E402
from orion.schemas.durable_run import DurableRunRequestV1  # noqa: E402
from orion.schemas.reading_turn import ReadingTurnResultV1  # noqa: E402
from app.admission_runtime import AdmissionRuntime  # noqa: E402
from app.pool_hold import PoolHolds  # noqa: E402
from app.settings import Settings  # noqa: E402
from pool_fixture import CFG, InProcessPool, PoolBus  # noqa: E402
from test_admission_runtime_postgres import DSN, Runner, with_database  # noqa: E402
from test_pool_refusal_postgres import fake_now, scripted  # noqa: E402

pg = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

NOBODY = "no listener on orion:reading:turn:request: request published to nobody"


class ReadingRunner(Runner):
    """The runtime's fake runner plus a Hub reading turn that is unheard ``unheard`` times."""

    def __init__(self, saver, unheard: int):
        super().__init__(saver)
        self.unheard = unheard
        self.reading_calls = []

    async def _run_reading_turn(self, request):
        self.reading_calls.append(request)
        if self.unheard > 0:
            self.unheard -= 1
            raise RpcPublishedToNobody(NOBODY)
        return ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, ok=True,
                                   text="A grounded reading.", source_fetches=[])


def reading_request(run_id):
    brief = {"seed_id": "reading:seed", "stage": 1, "prompt": "Read the source.", "session_id": "reading",
             "timeout_sec": 600}
    return DurableRunRequestV1(run_id=run_id, workflow="reading.turn", correlation_id=str(uuid4()),
                               admission={}, brief=brief)


def reading_runtime(pool, saver, store, gpu, runner, **overrides):
    """As test_admission_runtime_postgres.runtime, with ``runner``: the graphs bind the runner
    they are built with, so it cannot be swapped in afterwards."""
    settings = Settings(**{"_env_file": None, "POSTGRES_URI": DSN, "ORION_BUS_ENABLED": False,
                           "DURABLE_RUNS_ADMISSION_ENABLED": True, "DURABLE_RUNS_LEASE_HEARTBEAT_SEC": 0.03,
                           "DURABLE_RUNS_RETRY_BASE_SEC": 0.05, "DURABLE_RUNS_HOLD_STATUS_POLL_SEC": 60, **overrides})
    rt = AdmissionRuntime(settings, runner, pool, store=store,
                          holds=PoolHolds(PoolBus(gpu), source="orion-durable-runs", cfg=CFG))
    rt.gpu = gpu
    return rt


def setup(pool, saver, store, gpu, unheard, **overrides):
    rt = reading_runtime(pool, saver, store, gpu, ReadingRunner(saver, unheard), **overrides)
    clock = fake_now(rt)
    rt.deps.now = clock          # the graphs' own clock (listener_wait's retry_at) is the same one
    return rt, clock, scripted(rt)


async def history(store, run_id, name):
    return [e["detail"] for e in await store.history(run_id) if e["event"] == name]


@pg
def test_a_reading_published_to_nobody_waits_out_the_deploy_and_completes():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt, clock, bus = setup(pool, saver, store, gpu, unheard=1)
        req = reading_request("read-nobody-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))

        # Not failed: parked, hold handed back, the wait visible on the run's own event log.
        assert (await store.get_run(req.run_id))["terminal"] is None
        snap = await rt._graph_for("reading.turn").aget_state(rt.config(req.run_id))
        assert snap.next == ("resource_wait",)
        assert snap.values["listener_waits"] == 1 and int(snap.values.get("attempt") or 0) == 0
        assert snap.values["hold"]["listener_wait"] == 1 and "request_id" not in snap.values["hold"]
        [wait] = await history(store, req.run_id, "run.listener_wait")
        assert wait["waits"] == 1 and NOBODY in wait["error"]
        assert [lease["status"] for lease in gpu.leases()] == ["released"], "the seat is handed back, not held idle"
        assert len(rt.runner.reading_calls) == 1

        # Inside the wait: the pool is not asked again and Hub is not called again.
        asked = bus.acquires
        clock.advance(59)
        await rt._drive(await store.get_run(req.run_id))
        assert bus.acquires == asked and len(rt.runner.reading_calls) == 1

        # Hub is back: a fresh hold, the turn replays once, the reading completes.
        clock.advance(2)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert len(rt.runner.reading_calls) == 2
        assert [r["request_id"] for r in gpu.leases(holder="durable-runs:read-nobody-001")] == \
            ["read-nobody-001:1", "read-nobody-001:2"]
        await rt.close()
    asyncio.run(with_database(scenario))


@pg
def test_a_reading_that_stays_unheard_past_the_cap_fails_as_before():
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt, clock, _ = setup(pool, saver, store, gpu, unheard=10, DURABLE_RUNS_LISTENER_WAIT_MAX_TRIES=2,
                             DURABLE_RUNS_LISTENER_WAIT_BASE_SEC=10, DURABLE_RUNS_LISTENER_WAIT_MAX_SEC=10)
        req = reading_request("read-nobody-002")
        await rt.submit(req)
        for _ in range(3):
            await rt._drive(await store.get_run(req.run_id))
            clock.advance(11)
        row = await store.get_run(req.run_id)
        assert row["terminal"] == "failed"
        assert len(rt.runner.reading_calls) == 3, "two waits, then the third unheard turn fails"
        assert [w["waits"] for w in await history(store, req.run_id, "run.listener_wait")] == [1, 2]
        [failed] = await history(store, req.run_id, "run.failed")
        assert failed["error"].startswith("turn_exception:RpcPublishedToNobody: no listener")
        await rt.close()
    asyncio.run(with_database(scenario))
