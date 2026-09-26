"""A resumed run must never spin forever at one checkpoint.

Live 2026-09-22 -> 2026-09-25: run 54537b5b5ccc logged `durable_checkpoint_resume_failed` every
~7 s (53k events) because re-asking for its resource failed at the same checkpoint on every
reconcile tick, with no bound and no visible terminal reason. The broker's demand-immutability
trigger for that incident is gone with the broker (stage 4.5), but the bound is generic: any
failure re-asking for the run's GPU pool hold is counted, and N of them since the last real
progress, spanning a minimum time, fail the run with the error attached.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_fixture import InProcessPool  # noqa: E402
from test_admission_runtime_postgres import DSN, occupy_agent, request, runtime, with_database  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


async def _park_at_resource_request(rt, run_id):
    """Drive to resource_wait (another run holds the card), then put the graph where a worker
    recovery leaves a live run: retrying, no lease, next node resource_request."""
    await occupy_agent(rt.gpu)
    await rt._drive(await rt.store.get_run(run_id))
    cfg = rt.config(run_id)
    assert (await rt.graph.aget_state(cfg)).next == ("resource_wait",)
    await rt.graph.aupdate_state(cfg, {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
    assert (await rt.graph.aget_state(cfg)).next == ("resource_request",)


async def _failures(store, run_id):
    return [e for e in await store.history(run_id, limit=1000) if e["event"] == "run.checkpoint_resume_failed"]


def _break(rt, message="hold request unavailable"):
    real = rt.deps.register

    async def broken(state):
        raise RuntimeError(message)

    rt.deps.register = broken
    return real


def test_repeated_resume_failure_at_one_checkpoint_fails_the_run_with_a_visible_reason():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store, gpu=InProcessPool())
        rt.settings.resume_max_failures = 3
        rt.settings.resume_min_failure_span_sec = 0
        req = request("spin-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        _break(rt)
        for attempt in range(1, 3):
            await rt._drive(await store.get_run(req.run_id))
            assert len(await _failures(store, req.run_id)) == attempt
            assert (await store.get_run(req.run_id))["terminal"] is None

        await rt._drive(await store.get_run(req.run_id))
        row = await store.get_run(req.run_id)
        assert row["terminal"] == "failed"
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert not snap.next and snap.values["status"] == "failed"
        assert snap.values["last_error"].startswith("checkpoint_resume_failed")
        terminal = [e for e in await store.history(req.run_id) if e["event"] == "run.failed"]
        assert terminal and "hold request unavailable" in terminal[0]["detail"]["error"]
        # The failed run's queued hold is ended, never granted to nobody later.
        assert [r["status"] for r in rt.gpu.leases(holder="durable-runs:spin-001")] == ["released"]
        # A terminal run is no longer re-driven.
        assert req.run_id not in {r["run_id"] for r in await store.list_pending()}
        await rt.close()
    asyncio.run(with_database(scenario))


def test_wait_machinery_rewrites_do_not_reset_the_count_but_real_progress_does():
    """A grant -> fail -> worker_recovery -> re-request cycle writes a new checkpoint each time; it
    must still count toward the bound. Real node progress (e.g. harness_turn completing) resets it."""
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store, gpu=InProcessPool())
        rt.settings.resume_max_failures = 2
        rt.settings.resume_min_failure_span_sec = 0
        req = request("cycle-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        real = _break(rt)
        await rt._drive(await store.get_run(req.run_id))
        assert len(await _failures(store, req.run_id)) == 1
        # Recovery: re-request succeeds, graph moves to a NEW checkpoint.
        rt.deps.register = real
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        await rt.graph.aupdate_state(rt.config(req.run_id), {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
        # Real progress in between resets the count ...
        await store.record_event(req.run_id, "run.running", {"node": "harness_turn"})
        _break(rt)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        # ... but another wait-machinery rewrite does not.
        rt.deps.register = real
        await rt._drive(await store.get_run(req.run_id))
        await rt.graph.aupdate_state(rt.config(req.run_id), {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
        _break(rt)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_failures_must_span_the_minimum_time_before_the_run_is_failed():
    """A burst of failures during a short outage must not kill a healthy run."""
    async def scenario(pool, saver, store):
        now = [datetime(2026, 9, 25, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        rt = runtime(pool, saver, store, gpu=InProcessPool())
        rt.now = lambda: now[0]
        rt.settings.resume_max_failures = 2
        rt.settings.resume_min_failure_span_sec = 600
        req = request("span-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        _break(rt)
        for _ in range(5):
            now[0] += timedelta(seconds=5)
            await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        now[0] += timedelta(seconds=600)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_pool_status_failure_while_waiting_is_a_wait_not_a_resume_failure():
    """An unreachable pool while a run waits in line is not the run's fault: it stays waiting and
    nothing counts toward the failure bound."""
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store, gpu=InProcessPool(), DURABLE_RUNS_HOLD_STATUS_POLL_SEC=0.01)
        req = request("expiry-001")
        await rt.submit(req)
        await occupy_agent(rt.gpu)
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        rt.holds.bus.fail_next = 5
        await asyncio.sleep(0.02)
        await rt._drive(await store.get_run(req.run_id))  # must not raise
        assert await _failures(store, req.run_id) == []
        assert (await store.get_run(req.run_id))["terminal"] is None
        await rt.close()
    asyncio.run(with_database(scenario))


def test_projection_failure_on_a_finished_graph_is_never_relabelled_failed():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store, gpu=InProcessPool())
        rt.settings.resume_max_failures = 1
        rt.settings.resume_min_failure_span_sec = 0
        req = request("finished-001")
        await rt.submit(req)
        real = store.finish_projection
        async def broken_projection(*args, **kwargs):
            raise RuntimeError("projection unavailable")
        store.finish_projection = broken_projection
        await rt._drive(await store.get_run(req.run_id))
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert not snap.next and snap.values["status"] == "completed"
        assert (await store.get_run(req.run_id))["terminal"] is None
        store.finish_projection = real
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        await rt.close()
    asyncio.run(with_database(scenario))
