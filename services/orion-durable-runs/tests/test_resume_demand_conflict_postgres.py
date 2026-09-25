"""A resumed run must never spin forever on its own demand registration.

Live 2026-09-22 -> 2026-09-25: run 54537b5b5ccc logged
`durable_checkpoint_resume_failed` every ~7 s (53k events). Its accepted
request and demand rows had been rewritten out-of-band to add the
`chat-burst` alternative while the graph checkpoint kept the original
`["agent-burst"]` copy. A worker restart sent the graph back through
`resource_request`, which re-registered the *checkpoint* copy, hit
"run demand is immutable", and was re-driven on every reconcile tick with
no bound and no visible terminal reason.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_admission_runtime_postgres import DSN, request, runtime, with_database  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


async def _park_at_resource_request(rt, run_id):
    """Drive to resource_wait, then put the graph where worker_recovery left
    the live run: retrying, no lease, next node resource_request."""
    await rt._drive(await rt.store.get_run(run_id))
    cfg = rt.config(run_id)
    assert (await rt.graph.aget_state(cfg)).next == ("resource_wait",)
    await rt.graph.aupdate_state(cfg, {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
    assert (await rt.graph.aget_state(cfg)).next == ("resource_request",)


async def _failures(store, run_id):
    return [e for e in await store.history(run_id, limit=1000) if e["event"] == "run.checkpoint_resume_failed"]


def test_resume_after_out_of_band_alternatives_rewrite_registers_the_accepted_demand():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("rewrite-001")
        req.admission.alternatives = ["agent-burst"]
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        # The out-of-band repair: widen the accepted request and its demand,
        # leaving the checkpointed copy of `admission` untouched.
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE durable_admission_runs SET request=jsonb_set(request,'{admission,alternatives}',%s) WHERE run_id=%s",
                (Jsonb(["agent-burst", "chat-burst"]), req.run_id))
            await conn.execute(
                "UPDATE durable_resource_demands SET requirement=jsonb_set(requirement,'{alternatives}',%s) WHERE run_id=%s",
                (Jsonb(["agent-burst", "chat-burst"]), req.run_id))
        snap = await rt.graph.aget_state(rt.config(req.run_id))
        assert snap.values["admission"]["alternatives"] == ["agent-burst"]

        await rt._drive(await store.get_run(req.run_id))

        assert await _failures(store, req.run_id) == []
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        demand = await store.get_demand(req.run_id)
        assert demand["requirement"]["alternatives"] == ["agent-burst", "chat-burst"]
        assert demand["status"] == "pending"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_reregistration_accepts_a_stored_demand_equal_by_meaning():
    """A demand stored before a defaulted field existed is the same demand."""
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("meaning-001")
        await rt.submit(req)
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE durable_resource_demands SET requirement=requirement-'allow_elastic_activation' WHERE run_id=%s",
                (req.run_id,))
        await store.suspend_demand(req.run_id)
        demand = await store.register_demand(req.run_id, req.admission.model_dump(mode="json"))
        assert demand["status"] == "pending"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_repeated_resume_failure_at_one_checkpoint_fails_the_run_with_a_visible_reason():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        rt.settings.resume_max_failures = 3
        rt.settings.resume_min_failure_span_sec = 0
        req = request("spin-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)

        async def broken(*args, **kwargs):
            raise RuntimeError("demand store unavailable")
        store.register_demand = broken

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
        assert terminal and "demand store unavailable" in terminal[0]["detail"]["error"]
        assert (await store.get_demand(req.run_id))["status"] == "withdrawn"
        # A terminal run is no longer re-driven.
        assert req.run_id not in {r["run_id"] for r in await store.list_pending()}
        await rt.close()
    asyncio.run(with_database(scenario))


async def _broken(*args, **kwargs):
    raise RuntimeError("transient")


def test_wait_machinery_rewrites_do_not_reset_the_count_but_real_progress_does():
    """A grant -> fail -> worker_recovery -> re-request cycle writes a new
    checkpoint each time; it must still count toward the bound. Real node
    progress (e.g. harness_turn completing) resets it."""
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        rt.settings.resume_max_failures = 2
        rt.settings.resume_min_failure_span_sec = 0
        req = request("cycle-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        real = store.register_demand
        store.register_demand = _broken
        await rt._drive(await store.get_run(req.run_id))
        assert len(await _failures(store, req.run_id)) == 1
        # Recovery: re-request succeeds, graph moves to a NEW checkpoint.
        store.register_demand = real
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        await rt.graph.aupdate_state(rt.config(req.run_id), {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
        # Real progress in between resets the count ...
        await store.record_event(req.run_id, "run.running", {"node": "harness_turn"})
        store.register_demand = _broken
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        # ... but another wait-machinery rewrite does not.
        store.register_demand = real
        await rt._drive(await store.get_run(req.run_id))
        await rt.graph.aupdate_state(rt.config(req.run_id), {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
        store.register_demand = _broken
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_failures_must_span_the_minimum_time_before_the_run_is_failed():
    """A burst of failures during a short outage must not kill a healthy run."""
    async def scenario(pool, saver, store):
        from datetime import datetime, timedelta, timezone
        now = [datetime(2026, 9, 25, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        rt = runtime(pool, saver, store)
        rt.settings.resume_max_failures = 2
        rt.settings.resume_min_failure_span_sec = 600
        req = request("span-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        store.register_demand = _broken
        for _ in range(5):
            now[0] += timedelta(seconds=5)
            await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        now[0] += timedelta(seconds=600)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        await rt.close()
    asyncio.run(with_database(scenario))


def test_lease_expiry_reregistration_failure_is_counted_not_escaped():
    """The resource_wait re-register used to sit outside the bounded handler."""
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("expiry-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        store.register_demand = _broken
        await rt._drive(await store.get_run(req.run_id))  # must not raise
        assert len(await _failures(store, req.run_id)) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


def test_projection_failure_on_a_finished_graph_is_never_relabelled_failed():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        rt.settings.resume_max_failures = 1
        rt.settings.resume_min_failure_span_sec = 0
        req = request("finished-001")
        await rt.submit(req)
        await rt.broker.tick()
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
