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


def test_resume_failure_count_is_per_checkpoint_and_resets_on_progress():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        rt.settings.resume_max_failures = 2
        req = request("progress-001")
        await rt.submit(req)
        await _park_at_resource_request(rt, req.run_id)
        real = store.register_demand

        async def broken(*args, **kwargs):
            raise RuntimeError("transient")
        store.register_demand = broken
        await rt._drive(await store.get_run(req.run_id))
        assert len(await _failures(store, req.run_id)) == 1
        # The failure clears; the graph moves to a new checkpoint.
        store.register_demand = real
        await rt._drive(await store.get_run(req.run_id))
        assert (await rt.graph.aget_state(rt.config(req.run_id))).next == ("resource_wait",)
        # A later single failure at a different checkpoint does not inherit
        # the old count and must not terminate the run.
        await rt.graph.aupdate_state(rt.config(req.run_id), {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
        store.register_demand = broken
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] is None
        await rt.close()
    asyncio.run(with_database(scenario))
