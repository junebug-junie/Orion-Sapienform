"""reverie.visual end to end on real Postgres checkpoints, the real run registry/outbox and the REAL
GPU pool in process (tests/pool_fixture.py): the diffusion hold comes from gpu_pool.yaml
``hold_routes``, is released before caption, and every terminal reaches orion:durable:run:state."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_admission_runtime_postgres import DSN, runtime, with_database
from test_reverie_visual_graph import Thought
from orion.schemas.durable_run import DURABLE_RUN_STATE_KIND, DurableRunRequestV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_WORKFLOW, ReverieVisualRunBriefV1, reverie_visual_run_id,
)

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


class PoolView:
    """What Thought records at caption time: the pool's status for the run's hold."""

    def __init__(self):
        self.pool_status = None


def reverie_request(dispatch_id: str, deadline: timedelta = timedelta(minutes=90)) -> DurableRunRequestV1:
    brief = ReverieVisualRunBriefV1(visual_request=VisualRunRequestV1(dispatch_id=dispatch_id), timeout_sec=5.0)
    return DurableRunRequestV1(
        run_id=reverie_visual_run_id(dispatch_id), workflow=REVERIE_VISUAL_WORKFLOW, correlation_id=str(uuid4()),
        brief=brief, admission={"resource": "service.route.diffusion", "preferred_lane": "diffusion",
                                "deadline_at": datetime.now(timezone.utc) + deadline})


def states(rt):
    return [m for kind, m in rt.runner.events if kind == DURABLE_RUN_STATE_KIND]


def test_reverie_visual_completes_on_a_diffusion_hold_released_before_caption():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        await rt.gpu.boot()
        view = PoolView()
        thought = Thought(view)
        req = reverie_request("dispatch-e2e-1")
        rt.runner._run_reverie_visual_step = thought

        async def step(request, budget_sec=None):
            if request.step == "caption":
                [hold] = rt.gpu.leases(holder=f"durable-runs:{req.run_id}")
                view.pool_status = hold["status"]
            return await thought(request, budget_sec)

        rt.runner._run_reverie_visual_step = step
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert thought.steps() == ["prepare", "generate", "caption"]
        [hold] = rt.gpu.leases(holder=f"durable-runs:{req.run_id}")
        assert hold["work_class"] == "diffusion" and hold["status"] == "released"
        assert thought.pool_at_caption == ["released"]
        await rt.reconcile()
        [completed] = states(rt)
        assert completed.status == "completed" and completed.workflow == REVERIE_VISUAL_WORKFLOW
        assert completed.detail["dispatch_id"] == "dispatch-e2e-1" and completed.detail["visual_elapsed_sec"] == 15.0
        status = await rt.status(req.run_id)
        assert status["reverie_visual"]["attempt_id"] == "att-1" and status["work_started"] is True
        await rt.close()
    asyncio.run(with_database(scenario))


def test_reverie_visual_past_its_deadline_fails_abandons_and_publishes_the_failure():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        await rt.gpu.boot()
        thought = Thought(PoolView(), generate=[("retry", {"reason": "thermal_refused", "retry_after_sec": 3600})])
        rt.runner._run_reverie_visual_step = thought
        req = reverie_request("dispatch-e2e-2", deadline=timedelta(seconds=30))
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        snap = await rt._graph_for(REVERIE_VISUAL_WORKFLOW).aget_state(rt.config(req.run_id))
        assert snap.next == ("retry_wait",)
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        rt.now = lambda: later
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        assert thought.steps("abandon") == ["abandon"]
        await rt.reconcile()
        [failed] = states(rt)
        assert failed.status == "failed" and failed.node == "failed"
        assert failed.detail["last_error"] == "retry_window_expired" and failed.detail["retries"] == 1
        assert failed.detail["dispatch_id"] == "dispatch-e2e-2" and failed.detail["attempt_id"] == "att-1"
        await rt.close()
    asyncio.run(with_database(scenario))
