"""The store read dream.carry waits on: a child run's terminal status + the detail finish_projection
wrote (outcome, sha, caption), on real Postgres. None while the child is still running."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_admission_runtime_postgres import DSN, with_database
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_WORKFLOW, DreamHopImageV1, ReverieVisualRunBriefV1, reverie_visual_run_id,
)

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def test_terminal_detail_is_none_while_running_and_the_projection_detail_after():
    async def scenario(pool, saver, store):
        dispatch = "dream-carry:run-x:1"
        req = DurableRunRequestV1(
            run_id=reverie_visual_run_id(dispatch), workflow=REVERIE_VISUAL_WORKFLOW, correlation_id=str(uuid4()),
            brief=ReverieVisualRunBriefV1(visual_request=VisualRunRequestV1(dispatch_id=dispatch),
                                          dream_hop=DreamHopImageV1(carry_run_id="run-x", hop_index=1, prompt="p")),
            admission={"resource": "service.route.diffusion", "preferred_lane": "diffusion",
                       "deadline_at": datetime.now(timezone.utc) + timedelta(hours=1)})
        await store.submit(req.model_dump(mode="json"))
        assert await store.terminal_detail(req.run_id) is None
        assert await store.terminal_detail("no-such-run") is None
        detail = {"outcome": "produced", "artifact_sha256": "c" * 64, "caption": "a porch"}
        assert await store.finish_projection(req.run_id, "completed", detail) == "completed"
        status, got = await store.terminal_detail(req.run_id)
        assert status == "completed"
        assert {k: got[k] for k in detail} == detail
    asyncio.run(with_database(scenario))


def test_dream_carry_end_to_end_on_the_real_pool_store_and_driver():
    """The whole carry through AdmissionRuntime._drive: text hops on a real metacog_background hold,
    image hops as real child reverie.visual runs driven by the same runtime, the carry reading each
    child's terminal detail from the store, and one completed terminal on the state channel."""
    from test_admission_runtime_postgres import runtime
    from test_dream_carry_graph import Dream
    from test_reverie_visual_graph import Thought
    from test_reverie_visual_postgres import PoolView
    from orion.schemas.dream_carry import DREAM_CARRY_WORKFLOW, DreamCarryBriefV1, dream_carry_run_id
    from orion.schemas.durable_run import DURABLE_RUN_STATE_KIND

    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        await rt.gpu.boot()
        clock = [datetime.now(timezone.utc)]
        rt.now = lambda: clock[0]
        dream = Dream()
        thought = Thought(PoolView(), caption=[("done", {"caption": "A white railing in fog."})])
        rt.runner._run_dream_carry_step = dream
        rt.runner._run_reverie_visual_step = thought
        req = DurableRunRequestV1(
            run_id=dream_carry_run_id("sleep-e2e"), workflow=DREAM_CARRY_WORKFLOW, correlation_id=str(uuid4()),
            brief=DreamCarryBriefV1(trigger_id="sleep-e2e", timeout_sec=5.0),
            admission={"resource": "llm.route.metacog_background", "preferred_lane": "metacog_background",
                       "deadline_at": clock[0] + timedelta(hours=4)})
        await rt.submit(req)
        for _ in range(40):
            for row in await store.list_pending():
                await rt._drive(row)
            if (await store.get_run(req.run_id))["terminal"]:
                break
            clock[0] += timedelta(seconds=31)   # past the carry's child poll
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [r.hop_index for r in dream.calls if r.step == "text"] == [0, 2, 4]
        assert {req_.dream_hop.hop_index for _, req_ in thought.calls} == {1, 3, 5}
        assert all(req_.dream_hop.carry_run_id == req.run_id for _, req_ in thought.calls)
        # The carry's holds are metacog-class (llm.route.metacog_background), all handed back.
        holds = rt.gpu.leases(holder=f"durable-runs:{req.run_id}")
        assert len(holds) == 3 and {h["work_class"] for h in holds} == {"metacog"}
        assert {h["status"] for h in holds} == {"released"}
        await rt.reconcile()
        [done] = [m for kind, m in rt.runner.events if kind == DURABLE_RUN_STATE_KIND and m.run_id == req.run_id]
        assert done.status == "completed" and done.workflow == DREAM_CARRY_WORKFLOW
        assert done.detail["hops_made"] == 6 and done.detail["dream_id"] == "dream-42"
        assert len(done.detail["child_run_ids"]) == 3
        for child in done.detail["child_run_ids"]:
            status, detail = await store.terminal_detail(child)
            assert status == "completed" and detail["caption"] == "A white railing in fog."
        # A child poll is not a lifecycle fact: no resumed/status event per wake.
        nodes = [(e["event"], (e.get("detail") or {}).get("node")) for e in await store.history(req.run_id)]
        assert not [n for n in nodes if n[1] == "image_wait"]
        await rt.close()
    asyncio.run(with_database(scenario))
