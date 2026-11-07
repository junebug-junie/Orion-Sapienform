"""reminiscence.episode through the REAL AdmissionRuntime: real Postgres checkpoints and run registry, the real
GPU pool runtime in process (tests/pool_fixture.py), a real ledger file. Only the Hub turns are faked.

Proves the wiring the graph tests cannot: the runtime picks the reminiscence graph, all six turns run under
the pool's hold with their own source tags, each step's turn id is on its ``run.started`` event (the cost
script joins tokens through it), the run completes only when every receipt is backed by the ledger and all
five hops are live, ``status()`` shows the episode, and the run is an act of family durable.reminiscence.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI does).
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from test_admission_runtime_postgres import DSN, runtime, with_database  # noqa: E402
from test_harness_smoke_postgres import drive_until_terminal  # noqa: E402

from app.reminiscence_graph import ZONE  # noqa: E402
from app.reminiscence_ledger import ReminiscenceLedger  # noqa: E402
from orion.schemas.durable_run import CuriosityTurnResultV1, DurableRunRequestV1  # noqa: E402
from orion.schemas.reminiscence_run import (  # noqa: E402
    CROSS_READ_DONE, REMINISCENCE_HOPS, REMINISCENCE_SESSION_ID, REMINISCENCE_STEPS, step_tag,
)
from pool_fixture import InProcessPool  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def today() -> str:
    return datetime.now(timezone.utc).astimezone(ZONE).date().isoformat()


def hop_line(hop, hid, episode):
    return json.dumps({"schema_version": "reminiscence_hop.v1", "hop_id": hid, "episode": episode, "hop": hop,
                       "period_start": "2025-06-24", "period_end": "2025-06-30",
                       "note": "[reflection] A hop about my first week, long enough to pass.", "claims": [],
                       "written_at": datetime.now(timezone.utc).isoformat(), "supersedes": None})


def request(episode: str) -> DurableRunRequestV1:
    run_id = f"reminiscence-{episode}"
    return DurableRunRequestV1.model_validate({
        "run_id": run_id, "workflow": "reminiscence.episode", "correlation_id": "trace-" + run_id,
        "brief": {"reminiscence": "reminiscence.episode", "episode": episode, "period_start": "2025-06-24",
                  "period_end": "2025-06-30", "weeks_complete": 0, "weeks_total": 71, "timeout_sec": 120},
        "admission": {"resource": "llm.route.agent", "preferred_lane": "agent"},
    })


def with_turns(rt, ledger: ReminiscenceLedger, episode: str, *, mumble_once: str | None = None):
    """The Hub turns: a hop turn writes its hop (the dig write) and replies with the receipt."""
    calls = []

    async def run_turn(req):
        calls.append(req)
        step = req.source_tag.removeprefix("reminiscence_")
        if step == "cross_read":
            text = CROSS_READ_DONE
        else:
            hid = f"rm:{episode}:{step}:{len(calls):010x}"
            if mumble_once == step and sum(c.source_tag == req.source_tag for c in calls) == 1:
                text = "I wrote it."              # no receipt, and nothing written: a retried attempt
            else:
                with ledger.path.open("a") as fh:
                    fh.write(hop_line(step, hid, episode) + "\n")
                text = f"reminisce {step} saved {hid}; the week 2025-06-24..2025-06-30"
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=text, ok=True)

    rt.runner._run_turn = run_turn
    return calls


def test_a_reminiscence_run_writes_six_steps_under_the_hold_and_completes(tmp_path):
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, REMINISCENCE_LEDGER_DIR=str(tmp_path))
        episode = today()
        ledger = ReminiscenceLedger(tmp_path)
        calls = with_turns(rt, ledger, episode, mumble_once="where_i_am_now")
        req = request(episode)
        assert (await rt.submit(req))["workflow_kind"] == "reminiscence.episode"
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed"
        tags = [c.source_tag for c in calls]
        assert sorted(set(tags)) == sorted(step_tag(k) for k in REMINISCENCE_STEPS) and len(tags) == 7
        assert tags.count(step_tag("hop:where_i_am_now")) == 2      # no receipt -> retried, not completed
        assert {c.session_id for c in calls} == {REMINISCENCE_SESSION_ID}
        holds = gpu.leases(holder="durable-runs:" + req.run_id, kind="hold")
        assert all(c.gpu_lease is not None for c in calls)
        assert {c.gpu_lease.lease_id for c in calls} <= {h["lease_id"] for h in holds}
        status = await rt.status(req.run_id)
        assert status["reminiscence"]["episode"] == episode and status["reminiscence"]["outcome"] == "complete"
        assert set(status["reminiscence"]["hop_ids"]) == set(REMINISCENCE_HOPS)
        assert status["reminiscence"]["next_step"] is None
        assert status["harness_steps"]["hop:where_i_am_now"]["attempts"] == 2
        history = await store.history(req.run_id)
        started = {e["detail"].get("step_correlation_id") for e in history if e["event"] == "run.started"}
        assert {c.correlation_id for c in calls} <= started    # the cost script's token join
        done = next(e for e in history if e["event"] == "run.completed")
        assert done["detail"]["line"] == "reminiscence" and done["detail"]["turns"] == 6
        acts = [m for k, m in rt.runner.events if k == "grammar.event.v1"
                and m.act.object_ref == f"durable_run:{req.run_id}"]
        assert [a.act.actor for a in acts] == ["durable.reminiscence"] * 2   # opened, closed
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_complete_day_finishes_without_a_hold_or_a_turn(tmp_path):
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, REMINISCENCE_LEDGER_DIR=str(tmp_path))
        episode = today()
        ledger = ReminiscenceLedger(tmp_path)
        ledger.path.write_text("".join(hop_line(h, f"rm:{episode}:{h}:{i:010x}", episode) + "\n"
                                       for i, h in enumerate(REMINISCENCE_HOPS)))
        calls = with_turns(rt, ledger, episode)
        req = request(episode)
        await rt.submit(req)
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed" and calls == []
        assert gpu.leases(holder="durable-runs:" + req.run_id, kind="hold") == []
        assert (await rt.status(req.run_id))["reminiscence"]["outcome"] == "already_complete"
        await rt.close()
    asyncio.run(with_database(scenario))
