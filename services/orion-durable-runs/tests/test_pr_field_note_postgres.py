"""pr.field_note through the REAL AdmissionRuntime: real Postgres checkpoints and run registry, the real
GPU pool runtime in process (tests/pool_fixture.py), a real ledger file. Only the Hub turn is faked.

Proves the wiring the graph tests cannot: the runtime picks the field-note graph, the turn runs under
the pool's hold, the run completes only with a receipt the ledger backs, ``status()`` shows the note,
the run is an act of family durable.pr_field_note (opened at the grant, closed at the end), and a PR
that already has a record finishes without ever asking the pool for a hold.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI does).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]

from test_admission_runtime_postgres import DSN, runtime, with_database  # noqa: E402
from test_harness_smoke_postgres import drive_until_terminal  # noqa: E402

from app.pr_field_note_ledger import FieldNoteLedger  # noqa: E402
from orion.schemas.durable_run import CuriosityTurnResultV1, DurableRunRequestV1  # noqa: E402
from orion.schemas.pr_field_note_run import PR_FIELD_NOTE_SESSION_ID, PR_FIELD_NOTE_TAG  # noqa: E402
from pool_fixture import InProcessPool  # noqa: E402

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


def note_line(pr, note_id, note="[pr_text] A real note about the change in my body here."):
    return json.dumps({"schema_version": "pr_field_note.v1", "note_id": note_id, "pr_number": pr,
                       "merge_sha": "a" * 40, "merged_at": "2026-11-05T10:00:00+00:00", "title": "t",
                       "flags": {}, "body_map": [], "note": note, "written_at": "2026-11-05T11:00:00+00:00"})


def request(pr: int) -> DurableRunRequestV1:
    run_id = f"pr-field-note-{pr}"
    return DurableRunRequestV1.model_validate({
        "run_id": run_id, "workflow": "pr.field_note", "correlation_id": "trace-" + run_id,
        "brief": {"field_note": "pr.field_note", "pr_number": pr, "merged_at": "2026-11-05T10:00:00+00:00",
                  "title": "feat: a thing", "timeout_sec": 120},
        "admission": {"resource": "llm.route.agent", "preferred_lane": "agent"},
    })


def with_turn(rt, ledger: FieldNoteLedger, replies):
    """The Hub turn: each reply is (text, write_note_id_or_None) -- the dig write the turn ran."""
    calls = []

    async def run_turn(req):
        calls.append(req)
        text, write = replies.pop(0)
        if write:
            with ledger.path.open("a") as fh:
                fh.write(note_line(int(req.run_id.rsplit("-", 1)[1]), write) + "\n")
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text=text, ok=bool(text))

    rt.runner._run_turn = run_turn
    return calls


def test_a_note_run_completes_under_the_hold_only_with_a_ledger_backed_receipt(tmp_path):
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, PR_FIELD_NOTES_LEDGER_DIR=str(tmp_path))
        ledger = FieldNoteLedger(tmp_path)
        nid = "fn:3105:0123456789"
        calls = with_turn(rt, ledger, [("I wrote it.", None), (f"field-note #3105 saved {nid}", nid)])
        req = request(3105)
        assert (await rt.submit(req))["workflow_kind"] == "pr.field_note"
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed" and len(calls) == 2   # no receipt -> retried, not completed
        assert calls[1].session_id == PR_FIELD_NOTE_SESSION_ID and calls[1].source_tag == PR_FIELD_NOTE_TAG
        holds = gpu.leases(holder="durable-runs:" + req.run_id, kind="hold")   # a retry takes a fresh hold
        assert all(c.gpu_lease is not None for c in calls)
        assert {c.gpu_lease.lease_id for c in calls} <= {h["lease_id"] for h in holds}
        status = await rt.status(req.run_id)
        assert status["pr_field_note"] == {"pr_number": 3105, "outcome": "noted", "note_id": nid}
        assert status["harness_steps"]["field_note"]["attempts"] == 2
        events = {e["event"]: e for e in await store.history(req.run_id)}
        assert events["run.completed"]["detail"]["note_id"] == nid
        acts = [m for k, m in rt.runner.events if k == "grammar.event.v1"
                and m.act.object_ref == f"durable_run:{req.run_id}"]
        assert [a.act.actor for a in acts] == ["durable.pr_field_note"] * 2   # opened, closed
        await rt.close()
    asyncio.run(with_database(scenario))


def test_a_pr_with_a_record_finishes_without_a_hold_or_a_turn(tmp_path):
    async def scenario(pool, saver, store):
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu, PR_FIELD_NOTES_LEDGER_DIR=str(tmp_path))
        ledger = FieldNoteLedger(tmp_path)
        ledger.path.write_text(note_line(3106, "fn:3106:aaaaaaaaaa", note=None) + "\n")
        calls = with_turn(rt, ledger, [])
        req = request(3106)
        await rt.submit(req)
        row = await drive_until_terminal(rt, store, req.run_id)
        assert row["terminal"] == "completed" and calls == []
        assert gpu.leases(holder="durable-runs:" + req.run_id, kind="hold") == []
        assert (await rt.status(req.run_id))["pr_field_note"]["outcome"] == "already_recorded"
        await rt.close()
    asyncio.run(with_database(scenario))
