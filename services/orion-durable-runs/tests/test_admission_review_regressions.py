"""Independent review regressions: controls after inference and completion recovery."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app.graph import Deps
from app.settings import Settings
from orion.schemas.durable_run import DURABLE_RUN_STATE_KIND
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_admission_runtime_postgres import DSN, Runner, request, runtime, with_database
from pool_fixture import InProcessPool

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")


@pytest.mark.parametrize("action", ["pause", "cancel"])
def test_other_replica_control_after_inference_stops_tail_and_preserves_resume(monkeypatch, action):
    async def scenario(pool, saver, store):
        entered, release = asyncio.Event(), asyncio.Event()
        journal_calls = []
        real_deps = Runner._curiosity_deps
        def deps(self):
            original = real_deps(self)
            async def read(run_id):
                entered.set()
                await release.wait()
                return {"graph_readable": True}
            async def journal(entry):
                journal_calls.append(entry.entry_id)
                return entry.entry_id
            return Deps(original.run_turn, read, original.publish_attention_row, journal)
        monkeypatch.setattr(Runner, "_curiosity_deps", deps)
        gpu = InProcessPool()   # both replicas talk to the one pool
        owner = runtime(pool, saver, store, gpu=gpu)
        remote = runtime(pool, saver, store, gpu=gpu)
        req = request("review-control-"+action)
        await owner.submit(req)
        active = asyncio.create_task(owner._drive(await store.get_run(req.run_id)))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert len(owner.runner.calls) == 1
            await remote.control(req.run_id, action)
            release.set()
            await asyncio.wait_for(active, 2)
            assert journal_calls == []
            assert not any(e["event"] == "run.completed" for e in await store.history(req.run_id))
            # The control released the run's pool hold at once (Juniper's pause frees the card).
            assert all(r["status"] == "released" for r in gpu.leases(holder=f"durable-runs:{req.run_id}"))
            if action == "cancel":
                await remote._drive(await store.get_run(req.run_id))
                assert (await remote.status(req.run_id))["status"] == "cancelled"
            else:
                assert (await store.get_run(req.run_id))["terminal"] is None
                await remote.control(req.run_id, "resume")
                for _ in range(3):
                    await remote._drive(await store.get_run(req.run_id))
                    if (await store.get_run(req.run_id))["terminal"]:
                        break
                assert (await remote.status(req.run_id))["status"] == "completed"
                assert len(journal_calls) == 1
                assert not remote.runner.calls  # resumes the tail, never repeats inference
        finally:
            release.set()
            active.cancel()
            await asyncio.gather(active, return_exceptions=True)
            await owner.close()
            await remote.close()
    asyncio.run(with_database(scenario))


def test_finished_checkpoint_recovery_delivers_full_completion_through_outbox():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("review-finish-001")
        await rt.submit(req)
        state = {"run_id": req.run_id, "correlation_id": req.correlation_id,
            "brief": {**req.brief.model_dump(mode="json"), "line": "self_inquiry"},
            "admission": req.admission.model_dump(mode="json"), "requested_at": req.requested_at.isoformat(),
            "attempt": 1, "status": "completed", "text": "A grounded self inquiry finding",
            "journal_entry_id": "journal-one", "lease": None, "self_definition": {"definition": "A supported finding"}}
        # Death after checkpoint commit but before the old live finish callback.
        await rt.graph.aupdate_state(rt.config(req.run_id), state, as_node="finish")
        await rt._drive(await store.get_run(req.run_id))
        await rt.reconcile()
        completed = [model for kind, model in rt.runner.events if kind == DURABLE_RUN_STATE_KIND]
        assert len(completed) == 1
        assert completed[0].detail["line"] == "self_inquiry"
        assert completed[0].detail["finding_text"] == state["text"]
        assert completed[0].detail["self_definition"] == state["self_definition"]
        await rt.reconcile()
        assert len([model for kind, model in rt.runner.events if kind == DURABLE_RUN_STATE_KIND]) == 1
        await rt.close()
    asyncio.run(with_database(scenario))


@pytest.mark.parametrize("heartbeat", [15, 20])
def test_heartbeat_must_leave_time_to_renew_before_expiry(heartbeat):
    with pytest.raises(ValueError):
        Settings(postgres_uri="postgresql://unused/test", lease_seconds=15, lease_heartbeat_sec=heartbeat)


@pytest.mark.parametrize("control", ["paused", "cancelled"])
def test_control_wins_at_terminal_commit_without_false_completion(monkeypatch, control):
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("review-commit-"+control)
        await rt.submit(req)
        original = store.finish_projection
        async def control_at_boundary(run_id, status, detail):
            await store.set_control(run_id, control)
            return await original(run_id, status, detail)
        monkeypatch.setattr(store, "finish_projection", control_at_boundary)
        state = {"run_id": req.run_id, "brief": req.brief.model_dump(mode="json"),
                 "text": "A grounded finding", "status": "completed"}
        await rt._terminal(req.run_id, "completed", state)
        row = await store.get_run(req.run_id)
        assert row["terminal"] == ("cancelled" if control == "cancelled" else None)
        assert not any(event["event"] == "run.completed" for event in await store.history(req.run_id))
        if control == "paused":
            await store.set_control(req.run_id, None)
            assert req.run_id in {r["run_id"] for r in await store.list_pending()}   # resumable
        await rt.close()
    asyncio.run(with_database(scenario))


@pytest.mark.parametrize("phase", ["inference", "tail"])
def test_overall_deadline_expiring_mid_graph_fails_without_retry_or_journal(monkeypatch, phase):
    async def scenario(pool, saver, store):
        now = [datetime.now(timezone.utc)]
        journal_calls = []
        real_deps = Runner._curiosity_deps
        def deps(self):
            original = real_deps(self)
            async def turn(req):
                result = await original.run_turn(req)
                if phase == "inference":
                    now[0] += timedelta(seconds=20)
                return result
            async def read(run_id):
                if phase == "tail":
                    now[0] += timedelta(seconds=20)
                return {"graph_readable": True}
            async def journal(entry):
                journal_calls.append(entry.entry_id)
                return entry.entry_id
            return Deps(turn, read, original.publish_attention_row, journal)
        monkeypatch.setattr(Runner, "_curiosity_deps", deps)
        rt = runtime(pool, saver, store)
        rt.now = lambda: now[0]
        req = request("review-deadline-"+phase)
        req.admission.deadline_at = now[0]+timedelta(seconds=10)
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        checkpoint = await rt.graph.aget_state(rt.config(req.run_id))
        assert checkpoint.values["last_error"] == "workflow_deadline"
        assert journal_calls == []
        assert [r["status"] for r in rt.gpu.leases(holder=f"durable-runs:{req.run_id}")] == ["released"]
        assert not any(event["event"] == "run.retrying" for event in await store.history(req.run_id))
        await rt.close()
    asyncio.run(with_database(scenario))


def test_terminal_lifecycle_has_one_atomic_completion_event():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("review-terminal-001")
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        completed = [event for event in await store.history(req.run_id) if event["event"] == "run.completed"]
        assert [event["entry_id"] for event in completed] == [f"{req.run_id}:terminal:completed"]
        await rt.close()
    asyncio.run(with_database(scenario))


def test_status_initial_wait_stops_at_first_grant_across_release_and_retry():
    async def scenario(pool, saver, store):
        now = [datetime(2026, 9, 12, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        rt = runtime(pool, saver, store)
        rt.now = lambda: now[0]
        req = request("review-first-wait")
        await rt.submit(req)
        now[0] += timedelta(seconds=12)
        assert (await rt.status(req.run_id))["queue_wait_seconds"] == 12
        await rt._drive(await store.get_run(req.run_id))   # the pool grants: the wait ends here
        now[0] += timedelta(seconds=500)
        status = await rt.status(req.run_id)
        assert status["status"] == "completed" and status["queue_wait_seconds"] == 12
        await rt.close()
    asyncio.run(with_database(scenario))


def test_duplicate_receipt_of_a_pre_cutover_row_ignores_broker_lane_fields():
    """Rows accepted before 4.5 carry broker-derived ``alternatives``; a producer's duplicate
    receipt (which never sends them) is the same run, not a conflict. Any real change still is."""
    from orion.durable_admission.store import SubmissionConflict

    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("review-pre-cutover")
        stored = req.model_dump(mode="json")
        stored["admission"].update(alternatives=["agent-burst", "chat-burst"], allow_elastic_activation=True)
        await store.submit(stored)
        assert (await rt.submit(req))["status"] == "waiting_resource"
        changed = req.model_copy(update={"brief": req.brief.model_copy(update={"prompt": "Different study"})})
        with pytest.raises(SubmissionConflict):
            await rt.submit(changed)
        await rt.close()
    asyncio.run(with_database(scenario))
