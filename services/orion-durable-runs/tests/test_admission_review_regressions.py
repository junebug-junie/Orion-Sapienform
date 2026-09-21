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
        owner = runtime(pool, saver, store)
        remote = runtime(pool, saver, store)
        req = request("review-control-"+action)
        await owner.submit(req)
        await owner.broker.tick()
        active = asyncio.create_task(owner._drive(await store.get_run(req.run_id)))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert len(owner.runner.calls) == 1
            await remote.control(req.run_id, action)
            release.set()
            await asyncio.wait_for(active, 2)
            assert journal_calls == []
            assert not any(e["event"] == "run.completed" for e in await store.history(req.run_id))
            if action == "cancel":
                await remote._drive(await store.get_run(req.run_id))
                assert (await remote.status(req.run_id))["status"] == "cancelled"
            else:
                assert (await store.get_run(req.run_id))["terminal"] is None
                await remote.control(req.run_id, "resume")
                for _ in range(3):
                    await remote.broker.tick()
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
            await store.suspend_demand(run_id)
            return await original(run_id, status, detail)
        monkeypatch.setattr(store, "finish_projection", control_at_boundary)
        state = {"run_id": req.run_id, "brief": req.brief.model_dump(mode="json"),
                 "text": "A grounded finding", "status": "completed"}
        await rt._terminal(req.run_id, "completed", state)
        row = await store.get_run(req.run_id)
        assert row["terminal"] == ("cancelled" if control == "cancelled" else None)
        assert not any(event["event"] == "run.completed" for event in await store.history(req.run_id))
        if control == "paused":
            assert (await store.get_demand(req.run_id))["status"] == "suspended"
            await store.set_control(req.run_id, None)
            await store.register_demand(req.run_id, req.admission.model_dump(mode="json"))
            assert (await store.get_demand(req.run_id))["status"] == "pending"
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
        await rt.broker.tick()
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "failed"
        checkpoint = await rt.graph.aget_state(rt.config(req.run_id))
        assert checkpoint.values["last_error"] == "workflow_deadline"
        assert journal_calls == []
        assert await store.get_lease(req.run_id) is None
        assert not any(event["event"] == "run.retrying" for event in await store.history(req.run_id))
        await rt.close()
    asyncio.run(with_database(scenario))


def test_terminal_lifecycle_has_one_atomic_completion_event():
    async def scenario(pool, saver, store):
        rt = runtime(pool, saver, store)
        req = request("review-terminal-001")
        await rt.submit(req)
        await rt.broker.tick()
        await rt._drive(await store.get_run(req.run_id))
        completed = [event for event in await store.history(req.run_id) if event["event"] == "run.completed"]
        assert [event["entry_id"] for event in completed] == [f"{req.run_id}:terminal:completed"]
        await rt.close()
    asyncio.run(with_database(scenario))


def test_retry_preserves_first_assigned_lane_across_multiple_waiting_ticks():
    async def scenario(pool, saver, store):
        now = [datetime(2026, 9, 12, tzinfo=timezone.utc)]
        store.clock = lambda: now[0]
        rt = runtime(pool, saver, store)
        rt.broker.widening_enabled = True
        rt.broker.lanes["metacog"] = {"backend_key": "http://alternative", "configured": True,
            "healthy": True, "compatible_with": ["agent"], "capabilities": {}, "external_busy": True}
        study = request("review-pinned-study")
        study.admission.alternatives = ["metacog"]
        await rt.submit(study)
        original = (await rt.broker.tick())[0]
        assert original["lane"] == "agent"
        await store.release(original, "attempt_failed")
        holder = request("review-pinned-holder")
        holder.brief.timeout_sec = 3500
        await rt.submit(holder)
        holder_lease = (await rt.broker.tick())[0]
        assert holder_lease["run_id"] == holder.run_id
        await store.renew(holder_lease, 2000)
        now[0] += timedelta(seconds=1201)
        await store.register_demand(study.run_id, study.admission.model_dump(mode="json"))
        assert await rt.broker.tick() == []  # both lanes occupied
        rt.broker.lanes["metacog"]["external_busy"] = False
        assert await rt.broker.tick() == []  # prior waiting decision must not erase first assignment
        assert await store.get_lease(study.run_id) is None
        demand = await store.get_demand(study.run_id)
        assert demand["decision"]["requested_lane"] == "agent"
        assert demand["decision"]["retained_assigned_lane"] == "agent"
        assert demand["decision"]["suppressed"]["metacog"] == "run_assignment_locked"
        assert demand["requirement"]["pinned_lane"] is None  # preserve immutable operator request
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
        first = (await rt.broker.tick())[0]
        now[0] += timedelta(seconds=5)
        assert (await rt.status(req.run_id))["queue_wait_seconds"] == 12
        await store.release(first, "attempt_failed")
        now[0] += timedelta(seconds=200)
        assert (await rt.status(req.run_id))["queue_wait_seconds"] == 12
        await store.register_demand(req.run_id, req.admission.model_dump(mode="json"))
        retry = (await rt.broker.tick())[0]
        assert retry["generation"] > first["generation"]
        assert (await rt.status(req.run_id))["queue_wait_seconds"] == 12
        await store.finish_projection(req.run_id, "completed", {})
        now[0] += timedelta(seconds=500)
        assert (await rt.status(req.run_id))["queue_wait_seconds"] == 12
        await rt.close()
    asyncio.run(with_database(scenario))


def test_concurrent_policy_versions_share_first_committed_candidate_set(monkeypatch):
    import json
    from orion.durable_admission.store import SubmissionConflict
    async def scenario(pool, saver, store):
        first, second = runtime(pool, saver, store), runtime(pool, saver, store)
        first.settings.lane_policy_json = json.dumps({"metacog": {"compatible_with": ["agent"]}})
        second.settings.lane_policy_json = json.dumps({"chat": {"compatible_with": ["agent"]}})
        req = request("review-policy-race")
        original_get = store.get_run
        both_read = asyncio.Event()
        reads = 0
        async def synchronized_read(run_id):
            nonlocal reads
            value = await original_get(run_id)
            reads += 1
            if reads <= 2:
                assert value is None
                if reads == 2:
                    both_read.set()
                await both_read.wait()
            return value
        monkeypatch.setattr(store, "get_run", synchronized_read)
        receipts = await asyncio.gather(first.submit(req), second.submit(req))
        assert [receipt["run_id"] for receipt in receipts] == [req.run_id, req.run_id]
        row = await store.get_run(req.run_id)
        candidates = row["request"]["admission"]["alternatives"]
        assert candidates in (["metacog"], ["chat"])
        assert (await store.get_demand(req.run_id))["requirement"]["alternatives"] == candidates
        assert req.admission.alternatives == []
        changed = req.model_copy(update={"brief": req.brief.model_copy(update={"prompt": "Different study"})})
        with pytest.raises(SubmissionConflict):
            await first.submit(changed)
        explicit = req.model_copy(update={"admission": req.admission.model_copy(update={
            "alternatives": ["chat"] if candidates == ["metacog"] else ["metacog"]})})
        with pytest.raises(SubmissionConflict):
            await first.submit(explicit)
        await first.close()
        await second.close()
    asyncio.run(with_database(scenario))
