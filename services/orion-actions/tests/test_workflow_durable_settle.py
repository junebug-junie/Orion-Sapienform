"""Scheduled compactor runs settle from their ``compactor.digest`` durable run, not inside the RPC.

cortex-orch replies ``accepted`` once the durable run is registered; the schedule run stays
``dispatched`` (in flight) until that run's terminal ``orion:durable:run:state`` row arrives, which
settles it through the normal success / failure (retry budget, attention) paths. A terminal row that
never arrives is failed by the reaper only after the run's own deadline + grace, never at the
300 s claim TTL.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.main import (
    DURABLE_COMPLETION_GRACE, accepted_durable_run, durable_failure_notify_request, durable_settlement,
)
from app.workflow_schedule_store import WorkflowScheduleStore
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1

SERVICE = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 29, 12, 10, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 9, 30, 5, 59, 59, tzinfo=timezone.utc)


def _store(tmp_path, **kwargs) -> WorkflowScheduleStore:
    store = WorkflowScheduleStore(str(tmp_path / "wf.json"), claim_ttl_seconds=300, **kwargs)
    store.upsert_from_dispatch(WorkflowDispatchRequestV1.model_validate({
        "request_id": "bootstrap:github_compactor_pass",
        "workflow_id": "github_compactor_pass",
        "workflow_request": {"workflow_id": "github_compactor_pass"},
        "execution_policy": {"workflow_id": "github_compactor_pass", "invocation_mode": "scheduled",
                             "notify_on": "failure",
                             "schedule": {"kind": "recurring", "timezone": "America/Denver", "cadence": "daily",
                                          "hour_local": 6, "minute_local": 10}},
    }), now_utc=T0 - timedelta(days=1))
    return store


def _claim(store: WorkflowScheduleStore):
    [claimed] = store.claim_due(now_utc=T0)
    return claimed


def _accepted(run_id: str = "compactor:github_compactor_pass:2026-09-28:acme-widgets:abc") -> dict:
    return {"ok": True, "status": "accepted", "metadata": {"workflow": {
        "status": "accepted", "durable_run": {"run_id": run_id, "deadline_at": DEADLINE.isoformat()}}}}


def _state(status: str, *, run_id="compactor:github_compactor_pass:2026-09-28:acme-widgets:abc",
           workflow="compactor.digest", detail=None) -> dict:
    return DurableRunStateV1(run_id=run_id, workflow=workflow, thread_id=run_id, node="finish",
                             status=status, correlation_id="c", detail=detail or {}).model_dump(mode="json")


def _run(store, run_id):
    return next(r for r in store._runs if r.run_id == run_id)


def test_accepted_durable_run_reads_orch_reply_and_deadline():
    durable = accepted_durable_run(_accepted())
    assert durable["run_id"].startswith("compactor:") and durable["awaiting_until"] == DEADLINE + DURABLE_COMPLETION_GRACE
    # A workflow that finished inside the dispatch (or an already-finalized window) is not awaited.
    assert accepted_durable_run({"ok": True, "status": "success", "metadata": {"workflow": {"status": "completed"}}}) is None
    assert accepted_durable_run({"ok": True, "status": "accepted", "metadata": {}}) is None


def test_durable_settlement_only_for_compactor_terminal_rows():
    assert durable_settlement(_state("completed")) == {"durable_run_id": _state("completed")["run_id"],
                                                        "status": "completed", "error": None}
    failed = durable_settlement(_state("failed", detail={"error": "workflow_deadline"}))
    assert failed["status"] == "failed" and failed["error"] == "workflow_deadline"
    assert durable_settlement(_state("running")) is None
    assert durable_settlement(_state("completed", workflow="reading.turn")) is None
    assert durable_settlement({"not": "a state row"}) is None


def test_awaiting_run_survives_claim_ttl_then_completion_settles_it(tmp_path):
    store = _store(tmp_path)
    claimed = _claim(store)
    durable = accepted_durable_run(_accepted())
    store.mark_awaiting_durable(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id,
                                durable_run_id=durable["run_id"], awaiting_until=durable["awaiting_until"], now_utc=T0)
    # An hour later the 300 s claim TTL is long gone: an awaited run is NOT reaped.
    store.claim_due(now_utc=T0 + timedelta(hours=1))
    assert _run(store, claimed.run.run_id).status == "dispatched"
    # Restart-safe: the awaiting marker is persisted.
    reloaded = WorkflowScheduleStore(str(tmp_path / "wf.json"), claim_ttl_seconds=300)
    settled = reloaded.settle_durable_run(**durable_settlement(_state("completed")), now_utc=T0 + timedelta(hours=2))
    assert [item["run_id"] for item in settled] == [claimed.run.run_id]
    run = _run(reloaded, claimed.run.run_id)
    assert run.status == "completed" and run.error is None
    schedule = reloaded.list_schedules(include_inactive=True)[0]
    assert schedule.last_result_status == "completed"
    # A replayed terminal row (at-least-once outbox) is a no-op.
    assert reloaded.settle_durable_run(**durable_settlement(_state("completed"))) == []


def test_durable_failure_goes_through_the_retry_path(tmp_path):
    store = _store(tmp_path, retry_backoff_seconds=300)
    claimed = _claim(store)
    store.mark_awaiting_durable(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id,
                                durable_run_id=_state("failed")["run_id"], awaiting_until=DEADLINE, now_utc=T0)
    now = T0 + timedelta(hours=3)
    store.settle_durable_run(**durable_settlement(_state("failed", detail={"error": "workflow_deadline"})), now_utc=now)
    run = _run(store, claimed.run.run_id)
    assert run.status == "failed" and run.error == "durable_run_failed:workflow_deadline"
    schedule = store.list_schedules(include_inactive=True)[0]
    assert schedule.last_result_status == "failed"
    assert schedule.next_run_at == now + timedelta(seconds=300)          # bounded retry armed


def test_unobserved_completion_is_failed_only_after_the_run_deadline(tmp_path):
    store = _store(tmp_path)
    claimed = _claim(store)
    until = DEADLINE + DURABLE_COMPLETION_GRACE
    store.mark_awaiting_durable(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id,
                                durable_run_id="compactor:x", awaiting_until=until, now_utc=T0)
    store.claim_due(now_utc=until - timedelta(minutes=1))
    assert _run(store, claimed.run.run_id).status == "dispatched"
    store.claim_due(now_utc=until + timedelta(minutes=1))
    run = _run(store, claimed.run.run_id)
    assert run.status == "failed" and run.error == "durable_run_completion_unobserved:compactor:x"


def test_plain_dispatch_is_still_reaped_at_the_claim_ttl(tmp_path):
    store = _store(tmp_path)
    claimed = _claim(store)
    store.claim_due(now_utc=T0 + timedelta(seconds=301))
    assert _run(store, claimed.run.run_id).error == "claim_expired_after_300s"


def test_scheduler_awaits_accepted_durable_runs_and_subscribes_to_state():
    src = (SERVICE / "app" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_scheduler_loop")
    body = ast.unparse(loop)
    assert "accepted_durable_run(orch_payload)" in body and "mark_awaiting_durable(" in body
    assert "patterns.append(DURABLE_RUN_STATE_CHANNEL)" in src
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_envelope")
    assert "DURABLE_RUN_STATE_KIND" in ast.unparse(handler)


def test_terminal_row_before_the_accepted_reply_settles_on_mark(tmp_path):
    """A run can end before the scheduler loop records orch's accepted reply (instant refusal, or a
    re-dispatch finding a run that finishes in the gap): it must not wait ~18h for the reaper."""
    store = _store(tmp_path, retry_backoff_seconds=300)
    claimed = _claim(store)
    early = store.settle_durable_run(**durable_settlement(_state("failed", detail={"error": "gpu_pool_unavailable:x"})),
                                     now_utc=T0)
    assert early == []                                            # nothing waiting yet: remembered
    settled = store.mark_awaiting_durable(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id,
                                          durable_run_id=_state("failed")["run_id"], awaiting_until=DEADLINE,
                                          now_utc=T0 + timedelta(seconds=5))
    assert [item["status"] for item in settled] == ["failed"]
    assert _run(store, claimed.run.run_id).error == "durable_run_failed:gpu_pool_unavailable:x"


def test_failure_notifies_once_per_durable_run_per_policy(tmp_path):
    store = _store(tmp_path)
    claimed = _claim(store)
    store.mark_awaiting_durable(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id,
                                durable_run_id=_state("failed")["run_id"], awaiting_until=DEADLINE, now_utc=T0)
    [item] = store.settle_durable_run(**durable_settlement(_state("failed", detail={"error": "workflow_deadline"})),
                                      now_utc=T0 + timedelta(hours=1))
    req = durable_failure_notify_request(item)                    # schedule notify_on=failure
    assert req is not None and req.event_kind == "orion.workflow.failed"
    assert req.dedupe_key == f"workflow:github_compactor_pass:failed:{item['durable_run_id']}"
    assert "workflow_deadline" in req.body_text
    assert durable_failure_notify_request({**item, "status": "completed"}) is None
    assert durable_failure_notify_request({**item, "notify_on": "none"}) is None
    assert durable_failure_notify_request({**item, "notify_on": "success"}) is None
