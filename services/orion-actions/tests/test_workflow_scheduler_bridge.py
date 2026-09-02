from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.main import _publish_workflow_attention_signal, _schedule_attention_notify_request, workflow_schedule_metrics
from app.main import info as actions_info
from app.workflow_schedule_metrics import WorkflowScheduleMetrics
from app.workflow_schedule_store import WorkflowScheduleStore
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1, WorkflowScheduleManageRequestV1


def _dispatch_request(*, request_id: str, kind: str = "one_shot") -> WorkflowDispatchRequestV1:
    schedule = {
        "kind": kind,
        "timezone": "America/Denver",
        "run_at_utc": "2026-03-24T06:00:00Z",
        "label": "test",
    }
    if kind == "recurring":
        schedule = {
            "kind": "recurring",
            "timezone": "America/Denver",
            "cadence": "daily",
            "hour_local": 23,
            "minute_local": 0,
            "label": "nightly",
        }
    return WorkflowDispatchRequestV1.model_validate(
        {
            "request_id": request_id,
            "workflow_id": "journal_pass",
            "workflow_request": {"workflow_id": "journal_pass"},
            "execution_policy": {
                "workflow_id": "journal_pass",
                "invocation_mode": "scheduled",
                "notify_on": "completion",
                "recipient_group": "juniper_primary",
                "schedule": schedule,
            },
        }
    )


def test_claim_due_is_restart_safe(tmp_path) -> None:
    path = tmp_path / "wf-schedules.json"
    store = WorkflowScheduleStore(str(path))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-1", kind="one_shot"))
    assert len(store.claim_due(now_utc=datetime(2026, 3, 24, 6, 1, tzinfo=timezone.utc))) == 1

    reloaded = WorkflowScheduleStore(str(path))
    assert len(reloaded.claim_due(now_utc=datetime(2026, 3, 24, 6, 2, tzinfo=timezone.utc))) == 0


def test_recurring_schedule_advances_after_dispatch(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-2", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    before = store.list_schedules(include_inactive=True)[0].next_run_at
    store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    after = store.list_schedules(include_inactive=True)[0].next_run_at
    assert before is not None and after is not None and after > before


def test_recurring_dispatch_failure_requeues_claimed_slot(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    store.upsert_from_dispatch(_dispatch_request(request_id="req-3", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    assert len(claimed) == 1
    claimed_for = claimed[0].run.metadata.get("claimed_for_run_at")
    assert claimed_for is not None
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="downstream failure", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    reloaded = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    row = reloaded.list_schedules(include_inactive=True)[0]
    assert row.next_run_at is not None
    assert row.state == "scheduled"
    # The slot is retried, but at now + backoff -- NOT rewound to `claimed_for`,
    # which is already in the past and made the next scheduler tick re-claim it
    # immediately. First attempt: 07:01 + 300s.
    assert row.next_run_at == datetime(2026, 3, 25, 7, 6, tzinfo=timezone.utc)
    assert datetime.fromisoformat(claimed_for) < row.next_run_at


def test_dispatch_failure_backoff_grows_then_abandons_the_slot(tmp_path) -> None:
    """A persistently failing slot retries a bounded number of times, then stops.

    Before this bound existed, mark_dispatch_failed rewound next_run_at to a time
    already in the past, so the next scheduler poll (45s) re-claimed the same slot
    forever: 343 failure notifications from one schedule on 2026-08-20.
    """
    store = WorkflowScheduleStore(
        str(tmp_path / "wf-schedules.json"),
        max_dispatch_attempts=3,
        retry_backoff_seconds=300,
    )
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-backoff", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )

    def _fail_once(at: datetime) -> None:
        claimed = store.claim_due(now_utc=at)
        assert len(claimed) == 1
        store.mark_dispatch_failed(
            run_id=claimed[0].run.run_id,
            schedule_id=claimed[0].schedule.schedule_id,
            error="boom",
            now_utc=at,
        )

    # Attempt 1 -> 300 * 2**0 = 300s.
    _fail_once(datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    row = store.list_schedules(include_inactive=True)[0]
    assert row.next_run_at == datetime(2026, 3, 25, 7, 5, tzinfo=timezone.utc)

    # Attempt 2 -> 300 * 2**1 = 600s. Same slot, so the claimed_for key is stable.
    _fail_once(datetime(2026, 3, 25, 7, 5, tzinfo=timezone.utc))
    row = store.list_schedules(include_inactive=True)[0]
    assert row.next_run_at == datetime(2026, 3, 25, 7, 15, tzinfo=timezone.utc)

    # Attempt 3 spends the budget: the slot is abandoned to the next natural
    # occurrence claim_due already advanced to, not retried again.
    _fail_once(datetime(2026, 3, 25, 7, 15, tzinfo=timezone.utc))
    row = store.list_schedules(include_inactive=True)[0]
    assert row.next_run_at is not None
    assert row.next_run_at > datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)
    assert store.claim_due(now_utc=datetime(2026, 3, 25, 7, 30, tzinfo=timezone.utc)) == []


def test_attention_clears_once_a_degraded_schedule_recovers(tmp_path) -> None:
    """The live bug: `health` stays "degraded" for 5 runs, attention must not.

    github_compactor_pass sent 8 notifications/day from 2026-08-21 to 2026-09-01,
    including every day its run completed, because the attention condition paged
    off `health == "degraded"` -- a trailing-5-run property that outlives the
    failure by days.
    """
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), max_dispatch_attempts=1)
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-recover", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_failed(
        run_id=claimed[0].run.run_id,
        schedule_id=claimed[0].schedule.schedule_id,
        error="boom",
        now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc),
    )
    # Currently failing: attention fires.
    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc))
    assert [s.kind for s in signals] == ["degraded"]

    # Next day's run succeeds. health is still "degraded" (1 failure in the last
    # 5 runs), but the schedule is working again, so attention must clear exactly
    # once as "recovered" and then stay silent.
    claimed = store.claim_due(now_utc=datetime(2026, 3, 26, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id,
        schedule_id=claimed[0].schedule.schedule_id,
        now_utc=datetime(2026, 3, 26, 7, 1, tzinfo=timezone.utc),
    )
    # Read health the way the badge does, through the "list" management surface
    # that attaches analytics.
    listed = store.apply_management(
        WorkflowScheduleManageRequestV1.model_validate({"operation": "list", "request_id": "req-list"}),
        now_utc=datetime(2026, 3, 26, 7, 2, tzinfo=timezone.utc),
    )
    assert listed.schedules[0].analytics is not None
    assert listed.schedules[0].analytics.health == "degraded"

    recovered = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 26, 7, 2, tzinfo=timezone.utc))
    assert [s.transition for s in recovered] == ["recovered"]

    # ...and no further nagging as the daily cycle keeps succeeding, even though
    # `health` stays "degraded" until the failure ages out of the 5-run window.
    # This is the steady state that produced 8 messages/day live.
    for day in (27, 28):
        claimed = store.claim_due(now_utc=datetime(2026, 3, day, 7, 0, tzinfo=timezone.utc))
        assert len(claimed) == 1
        store.mark_dispatch_succeeded(
            run_id=claimed[0].run.run_id,
            schedule_id=claimed[0].schedule.schedule_id,
            now_utc=datetime(2026, 3, day, 7, 1, tzinfo=timezone.utc),
        )
        assert store.evaluate_attention_signals(now_utc=datetime(2026, 3, day, 7, 2, tzinfo=timezone.utc)) == []


def test_set_notify_on_updates_the_copy_a_dispatch_actually_reads(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    request = _dispatch_request(request_id="req-notify", kind="recurring")
    request.workflow_request["execution_policy"] = request.execution_policy.model_dump(mode="json")
    created = store.upsert_from_dispatch(request, now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None
    assert created.workflow_request["execution_policy"]["notify_on"] == "completion"

    assert store.set_notify_on(schedule_id=created.schedule_id, notify_on="failure") is True

    reloaded = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    row = reloaded.list_schedules(include_inactive=True)[0]
    assert row.notify_on == "failure"
    assert row.execution_policy.notify_on == "failure"
    # _dispatch_scheduled_workflow forwards this copy to cortex-orch, and it is what
    # _emit_workflow_notify reads -- updating only the record field is a no-op live.
    assert row.workflow_request["execution_policy"]["notify_on"] == "failure"

    # Idempotent: a second call reports "nothing changed".
    assert reloaded.set_notify_on(schedule_id=created.schedule_id, notify_on="failure") is False
    assert reloaded.set_notify_on(schedule_id="missing", notify_on="failure") is False


def test_attention_notify_integration_dedupe_payload_and_no_spam(tmp_path) -> None:
    metrics = WorkflowScheduleMetrics()
    # Attention pages when the retry budget is spent, not on the first failure;
    # a 1-attempt budget makes a single failure reach that condition.
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), metrics=metrics, max_dispatch_attempts=1)
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-integration", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc))
    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(signals) == 1
    signal = signals[0]
    sent: list = []
    attention_requests: list = []
    chat_messages: list = []

    class _Notify:
        def send(self, req):
            sent.append(req)
            return SimpleNamespace(ok=True, detail=None)

        def attention_request(self, **kwargs):
            attention_requests.append(kwargs)
            return SimpleNamespace(ok=True, detail=None, notification_id=None)

        def chat_message(self, **kwargs):
            chat_messages.append(kwargs)
            return SimpleNamespace(ok=True, detail=None, notification_id=None)

    before_entered = workflow_schedule_metrics.get("workflow_schedule_attention_entered_total")
    asyncio.run(_publish_workflow_attention_signal(signal=signal, notify=_Notify()))
    assert workflow_schedule_metrics.get("workflow_schedule_attention_entered_total") == before_entered + 1
    assert len(sent) == 1
    assert len(attention_requests) == 1
    req = sent[0]
    # One failure with a retry still pending is "degraded", not "failing" -- the old
    # "failing" label came from mark_dispatch_failed rewinding next_run_at into the
    # past, which made a retry-pending schedule read as overdue. Escalation to
    # "failing" now happens on real repeat failure, not on that artifact.
    assert req.dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:degraded"
    assert req.context["transition"] == "entered"
    assert req.context["condition"] == "degraded"
    assert req.context["state"] == "active"
    assert req.context["schedule_id_short"] == created.schedule_id[-8:]
    assert attention_requests[0]["context"]["reason"].startswith("Workflow schedule needs attention:")

    again = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 3, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert again == []

    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id,
        schedule_id=claimed[0].schedule.schedule_id,
        now_utc=datetime(2026, 3, 25, 7, 4, tzinfo=timezone.utc),
    )
    recovered = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 5, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(recovered) == 1
    sent.clear()
    attention_requests.clear()
    chat_messages.clear()
    before_recovered = workflow_schedule_metrics.get("workflow_schedule_attention_recovered_total")
    asyncio.run(_publish_workflow_attention_signal(signal=recovered[0], notify=_Notify()))
    assert workflow_schedule_metrics.get("workflow_schedule_attention_recovered_total") == before_recovered + 1
    assert sent[0].dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:recovered"
    assert sent[0].context["transition"] == "recovered"
    assert sent[0].context["condition"] == "ok"
    assert attention_requests == []
    assert len(chat_messages) == 1


def test_attention_notify_integration_overdue_transition(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    created = store.upsert_from_dispatch(_dispatch_request(request_id="req-overdue", kind="recurring"), now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None
    signals = store.evaluate_attention_signals(
        now_utc=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc),
        overdue_min_seconds=120,
        reminder_cooldown_seconds=9999,
    )
    assert len(signals) == 1
    req = _schedule_attention_notify_request(signal=signals[0], correlation_id="corr-overdue")
    assert req.dedupe_key == f"workflow:schedule:attention:{created.schedule_id}:overdue"
    assert req.context["transition"] == "entered"
    assert req.context["condition"] == "overdue"
    assert req.context["is_overdue"] is True


def test_actions_info_surface_exposes_runtime_identity() -> None:
    payload = asyncio.run(actions_info())
    assert payload["service"]
    assert payload["version"]
    assert payload["process_started_at"]


def test_hung_dispatch_is_reaped_and_still_pages(tmp_path) -> None:
    """A dispatch that never terminates must not silence the attention signal.

    `_claim_ttl` was accepted by the constructor and never read, so nothing reaped
    a hung dispatch: the live store still holds an orphaned `dispatched` row from
    2026-08-20T16:24:52Z. That orphan is the newest run for its schedule, which is
    exactly the shape that makes a "did the newest run fail?" attention condition
    go quiet on a schedule that is genuinely stuck.
    """
    store = WorkflowScheduleStore(
        str(tmp_path / "wf-schedules.json"),
        claim_ttl_seconds=300,
        max_dispatch_attempts=1,
    )
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-hang", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    assert len(claimed) == 1
    # Neither mark_dispatch_succeeded nor mark_dispatch_failed is ever called.
    assert store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 2, tzinfo=timezone.utc)) == []

    # Inside the TTL the claim is still legitimately in flight and is left alone.
    store.claim_due(now_utc=datetime(2026, 3, 25, 7, 4, tzinfo=timezone.utc))
    run = [r for r in store._runs if r.run_id == claimed[0].run.run_id][0]
    assert run.status == "dispatched"

    # Past the TTL it is reaped through the normal failure path.
    store.claim_due(now_utc=datetime(2026, 3, 25, 7, 6, tzinfo=timezone.utc))
    run = [r for r in store._runs if r.run_id == claimed[0].run.run_id][0]
    assert run.status == "failed"
    assert run.error == "claim_expired_after_300s"
    assert run.completed_at is not None

    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 7, 7, tzinfo=timezone.utc))
    assert [s.kind for s in signals] == ["degraded"]


def test_retry_budget_survives_run_history_truncation(tmp_path) -> None:
    """A restart must not hand back a retry budget that was already spent.

    The persisted run list is truncated to `history_limit` on every save while the
    in-memory list is not, so a count derived from history answers differently
    before and after a restart -- always in the "grant more retries" direction. The
    live store is already sitting exactly at its 200-run cap, so this is reachable,
    and it is the same shape as the restart-resets-the-daily-cap incident.
    """
    path = str(tmp_path / "wf-schedules.json")
    store = WorkflowScheduleStore(path, history_limit=20, max_dispatch_attempts=3, retry_backoff_seconds=60)
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-trunc", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    at = datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc)
    for _ in range(3):
        claimed = store.claim_due(now_utc=at)
        assert len(claimed) == 1
        store.mark_dispatch_failed(
            run_id=claimed[0].run.run_id,
            schedule_id=claimed[0].schedule.schedule_id,
            error="boom",
            now_utc=at,
        )
        at = store.list_schedules(include_inactive=True)[0].next_run_at

    # Push the failures out of the persisted window with unrelated runs.
    for i in range(25):
        store.upsert_from_dispatch(
            _dispatch_request(request_id=f"noise-{i}", kind="recurring"),
            now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
        )
        store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))

    reloaded = WorkflowScheduleStore(path, history_limit=20, max_dispatch_attempts=3, retry_backoff_seconds=60)
    row = [s for s in reloaded.list_schedules(include_inactive=True) if s.request_id == "req-trunc"][0]
    assert reloaded._consecutive_failures(row) == 3
    # Budget still spent: the slot is not re-armed for another backoff retry.
    claimed = reloaded.claim_due(now_utc=datetime(2026, 3, 26, 7, 0, tzinfo=timezone.utc))
    mine = [c for c in claimed if c.schedule.request_id == "req-trunc"]
    assert len(mine) == 1
    reloaded.mark_dispatch_failed(
        run_id=mine[0].run.run_id,
        schedule_id=mine[0].schedule.schedule_id,
        error="boom",
        now_utc=datetime(2026, 3, 26, 7, 0, tzinfo=timezone.utc),
    )
    row = [s for s in reloaded.list_schedules(include_inactive=True) if s.request_id == "req-trunc"][0]
    assert row.next_run_at > datetime(2026, 3, 26, 8, 0, tzinfo=timezone.utc)


def test_a_success_clears_the_retry_budget(tmp_path) -> None:
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"), max_dispatch_attempts=3, retry_backoff_seconds=60)
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-reset", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    at = datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc)
    claimed = store.claim_due(now_utc=at)
    store.mark_dispatch_failed(
        run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=at
    )
    row = store.list_schedules(include_inactive=True)[0]
    assert store._consecutive_failures(row) == 1

    claimed = store.claim_due(now_utc=row.next_run_at)
    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, now_utc=row.next_run_at
    )
    row = store.list_schedules(include_inactive=True)[0]
    assert store._consecutive_failures(row) == 0
    assert "consecutive_failures" not in (row.metadata or {})


def test_a_failed_dispatch_does_not_revive_a_cancelled_schedule(tmp_path) -> None:
    """Cancel while a dispatch is in flight; the dispatch then fails.

    The failure must not undo the cancel -- least of all arm the schedule to run
    again one backoff from now.
    """
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    created = store.upsert_from_dispatch(
        _dispatch_request(request_id="req-cancel", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    assert created is not None
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc))
    assert len(claimed) == 1
    resp = store.apply_management(
        WorkflowScheduleManageRequestV1.model_validate(
            {"operation": "cancel", "request_id": "m-cancel", "schedule_id": created.schedule_id}
        )
    )
    assert resp.ok is True

    store.mark_dispatch_failed(
        run_id=claimed[0].run.run_id,
        schedule_id=created.schedule_id,
        error="boom",
        now_utc=datetime(2026, 3, 25, 7, 1, tzinfo=timezone.utc),
    )

    row = [s for s in store.list_schedules(include_inactive=True) if s.schedule_id == created.schedule_id][0]
    assert row.state == "cancelled"
    assert store.claim_due(now_utc=datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)) == []


def test_management_update_of_notify_on_reaches_the_authoritative_copy(tmp_path) -> None:
    """The operator-facing path must write the copy a dispatch actually reads.

    It previously wrote only the record field and execution_policy, leaving the
    embedded copy stale -- so an operator changing notify_on saw no change in what
    notified them, and the bootstrap reconcile would then overwrite their setting.
    """
    store = WorkflowScheduleStore(str(tmp_path / "wf-schedules.json"))
    request = _dispatch_request(request_id="req-mgmt-notify", kind="recurring")
    request.workflow_request["execution_policy"] = request.execution_policy.model_dump(mode="json")
    created = store.upsert_from_dispatch(request, now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc))
    assert created is not None

    resp = store.apply_management(
        WorkflowScheduleManageRequestV1.model_validate(
            {
                "operation": "update",
                "request_id": "m-update",
                "schedule_id": created.schedule_id,
                "patch": {"notify_on": "none"},
            }
        )
    )
    assert resp.ok is True

    row = [s for s in store.list_schedules(include_inactive=True) if s.schedule_id == created.schedule_id][0]
    assert row.notify_on == "none"
    assert row.execution_policy.notify_on == "none"
    assert row.workflow_request["execution_policy"]["notify_on"] == "none"


def test_a_blip_that_the_retry_fixes_never_pages(tmp_path) -> None:
    """A failure the store recovers from on its own is not Juniper's problem.

    Paging on the first failure would send four messages for a transient blip:
    one workflow-failed notify, the attention "entered" signal (published twice,
    generic + pending-attention), then a "recovered" async message. Attention is
    reserved for the retry budget actually running out.
    """
    store = WorkflowScheduleStore(
        str(tmp_path / "wf-schedules.json"),
        max_dispatch_attempts=3,
        retry_backoff_seconds=60,
    )
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-blip", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    at = datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc)
    claimed = store.claim_due(now_utc=at)
    store.mark_dispatch_failed(
        run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="blip", now_utc=at
    )
    # Budget is not spent (1 of 3), so nothing pages while the retry is pending.
    assert store.evaluate_attention_signals(now_utc=at + timedelta(seconds=1)) == []

    retry_at = store.list_schedules(include_inactive=True)[0].next_run_at
    assert retry_at == at + timedelta(seconds=60)
    claimed = store.claim_due(now_utc=retry_at)
    assert len(claimed) == 1
    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, now_utc=retry_at
    )
    # ...and nothing pages after it recovers either: no "entered", so no
    # "recovered" to clear, so zero messages for the whole episode.
    assert store.evaluate_attention_signals(now_utc=retry_at + timedelta(seconds=60)) == []


def test_repeated_failures_still_page_once_the_budget_is_spent(tmp_path) -> None:
    """The other half of the blip rule: a genuinely stuck schedule must still page.

    Seeded with one success so the schedule sits on the `degraded` branch this rule
    governs. A schedule with *no* success in the window takes the separate `failing`
    branch at two failures, which is untouched here and correct as it stands --
    "two failures and nothing has ever worked" needs no retry budget to be alarming.
    """
    store = WorkflowScheduleStore(
        str(tmp_path / "wf-schedules.json"),
        max_dispatch_attempts=3,
        retry_backoff_seconds=60,
    )
    store.upsert_from_dispatch(
        _dispatch_request(request_id="req-stuck", kind="recurring"),
        now_utc=datetime(2026, 3, 24, 7, 0, tzinfo=timezone.utc),
    )
    seed_at = datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc)
    seeded = store.claim_due(now_utc=seed_at)
    store.mark_dispatch_succeeded(
        run_id=seeded[0].run.run_id, schedule_id=seeded[0].schedule.schedule_id, now_utc=seed_at
    )
    at = store.list_schedules(include_inactive=True)[0].next_run_at
    for attempt in (1, 2, 3):
        claimed = store.claim_due(now_utc=at)
        assert len(claimed) == 1
        store.mark_dispatch_failed(
            run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=at
        )
        signals = store.evaluate_attention_signals(now_utc=at + timedelta(seconds=1))
        if attempt < 3:
            assert signals == [], f"paged too early on attempt {attempt}"
        else:
            assert [s.kind for s in signals] == ["degraded"]
        at = store.list_schedules(include_inactive=True)[0].next_run_at
