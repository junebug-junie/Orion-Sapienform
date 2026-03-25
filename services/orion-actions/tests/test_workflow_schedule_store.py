from __future__ import annotations

from datetime import datetime, timezone

from app.workflow_schedule_store import WorkflowScheduleStore
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1, WorkflowScheduleManageRequestV1, WorkflowScheduleUpdatePatchV1


def _dispatch(request_id: str, *, workflow_id: str = "journal_pass", recurring: bool = False) -> WorkflowDispatchRequestV1:
    schedule = {
        "kind": "one_shot",
        "timezone": "America/Denver",
        "run_at_utc": "2026-03-24T06:00:00Z",
    }
    if recurring:
        schedule = {"kind": "recurring", "cadence": "daily", "timezone": "America/Denver", "hour_local": 22, "minute_local": 0}
    return WorkflowDispatchRequestV1.model_validate(
        {
            "request_id": request_id,
            "workflow_id": workflow_id,
            "workflow_request": {"workflow_id": workflow_id},
            "execution_policy": {
                "workflow_id": workflow_id,
                "invocation_mode": "scheduled",
                "notify_on": "completion",
                "recipient_group": "juniper_primary",
                "schedule": schedule,
            },
        }
    )


def test_durable_create_load(tmp_path):
    path = tmp_path / "schedules.json"
    store = WorkflowScheduleStore(str(path))
    created = store.upsert_from_dispatch(_dispatch("r1"))
    assert created is not None
    reloaded = WorkflowScheduleStore(str(path))
    rows = reloaded.list_schedules(include_inactive=True)
    assert len(rows) == 1
    assert rows[0].schedule_id == created.schedule_id


def test_one_shot_claim_marks_completed(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    store.upsert_from_dispatch(_dispatch("r1"))
    claimed = store.claim_due(now_utc=datetime(2026, 3, 24, 6, 1, tzinfo=timezone.utc))
    assert len(claimed) == 1
    listed = store.list_schedules(include_inactive=True)[0]
    assert listed.state == "completed"


def test_recurring_claim_advances_next_run(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    store.upsert_from_dispatch(_dispatch("r2", recurring=True), now_utc=datetime(2026, 3, 24, 1, 0, tzinfo=timezone.utc))
    before = store.list_schedules(include_inactive=True)[0].next_run_at
    store.claim_due(now_utc=datetime(2026, 3, 25, 6, 1, tzinfo=timezone.utc))
    after = store.list_schedules(include_inactive=True)[0].next_run_at
    assert after is not None and before is not None and after > before


def test_list_cancel_update_and_ambiguity(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    one = store.upsert_from_dispatch(_dispatch("r1", workflow_id="self_review", recurring=True))
    store.upsert_from_dispatch(_dispatch("r2", workflow_id="self_review", recurring=True))

    listed = store.apply_management(WorkflowScheduleManageRequestV1(operation="list", request_id="m1"))
    assert listed.ok is True and len(listed.schedules) == 2

    ambiguous = store.apply_management(WorkflowScheduleManageRequestV1(operation="cancel", request_id="m2", workflow_id="self_review"))
    assert ambiguous.ok is False and ambiguous.ambiguous is True

    updated = store.apply_management(
        WorkflowScheduleManageRequestV1(
            operation="update",
            request_id="m3",
            schedule_id=one.schedule_id if one else None,
            patch=WorkflowScheduleUpdatePatchV1(hour_local=22, minute_local=0),
        )
    )
    assert updated.ok is True

    cancelled = store.apply_management(WorkflowScheduleManageRequestV1(operation="cancel", request_id="m4", schedule_id=one.schedule_id if one else None))
    assert cancelled.ok is True and cancelled.schedule is not None and cancelled.schedule.state == "cancelled"


def test_update_conflict_returns_clear_message(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    created = store.upsert_from_dispatch(_dispatch("r1", workflow_id="journal_pass", recurring=True))
    assert created is not None

    conflict = store.apply_management(
        WorkflowScheduleManageRequestV1(
            operation="update",
            request_id="m-conflict",
            schedule_id=created.schedule_id,
            patch=WorkflowScheduleUpdatePatchV1(hour_local=21, expected_revision=created.revision - 1),
        )
    )
    assert conflict.ok is False
    assert "revision conflict" in conflict.message.lower()
    assert conflict.error_code == "schedule_revision_conflict"


def test_history_includes_lifecycle_events(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    created = store.upsert_from_dispatch(_dispatch("r1", recurring=True))
    assert created is not None
    store.apply_management(WorkflowScheduleManageRequestV1(operation="pause", request_id="m-pause", schedule_id=created.schedule_id))
    history = store.apply_management(WorkflowScheduleManageRequestV1(operation="history", request_id="m-history", schedule_id=created.schedule_id))
    assert history.ok is True
    assert any(evt.kind == "schedule_paused" for evt in history.events)


def test_transition_errors_are_structured(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    created = store.upsert_from_dispatch(_dispatch("r1", recurring=True))
    assert created is not None
    paused = store.apply_management(WorkflowScheduleManageRequestV1(operation="pause", request_id="m1", schedule_id=created.schedule_id))
    assert paused.ok is True
    already_paused = store.apply_management(WorkflowScheduleManageRequestV1(operation="pause", request_id="m2", schedule_id=created.schedule_id))
    assert already_paused.ok is False
    assert already_paused.error_code == "already_paused"
    cancelled = store.apply_management(WorkflowScheduleManageRequestV1(operation="cancel", request_id="m3", schedule_id=created.schedule_id))
    assert cancelled.ok is True
    resume_cancelled = store.apply_management(WorkflowScheduleManageRequestV1(operation="resume", request_id="m4", schedule_id=created.schedule_id))
    assert resume_cancelled.ok is False
    assert resume_cancelled.error_code == "unsupported_transition"


def test_attention_signals_dedupe_and_recovery(tmp_path):
    store = WorkflowScheduleStore(str(tmp_path / "schedules.json"))
    created = store.upsert_from_dispatch(_dispatch("r1", recurring=True), now_utc=datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc))
    assert created is not None
    claimed = store.claim_due(now_utc=datetime(2026, 3, 25, 6, 0, tzinfo=timezone.utc))
    assert len(claimed) == 1
    store.mark_dispatch_failed(run_id=claimed[0].run.run_id, schedule_id=claimed[0].schedule.schedule_id, error="boom", now_utc=datetime(2026, 3, 25, 6, 1, tzinfo=timezone.utc))
    signals = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 6, 2, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(signals) == 1
    assert signals[0].state == "active"
    again = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 6, 3, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert again == []
    store.mark_dispatch_succeeded(
        run_id=claimed[0].run.run_id,
        schedule_id=claimed[0].schedule.schedule_id,
        now_utc=datetime(2026, 3, 25, 6, 4, tzinfo=timezone.utc),
    )
    recovered = store.evaluate_attention_signals(now_utc=datetime(2026, 3, 25, 6, 5, tzinfo=timezone.utc), reminder_cooldown_seconds=9999)
    assert len(recovered) == 1
    assert recovered[0].transition == "recovered"
