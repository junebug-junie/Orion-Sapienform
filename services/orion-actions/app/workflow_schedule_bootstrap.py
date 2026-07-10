from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from orion.schemas.workflow_execution import (
    WorkflowDispatchRequestV1,
    WorkflowExecutionPolicyV1,
    WorkflowScheduleSpecV1,
)

from .workflow_schedule_store import WorkflowScheduleStore

logger = logging.getLogger("orion.actions.workflow_schedule_bootstrap")

CHAT_HISTORY_COMPACTOR_WORKFLOW_ID = "chat_history_compactor_pass"
_BOOTSTRAP_REQUEST_ID = "bootstrap:chat_history_compactor_pass:daily:06:00:America/Denver"


def _is_matching_daily_chat_compactor(schedule) -> bool:
    if schedule.workflow_id != CHAT_HISTORY_COMPACTOR_WORKFLOW_ID:
        return False
    if schedule.state in {"cancelled", "completed"}:
        return False
    spec = schedule.execution_policy.schedule if schedule.execution_policy else None
    if spec is None or spec.kind != "recurring" or spec.cadence != "daily":
        return False
    # minute_local=0 is valid; do not use `or` (0 is falsy).
    if int(-1 if spec.hour_local is None else spec.hour_local) != 6:
        return False
    if int(-1 if spec.minute_local is None else spec.minute_local) != 0:
        return False
    if (spec.timezone or "") != "America/Denver":
        return False
    return True


def ensure_chat_history_compactor_daily_schedule(
    store: WorkflowScheduleStore,
) -> object | None:
    """Idempotently seed daily 06:00 America/Denver chat history compactor schedule.

    Returns the created record, or None if a matching schedule already exists.
    """
    existing = [s for s in store.list_schedules(include_inactive=True) if _is_matching_daily_chat_compactor(s)]
    if existing:
        logger.info(
            "chat_history_compactor_schedule_bootstrap_skip existing_schedule_id=%s",
            existing[0].schedule_id,
        )
        return None

    policy = WorkflowExecutionPolicyV1(
        workflow_id=CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
        invocation_mode="scheduled",
        schedule=WorkflowScheduleSpecV1(
            kind="recurring",
            timezone="America/Denver",
            cadence="daily",
            hour_local=6,
            minute_local=0,
            label="Chat history compactor (daily Denver)",
        ),
        notify_on="completion",
        recipient_group="juniper_primary",
        requested_from_chat=False,
        policy_summary="Bootstrap: daily chat_history_log digest into indexed memory card",
    )
    request = WorkflowDispatchRequestV1(
        request_id=_BOOTSTRAP_REQUEST_ID,
        workflow_id=CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
        workflow_request={
            "workflow_id": CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
            "workflow_display_name": "Chat History Compactor",
            "window_mode": "day",
            "execution_policy": policy.model_dump(mode="json"),
        },
        execution_policy=policy,
        correlation_id=str(uuid4()),
        source_service="orion-actions",
        source_kind="actions_scheduler",
        created_at=datetime.now(timezone.utc),
    )
    record = store.upsert_from_dispatch(request)
    logger.info(
        "chat_history_compactor_schedule_bootstrap_created schedule_id=%s next_run_at=%s",
        getattr(record, "schedule_id", None),
        getattr(record, "next_run_at", None),
    )
    return record
