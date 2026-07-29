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
_CHAT_HISTORY_COMPACTOR_BOOTSTRAP_REQUEST_ID = "bootstrap:chat_history_compactor_pass:daily:06:00:America/Denver"

GITHUB_COMPACTOR_WORKFLOW_ID = "github_compactor_pass"
_GITHUB_COMPACTOR_BOOTSTRAP_REQUEST_ID = "bootstrap:github_compactor_pass:daily:06:10:America/Denver"


def _blocks_bootstrap_seed(schedule, *, workflow_id: str, bootstrap_request_id: str) -> bool:
    """True when an existing record means bootstrap must not (re-)seed.

    Two cases block seeding:
    - Any record ever created by this bootstrap (matched by request_id), in ANY
      state — so an operator cancel or time edit sticks across restarts instead
      of being resurrected or duplicated at the bootstrap slot.
    - Any live recurring schedule for the workflow (operator-created), so a
      manual replacement schedule is not doubled up by bootstrap.
    """
    if getattr(schedule, "request_id", None) == bootstrap_request_id:
        return True
    if schedule.workflow_id != workflow_id:
        return False
    if schedule.state in {"cancelled", "completed"}:
        return False
    spec = schedule.execution_policy.schedule if schedule.execution_policy else None
    return spec is not None and spec.kind == "recurring"


def _ensure_daily_schedule(
    store: WorkflowScheduleStore,
    *,
    workflow_id: str,
    workflow_display_name: str,
    bootstrap_request_id: str,
    hour_local: int,
    minute_local: int,
    label: str,
    policy_summary: str,
    log_prefix: str,
) -> object | None:
    """Idempotently seed a daily America/Denver schedule for a workflow.

    Returns the created record, or None if a matching schedule already exists.
    """
    existing = [
        s
        for s in store.list_schedules(include_inactive=True)
        if _blocks_bootstrap_seed(s, workflow_id=workflow_id, bootstrap_request_id=bootstrap_request_id)
    ]
    if existing:
        logger.info(
            "%s_schedule_bootstrap_skip existing_schedule_id=%s",
            log_prefix,
            existing[0].schedule_id,
        )
        return None

    policy = WorkflowExecutionPolicyV1(
        workflow_id=workflow_id,
        invocation_mode="scheduled",
        schedule=WorkflowScheduleSpecV1(
            kind="recurring",
            timezone="America/Denver",
            cadence="daily",
            hour_local=hour_local,
            minute_local=minute_local,
            label=label,
        ),
        notify_on="completion",
        recipient_group="juniper_primary",
        requested_from_chat=False,
        policy_summary=policy_summary,
    )
    request = WorkflowDispatchRequestV1(
        request_id=bootstrap_request_id,
        workflow_id=workflow_id,
        workflow_request={
            "workflow_id": workflow_id,
            "workflow_display_name": workflow_display_name,
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
        "%s_schedule_bootstrap_created schedule_id=%s next_run_at=%s",
        log_prefix,
        getattr(record, "schedule_id", None),
        getattr(record, "next_run_at", None),
    )
    return record


def ensure_chat_history_compactor_daily_schedule(
    store: WorkflowScheduleStore,
) -> object | None:
    """Idempotently seed daily 06:00 America/Denver chat history compactor schedule."""
    return _ensure_daily_schedule(
        store,
        workflow_id=CHAT_HISTORY_COMPACTOR_WORKFLOW_ID,
        workflow_display_name="Chat History Compactor",
        bootstrap_request_id=_CHAT_HISTORY_COMPACTOR_BOOTSTRAP_REQUEST_ID,
        hour_local=6,
        minute_local=0,
        label="Chat history compactor (daily Denver)",
        policy_summary="Bootstrap: daily chat_history_log digest into indexed memory card",
        log_prefix="chat_history_compactor",
    )


def ensure_github_compactor_daily_schedule(
    store: WorkflowScheduleStore,
) -> object | None:
    """Idempotently seed daily 06:10 America/Denver GitHub compactor schedule.

    Runs 10 minutes after the chat history compactor's 06:00 slot so the two
    daily digests don't contend for the same cortex-orch dispatch window.
    """
    return _ensure_daily_schedule(
        store,
        workflow_id=GITHUB_COMPACTOR_WORKFLOW_ID,
        workflow_display_name="GitHub Compactor",
        bootstrap_request_id=_GITHUB_COMPACTOR_BOOTSTRAP_REQUEST_ID,
        hour_local=6,
        minute_local=10,
        label="GitHub compactor (daily Denver, +10m after chat)",
        policy_summary="Bootstrap: daily merged-PR digest into indexed memory card",
        log_prefix="github_compactor",
    )
