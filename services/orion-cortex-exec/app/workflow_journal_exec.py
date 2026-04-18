"""Emit journal.entry.write.v1 from exec for workflow-owned journal.compose runs."""

from __future__ import annotations

import logging

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler.schemas import JournalEntryWriteV1, JournalTriggerV1
from orion.journaler.workflow_journal_delegate import workflow_journal_write_delegated_to_exec
from orion.journaler.worker import (
    JOURNAL_WRITE_KIND,
    build_write_payload,
    coerce_journal_title,
    draft_from_cortex_result,
    stable_journal_entry_id_for_workflow_compose,
)
from orion.schemas.cortex.schemas import PlanExecutionRequest, PlanExecutionResult

logger = logging.getLogger("orion.cortex.exec.workflow_journal")

JOURNAL_WRITE_CHANNEL = "orion:journal:write"


async def maybe_publish_workflow_journal_write(
    *,
    bus: OrionBusAsync | None,
    source: ServiceRef,
    correlation_id: str,
    payload: PlanExecutionRequest,
    result: PlanExecutionResult,
) -> str | None:
    """
    If this journal.compose was invoked from a chat workflow that delegates persistence to exec,
    parse the draft, build JournalEntryWriteV1, and publish to orion:journal:write.

    Returns None on success, or a short error string on failure (caller should mark plan failed).
    """
    if bus is None or not getattr(bus, "enabled", True):
        return "journal_write_exec_missing_bus"
    if str(result.status or "").lower() != "success":
        return None
    if str(payload.plan.verb_name or "").strip() != "journal.compose":
        return None
    ctx = payload.context or {}
    md = ctx.get("metadata") if isinstance(ctx, dict) else None
    if not isinstance(md, dict) or not workflow_journal_write_delegated_to_exec(md):
        return None
    raw_trigger = md.get("journal_trigger")
    if not isinstance(raw_trigger, dict):
        return "journal_write_exec_missing_journal_trigger"
    try:
        trigger = JournalTriggerV1.model_validate(raw_trigger)
    except Exception as exc:
        return f"journal_write_exec_invalid_trigger:{exc}"
    wf = md.get("workflow_execution") if isinstance(md.get("workflow_execution"), dict) else {}
    wf_id = str(md.get("workflow_id") or wf.get("workflow_id") or "journal_workflow")
    compose_payload = {
        "final_text": result.final_text,
        "status": result.status,
        "metadata": result.metadata if isinstance(result.metadata, dict) else {},
    }
    try:
        draft = draft_from_cortex_result(compose_payload)
    except Exception as exc:
        return f"journal_write_exec_draft_parse:{exc}"
    title = coerce_journal_title(
        raw_title=draft.title, fallback_summary=trigger.summary, correlation_id=correlation_id
    )
    body = str(draft.body or "").strip()
    if not body:
        return "journal_write_exec_empty_body"
    draft = draft.model_copy(update={"title": title, "body": body}, deep=True)
    author = str(payload.args.user_id or "").strip() or "orion"
    entry_id = stable_journal_entry_id_for_workflow_compose(
        correlation_id=correlation_id, workflow_id=wf_id, draft=draft, trigger=trigger
    )
    write = build_write_payload(
        draft, trigger=trigger, correlation_id=correlation_id, author=author, entry_id=entry_id
    )
    payload_dict = write.model_dump(mode="json")
    try:
        JournalEntryWriteV1.model_validate(payload_dict)
    except Exception as exc:
        return f"journal_write_exec_invalid_write_payload:{exc}"
    envelope = BaseEnvelope(
        kind=JOURNAL_WRITE_KIND,
        source=source,
        correlation_id=correlation_id,
        causality_chain=[],
        trace={"trace_id": correlation_id, "workflow_id": wf_id, "workflow_journal": "exec"},
        payload=payload_dict,
    )
    try:
        await bus.publish(JOURNAL_WRITE_CHANNEL, envelope)
    except ValueError as exc:
        logger.exception("journal_write_exec_catalog_rejected corr=%s err=%s", correlation_id, exc)
        return f"journal_write_exec_catalog_rejected:{exc}"
    except Exception as exc:
        logger.exception("journal_write_exec_publish_failed corr=%s", correlation_id)
        return f"journal_write_exec_publish_failed:{exc}"
    logger.info(
        "journal_write_exec_published corr=%s channel=%s entry_id=%s workflow_id=%s",
        correlation_id,
        JOURNAL_WRITE_CHANNEL,
        write.entry_id,
        wf_id,
    )
    return None
