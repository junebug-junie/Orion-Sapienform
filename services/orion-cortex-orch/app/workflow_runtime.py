from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Tuple
from uuid import uuid4

from orion.cognition.workflows import get_workflow_definition, workflow_registry_payload
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs import VerbResultV1
from orion.journaler.schemas import JournalEntryDraftV1
from orion.journaler.worker import (
    JOURNAL_WRITE_KIND,
    build_compose_request,
    build_manual_trigger,
    build_write_payload,
    draft_from_cortex_result,
)
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.spark.concept_induction.profile_repository import build_concept_profile_repository
from orion.spark.concept_induction.settings import DEFAULT_CONCEPT_STORE_PATH
from .concept_profile_config import build_orch_concept_profile_settings
from .settings import get_settings
from orion.schemas.notify import NotificationRequest
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1, WorkflowExecutionPolicyV1
from orion.schemas.workflow_execution import WorkflowScheduleManageRequestV1, WorkflowScheduleManageResponseV1

logger = logging.getLogger("orion.cortex.orch.workflow_runtime")
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
ACTIONS_WORKFLOW_TRIGGER_CHANNEL = "orion:actions:trigger:workflow.v1"
ACTIONS_WORKFLOW_MANAGE_CHANNEL = "orion:actions:manage:workflow.v1"
NOTIFY_PERSISTENCE_REQUEST_CHANNEL = "orion:notify:persistence:request"
ConceptProfileBackendKind = Literal["local", "graph", "shadow"]
CutoverFallbackPolicyKind = Literal["fail_open_local", "fail_closed"]


class WorkflowExecutionError(RuntimeError):
    pass


def has_explicit_workflow_request(req: CortexClientRequest) -> bool:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    workflow_request = metadata.get("workflow_request")
    return isinstance(workflow_request, dict) and bool(workflow_request.get("workflow_id"))


def has_workflow_schedule_management_request(req: CortexClientRequest) -> bool:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    manage = metadata.get("workflow_schedule_management")
    return isinstance(manage, dict) and bool(manage.get("operation"))


def _workflow_schedule_management_request(req: CortexClientRequest) -> Dict[str, Any]:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    manage = metadata.get("workflow_schedule_management")
    return dict(manage) if isinstance(manage, dict) else {}


def _workflow_request(req: CortexClientRequest) -> Dict[str, Any]:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    workflow_request = metadata.get("workflow_request")
    return dict(workflow_request) if isinstance(workflow_request, dict) else {}


def _execution_policy(req: CortexClientRequest, workflow_id: str) -> WorkflowExecutionPolicyV1:
    request = _workflow_request(req)
    raw = request.get("execution_policy")
    if isinstance(raw, dict):
        try:
            return WorkflowExecutionPolicyV1.model_validate(raw)
        except Exception:
            pass
    return WorkflowExecutionPolicyV1(
        workflow_id=workflow_id,
        invocation_mode="immediate",
        notify_on="none",
        recipient_group="juniper_primary",
        session_id=req.context.session_id,
        origin_user_id=req.context.user_id,
        requested_from_chat=True,
    )


def _workflow_metadata_base(*, request: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {
        "workflow_request": request,
        "workflow_status": status,
        "available_workflows": workflow_registry_payload(user_invocable_only=True),
    }


def _extract_result_payload(verb_result: VerbResultV1) -> Dict[str, Any]:
    payload = verb_result.output if isinstance(verb_result.output, dict) else {}
    if isinstance(payload.get("result"), dict):
        return dict(payload.get("result") or {})
    return dict(payload)


def _parse_json_text(text: str | None) -> Dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _ensure_trace(trace: dict | None, *, correlation_id: str, workflow_id: str) -> dict[str, Any]:
    base = dict(trace or {})
    base.setdefault("trace_id", correlation_id)
    base.setdefault("workflow_id", workflow_id)
    return base


def _resolve_personality_file(metadata: Dict[str, Any] | None) -> str | None:
    metadata = metadata if isinstance(metadata, dict) else {}
    direct = metadata.get("personality_file")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    plan_metadata = metadata.get("plan_metadata")
    if isinstance(plan_metadata, dict):
        nested = plan_metadata.get("personality_file")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _workflow_execution_envelope(
    *,
    req: CortexClientRequest,
    correlation_id: str,
    workflow_id: str,
    workflow_subverb: str | None,
) -> Dict[str, Any]:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    workflow_request = _workflow_request(req)
    personality_file = _resolve_personality_file(metadata)
    return {
        "correlation_id": correlation_id,
        "trace_id": req.context.trace_id or correlation_id,
        "session_id": req.context.session_id,
        "user_id": req.context.user_id,
        "workflow_id": workflow_id,
        "workflow_execution": {
            "workflow_id": workflow_id,
            "workflow_subverb": workflow_subverb,
            "resolver": workflow_request.get("resolver"),
            "matched_alias": workflow_request.get("matched_alias"),
        },
        "personality_file": personality_file,
    }


def _log_primary_metadata_status(
    *,
    correlation_id: str,
    workflow_id: str,
    verb: str,
    metadata: Dict[str, Any] | None,
) -> None:
    metadata = metadata if isinstance(metadata, dict) else {}
    workflow_execution = metadata.get("workflow_execution")
    personality_file = _resolve_personality_file(metadata)
    missing = [
        field
        for field in ("trace_id", "session_id", "user_id")
        if not metadata.get(field)
    ]
    if not isinstance(workflow_execution, dict):
        missing.append("workflow_execution")
    fallback_reason = "none"
    if not personality_file:
        fallback_reason = "missing_personality_file"
        missing.append("personality_file")
    logger.info(
        "workflow_primary_metadata_status %s",
        json.dumps(
            {
                "correlation_id": correlation_id,
                "workflow_id": workflow_id,
                "verb": verb,
                "personality_file_present": bool(personality_file),
                "workflow_metadata_present": isinstance(workflow_execution, dict),
                "missing_fields": missing,
                "fallback_reason": fallback_reason,
            },
            sort_keys=True,
            default=str,
        ),
    )


def _shape_workflow_result_summary(*, workflow_meta: Dict[str, Any], result: CortexClientResult) -> Dict[str, Any]:
    persisted = workflow_meta.get("persisted") if isinstance(workflow_meta, dict) else []
    result_shape = "structured" if isinstance(workflow_meta.get("skill_result"), dict) else "summary"
    low_value = False
    reason = None
    if workflow_meta.get("workflow_id") == "self_review":
        finding_count = int(workflow_meta.get("finding_count") or 0)
        summary = str(workflow_meta.get("main_result") or "").strip().lower()
        if finding_count == 0:
            low_value = "no notable self-review findings" not in summary
            reason = "empty_finding_summary" if low_value else "no_findings"
    return {
        "executed": bool(workflow_meta.get("executed", result.ok)),
        "persisted": bool(persisted),
        "result_shape": result_shape,
        "empty_or_low_value": low_value,
        "reason": reason,
    }


def _workflow_summary_text(*, title: str, status: str, main_result: str, persisted: Iterable[str] | None = None, scheduled: Iterable[str] | None = None) -> str:
    persisted_items = [item for item in (persisted or []) if item]
    scheduled_items = [item for item in (scheduled or []) if item]
    lines = [
        f"Workflow: {title}",
        f"Status: {status}",
        f"Result: {main_result}",
        f"Persisted: {', '.join(persisted_items) if persisted_items else 'none'}",
        f"Scheduled: {', '.join(scheduled_items) if scheduled_items else 'none'}",
    ]
    return "\n".join(lines)


def _coerce_journal_title(*, raw_title: Any, fallback_summary: str, correlation_id: str) -> str:
    title = str(raw_title or "").strip()
    if title:
        return title
    seed = str(fallback_summary or "").strip() or "Journal Pass"
    return f"Journal Pass · {seed[:64]} · {correlation_id[:8]}"


def _should_notify(*, notify_on: str, ok: bool) -> bool:
    normalized = str(notify_on or "none").lower()
    if normalized == "none":
        return False
    if normalized == "completion":
        return True
    if normalized == "success":
        return bool(ok)
    if normalized == "failure":
        return not bool(ok)
    return False


async def _emit_workflow_notify(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    req: CortexClientRequest,
    workflow_id: str,
    workflow_name: str,
    correlation_id: str,
    ok: bool,
    final_text: str,
    notify_on: str,
    recipient_group: str,
    execution_source: str,
) -> None:
    if not _should_notify(notify_on=notify_on, ok=ok):
        return
    status = "completed" if ok else "failed"
    event_kind = "orion.workflow.completed" if ok else "orion.workflow.failed"
    title = f"Workflow {workflow_name} {status}"
    preview = f"{workflow_name} {status}. {final_text}".strip()[:280]
    notification = NotificationRequest(
        source_service=source.name or "orion-cortex-orch",
        event_kind=event_kind,
        severity="info" if ok else "warning",
        title=title,
        body_text=preview,
        body_md=final_text,
        recipient_group=recipient_group or "juniper_primary",
        session_id=req.context.session_id or "workflow",
        correlation_id=correlation_id,
        dedupe_key=f"workflow:{workflow_id}:{status}:{correlation_id}",
        dedupe_window_seconds=86400,
        tags=["workflow", workflow_id, status],
        context={
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "status": status,
            "execution_source": execution_source,
            "notify_on": notify_on,
            "preview_text": preview,
        },
    )
    env = BaseEnvelope(
        kind="notify.notification.request.v1",
        source=source,
        correlation_id=correlation_id,
        payload=notification.model_dump(mode="json"),
    )
    await bus.publish(NOTIFY_PERSISTENCE_REQUEST_CHANNEL, env)


async def _schedule_workflow_dispatch(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    req: CortexClientRequest,
    correlation_id: str,
    workflow_request: Dict[str, Any],
    policy: WorkflowExecutionPolicyV1,
) -> CortexClientResult:
    request_id = str(uuid4())
    dispatch = WorkflowDispatchRequestV1(
        request_id=request_id,
        workflow_id=policy.workflow_id,
        workflow_request=workflow_request,
        execution_policy=policy,
        correlation_id=correlation_id,
        source_service=source.name or "orion-cortex-orch",
        source_kind="chat_schedule_request",
    )
    env = BaseEnvelope(
        kind="orion.actions.trigger.workflow.v1",
        source=source,
        correlation_id=correlation_id,
        payload=dispatch.model_dump(mode="json"),
    )
    await bus.publish(ACTIONS_WORKFLOW_TRIGGER_CHANNEL, env)
    schedule = policy.schedule
    schedule_label = schedule.label if schedule else "scheduled"
    scheduled_for = schedule.run_at_utc.isoformat() if schedule and schedule.run_at_utc else schedule_label
    notify_text = f"notify_on={policy.notify_on}" if policy.notify_on != "none" else "notify_on=none"
    summary = f"Workflow request accepted: {policy.workflow_id} ({schedule_label}). {notify_text}."
    metadata = _workflow_metadata_base(request=workflow_request, status="scheduled")
    metadata["workflow"] = {
        "workflow_id": policy.workflow_id,
        "display_name": get_workflow_definition(policy.workflow_id).display_name if get_workflow_definition(policy.workflow_id) else policy.workflow_id,
        "status": "scheduled",
        "subverb": None,
        "persisted": [],
        "scheduled": [f"{request_id}:{scheduled_for}"],
        "main_result": summary,
        "execution_policy": policy.model_dump(mode="json"),
        "dispatch_request_id": request_id,
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
    }
    return CortexClientResult(
        ok=True,
        mode="brain",
        verb=policy.workflow_id,
        status="success",
        final_text=_workflow_summary_text(
            title=metadata["workflow"]["display_name"],
            status="scheduled",
            main_result=summary,
            scheduled=metadata["workflow"]["scheduled"],
        ),
        memory_used=False,
        recall_debug={},
        steps=[],
        error=None,
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def execute_workflow_schedule_management(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    req: CortexClientRequest,
    correlation_id: str,
) -> CortexClientResult:
    raw = _workflow_schedule_management_request(req)
    request = WorkflowScheduleManageRequestV1.model_validate(raw)
    reply_channel = f"orion:actions:manage:result:{correlation_id}"
    env = BaseEnvelope(
        kind="orion.actions.manage.workflow.v1",
        source=source,
        correlation_id=correlation_id,
        payload=request.model_dump(mode="json"),
        reply_to=reply_channel,
    )
    msg = await bus.rpc_request(
        ACTIONS_WORKFLOW_MANAGE_CHANNEL,
        env,
        reply_channel=reply_channel,
        timeout_sec=20.0,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok or decoded.envelope is None:
        raise WorkflowExecutionError(f"workflow_manage_decode_failed:{decoded.error}")
    payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
    response = WorkflowScheduleManageResponseV1.model_validate(payload)

    lines: list[str] = [f"Workflow schedule {response.operation}: {'ok' if response.ok else 'failed'}", response.message]
    if response.error_code:
        lines.append(f"error_code={response.error_code}")
    if response.schedule is not None:
        lines.append(
            f"schedule_id={response.schedule.schedule_id} workflow={response.schedule.workflow_id} state={response.schedule.state} next_run={response.schedule.next_run_at}"
        )
    if response.schedules:
        lines.append("Schedules:")
        for item in response.schedules[:10]:
            cadence = item.execution_policy.schedule.label if item.execution_policy.schedule and item.execution_policy.schedule.label else (
                item.execution_policy.schedule.cadence if item.execution_policy.schedule else "one-shot"
            )
            short_id = item.schedule_id[-8:] if len(item.schedule_id) > 8 else item.schedule_id
            lines.append(
                f"- {item.workflow_display_name or item.workflow_id} | {item.state} | next={item.next_run_at} | cadence={cadence} | notify={item.notify_on} | id={short_id}"
            )
    if response.ambiguous:
        lines.append("Multiple matching schedules found. Please specify schedule_id.")
    final_text = "\n".join(lines)
    metadata = _workflow_metadata_base(request=raw, status="completed" if response.ok else "failed")
    metadata["workflow_schedule_management"] = response.model_dump(mode="json")
    return CortexClientResult(
        ok=response.ok,
        mode="brain",
        verb="workflow.schedule.management",
        status="success" if response.ok else "fail",
        final_text=final_text,
        memory_used=False,
        recall_debug={},
        steps=[],
        error=None if response.ok else {"message": response.message, "code": response.error_code, "details": response.error_details},
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def _run_workflow_subverb(
    call_verb_runtime,
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    base_request: CortexClientRequest,
    workflow_id: str,
    verb: str,
    packs: list[str] | None = None,
    options_patch: dict[str, Any] | None = None,
) -> Tuple[VerbResultV1, Dict[str, Any]]:
    subrequest = base_request.model_copy(deep=True)
    subrequest.mode = "brain"
    subrequest.route_intent = "none"
    subrequest.verb = verb
    subrequest.packs = list(packs if packs is not None else (base_request.packs or []))
    subrequest.options = dict(subrequest.options or {})
    subrequest.options.update(options_patch or {})
    subrequest.context.metadata = dict(subrequest.context.metadata or {})
    subrequest.context.metadata.setdefault("workflow_request", _workflow_request(base_request))
    subrequest.context.metadata["workflow_subverb"] = verb
    subrequest.context.metadata["workflow_id"] = workflow_id
    subrequest.context.metadata.update(
        _workflow_execution_envelope(
            req=base_request,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            workflow_subverb=verb,
        )
    )
    _log_primary_metadata_status(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        verb=verb,
        metadata=subrequest.context.metadata,
    )
    result = await call_verb_runtime(
        bus,
        source=source,
        client_request=subrequest,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=_ensure_trace(trace, correlation_id=correlation_id, workflow_id=workflow_id),
        timeout_sec=float((subrequest.options or {}).get("timeout_sec", 900.0)),
    )
    return result, _extract_result_payload(result)


async def _execute_dream_cycle(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    call_verb_runtime,
) -> CortexClientResult:
    workflow_id = "dream_cycle"
    verb_result, payload = await _run_workflow_subverb(
        call_verb_runtime,
        bus=bus,
        source=source,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=trace,
        base_request=req,
        workflow_id=workflow_id,
        verb="dream_cycle",
        packs=list(req.packs or ["emergent_pack"]),
        options_patch={"workflow_execution": True},
    )
    final_text = payload.get("final_text") or "Dream cycle completed through the existing dream verb."
    payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dream_persisted = bool(payload_metadata.get("dream_result_published") or payload_metadata.get("dream_persisted"))
    persisted = ["dream.result.v1"] if dream_persisted else []
    if not dream_persisted:
        final_text = f"{final_text} Dream persistence was not confirmed by the execution payload."
    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Dream Cycle",
        "status": "completed" if verb_result.ok else "failed",
        "executed": True,
        "subverb": "dream_cycle",
        "persisted": persisted,
        "scheduled": [],
        "main_result": final_text,
        "dream_persistence_confirmed": dream_persisted,
    }
    return CortexClientResult(
        ok=verb_result.ok,
        mode="brain",
        verb=workflow_id,
        status="success" if verb_result.ok else "fail",
        final_text=_workflow_summary_text(
            title="Dream Cycle",
            status="completed" if verb_result.ok else "failed",
            main_result=final_text,
            persisted=persisted,
        ),
        memory_used=bool(payload.get("memory_used")),
        recall_debug=payload.get("recall_debug") or {},
        steps=payload.get("steps") or [],
        error=None if verb_result.ok else {"message": verb_result.error or "workflow_failed"},
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def _execute_journal_pass(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    call_verb_runtime,
) -> CortexClientResult:
    workflow_id = "journal_pass"
    trigger = build_manual_trigger(
        summary=req.context.raw_user_text or req.context.user_message or "Journal pass requested",
        prompt_seed=req.context.raw_user_text or req.context.user_message,
        source_ref=correlation_id,
    )
    compose_req = build_compose_request(
        trigger,
        session_id=req.context.session_id or "workflow-journal-session",
        trace_id=req.context.trace_id or correlation_id,
        user_id=req.context.user_id,
        options={"workflow_execution": True, "workflow_id": workflow_id},
    )
    compose_req.context.metadata = {**dict(req.context.metadata or {}), **dict(compose_req.context.metadata or {})}
    compose_req.context.metadata.update(
        _workflow_execution_envelope(
            req=req,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            workflow_subverb="journal.compose",
        )
    )
    compose_req.context.metadata["workflow_subverb"] = "journal.compose"
    compose_req.context.metadata["workflow_id"] = workflow_id
    _log_primary_metadata_status(
        correlation_id=correlation_id,
        workflow_id=workflow_id,
        verb="journal.compose",
        metadata=compose_req.context.metadata,
    )
    verb_result = await call_verb_runtime(
        bus,
        source=source,
        client_request=compose_req,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=_ensure_trace(trace, correlation_id=correlation_id, workflow_id=workflow_id),
        timeout_sec=float((compose_req.options or {}).get("timeout_sec", 900.0)),
    )
    compose_payload = _extract_result_payload(verb_result)
    draft = draft_from_cortex_result(compose_payload)
    title = _coerce_journal_title(raw_title=draft.title, fallback_summary=trigger.summary, correlation_id=correlation_id)
    body = str(draft.body or "").strip()
    if not body:
        raise WorkflowExecutionError("journal_pass_empty_body")
    draft = draft.model_copy(update={"title": title, "body": body}, deep=True)
    write = build_write_payload(
        draft,
        trigger=trigger,
        correlation_id=correlation_id,
        author=req.context.user_id or "orion",
    )
    await bus.publish(
        JOURNAL_WRITE_CHANNEL,
        BaseEnvelope(
            kind=JOURNAL_WRITE_KIND,
            source=source,
            correlation_id=correlation_id,
            causality_chain=list(causality_chain or []),
            trace=_ensure_trace(trace, correlation_id=correlation_id, workflow_id=workflow_id),
            payload=write.model_dump(mode="json"),
        ),
    )
    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Journal Pass",
        "status": "completed",
        "executed": True,
        "subverb": "journal.compose",
        "persisted": [f"journal.entry.write.v1:{write.entry_id}"],
        "scheduled": [],
        "main_result": draft.title,
        "journal_entry": write.model_dump(mode="json"),
        "draft": draft.model_dump(mode="json"),
    }
    return CortexClientResult(
        ok=True,
        mode="brain",
        verb=workflow_id,
        status="success",
        final_text=_workflow_summary_text(
            title="Journal Pass",
            status="completed",
            main_result=f"Drafted '{draft.title}' and sent it through the append-only journal write path.",
            persisted=metadata["workflow"]["persisted"],
        ),
        memory_used=bool(compose_payload.get("memory_used")),
        recall_debug=compose_payload.get("recall_debug") or {},
        steps=compose_payload.get("steps") or [],
        error=None,
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def _execute_self_review(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    call_verb_runtime,
) -> CortexClientResult:
    workflow_id = "self_review"
    verb_result, payload = await _run_workflow_subverb(
        call_verb_runtime,
        bus=bus,
        source=source,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=trace,
        base_request=req,
        workflow_id=workflow_id,
        verb="self_concept_reflect",
        options_patch={"workflow_execution": True},
    )
    skill_result = ((payload.get("metadata") or {}).get("skill_result") or _parse_json_text(payload.get("final_text"))) or {}
    findings = list(skill_result.get("findings") or []) if isinstance(skill_result, dict) else []
    summary = str((skill_result.get("summary") if isinstance(skill_result, dict) else None) or payload.get("final_text") or "Self review completed.").strip()
    persisted: List[str] = []
    if isinstance(skill_result, dict):
        graph_write = skill_result.get("graph_write") or {}
        journal_write = skill_result.get("journal_write") or {}
        if graph_write.get("graph"):
            persisted.append(f"graph:{graph_write['graph']}")
        if journal_write.get("channel"):
            persisted.append(f"journal:{journal_write['channel']}")
    if not summary:
        summary = "Self review completed."
    if not findings:
        summary = "No notable self-review findings were identified from the available window/context."
    else:
        concise_findings = []
        for item in findings[:3]:
            if isinstance(item, dict):
                concise_findings.append(str(item.get("kind") or item.get("summary") or "finding").strip())
            else:
                concise_findings.append(str(item).strip())
        summary = f"Self review completed with {len(findings)} findings: {', '.join([f for f in concise_findings if f])}."

    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Self Review",
        "status": "completed" if verb_result.ok else "failed",
        "executed": True,
        "subverb": "self_concept_reflect",
        "persisted": persisted,
        "scheduled": [],
        "main_result": summary,
        "finding_count": len(findings),
        "skill_result": skill_result,
    }
    return CortexClientResult(
        ok=verb_result.ok,
        mode="brain",
        verb=workflow_id,
        status="success" if verb_result.ok else "fail",
        final_text=_workflow_summary_text(
            title="Self Review",
            status="completed" if verb_result.ok else "failed",
            main_result=summary,
            persisted=persisted,
        ),
        memory_used=bool(payload.get("memory_used")),
        recall_debug=payload.get("recall_debug") or {},
        steps=payload.get("steps") or [],
        error=None if verb_result.ok else {"message": verb_result.error or "workflow_failed"},
        correlation_id=correlation_id,
        metadata=metadata,
    )


def _state_summary(snapshot: SparkStateSnapshotV1 | dict[str, Any] | None) -> str | None:
    if snapshot is None:
        return None
    raw = snapshot if isinstance(snapshot, dict) else snapshot.model_dump(mode="json")
    coherence = raw.get("coherence")
    novelty = raw.get("novelty")
    focus = raw.get("focus")
    parts = []
    if coherence is not None:
        parts.append(f"coherence={coherence}")
    if novelty is not None:
        parts.append(f"novelty={novelty}")
    if focus is not None:
        parts.append(f"focus={focus}")
    return ", ".join(parts) if parts else None


def _resolve_concept_profile_backend_for_consumer(settings: Any, *, consumer: str) -> ConceptProfileBackendKind:
    global_backend = str(getattr(settings, "concept_profile_repository_backend", "local") or "local").strip().lower()
    if global_backend not in {"local", "graph", "shadow"}:
        global_backend = "local"
    if consumer == "concept_induction_pass":
        override = str(getattr(settings, "concept_profile_backend_concept_induction_pass", "") or "").strip().lower()
        if override in {"local", "graph", "shadow"}:
            return override  # type: ignore[return-value]
    return global_backend  # type: ignore[return-value]


def _resolve_graph_cutover_fallback_policy(settings: Any) -> CutoverFallbackPolicyKind:
    policy = str(getattr(settings, "concept_profile_graph_cutover_fallback_policy", "fail_open_local") or "fail_open_local")
    policy = policy.strip().lower()
    if policy not in {"fail_open_local", "fail_closed"}:
        return "fail_open_local"
    return policy  # type: ignore[return-value]


def _log_concept_profile_resolution(
    *,
    consumer: str,
    requested_backend: str,
    resolved_backend: str,
    fallback_policy: str,
    fallback_used: bool,
    unavailable_reason: str | None,
    subjects_requested: list[str],
    profiles_returned: int,
    correlation_id: str,
    session_id: str | None,
) -> None:
    logger.info(
        "concept_profile_repository_resolution %s",
        json.dumps(
            {
                "consumer": consumer,
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
                "fallback_policy": fallback_policy,
                "fallback_used": fallback_used,
                "unavailable_reason": unavailable_reason,
                "subjects_requested": len(subjects_requested),
                "profiles_returned": profiles_returned,
                "correlation_id": correlation_id,
                "session_id": session_id or None,
            },
            sort_keys=True,
        ),
    )


def _concept_item_payload(concept: Any) -> Dict[str, Any]:
    evidence_count = len(getattr(concept, "evidence", []) or [])
    return {
        "concept_id": getattr(concept, "concept_id", None),
        "label": getattr(concept, "label", None),
        "type": getattr(concept, "type", None),
        "aliases": list(getattr(concept, "aliases", []) or []),
        "salience": getattr(concept, "salience", None),
        "confidence": getattr(concept, "confidence", None),
        "evidence_count": evidence_count,
    }


def _cluster_item_payload(cluster: Any, *, concepts: list[dict[str, Any]]) -> Dict[str, Any]:
    concept_ids = list(getattr(cluster, "concept_ids", []) or [])
    labels_by_id = {str(item.get("concept_id") or ""): item.get("label") for item in concepts}
    representative_labels = [labels_by_id.get(cid) for cid in concept_ids[:5] if labels_by_id.get(cid)]
    return {
        "cluster_id": getattr(cluster, "cluster_id", None),
        "label": getattr(cluster, "label", None),
        "summary": getattr(cluster, "summary", None),
        "concept_ids": concept_ids,
        "representative_labels": representative_labels,
        "cohesion_score": getattr(cluster, "cohesion_score", None),
    }


def _profile_detail_payload(lookup: Any) -> Dict[str, Any] | None:
    profile = lookup.profile
    if profile is None:
        return None
    created_at = getattr(profile, "created_at", None)
    window_start = getattr(profile, "window_start", None)
    window_end = getattr(profile, "window_end", None)
    created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else (
        window_end.isoformat() if hasattr(window_end, "isoformat") else None
    )
    window_start_iso = window_start.isoformat() if hasattr(window_start, "isoformat") else None
    window_end_iso = window_end.isoformat() if hasattr(window_end, "isoformat") else None
    concepts = [_concept_item_payload(item) for item in list(profile.concepts or [])[:25]]
    clusters = [_cluster_item_payload(cluster, concepts=concepts) for cluster in list(profile.clusters or [])[:15]]
    return {
        "subject": lookup.subject,
        "profile_id": profile.profile_id,
        "revision": profile.revision,
        "created_at": created_at_iso,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
        "concept_count": len(profile.concepts),
        "cluster_count": len(profile.clusters),
        "concepts": concepts,
        "clusters": clusters,
        "state_estimate": _state_summary(profile.state_estimate),
        "provenance": {
            "materialization_ref": str((getattr(profile, "metadata", {}) or {}).get("materialization_ref") or ""),
            "source_kind": str((getattr(profile, "metadata", {}) or {}).get("source_kind") or ""),
        },
    }


def _concept_induction_trace_payload(*, lookups: list[Any], observer: dict[str, str], repository_status: Any, requested_backend: str, resolved_backend: str, fallback_used: bool, fallback_policy: str, unavailable_reason: str | None) -> Dict[str, Any]:
    profile_rows = []
    for lookup in lookups[:8]:
        row = {
            "subject": lookup.subject,
            "availability": lookup.availability,
            "unavailable_reason": lookup.unavailable_reason,
        }
        if lookup.profile is not None:
            row["profile_id"] = lookup.profile.profile_id
            row["revision"] = lookup.profile.revision
        profile_rows.append(row)
    return {
        "repository_resolution": {
            "observer": dict(observer),
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "fallback_used": fallback_used,
            "fallback_policy": fallback_policy,
            "unavailable_reason": unavailable_reason,
            "source_path": repository_status.source_path,
        },
        "artifacts": {
            "lookup_rows": profile_rows,
        },
    }


def _build_concept_induction_synthesis_body(*, details: Dict[str, Any]) -> str:
    by_subject = {str(item.get("subject") or ""): item for item in (details.get("profiles") or []) if isinstance(item, dict)}

    def _subject_line(subject: str) -> str:
        item = by_subject.get(subject) or {}
        top_concepts = [str(c.get("label") or c.get("concept_id") or "").strip() for c in (item.get("concepts") or [])[:3] if isinstance(c, dict)]
        top_clusters = [str(c.get("label") or c.get("cluster_id") or "").strip() for c in (item.get("clusters") or [])[:2] if isinstance(c, dict)]
        state = item.get("state_estimate") if isinstance(item.get("state_estimate"), dict) else {}
        return (
            f"- {subject}: rev {item.get('revision', '--')} "
            f"({item.get('concept_count', 0)} concepts / {item.get('cluster_count', 0)} clusters). "
            f"Concepts: {', '.join([x for x in top_concepts if x]) or 'none'}. "
            f"Clusters: {', '.join([x for x in top_clusters if x]) or 'none'}. "
            f"State: {_state_summary(state) if state else 'n/a'}."
        )

    review_lines = [_subject_line(name) for name in ("orion", "juniper", "relationship")]
    trace = details.get("trace") if isinstance(details.get("trace"), dict) else {}
    resolution = trace.get("repository_resolution") if isinstance(trace.get("repository_resolution"), dict) else {}
    backend_line = (
        f"Backend resolution: requested={resolution.get('requested_backend') or '--'}, "
        f"resolved={resolution.get('resolved_backend') or '--'}, "
        f"fallback_used={bool(resolution.get('fallback_used'))}."
    )
    return "\n".join(
        [
            "Concept induction review synthesis",
            "",
            "Reviewed subjects:",
            *review_lines,
            "",
            "Repository trace:",
            f"- {backend_line}",
            "",
            "Reflection:",
            "The current concept-profile revisions show what appears most salient across Orion, Juniper, and relationship context with bounded artifact-backed traceability.",
        ]
    ).strip()


def _concept_induction_grounding_payload(*, details: Dict[str, Any], workflow_id: str, resolved_backend: str) -> Dict[str, Any]:
    profiles = [item for item in (details.get("profiles") or []) if isinstance(item, dict)]
    trace = details.get("trace") if isinstance(details.get("trace"), dict) else {}
    reviewed_subjects = [str(item.get("subject") or "").strip() for item in profiles if str(item.get("subject") or "").strip()]
    return {
        "workflow_id": workflow_id,
        "reviewed_subjects": reviewed_subjects,
        "generated_at": details.get("generated_at"),
        "profiles": profiles,
        "trace": trace,
        "provenance": {
            "repository_backend": resolved_backend,
            "synthesis_mode": "brain_grounded",
            "synthesis_prompt_version": "concept_induction_journal_grounded.v1",
        },
    }


def _extract_concept_induction_artifact_terms(*, grounding_payload: Dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for subject in grounding_payload.get("reviewed_subjects") or []:
        normalized = str(subject or "").strip().lower()
        if normalized:
            allowed.add(normalized)
    for profile in grounding_payload.get("profiles") or []:
        if not isinstance(profile, dict):
            continue
        for key in ("subject", "profile_id"):
            value = str(profile.get(key) or "").strip().lower()
            if value:
                allowed.add(value)
        for concept in profile.get("concepts") or []:
            if not isinstance(concept, dict):
                continue
            for key in ("concept_id", "label"):
                value = str(concept.get(key) or "").strip().lower()
                if value:
                    allowed.add(value)
        for cluster in profile.get("clusters") or []:
            if not isinstance(cluster, dict):
                continue
            for key in ("cluster_id", "label"):
                value = str(cluster.get(key) or "").strip().lower()
                if value:
                    allowed.add(value)
    return allowed


def _grounding_violations(*, text: str, grounding_payload: Dict[str, Any]) -> list[str]:
    allowed = _extract_concept_induction_artifact_terms(grounding_payload=grounding_payload)
    if not allowed:
        return []
    violations: list[str] = []
    for line in (text or "").splitlines():
        lower = line.strip().lower()
        if "concept " in lower and ":" in lower:
            token = lower.split("concept ", 1)[1].split(":", 1)[0].strip(" -_*`")
            if token and token not in allowed:
                violations.append(f"unsupported_concept:{token}")
        if "cluster " in lower and ":" in lower:
            token = lower.split("cluster ", 1)[1].split(":", 1)[0].strip(" -_*`")
            if token and token not in allowed:
                violations.append(f"unsupported_cluster:{token}")
    return violations[:8]


async def _run_concept_induction_grounded_journal_synthesis(
    *,
    call_verb_runtime,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    workflow_id: str,
    grounding_payload: Dict[str, Any],
) -> JournalEntryDraftV1:
    synth_req = req.model_copy(deep=True)
    synth_req.mode = "brain"
    synth_req.route_intent = "none"
    synth_req.verb = "concept_induction_journal_synthesize"
    synth_req.packs = []
    synth_req.context.messages = []
    synth_req.context.raw_user_text = "Synthesize reviewed concept-induction payload into grounded journal entry."
    synth_req.context.user_message = synth_req.context.raw_user_text
    synth_req.context.metadata = dict(synth_req.context.metadata or {})
    synth_req.context.metadata["workflow_subverb"] = "concept_induction_journal_synthesize"
    synth_req.context.metadata["workflow_id"] = workflow_id
    synth_req.context.metadata["concept_induction_journal_grounding"] = grounding_payload
    synth_req.context.metadata.update(
        _workflow_execution_envelope(
            req=req,
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            workflow_subverb="concept_induction_journal_synthesize",
        )
    )
    synth_req.context.metadata["workflow_request"] = {
        "workflow_id": workflow_id,
        "action": "synthesize_to_journal",
    }
    synth_req.recall.enabled = False
    synth_req.recall.required = False
    synth_req.recall.max_items = 0
    synth_req.options = dict(synth_req.options or {})
    synth_req.options.update({"workflow_execution": True})
    verb_result = await call_verb_runtime(
        bus,
        source=source,
        client_request=synth_req,
        correlation_id=correlation_id,
        causality_chain=causality_chain,
        trace=_ensure_trace(trace, correlation_id=correlation_id, workflow_id=workflow_id),
        timeout_sec=float((synth_req.options or {}).get("timeout_sec", 120.0)),
    )
    payload = _extract_result_payload(verb_result)
    draft = draft_from_cortex_result(payload)
    title = _coerce_journal_title(
        raw_title=draft.title,
        fallback_summary="Concept Induction Review",
        correlation_id=correlation_id,
    )
    body = str(draft.body or "").strip()
    if not body:
        raise WorkflowExecutionError("concept_induction_synthesis_empty_body")
    violations = _grounding_violations(text=body, grounding_payload=grounding_payload)
    if violations:
        raise WorkflowExecutionError(f"concept_induction_synthesis_grounding_violation:{','.join(violations)}")
    return draft.model_copy(update={"title": title, "body": body}, deep=True)


def _coerce_reviewed_concept_details(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    profiles = value.get("profiles")
    if not isinstance(profiles, list):
        return None
    bounded_profiles = [item for item in profiles if isinstance(item, dict)][:6]
    trace = value.get("trace") if isinstance(value.get("trace"), dict) else {}
    return {
        "generated_at": value.get("generated_at"),
        "profiles": bounded_profiles,
        "trace": trace,
    }


async def _execute_concept_induction_pass(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    req: CortexClientRequest,
    call_verb_runtime,
) -> CortexClientResult:
    workflow_id = "concept_induction_pass"
    settings = build_orch_concept_profile_settings(get_settings())
    consumer = "concept_induction_pass"
    requested_backend = _resolve_concept_profile_backend_for_consumer(settings, consumer=consumer)
    fallback_policy = _resolve_graph_cutover_fallback_policy(settings)
    repository = build_concept_profile_repository(settings, backend_override=requested_backend)
    repository_status = repository.status()
    resolved_backend = repository_status.backend
    fallback_used = False
    fallback_reason: str | None = None
    using_placeholder_store = repository_status.placeholder_default_in_use
    subjects = list(settings.subjects or ["orion", "juniper", "relationship"])
    session_id = str(req.context.session_id or "")
    observer = {
        "consumer": consumer,
        "correlation_id": correlation_id,
        "session_id": session_id,
    }
    lookups = repository.list_latest(
        subjects,
        observer=observer,
    )
    graph_unavailable = requested_backend == "graph" and any(lookup.availability == "unavailable" for lookup in lookups)
    if graph_unavailable:
        fallback_reason = next((lookup.unavailable_reason for lookup in lookups if lookup.availability == "unavailable"), None)
        if fallback_policy == "fail_open_local":
            fallback_repository = build_concept_profile_repository(settings, backend_override="local")
            fallback_status = fallback_repository.status()
            lookups = fallback_repository.list_latest(subjects, observer=observer)
            repository_status = fallback_status
            resolved_backend = fallback_status.backend
            fallback_used = True
            using_placeholder_store = fallback_status.placeholder_default_in_use
        else:
            status = "failed"
            main_result = (
                "Concept profile graph retrieval is unavailable and cutover policy is fail_closed "
                f"(reason={fallback_reason or 'unknown'})."
            )
            metadata = _workflow_metadata_base(request=_workflow_request(req), status=status)
            metadata["workflow"] = {
                "workflow_id": workflow_id,
                "display_name": "Concept Induction Pass",
                "status": status,
                "executed": True,
                "subverb": None,
                "persisted": [],
                "scheduled": [],
                "main_result": main_result,
                "profile_store_path": settings.store_path,
                "profile_store_placeholder_path": using_placeholder_store,
                "profile_store_env_var": "CONCEPT_STORE_PATH",
                "profiles_reviewed": [],
                "concept_profile_resolution": {
                    "consumer": consumer,
                    "requested_backend": requested_backend,
                    "resolved_backend": resolved_backend,
                    "fallback_policy": fallback_policy,
                    "fallback_used": False,
                    "unavailable_reason": fallback_reason,
                },
            }
            _log_concept_profile_resolution(
                consumer=consumer,
                requested_backend=requested_backend,
                resolved_backend=resolved_backend,
                fallback_policy=fallback_policy,
                fallback_used=False,
                unavailable_reason=fallback_reason,
                subjects_requested=subjects,
                profiles_returned=0,
                correlation_id=correlation_id,
                session_id=session_id,
            )
            return CortexClientResult(
                ok=False,
                mode="brain",
                verb=workflow_id,
                status="fail",
                final_text=_workflow_summary_text(
                    title="Concept Induction Pass",
                    status=status,
                    main_result=main_result,
                    persisted=[],
                ),
                memory_used=False,
                recall_debug={},
                steps=[],
                error={"message": main_result, "code": "concept_profiles_graph_unavailable"},
                correlation_id=correlation_id,
                metadata=metadata,
            )

    unavailable_reason = next((lookup.unavailable_reason for lookup in lookups if lookup.availability == "unavailable"), None)
    logger.info(
        "concept_profile_repository_status %s",
        json.dumps(
            {
                "backend": repository_status.backend,
                "source_path": repository_status.source_path,
                "placeholder_default_in_use": repository_status.placeholder_default_in_use,
                "subjects_requested": len(subjects),
                "profiles_returned": sum(1 for lookup in lookups if lookup.profile is not None),
            },
            sort_keys=True,
        ),
    )
    reviews: List[Dict[str, Any]] = []
    profile_details: List[Dict[str, Any]] = []
    for lookup in lookups:
        subject = lookup.subject
        profile = lookup.profile
        if profile is None:
            continue
        detail = _profile_detail_payload(lookup)
        if detail is not None:
            profile_details.append(detail)
        reviews.append(
            {
                "subject": subject,
                "profile_id": profile.profile_id,
                "revision": profile.revision,
                "concept_count": len(profile.concepts),
                "cluster_count": len(profile.clusters),
                "top_concepts": [concept.label for concept in profile.concepts[:5]],
                "cluster_labels": [cluster.label for cluster in profile.clusters[:3]],
                "state_estimate": _state_summary(profile.state_estimate),
                "window_start": profile.window_start.isoformat(),
                "window_end": profile.window_end.isoformat(),
            }
        )
    ok = bool(reviews)
    status = "completed" if ok else "failed"
    if reviews:
        main_result = "; ".join(
            f"{item['subject']} rev {item['revision']} with {item['concept_count']} concepts / {item['cluster_count']} clusters"
            for item in reviews
        )
    elif using_placeholder_store:
        main_result = (
            "Concept induction profiles are not configured in this environment. "
            f"Set CONCEPT_STORE_PATH to a real profile store path (current placeholder: {DEFAULT_CONCEPT_STORE_PATH})."
        )
    else:
        main_result = f"Concept induction profiles are not available in the configured store ({settings.store_path})."
    metadata = _workflow_metadata_base(request=_workflow_request(req), status=status)
    computed_details = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profile_details,
        "trace": _concept_induction_trace_payload(
            lookups=lookups,
            observer=observer,
            repository_status=repository_status,
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
            fallback_used=fallback_used,
            fallback_policy=fallback_policy,
            unavailable_reason=fallback_reason or unavailable_reason,
        ),
    }
    workflow_request = _workflow_request(req)
    requested_details = _coerce_reviewed_concept_details(workflow_request.get("concept_induction_details"))
    concept_induction_details = requested_details or computed_details
    synthesize_requested = str(workflow_request.get("action") or "").strip().lower() == "synthesize_to_journal"
    persisted: list[str] = []
    synthesis_status: Dict[str, Any] | None = None
    if synthesize_requested and ok:
        logger.info(
            "concept_induction_journal_handoff %s",
            json.dumps(
                {
                    "stage": "synth_result_requested",
                    "workflow_id": workflow_id,
                    "correlation_id": correlation_id,
                },
                sort_keys=True,
            ),
        )
        grounding_payload = _concept_induction_grounding_payload(
            details=concept_induction_details,
            workflow_id=workflow_id,
            resolved_backend=resolved_backend,
        )
        try:
            synthesis_draft = await _run_concept_induction_grounded_journal_synthesis(
                call_verb_runtime=call_verb_runtime,
                bus=bus,
                source=source,
                correlation_id=correlation_id,
                causality_chain=causality_chain,
                trace=trace,
                req=req,
                workflow_id=workflow_id,
                grounding_payload=grounding_payload,
            )
            logger.info(
                "concept_induction_journal_handoff %s",
                json.dumps(
                    {
                        "stage": "synth_result_received",
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                    },
                    sort_keys=True,
                ),
            )
            logger.info(
                "concept_induction_journal_handoff %s",
                json.dumps(
                    {
                        "stage": "synth_result_parsed",
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                    },
                    sort_keys=True,
                ),
            )
            logger.info(
                "concept_induction_journal_handoff %s",
                json.dumps(
                    {
                        "stage": "grounding_check_passed",
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                    },
                    sort_keys=True,
                ),
            )
        except Exception as exc:
            message = str(exc)
            stage = "grounding_check_failed" if "grounding_violation" in message else "synth_result_parse_failed"
            logger.warning(
                "concept_induction_journal_handoff %s",
                json.dumps(
                    {
                        "stage": stage,
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                        "error": message,
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            raise
        reviewed_refs = [
            f"{item.get('subject')}:{item.get('profile_id')}@{item.get('revision')}"
            for item in (profile_details or [])
        ]
        trigger = build_manual_trigger(
            summary="Concept induction review synthesis",
            source_ref=f"concept_induction_pass:{','.join(reviewed_refs[:6])}",
        )
        write = build_write_payload(
            synthesis_draft,
            trigger=trigger,
            correlation_id=correlation_id,
            author=req.context.user_id or "orion",
        )
        envelope = BaseEnvelope(
            kind=JOURNAL_WRITE_KIND,
            source=source,
            payload=write.model_dump(mode="json"),
            correlation_id=correlation_id,
            causality_chain=list(causality_chain or []),
            trace=_ensure_trace(trace, correlation_id=correlation_id, workflow_id=workflow_id),
        )
        logger.info(
            "concept_induction_journal_handoff %s",
            json.dumps(
                {
                    "stage": "journal_write_requested",
                    "workflow_id": workflow_id,
                    "correlation_id": correlation_id,
                    "kind": JOURNAL_WRITE_KIND,
                    "channel": JOURNAL_WRITE_CHANNEL,
                    "entry_id": write.entry_id,
                },
                sort_keys=True,
            ),
        )
        try:
            await bus.publish(JOURNAL_WRITE_CHANNEL, envelope)
        except Exception as exc:
            logger.exception(
                "concept_induction_journal_handoff %s",
                json.dumps(
                    {
                        "stage": "journal_write_publish_failed",
                        "workflow_id": workflow_id,
                        "correlation_id": correlation_id,
                        "kind": JOURNAL_WRITE_KIND,
                        "channel": JOURNAL_WRITE_CHANNEL,
                        "entry_id": write.entry_id,
                        "error": str(exc),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            raise WorkflowExecutionError("concept_induction_journal_write_publish_failed") from exc
        logger.info(
            "concept_induction_journal_handoff %s",
            json.dumps(
                {
                    "stage": "journal_write_published",
                    "workflow_id": workflow_id,
                    "correlation_id": correlation_id,
                    "kind": JOURNAL_WRITE_KIND,
                    "channel": JOURNAL_WRITE_CHANNEL,
                    "entry_id": write.entry_id,
                },
                sort_keys=True,
            ),
        )
        persisted.append(f"journal.entry.write.v1:{write.entry_id}")
        synthesis_status = {
            "ok": True,
            "synthesis_completed": True,
            "journal_write_emitted": True,
            "journal_write_confirmed": False,
            "journal_entry": write.model_dump(mode="json"),
            "provenance": {
                "source_workflow_id": workflow_id,
                "reviewed_subjects": [item.get("subject") for item in profile_details],
                "reviewed_profiles": [
                    {
                        "subject": item.get("subject"),
                        "profile_id": item.get("profile_id"),
                        "revision": item.get("revision"),
                    }
                    for item in profile_details
                ],
                "repository_backend": resolved_backend,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "synthesis_mode": "brain_grounded",
                "synthesis_prompt_version": "concept_induction_journal_grounded.v1",
            },
        }
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Concept Induction Pass",
        "status": status,
        "executed": True,
        "subverb": None,
        "persisted": persisted,
        "scheduled": [],
        "main_result": main_result,
        "profile_store_path": settings.store_path,
        "profile_store_placeholder_path": using_placeholder_store,
        "profile_store_env_var": "CONCEPT_STORE_PATH",
        "profiles_reviewed": reviews,
        "concept_induction_details": concept_induction_details,
        "concept_profile_resolution": {
            "consumer": consumer,
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "fallback_policy": fallback_policy,
            "fallback_used": fallback_used,
            "unavailable_reason": fallback_reason or unavailable_reason,
        },
        "synthesis_to_journal": synthesis_status,
    }
    _log_concept_profile_resolution(
        consumer=consumer,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        fallback_policy=fallback_policy,
        fallback_used=fallback_used,
        unavailable_reason=fallback_reason or unavailable_reason,
        subjects_requested=subjects,
        profiles_returned=len(reviews),
        correlation_id=correlation_id,
        session_id=session_id,
    )
    return CortexClientResult(
        ok=ok,
        mode="brain",
        verb=workflow_id,
        status="success" if ok else "fail",
        final_text=_workflow_summary_text(
            title="Concept Induction Pass",
            status=status,
            main_result=main_result,
            persisted=[],
        ),
        memory_used=False,
        recall_debug={},
        steps=[],
        error=None if ok else {"message": main_result, "code": "concept_profiles_unavailable"},
        correlation_id=correlation_id,
        metadata=metadata,
    )


async def execute_chat_workflow(
    *,
    bus: OrionBusAsync,
    source: ServiceRef,
    req: CortexClientRequest,
    correlation_id: str,
    causality_chain: list | None,
    trace: dict | None,
    call_verb_runtime,
) -> CortexClientResult:
    request = _workflow_request(req)
    workflow_id = str(request.get("workflow_id") or "").strip()
    definition = get_workflow_definition(workflow_id)
    if definition is None:
        raise WorkflowExecutionError(f"unknown_workflow:{workflow_id or 'missing'}")

    policy = _execution_policy(req, workflow_id)
    logger.info("workflow_requested corr=%s workflow_id=%s alias=%s session_id=%s mode=%s notify_on=%s", correlation_id, workflow_id, request.get("matched_alias"), req.context.session_id, policy.invocation_mode, policy.notify_on)
    logger.info(
        "workflow_path_decision corr=%s workflow_id=%s invocation_mode=%s schedule_kind=%s",
        correlation_id,
        workflow_id,
        policy.invocation_mode,
        policy.schedule.kind if policy.schedule else None,
    )
    if policy.invocation_mode == "scheduled":
        logger.info("workflow_scheduled corr=%s workflow_id=%s schedule=%s", correlation_id, workflow_id, policy.schedule.model_dump(mode="json") if policy.schedule else {})
        return await _schedule_workflow_dispatch(
            bus=bus,
            source=source,
            req=req,
            correlation_id=correlation_id,
            workflow_request=request,
            policy=policy,
        )
    logger.info("workflow_started corr=%s workflow_id=%s", correlation_id, workflow_id)
    try:
        if workflow_id == "dream_cycle":
            result = await _execute_dream_cycle(
                bus=bus,
                source=source,
                correlation_id=correlation_id,
                causality_chain=causality_chain,
                trace=trace,
                req=req,
                call_verb_runtime=call_verb_runtime,
            )
        elif workflow_id == "journal_pass":
            result = await _execute_journal_pass(
                bus=bus,
                source=source,
                correlation_id=correlation_id,
                causality_chain=causality_chain,
                trace=trace,
                req=req,
                call_verb_runtime=call_verb_runtime,
            )
        elif workflow_id == "self_review":
            result = await _execute_self_review(
                bus=bus,
                source=source,
                correlation_id=correlation_id,
                causality_chain=causality_chain,
                trace=trace,
                req=req,
                call_verb_runtime=call_verb_runtime,
            )
        elif workflow_id == "concept_induction_pass":
            result = await _execute_concept_induction_pass(
                bus=bus,
                source=source,
                correlation_id=correlation_id,
                causality_chain=causality_chain,
                trace=trace,
                req=req,
                call_verb_runtime=call_verb_runtime,
            )
        else:
            raise WorkflowExecutionError(f"unimplemented_workflow:{workflow_id}")
    except Exception:
        logger.exception("workflow_failed corr=%s workflow_id=%s", correlation_id, workflow_id)
        logger.info(
            "workflow_execution_truth %s",
            json.dumps(
                {
                    "correlation_id": correlation_id,
                    "workflow_id": workflow_id,
                    "executed": False,
                    "persisted": False,
                    "persistence_kind": [],
                },
                sort_keys=True,
                default=str,
            ),
        )
        await _emit_workflow_notify(
            bus=bus,
            source=source,
            req=req,
            workflow_id=workflow_id,
            workflow_name=definition.display_name,
            correlation_id=correlation_id,
            ok=False,
            final_text="Workflow failed before completion.",
            notify_on=policy.notify_on,
            recipient_group=policy.recipient_group,
            execution_source="immediate",
        )
        raise
    await _emit_workflow_notify(
        bus=bus,
        source=source,
        req=req,
        workflow_id=workflow_id,
        workflow_name=definition.display_name,
        correlation_id=correlation_id,
        ok=result.ok,
        final_text=result.final_text or "",
        notify_on=policy.notify_on,
        recipient_group=policy.recipient_group,
        execution_source="immediate",
    )
    logger.info("workflow_completed corr=%s workflow_id=%s ok=%s", correlation_id, workflow_id, result.ok)
    workflow_meta = (result.metadata or {}).get("workflow") if isinstance(result.metadata, dict) else {}
    usefulness = _shape_workflow_result_summary(workflow_meta=workflow_meta if isinstance(workflow_meta, dict) else {}, result=result)
    logger.info(
        "workflow_result_usefulness %s",
        json.dumps(
            {
                "correlation_id": correlation_id,
                "workflow_id": workflow_id,
                **usefulness,
            },
            sort_keys=True,
            default=str,
        ),
    )
    logger.info(
        "workflow_execution_truth %s",
        json.dumps(
            {
                "correlation_id": correlation_id,
                "workflow_id": workflow_id,
                "executed": bool((workflow_meta or {}).get("executed", True)),
                "persisted": bool((workflow_meta or {}).get("persisted")),
                "persistence_kind": (workflow_meta or {}).get("persisted") or [],
            },
            sort_keys=True,
            default=str,
        ),
    )
    logger.info(
        "workflow_response_shape corr=%s workflow_id=%s status=%s scheduled_count=%s persisted_count=%s",
        correlation_id,
        workflow_id,
        (result.metadata.get("workflow") or {}).get("status") if isinstance(result.metadata, dict) else None,
        len(((result.metadata.get("workflow") or {}).get("scheduled") or [])) if isinstance(result.metadata, dict) else 0,
        len(((result.metadata.get("workflow") or {}).get("persisted") or [])) if isinstance(result.metadata, dict) else 0,
    )
    return result
