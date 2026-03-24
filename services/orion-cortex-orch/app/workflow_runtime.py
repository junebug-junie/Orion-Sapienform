from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

from orion.cognition.workflows import get_workflow_definition, workflow_registry_payload
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.verbs import VerbResultV1
from orion.journaler.worker import (
    JOURNAL_WRITE_KIND,
    build_compose_request,
    build_manual_trigger,
    build_write_payload,
    draft_from_cortex_result,
)
from orion.schemas.cortex.contracts import CortexClientRequest, CortexClientResult
from orion.schemas.telemetry.spark import SparkStateSnapshotV1
from orion.spark.concept_induction.settings import get_settings as get_concept_settings
from orion.spark.concept_induction.store import LocalProfileStore
from orion.schemas.notify import NotificationRequest
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1, WorkflowExecutionPolicyV1

logger = logging.getLogger("orion.cortex.orch.workflow_runtime")
JOURNAL_WRITE_CHANNEL = "orion:journal:write"
ACTIONS_WORKFLOW_TRIGGER_CHANNEL = "orion:actions:trigger:workflow.v1"
NOTIFY_PERSISTENCE_REQUEST_CHANNEL = "orion:notify:persistence:request"


class WorkflowExecutionError(RuntimeError):
    pass


def has_explicit_workflow_request(req: CortexClientRequest) -> bool:
    metadata = req.context.metadata if isinstance(req.context.metadata, dict) else {}
    workflow_request = metadata.get("workflow_request")
    return isinstance(workflow_request, dict) and bool(workflow_request.get("workflow_id"))


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
    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Dream Cycle",
        "status": "completed" if verb_result.ok else "failed",
        "subverb": "dream_cycle",
        "persisted": ["dream.result.v1"] if verb_result.ok else [],
        "scheduled": [],
        "main_result": final_text,
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
            persisted=metadata["workflow"]["persisted"],
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
    summary = str((skill_result.get("summary") if isinstance(skill_result, dict) else None) or payload.get("final_text") or "Self review completed.")
    persisted: List[str] = []
    if isinstance(skill_result, dict):
        graph_write = skill_result.get("graph_write") or {}
        journal_write = skill_result.get("journal_write") or {}
        if graph_write.get("graph"):
            persisted.append(f"graph:{graph_write['graph']}")
        if journal_write.get("channel"):
            persisted.append(f"journal:{journal_write['channel']}")
    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Self Review",
        "status": "completed" if verb_result.ok else "failed",
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
            main_result=f"{summary} Findings: {len(findings)}.",
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
    del bus, source, causality_chain, trace, call_verb_runtime
    workflow_id = "concept_induction_pass"
    settings = get_concept_settings()
    store = LocalProfileStore(settings.store_path)
    subjects = list(settings.subjects or ["orion", "juniper", "relationship"])
    reviews: List[Dict[str, Any]] = []
    for subject in subjects:
        profile = store.load(subject)
        if profile is None:
            continue
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
    if reviews:
        main_result = "; ".join(
            f"{item['subject']} rev {item['revision']} with {item['concept_count']} concepts / {item['cluster_count']} clusters"
            for item in reviews
        )
    else:
        main_result = f"No concept induction profiles were available in {settings.store_path}."
    metadata = _workflow_metadata_base(request=_workflow_request(req), status="completed")
    metadata["workflow"] = {
        "workflow_id": workflow_id,
        "display_name": "Concept Induction Pass",
        "status": "completed",
        "subverb": None,
        "persisted": [],
        "scheduled": [],
        "main_result": main_result,
        "profile_store_path": settings.store_path,
        "profiles_reviewed": reviews,
    }
    return CortexClientResult(
        ok=True,
        mode="brain",
        verb=workflow_id,
        status="success",
        final_text=_workflow_summary_text(
            title="Concept Induction Pass",
            status="completed",
            main_result=main_result,
            persisted=[],
        ),
        memory_used=False,
        recall_debug={},
        steps=[],
        error=None,
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
    return result
