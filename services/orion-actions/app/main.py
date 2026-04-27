from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import requests
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict
from uuid import uuid4
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.llm_json import parse_json_object
from orion.journaler import (
    JOURNAL_WRITE_KIND,
    JournalTriggerV1,
    build_collapse_stored_trigger,
    build_compose_request,
    build_manual_trigger,
    build_metacog_trigger,
    build_notify_summary_trigger,
    build_scheduler_trigger,
    build_write_payload,
    cooldown_key_for_trigger,
    draft_from_cortex_result,
)
from orion.schemas.actions.daily import DailyMetacogV1, DailyPulseV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, CollapseMirrorStoredV1
from orion.schemas.cortex.schemas import PlanExecutionArgs, PlanExecutionRequest
from orion.schemas.notify import NotificationRecord, NotificationRequest
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

from .logic import (
    ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
    SKILL_BIOMETRICS_SNAPSHOT_V1,
    SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1,
    SKILL_NOTIFY_CHAT_MESSAGE_V1,
    ActionDedupe,
    build_audit_envelope,
    build_cortex_orch_envelope,
    build_skill_cortex_orch_envelope,
    dispatch_cortex_request,
    dedupe_key_for,
    extract_message_sections,
    new_reply_channel,
    should_trigger,
)
from .settings import settings
from orion.schemas.workflow_execution import WorkflowDispatchRequestV1, WorkflowScheduleManageRequestV1, WorkflowScheduleManageResponseV1

from .workflow_schedule_metrics import WorkflowScheduleMetrics
from .workflow_schedule_store import ClaimedSchedule, ScheduleAttentionSignal, WorkflowScheduleStore

logger = logging.getLogger("orion-actions")
PROCESS_STARTED_AT_UTC = datetime.now(timezone.utc)

ACTION_DAILY_PULSE_V1 = "daily_pulse_v1"
ACTION_DAILY_METACOG_V1 = "daily_metacog_v1"
ACTION_WORKFLOW_SCHEDULE_V1 = "workflow.schedule.v1"
ACTION_WORKFLOW_MANAGE_V1 = "workflow.manage.v1"
WORKFLOW_TRIGGER_KIND = "orion.actions.trigger.workflow.v1"
WORKFLOW_TRIGGER_CHANNEL = "orion:actions:trigger:workflow.v1"
WORKFLOW_MANAGE_KIND = "orion.actions.manage.workflow.v1"
WORKFLOW_MANAGE_CHANNEL = "orion:actions:manage:workflow.v1"


def _runtime_identity() -> dict[str, str]:
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
        "git_sha": os.getenv("GIT_SHA") or os.getenv("SOURCE_COMMIT") or "unknown",
        "build_timestamp": os.getenv("BUILD_TIMESTAMP") or "unknown",
        "environment": os.getenv("ORION_ENV") or os.getenv("ENVIRONMENT") or "unknown",
        "process_started_at": PROCESS_STARTED_AT_UTC.isoformat(),
        "now_utc": datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class DailyWindow:
    request_date: str
    window_start_utc: str
    window_end_utc: str
    timezone: str


def _ensure_logging() -> None:
    level_name = (settings.log_level or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s - %(message)s")
    else:
        root.setLevel(level)
    logger.setLevel(level)


_ensure_logging()
workflow_schedule_metrics = WorkflowScheduleMetrics()


def _normalized_llm_route(preferred: str | None, fallback: str) -> str:
    route = str(preferred or fallback or "").strip().lower()
    if route in {"chat_quick", "quick_chat"}:
        return "quick"
    if route in {"chat", "quick", "metacog"}:
        return route
    return "chat"


def _cfg() -> ChassisConfig:
    os.environ["ORION_BUS_ENFORCE_CATALOG"] = "true" if settings.orion_bus_enforce_catalog else "false"
    return ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
        heartbeat_interval_sec=30.0,
    )


def _source_ref() -> ServiceRef:
    return ServiceRef(name=settings.service_name, version=settings.service_version, node=settings.node_name)


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_daily_window(*, now_utc: datetime | None = None, tz_name: str, override_date: str | None = None) -> DailyWindow:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    tz = ZoneInfo(tz_name)
    local_now = now.astimezone(tz)

    if override_date:
        local_date = datetime.strptime(override_date, "%Y-%m-%d").date()
    else:
        local_date = (local_now - timedelta(days=1)).date()

    local_start = datetime(local_date.year, local_date.month, local_date.day, tzinfo=tz)
    local_end = local_start + timedelta(days=1)
    return DailyWindow(
        request_date=local_date.isoformat(),
        window_start_utc=_iso(local_start),
        window_end_utc=_iso(local_end),
        timezone=tz_name,
    )


def should_run_daily(*, now_utc: datetime, tz_name: str, hour_local: int, minute_local: int, last_ran_date: str | None) -> tuple[bool, str]:
    tz = ZoneInfo(tz_name)
    local_now = now_utc.astimezone(tz)
    today = local_now.date().isoformat()
    if last_ran_date == today:
        return False, today
    threshold = local_now.replace(hour=hour_local, minute=minute_local, second=0, microsecond=0)
    return local_now >= threshold, today


def _extract_plan_final_text(result_payload: dict[str, Any]) -> str:
    result = result_payload.get("result") if isinstance(result_payload, dict) else None
    if isinstance(result, dict):
        final = result.get("final_text")
        if isinstance(final, str) and final.strip():
            return final.strip()
        steps = result.get("steps")
        if isinstance(steps, list):
            for step in reversed(steps):
                step_result = step.get("result") if isinstance(step, dict) else None
                if not isinstance(step_result, dict):
                    continue
                for payload in step_result.values():
                    if not isinstance(payload, dict):
                        continue
                    text = payload.get("text") or payload.get("content")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
                    raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
                    raw_text = raw.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        return raw_text.strip()
    return ""


def _json_loads_strict(text: str) -> dict[str, Any]:
    data = parse_json_object(text)
    if not isinstance(data, dict):
        raise ValueError("llm output must be a JSON object")
    return data


def _clamp_daily_metacog_payload(parsed: dict[str, Any], *, action_name: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return parsed
    if action_name != ACTION_DAILY_METACOG_V1:
        return parsed

    clamped = dict(parsed)
    max_lengths = {
        "course_correction": 300,
        "tomorrow_experiment": 240,
    }
    for field, limit in max_lengths.items():
        value = clamped.get(field)
        if isinstance(value, str) and len(value) > limit:
            logger.warning(
                "daily payload field exceeded max length; clamping action=%s field=%s original_len=%s max_len=%s",
                action_name,
                field,
                len(value),
                limit,
            )
            clamped[field] = value[:limit].rstrip()
    return clamped


def _extract_daily_llm_diagnostics(plan_result_payload: dict[str, Any] | None) -> dict[str, Any]:
    result = plan_result_payload.get("result") if isinstance(plan_result_payload, dict) else None
    if not isinstance(result, dict):
        return {}

    steps = result.get("steps")
    if not isinstance(steps, list):
        return {}

    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_result = step.get("result")
        if not isinstance(step_result, dict):
            continue
        payload = step_result.get("LLMGatewayService")
        if not isinstance(payload, dict):
            continue

        raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
        choices = raw.get("choices") if isinstance(raw, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        if not usage and isinstance(raw.get("usage"), dict):
            usage = raw.get("usage") or {}

        return {
            "model": payload.get("model_used") or payload.get("model"),
            "finish_reason": first_choice.get("finish_reason") or raw.get("finish_reason"),
            "stop_reason": first_choice.get("stop_reason") or raw.get("stop_reason"),
            "usage": usage,
        }
    return {}


def _is_likely_incomplete_json(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or not raw.startswith("{"):
        return False
    if raw.endswith("}"):
        return False
    return True


def _should_retry_for_truncated_generation(*, text: str, diagnostics: dict[str, Any]) -> bool:
    finish_reason = str(diagnostics.get("finish_reason") or "").strip().lower()
    stop_reason = str(diagnostics.get("stop_reason") or "").strip().lower()
    if finish_reason in {"length", "max_tokens", "timeout"}:
        return True
    if stop_reason in {"length", "max_tokens", "timeout"}:
        return True
    return _is_likely_incomplete_json(text)


def _log_daily_json_parse_failure(
    *,
    action_name: str,
    final_text: str,
    diagnostics: dict[str, Any],
    timeout_sec: float,
    requested_max_tokens: Any,
    parse_error: Exception,
) -> None:
    logger.error(
        "daily json parse failed action=%s len=%s tail=%r model=%s finish_reason=%s stop_reason=%s timeout_sec=%s requested_max_tokens=%s parse_error=%s",
        action_name,
        len(final_text),
        final_text[-240:],
        diagnostics.get("model"),
        diagnostics.get("finish_reason"),
        diagnostics.get("stop_reason"),
        timeout_sec,
        requested_max_tokens,
        parse_error,
    )


async def _rpc_request_with_retry(
    *,
    bus: Any,
    request_channel: str,
    reply_prefix: str,
    timeout_sec: float,
    envelope_factory: Callable[[str, int], BaseEnvelope],
    operation_name: str,
    max_attempts: int = 2,
) -> dict[str, Any]:
    last_error: TimeoutError | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        reply_channel = new_reply_channel(reply_prefix)
        envelope = envelope_factory(reply_channel, attempt)
        try:
            return await bus.rpc_request(
                request_channel,
                envelope,
                reply_channel=reply_channel,
                timeout_sec=timeout_sec,
            )
        except TimeoutError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "%s timed out attempt=%s/%s request_channel=%s reply_channel=%s timeout_sec=%.1f; retrying",
                operation_name,
                attempt,
                max_attempts,
                request_channel,
                reply_channel,
                timeout_sec,
            )
    assert last_error is not None
    raise last_error


def _daily_pulse_dedupe_key(window: DailyWindow) -> str:
    return f"actions:daily_pulse:{window.request_date}:{settings.node_name}:{settings.actions_recipient_group}"


def _daily_metacog_dedupe_key(window: DailyWindow) -> str:
    return f"actions:daily_metacog:{window.request_date}:{settings.node_name}"


def _daily_notify_request(
    *,
    event_kind: str,
    title: str,
    dedupe_key: str,
    correlation_id: str,
    payload: dict[str, Any],
    include_email_channel: bool,
) -> NotificationRequest:
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    preview = "; ".join([f"{k}: {v}" for k, v in list(payload.items())[:3]])[:280]
    full_markdown = f"## {title}\n\n```json\n{pretty}\n```\n"
    payload_fingerprint = hashlib.sha1(pretty.encode("utf-8")).hexdigest()[:12]
    return NotificationRequest(
        source_service=settings.service_name,
        event_kind=event_kind,
        severity="info",
        title=title,
        # Email transport currently prefers body_text, so keep it full for parity with Hub full view.
        body_text=full_markdown,
        body_md=full_markdown,
        recipient_group=settings.actions_recipient_group,
        session_id=settings.actions_session_id,
        correlation_id=correlation_id,
        dedupe_key=f"{dedupe_key}:{payload_fingerprint}",
        dedupe_window_seconds=int(settings.actions_notify_dedupe_window_seconds),
        tags=["actions", "daily", "json"],
        context={"payload": payload, "preview_text": preview},
        channels_requested=["email"] if include_email_channel else None,
    )


def _daily_preview_and_markdown(*, title: str, payload: dict[str, Any]) -> tuple[str, str]:
    preview = "; ".join([f"{k}: {v}" for k, v in list(payload.items())[:3]])[:280]
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    markdown = f"## {title}\n\n```json\n{pretty}\n```\n"
    return preview, markdown


def _send_orion_async_message(
    *,
    notify,
    title: str,
    preview_text: str,
    full_text: str,
    severity: str = "info",
    tags: list[str] | None = None,
    session_id: str | None = None,
    source_service: str | None = None,
    correlation_id: str | None = None,
):
    return notify.chat_message(
        session_id=session_id or settings.actions_session_id,
        title=title,
        preview_text=preview_text[:280],
        full_text=full_text,
        severity=severity,
        require_read_receipt=True,
        tags=tags or ["actions", "async-message"],
        source_service=source_service or settings.service_name,
        correlation_id=correlation_id,
    )


def _send_pending_attention(
    *,
    notify,
    reason: str,
    message: str,
    severity: str = "warning",
    context: dict[str, Any] | None = None,
    require_ack: bool = True,
    expires_in_minutes: int | None = None,
):
    payload_context = dict(context or {})
    payload_context.setdefault("source_service", settings.service_name)
    payload_context.setdefault("reason", reason)
    return notify.attention_request(
        message=message,
        severity=severity,
        require_ack=require_ack,
        context=payload_context,
        expires_in_minutes=expires_in_minutes,
    )


def _publish_daily_outputs(
    *,
    notify,
    action_name: str,
    title: str,
    preview_text: str,
    full_text: str,
    notify_req: NotificationRequest,
    correlation_id: str,
) -> dict[str, Any]:
    accepted = None
    if settings.actions_preserve_generic_notify_enabled:
        accepted = notify.send(notify_req)

    chat_message_accepted = None
    if settings.actions_async_messages_enabled and settings.actions_daily_async_messages_enabled:
        logger.info("daily_async_message_attempted action=%s correlation_id=%s", action_name, correlation_id)
        chat_tags = ["actions", "daily", "pulse"] if action_name == ACTION_DAILY_PULSE_V1 else ["actions", "daily", "metacog"]
        chat_message_accepted = _send_orion_async_message(
            notify=notify,
            title=title,
            preview_text=preview_text,
            full_text=full_text,
            severity="info",
            tags=chat_tags,
            session_id=settings.actions_session_id,
            source_service=settings.service_name,
            correlation_id=correlation_id,
        )
        if chat_message_accepted.ok:
            logger.info("daily_async_message_succeeded action=%s correlation_id=%s", action_name, correlation_id)
        else:
            logger.info(
                "daily_async_message_failed action=%s correlation_id=%s detail=%s",
                action_name,
                correlation_id,
                chat_message_accepted.detail,
            )

    return {
        "generic": accepted,
        "chat_message": chat_message_accepted,
    }


def _schedule_attention_notify_request(*, signal: ScheduleAttentionSignal, correlation_id: str) -> NotificationRequest:
    schedule = signal.schedule
    analytics = signal.analytics
    short_id = schedule.schedule_id[-8:] if len(schedule.schedule_id) > 8 else schedule.schedule_id
    health = str(analytics.health or "idle")
    status_word = "recovered" if signal.transition == "recovered" else "needs attention"
    overdue = "none"
    if analytics.is_overdue:
        overdue = f"{int(analytics.overdue_seconds or 0)}s"
    title = f"Workflow schedule {status_word}: {schedule.workflow_display_name or schedule.workflow_id}"
    body = (
        f"Schedule #{short_id} ({schedule.workflow_id}) {status_word}. "
        f"health={health}, condition={signal.kind}, overdue={overdue}, "
        f"recent={int(analytics.recent_success_count or 0)} success/{int(analytics.recent_failure_count or 0)} failure."
    )
    severity = "warning"
    if signal.kind == "failing":
        severity = "error"
    if signal.transition == "recovered":
        severity = "info"
    dedupe_condition = signal.kind if signal.state == "active" else "recovered"
    return NotificationRequest(
        source_service=settings.service_name,
        event_kind="workflow.schedule.attention.v1",
        severity=severity,
        title=title,
        body_text=body,
        context={
            "schedule_id": schedule.schedule_id,
            "schedule_id_short": short_id,
            "workflow_id": schedule.workflow_id,
            "workflow_display_name": schedule.workflow_display_name,
            "health": health,
            "condition": signal.kind,
            "transition": signal.transition,
            "state": signal.state,
            "is_overdue": bool(analytics.is_overdue),
            "overdue_seconds": analytics.overdue_seconds,
            "missed_run_count": analytics.missed_run_count,
            "needs_attention": bool(analytics.needs_attention),
        },
        tags=["workflow", "schedule", "attention", health],
        recipient_group=settings.actions_recipient_group,
        dedupe_key=f"workflow:schedule:attention:{schedule.schedule_id}:{dedupe_condition}",
        dedupe_window_seconds=int(settings.actions_notify_dedupe_window_seconds),
        correlation_id=correlation_id,
        session_id=schedule.execution_policy.session_id or settings.actions_session_id,
    )


def _journal_daily_dedupe_key(window: DailyWindow) -> str:
    return f"actions:journal:daily:{window.request_date}:{settings.node_name}"


def _world_pulse_daily_dedupe_key(window: DailyWindow) -> str:
    return f"actions:world_pulse:daily:{window.request_date}:{settings.node_name}"


def _trigger_world_pulse_run(*, date: str, requested_by: str) -> bool:
    try:
        resp = requests.post(
            f"{settings.world_pulse_base_url.rstrip('/')}/api/world-pulse/run",
            json={"date": date, "dry_run": True, "requested_by": requested_by},
            timeout=20,
        )
        return resp.ok
    except Exception:
        logger.exception("world pulse trigger failed date=%s", date)
        return False


def _is_scheduler_daily_journal(*, trigger: Any, write_payload: dict[str, Any], draft: dict[str, Any]) -> bool:
    mode = str(write_payload.get("mode") or draft.get("mode") or "").strip().lower()
    trigger_kind = str(getattr(trigger, "trigger_kind", "") or "").strip().lower()
    source_kind = str(write_payload.get("source_kind") or getattr(trigger, "source_kind", "") or "").strip().lower()
    if mode != "daily":
        return False
    if trigger_kind == "daily_summary" and source_kind == "scheduler":
        return True
    logger.info(
        "scheduler_daily_journal_unexpected_fields mode=%s trigger_kind=%s source_kind=%s source_ref=%s",
        mode or None,
        trigger_kind or None,
        source_kind or None,
        str(write_payload.get("source_ref") or getattr(trigger, "source_ref", None) or ""),
    )
    return False


def _build_scheduler_daily_journal_message_payload(
    *,
    trigger: Any,
    write_payload: dict[str, Any],
    draft: dict[str, Any],
    correlation_id: str,
) -> dict[str, Any]:
    title_text = str(write_payload.get("title") or draft.get("title") or "Daily Journal").strip()
    body_text = str(write_payload.get("body") or draft.get("body") or "").strip()
    source_ref = str(write_payload.get("source_ref") or getattr(trigger, "source_ref", "") or "").strip()
    mode = str(write_payload.get("mode") or draft.get("mode") or "").strip()
    source_kind = str(write_payload.get("source_kind") or getattr(trigger, "source_kind", "") or "").strip()
    preview = f"{title_text}: {body_text}".strip(": ").replace("\n", " ")[:280]
    full_text = (
        "## Orion — Daily Journal\n\n"
        f"**Title:** {title_text}\n\n"
        f"**Mode:** {mode or 'daily'}\n\n"
        f"**Source:** {source_kind or 'scheduler'}"
        f"{f' ({source_ref})' if source_ref else ''}\n\n"
        f"{body_text}"
    )
    return {
        "title": "Orion — Daily Journal",
        "preview_text": preview,
        "full_text": full_text,
        "severity": "info",
        "tags": ["actions", "journal", "daily", "scheduler"],
        "session_id": settings.actions_journal_session_id or settings.actions_session_id,
        "source_service": settings.service_name,
        "correlation_id": correlation_id,
    }


def _build_scheduler_daily_journal_email_request(
    *,
    trigger: Any,
    write_payload: dict[str, Any],
    draft: dict[str, Any],
    correlation_id: str,
) -> NotificationRequest:
    message_payload = _build_scheduler_daily_journal_message_payload(
        trigger=trigger,
        write_payload=write_payload,
        draft=draft,
        correlation_id=correlation_id,
    )
    dedupe_seed = str(write_payload.get("entry_id") or correlation_id)
    return NotificationRequest(
        source_service=settings.service_name,
        event_kind="orion.journal.daily.scheduler",
        severity="info",
        title="Orion — Daily Journal",
        body_text=message_payload["full_text"],
        body_md=message_payload["full_text"],
        recipient_group=settings.actions_recipient_group,
        session_id=settings.actions_journal_session_id or settings.actions_session_id,
        correlation_id=correlation_id,
        tags=["actions", "journal", "daily", "scheduler"],
        channels_requested=["email"],
        dedupe_key=f"actions:journal:daily:scheduler:{dedupe_seed}",
        dedupe_window_seconds=int(settings.actions_notify_dedupe_window_seconds),
        context={
            "trigger_kind": str(getattr(trigger, "trigger_kind", "") or ""),
            "source_kind": str(write_payload.get("source_kind") or getattr(trigger, "source_kind", "") or ""),
            "source_ref": str(write_payload.get("source_ref") or getattr(trigger, "source_ref", "") or ""),
            "entry_id": str(write_payload.get("entry_id") or ""),
        },
    )


async def _publish_workflow_attention_signal(*, signal: ScheduleAttentionSignal, notify) -> None:
    req = _schedule_attention_notify_request(signal=signal, correlation_id=str(uuid4()))
    if settings.actions_preserve_generic_notify_enabled:
        generic_accepted = await asyncio.to_thread(notify.send, req)
        if not generic_accepted.ok:
            logger.warning(
                "workflow attention notify failed schedule_id=%s transition=%s condition=%s detail=%s",
                signal.schedule.schedule_id,
                signal.transition,
                signal.kind,
                generic_accepted.detail,
            )

    if signal.transition == "recovered":
        if settings.actions_async_messages_enabled:
            await asyncio.to_thread(
                _send_orion_async_message,
                notify=notify,
                title=req.title,
                preview_text=req.body_text or req.title,
                full_text=req.body_text or req.title,
                severity="info",
                tags=["workflow", "schedule", "recovered"],
                session_id=req.session_id or settings.actions_session_id,
                source_service=settings.service_name,
                correlation_id=req.correlation_id,
            )
        workflow_schedule_metrics.incr_attention(signal.transition)
        return

    if settings.actions_pending_attention_enabled:
        attention_context = dict(req.context or {})
        attention_context.update(
            {
                "source_service": settings.service_name,
                "reason": req.title,
                "event_kind": "workflow.schedule.attention.v1",
                "tags": ["workflow", "schedule", "attention"],
                "correlation_id": req.correlation_id,
            }
        )
        await asyncio.to_thread(
            _send_pending_attention,
            notify=notify,
            reason=req.title,
            message=req.body_text or req.title,
            severity=req.severity if req.severity in {"error", "warning"} else "warning",
            context=attention_context,
            require_ack=True,
        )

    workflow_schedule_metrics.incr_attention(signal.transition)


def should_journal_from_collapse(is_causally_dense: bool, *, dense_only: bool) -> bool:
    return bool(is_causally_dense) or not dense_only


def should_run_interval(*, now_monotonic: float, last_run_monotonic: float | None, interval_seconds: int, run_on_startup: bool) -> bool:
    if last_run_monotonic is None:
        return bool(run_on_startup)
    return (now_monotonic - last_run_monotonic) >= max(1, int(interval_seconds))


def _extract_skill_result_from_orch(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    if isinstance(metadata, dict) and isinstance(metadata.get("skill_result"), dict):
        return metadata.get("skill_result") or {}
    final_text = payload.get("final_text") if isinstance(payload, dict) else None
    if isinstance(final_text, str) and final_text.strip():
        try:
            parsed = json.loads(final_text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _scheduler_threshold_findings(*, biometrics_snapshot: Dict[str, Any] | None, gpu_snapshot: Dict[str, Any] | None) -> list[str]:
    findings: list[str] = []
    if isinstance(biometrics_snapshot, dict):
        cluster = biometrics_snapshot.get("cluster") if isinstance(biometrics_snapshot.get("cluster"), dict) else {}
        composite = cluster.get("composite") if isinstance(cluster.get("composite"), dict) else {}
        stability = composite.get("stability")
        try:
            if stability is not None and float(stability) < float(settings.actions_skills_biometrics_stability_threshold):
                findings.append(f"stability_below:{float(stability):.3f}")
        except Exception:
            pass
    if isinstance(gpu_snapshot, dict):
        gpus = gpu_snapshot.get("gpus") if isinstance(gpu_snapshot.get("gpus"), list) else []
        for gpu in gpus:
            try:
                ratio = float(gpu.get("memory_used_ratio") or 0.0)
            except Exception:
                ratio = 0.0
            if ratio > float(settings.actions_skills_gpu_mem_threshold):
                findings.append(f"gpu_mem_above:{gpu.get('name') or gpu.get('index')}:{ratio:.3f}")
    return findings


def _threshold_notify_skill_args(*, findings: list[str], correlation_id: str) -> dict[str, Any]:
    body_text = "Threshold findings: " + "; ".join(findings)
    return {
        "title": "Orion Skills Threshold Alert",
        "body_text": body_text,
        "body_md": body_text,
        "recipient_group": settings.actions_recipient_group,
        "session_id": settings.actions_session_id,
        "dedupe_key": f"actions:skills:threshold:{correlation_id}",
        "dedupe_window_seconds": int(settings.actions_notify_dedupe_window_seconds),
        "tags": ["actions", "skills", "threshold"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    deduper = ActionDedupe(ttl_seconds=settings.actions_dedupe_ttl_seconds)
    journal_deduper = ActionDedupe(ttl_seconds=settings.actions_journaling_cooldown_seconds)
    sem = asyncio.Semaphore(max(1, int(settings.actions_max_concurrency)))
    from orion.notify.client import NotifyClient
    notify = NotifyClient(base_url=settings.notify_url, api_token=settings.notify_api_token, timeout=10)
    src = _source_ref()
    last_daily_run: dict[str, str] = {}
    last_skill_run_monotonic: float | None = None
    last_journal_run: str | None = None
    workflow_schedule_store = WorkflowScheduleStore(
        settings.actions_workflow_schedule_store_path,
        metrics=workflow_schedule_metrics,
    )

    async def _audit(
        parent: BaseEnvelope,
        *,
        status: str,
        event_id: str,
        action_name: str,
        reason: str | None = None,
        extra: dict | None = None,
    ):
        try:
            env = build_audit_envelope(
                parent,
                source=src,
                status=status,
                action_name=action_name,
                event_id=event_id,
                reason=reason,
                extra=extra,
            )
            await hunter.bus.publish(settings.actions_audit_channel, env)
        except Exception:
            logger.debug("audit publish failed", exc_info=True)

    async def _run_plan(parent: BaseEnvelope, *, verb_name: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        plan = build_plan_for_verb(verb_name)
        request_id = str(parent.correlation_id)
        timeout_sec = float(settings.actions_exec_timeout_seconds)
        daily_llm_route = _normalized_llm_route(settings.actions_daily_llm_route, settings.actions_llm_route)

        def _plan_envelope(reply_channel: str, attempt: int) -> BaseEnvelope:
            req = PlanExecutionRequest(
                plan=plan,
                args=PlanExecutionArgs(
                    request_id=request_id,
                    trigger_source=settings.service_name,
                    user_id=settings.actions_recipient_group,
                    extra={
                        "mode": "brain",
                        "llm_route": daily_llm_route,
                        "session_id": settings.actions_session_id,
                        "verb": verb_name,
                        "trace_id": request_id,
                        "rpc_attempt": attempt,
                    },
                ),
                context=context,
            )
            return parent.derive_child(kind=req.kind, source=src, payload=req, reply_to=reply_channel)

        msg = await _rpc_request_with_retry(
            bus=hunter.bus,
            request_channel=settings.cortex_exec_request_channel,
            reply_prefix="orion:exec:result",
            timeout_sec=timeout_sec,
            envelope_factory=_plan_envelope,
            operation_name=f"daily plan {verb_name}",
        )
        decoded = hunter.bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            raise RuntimeError(f"cortex_exec_decode_failed:{decoded.error}")
        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        final_text = _extract_plan_final_text(payload)
        if not final_text:
            raise RuntimeError("cortex_exec_missing_final_text")
        return final_text, payload

    async def _run_journal(parent: BaseEnvelope, *, trigger) -> dict[str, Any]:
        journal_llm_route = _normalized_llm_route(settings.actions_journal_llm_route, settings.actions_llm_route)
        req = build_compose_request(
            trigger,
            session_id=settings.actions_journal_session_id,
            user_id=settings.actions_recipient_group,
            trace_id=str(parent.correlation_id),
            recall_profile=settings.actions_recall_profile,
            options={
                "timeout_sec": float(settings.actions_exec_timeout_seconds),
                "llm_route": journal_llm_route,
            },
        )
        reply_channel = new_reply_channel("orion:cortex:result")
        req_env = parent.derive_child(kind="cortex.orch.request", source=src, payload=req, reply_to=reply_channel)
        msg = await hunter.bus.rpc_request(
            settings.cortex_request_channel,
            req_env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.actions_exec_timeout_seconds),
        )
        decoded = hunter.bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            raise RuntimeError(f"cortex_orch_decode_failed:{decoded.error}")
        orch_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        if not orch_payload.get("ok", False):
            raise RuntimeError(f"journal_compose_failed:{orch_payload.get('error') or orch_payload.get('status')}")
        draft = draft_from_cortex_result(orch_payload)
        write = build_write_payload(
            draft,
            trigger=trigger,
            correlation_id=str(parent.correlation_id),
            author=settings.actions_journal_author,
        )
        write_env = parent.derive_child(kind=JOURNAL_WRITE_KIND, source=src, payload=write.model_dump(mode="json"), reply_to=None)
        await hunter.bus.publish(settings.actions_journal_write_channel, write_env)
        return {
            "draft": draft.model_dump(mode="json"),
            "write": write.model_dump(mode="json"),
            "orch_payload": orch_payload,
        }

    async def _dispatch_journal(parent: BaseEnvelope, *, trigger, audit_action: str, dedupe_key: str, reason: str | None = None) -> None:
        if not settings.actions_journaling_enabled:
            await _audit(parent, status="skipped", event_id=dedupe_key, action_name=audit_action, reason="journaling_disabled")
            return
        if not journal_deduper.try_acquire(dedupe_key):
            await _audit(parent, status="skipped", event_id=dedupe_key, action_name=audit_action, reason=reason or "journal_cooldown")
            return

        acquired = False
        t0 = time.monotonic()
        try:
            await sem.acquire()
            acquired = True
            result = await _run_journal(parent, trigger=trigger)
            draft = result.get("draft") if isinstance(result, dict) else {}
            write_payload = result.get("write") if isinstance(result, dict) else {}
            scheduler_daily = _is_scheduler_daily_journal(trigger=trigger, write_payload=write_payload, draft=draft)
            scheduler_daily_msg_accepted = None
            scheduler_daily_email_accepted = None
            scheduler_daily_msg_skip_reason = None
            scheduler_daily_email_skip_reason = None

            if scheduler_daily:
                if settings.actions_async_messages_enabled and settings.actions_scheduler_daily_journal_messages_enabled:
                    logger.info(
                        "scheduler_daily_journal_message_attempted correlation_id=%s entry_id=%s",
                        parent.correlation_id,
                        write_payload.get("entry_id"),
                    )
                    message_payload = _build_scheduler_daily_journal_message_payload(
                        trigger=trigger,
                        write_payload=write_payload,
                        draft=draft,
                        correlation_id=str(parent.correlation_id),
                    )
                    scheduler_daily_msg_accepted = await asyncio.to_thread(
                        _send_orion_async_message,
                        notify=notify,
                        **message_payload,
                    )
                    if scheduler_daily_msg_accepted.ok:
                        logger.info(
                            "scheduler_daily_journal_message_succeeded correlation_id=%s notification_id=%s",
                            parent.correlation_id,
                            scheduler_daily_msg_accepted.notification_id,
                        )
                    else:
                        logger.info(
                            "scheduler_daily_journal_message_failed correlation_id=%s detail=%s",
                            parent.correlation_id,
                            scheduler_daily_msg_accepted.detail,
                        )
                else:
                    scheduler_daily_msg_skip_reason = "scheduler_daily_journal_messages_disabled"
                    logger.info(
                        "scheduler_daily_journal_message_skipped correlation_id=%s reason=%s",
                        parent.correlation_id,
                        scheduler_daily_msg_skip_reason,
                    )

                if settings.actions_scheduler_daily_journal_email_enabled:
                    logger.info(
                        "scheduler_daily_journal_email_attempted correlation_id=%s entry_id=%s",
                        parent.correlation_id,
                        write_payload.get("entry_id"),
                    )
                    email_req = _build_scheduler_daily_journal_email_request(
                        trigger=trigger,
                        write_payload=write_payload,
                        draft=draft,
                        correlation_id=str(parent.correlation_id),
                    )
                    scheduler_daily_email_accepted = await asyncio.to_thread(notify.send, email_req)
                    if scheduler_daily_email_accepted.ok:
                        logger.info(
                            "scheduler_daily_journal_email_succeeded correlation_id=%s notification_id=%s",
                            parent.correlation_id,
                            scheduler_daily_email_accepted.notification_id,
                        )
                    else:
                        logger.info(
                            "scheduler_daily_journal_email_failed correlation_id=%s detail=%s",
                            parent.correlation_id,
                            scheduler_daily_email_accepted.detail,
                        )
                else:
                    scheduler_daily_email_skip_reason = "scheduler_daily_journal_email_disabled"
                    logger.info(
                        "scheduler_daily_journal_email_skipped correlation_id=%s reason=%s",
                        parent.correlation_id,
                        scheduler_daily_email_skip_reason,
                    )
            else:
                scheduler_daily_msg_skip_reason = "not_scheduler_daily_journal"
                scheduler_daily_email_skip_reason = "not_scheduler_daily_journal"
                logger.info(
                    "scheduler_daily_journal_message_skipped correlation_id=%s reason=%s",
                    parent.correlation_id,
                    scheduler_daily_msg_skip_reason,
                )
                logger.info(
                    "scheduler_daily_journal_email_skipped correlation_id=%s reason=%s",
                    parent.correlation_id,
                    scheduler_daily_email_skip_reason,
                )
            dt_ms = int((time.monotonic() - t0) * 1000)
            journal_deduper.mark_done(dedupe_key)
            await _audit(
                parent,
                status="completed",
                event_id=dedupe_key,
                action_name=audit_action,
                extra={
                    "duration_ms": dt_ms,
                    "journal_mode": result["draft"]["mode"],
                    "write_channel": settings.actions_journal_write_channel,
                    "journal_entry_id": result["write"]["entry_id"],
                    "scheduler_daily_journal_message_ok": scheduler_daily_msg_accepted.ok if scheduler_daily_msg_accepted else None,
                    "scheduler_daily_journal_message_notification_id": str(scheduler_daily_msg_accepted.notification_id) if scheduler_daily_msg_accepted and scheduler_daily_msg_accepted.notification_id else None,
                    "scheduler_daily_journal_message_skipped_reason": scheduler_daily_msg_skip_reason,
                    "scheduler_daily_journal_email_ok": scheduler_daily_email_accepted.ok if scheduler_daily_email_accepted else None,
                    "scheduler_daily_journal_email_notification_id": str(scheduler_daily_email_accepted.notification_id) if scheduler_daily_email_accepted and scheduler_daily_email_accepted.notification_id else None,
                    "scheduler_daily_journal_email_skipped_reason": scheduler_daily_email_skip_reason,
                },
            )
        except Exception as exc:
            dt_ms = int((time.monotonic() - t0) * 1000)
            await _audit(
                parent,
                status="failed",
                event_id=dedupe_key,
                action_name=audit_action,
                reason=str(exc),
                extra={"duration_ms": dt_ms},
            )
            logger.exception("Journal dispatch failed action=%s corr=%s", audit_action, parent.correlation_id)
        finally:
            if acquired:
                sem.release()
            journal_deduper.release(dedupe_key)

    async def _execute_daily(parent: BaseEnvelope, *, action_name: str, window: DailyWindow, dedupe_key: str):
        if not deduper.try_acquire(dedupe_key):
            await _audit(parent, status="skipped", event_id=dedupe_key, action_name=action_name, reason="deduped")
            return

        t0 = time.monotonic()
        acquired = False
        try:
            await sem.acquire()
            acquired = True
            await _audit(parent, status="started", event_id=dedupe_key, action_name=action_name)
            context = {
                "request_date": window.request_date,
                "timezone": window.timezone,
                "window_start_utc": window.window_start_utc,
                "window_end_utc": window.window_end_utc,
                "node": settings.node_name,
                "recipient_group": settings.actions_recipient_group,
                "session_id": settings.actions_session_id,
                "trace_id": str(parent.correlation_id),
            }
            parse_retry_used = False
            plan_result_payload: dict[str, Any] = {}
            daily_llm_diag: dict[str, Any] = {}
            final_text = ""
            parsed: dict[str, Any] | None = None
            for attempt in (1, 2):
                attempt_context = dict(context)
                if attempt > 1:
                    parse_retry_used = True
                    attempt_context["max_tokens"] = int(attempt_context.get("max_tokens") or 1024)
                    attempt_context["daily_parse_retry"] = "truncated_generation"

                final_text, plan_result_payload = await _run_plan(parent, verb_name=action_name, context=attempt_context)
                daily_llm_diag = _extract_daily_llm_diagnostics(plan_result_payload)

                if _should_retry_for_truncated_generation(text=final_text, diagnostics=daily_llm_diag):
                    if attempt == 1:
                        logger.warning(
                            "daily json looks truncated; retrying action=%s len=%s tail=%r model=%s finish_reason=%s stop_reason=%s",
                            action_name,
                            len(final_text),
                            final_text[-240:],
                            daily_llm_diag.get("model"),
                            daily_llm_diag.get("finish_reason"),
                            daily_llm_diag.get("stop_reason"),
                        )
                        continue
                    raise RuntimeError("truncated_generation")

                try:
                    parsed = _json_loads_strict(final_text)
                    break
                except Exception as parse_exc:
                    _log_daily_json_parse_failure(
                        action_name=action_name,
                        final_text=final_text,
                        diagnostics=daily_llm_diag,
                        timeout_sec=float(settings.actions_exec_timeout_seconds),
                        requested_max_tokens=attempt_context.get("max_tokens"),
                        parse_error=parse_exc,
                    )
                    if attempt == 1 and _should_retry_for_truncated_generation(text=final_text, diagnostics=daily_llm_diag):
                        continue
                    raise

            if parsed is None:
                raise RuntimeError("daily_json_parse_unavailable")
            parsed = _clamp_daily_metacog_payload(parsed, action_name=action_name)

            if action_name == ACTION_DAILY_PULSE_V1:
                model = DailyPulseV1.model_validate(parsed)
                event_kind = "orion.daily.pulse"
                title = "Orion — Daily Pulse"
            else:
                model = DailyMetacogV1.model_validate(parsed)
                event_kind = "orion.daily.metacog"
                title = "Orion — Daily Metacog"

            model_payload = model.model_dump(mode="json")
            preview_text, full_text = _daily_preview_and_markdown(title=title, payload=model_payload)
            notify_req = _daily_notify_request(
                event_kind=event_kind,
                title=title,
                dedupe_key=dedupe_key,
                correlation_id=str(parent.correlation_id),
                payload=model_payload,
                include_email_channel=settings.actions_daily_email_enabled,
            )
            if settings.actions_daily_email_enabled:
                logger.info("daily_email_requested action=%s correlation_id=%s", action_name, parent.correlation_id)
            publish_results = await asyncio.to_thread(
                _publish_daily_outputs,
                notify=notify,
                action_name=action_name,
                title=title,
                preview_text=preview_text,
                full_text=full_text,
                notify_req=notify_req,
                correlation_id=str(parent.correlation_id),
            )
            accepted = publish_results.get("generic")
            chat_message_accepted = publish_results.get("chat_message")

            dt_ms = int((time.monotonic() - t0) * 1000)
            extra = {
                "duration_ms": dt_ms,
                "generic_notify_ok": accepted.ok if accepted else None,
                "generic_notify_status": accepted.status if accepted else None,
                "generic_notification_id": str(accepted.notification_id) if accepted and accepted.notification_id else None,
                "chat_message_ok": chat_message_accepted.ok if chat_message_accepted else None,
                "chat_message_notification_id": str(chat_message_accepted.notification_id) if chat_message_accepted and chat_message_accepted.notification_id else None,
                "parse_retry_used": parse_retry_used,
                "llm_finish_reason": daily_llm_diag.get("finish_reason"),
                "llm_stop_reason": daily_llm_diag.get("stop_reason"),
                "llm_model": daily_llm_diag.get("model"),
                "plan_result_status": (plan_result_payload.get("result") or {}).get("status")
                if isinstance(plan_result_payload, dict)
                else None,
            }

            generic_ok = accepted.ok if accepted else True
            chat_ok = chat_message_accepted.ok if chat_message_accepted else True

            if generic_ok and chat_ok:
                deduper.mark_done(dedupe_key)
                await _audit(parent, status="completed", event_id=dedupe_key, action_name=action_name, extra=extra)
            else:
                reason = None
                if accepted and not accepted.ok:
                    reason = accepted.detail
                elif chat_message_accepted and not chat_message_accepted.ok:
                    reason = chat_message_accepted.detail
                await _audit(parent, status="failed", event_id=dedupe_key, action_name=action_name, reason=reason, extra=extra)

        except Exception as exc:
            dt_ms = int((time.monotonic() - t0) * 1000)
            await _audit(
                parent,
                status="failed",
                event_id=dedupe_key,
                action_name=action_name,
                reason=str(exc),
                extra={"duration_ms": dt_ms},
            )
            logger.exception("Daily action failed action=%s dedupe=%s", action_name, dedupe_key)
        finally:
            if acquired:
                sem.release()
            deduper.release(dedupe_key)

    async def _handle_collapse(env: BaseEnvelope) -> None:
        try:
            entry = CollapseMirrorEntryV2.model_validate(env.payload)
        except Exception:
            return

        event_id = dedupe_key_for(entry, env)
        if not should_trigger(entry):
            await _audit(env, status="skipped", event_id=event_id, action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1, reason="observer_not_juniper")
            return
        if not deduper.try_acquire(event_id):
            await _audit(env, status="skipped", event_id=event_id, action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1, reason="deduped")
            return

        acquired = False
        t0 = time.monotonic()
        try:
            await sem.acquire()
            acquired = True
            req_env = build_cortex_orch_envelope(
                env,
                source=src,
                entry=entry,
                session_id=settings.actions_session_id,
                recipient_group=settings.actions_recipient_group,
                dedupe_key=event_id,
                dedupe_window_seconds=settings.actions_notify_dedupe_window_seconds,
                recall_profile=settings.actions_recall_profile,
                verb=settings.actions_verb,
            )
            await dispatch_cortex_request(
                bus=hunter.bus,
                channel=settings.cortex_request_channel,
                envelope=req_env,
            )

            dt_ms = int((time.monotonic() - t0) * 1000)
            deduper.mark_done(event_id)
            await _audit(
                env,
                status="dispatched",
                event_id=event_id,
                action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
                extra={
                    "duration_ms": dt_ms,
                    "verb": settings.actions_verb,
                    "channel": settings.cortex_request_channel,
                },
            )
            logger.info(
                "dispatched cortex.orch.request verb=%s event_id=%s corr=%s",
                settings.actions_verb,
                event_id,
                env.correlation_id,
            )

        except Exception as exc:
            dt_ms = int((time.monotonic() - t0) * 1000)
            await _audit(
                env,
                status="failed",
                event_id=event_id,
                action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
                reason=str(exc),
                extra={"duration_ms": dt_ms},
            )
            logger.exception("Collapse action failed event_id=%s corr=%s", event_id, env.correlation_id)
        finally:
            if acquired:
                sem.release()
            deduper.release(event_id)

    async def _handle_manual_daily(env: BaseEnvelope, *, action_name: str):
        payload = env.payload if isinstance(env.payload, dict) else {}
        override_date = payload.get("date") if isinstance(payload.get("date"), str) else settings.actions_daily_run_once_date
        window = build_daily_window(tz_name=settings.actions_daily_timezone, override_date=override_date)
        dedupe_key = _daily_pulse_dedupe_key(window) if action_name == ACTION_DAILY_PULSE_V1 else _daily_metacog_dedupe_key(window)
        await _execute_daily(env, action_name=action_name, window=window, dedupe_key=dedupe_key)

    async def _handle_journal_manual(env: BaseEnvelope) -> None:
        payload = env.payload if isinstance(env.payload, dict) else {}
        trigger: JournalTriggerV1 | None = None
        try:
            trigger = JournalTriggerV1.model_validate(payload)
        except Exception:
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                await _audit(env, status="skipped", event_id=str(env.correlation_id), action_name="journal.manual", reason="missing_summary")
                return
            trigger = build_manual_trigger(
                summary=summary,
                prompt_seed=payload.get("prompt_seed"),
                source_ref=payload.get("source_ref"),
            )
        await _dispatch_journal(
            env,
            trigger=trigger,
            audit_action="journal.manual",
            dedupe_key=cooldown_key_for_trigger(trigger),
        )

    async def _handle_journal_notify(env: BaseEnvelope) -> None:
        try:
            record = NotificationRecord.model_validate(env.payload)
        except Exception:
            return
        if record.event_kind != "orion.digest.daily":
            return
        trigger = build_notify_summary_trigger(record)
        await _dispatch_journal(
            env,
            trigger=trigger,
            audit_action="journal.notify_summary",
            dedupe_key=cooldown_key_for_trigger(trigger),
        )

    async def _handle_journal_metacog(env: BaseEnvelope) -> None:
        try:
            trigger_payload = MetacogTriggerV1.model_validate(env.payload)
        except Exception:
            return
        trigger = build_metacog_trigger(trigger_payload)
        await _dispatch_journal(
            env,
            trigger=trigger,
            audit_action="journal.metacog_digest",
            dedupe_key=cooldown_key_for_trigger(trigger),
        )

    async def _handle_journal_collapse_stored(env: BaseEnvelope) -> bool:
        try:
            stored = CollapseMirrorStoredV1.model_validate(env.payload)
        except Exception:
            return False
        if not should_journal_from_collapse(stored.is_causally_dense, dense_only=settings.actions_journaling_collapse_dense_only):
            await _audit(
                env,
                status="skipped",
                event_id=str(stored.mirror_id or env.correlation_id),
                action_name="journal.collapse_response",
                reason="collapse_not_dense",
            )
            return True
        trigger = build_collapse_stored_trigger(stored)
        await _dispatch_journal(
            env,
            trigger=trigger,
            audit_action="journal.collapse_response",
            dedupe_key=cooldown_key_for_trigger(trigger),
        )
        return True

    async def _dispatch_scheduled_workflow(claimed: ClaimedSchedule) -> None:
        entry = claimed.schedule
        workflow_request = dict(entry.workflow_request or {})
        policy = dict(workflow_request.get("execution_policy") or {})
        policy["invocation_mode"] = "immediate"
        workflow_request["execution_policy"] = policy
        workflow_request["scheduled_dispatch"] = {
            "request_id": entry.request_id,
            "source": "orion-actions",
            "scheduled_label": (entry.next_run_at.isoformat() if entry.next_run_at else "scheduled"),
        }
        env = build_skill_cortex_orch_envelope(
            BaseEnvelope(kind=WORKFLOW_TRIGGER_KIND, source=src, correlation_id=str(uuid4()), payload={}),
            source=src,
            verb="chat_general",
            session_id=entry.execution_policy.session_id or settings.actions_session_id,
            user_id=entry.execution_policy.origin_user_id or settings.actions_recipient_group,
            metadata={
                "workflow_request": workflow_request,
                "workflow_dispatch_source": "orion-actions-scheduler",
            },
            options={"source": "orion-actions", "policy_dispatch_only": True},
            recall_enabled=False,
        )
        payload = env.payload.model_dump(mode="json") if hasattr(env.payload, "model_dump") else dict(env.payload or {})
        payload["verb"] = None
        payload["route_intent"] = "none"
        payload["mode"] = "brain"
        await dispatch_cortex_request(
            bus=hunter.bus,
            channel=settings.cortex_request_channel,
            envelope=env.model_copy(update={"payload": payload}),
        )

    async def _handle_workflow_schedule(env: BaseEnvelope) -> None:
        try:
            request = WorkflowDispatchRequestV1.model_validate(env.payload)
        except Exception:
            await _audit(env, status="failed", event_id=str(env.correlation_id), action_name=ACTION_WORKFLOW_SCHEDULE_V1, reason="invalid_workflow_dispatch_payload")
            return
        logger.info(
            "actions_workflow_registration corr=%s workflow_id=%s invocation_mode=%s schedule_kind=%s",
            env.correlation_id,
            request.workflow_id,
            request.execution_policy.invocation_mode,
            request.execution_policy.schedule.kind if request.execution_policy.schedule else None,
        )
        entry = workflow_schedule_store.upsert_from_dispatch(request)
        if entry is None:
            logger.info(
                "actions_workflow_registration_result corr=%s workflow_id=%s status=failed reason=invalid_schedule",
                env.correlation_id,
                request.workflow_id,
            )
            await _audit(env, status="failed", event_id=request.request_id, action_name=ACTION_WORKFLOW_SCHEDULE_V1, reason="invalid_schedule")
            return
        logger.info(
            "actions_workflow_registration_result corr=%s workflow_id=%s status=scheduled schedule_id=%s next_run_utc=%s",
            env.correlation_id,
            request.workflow_id,
            entry.schedule_id,
            entry.next_run_at.isoformat() if entry.next_run_at else None,
        )
        await _audit(
            env,
            status="scheduled",
            event_id=request.request_id,
            action_name=ACTION_WORKFLOW_SCHEDULE_V1,
            extra={
                "workflow_id": request.workflow_id,
                "schedule_id": entry.schedule_id,
                "notify_on": request.execution_policy.notify_on,
                "next_run_utc": entry.next_run_at.isoformat() if entry.next_run_at else None,
            },
        )

    async def _reply_management(env: BaseEnvelope, response: WorkflowScheduleManageResponseV1) -> None:
        if not env.reply_to:
            return
        reply = env.derive_child(kind=WORKFLOW_MANAGE_KIND, source=src, payload=response.model_dump(mode="json"), reply_to=None)
        await hunter.bus.publish(str(env.reply_to), reply)

    async def _handle_workflow_management(env: BaseEnvelope) -> None:
        try:
            request = WorkflowScheduleManageRequestV1.model_validate(env.payload)
        except Exception:
            response = WorkflowScheduleManageResponseV1(
                ok=False,
                operation="list",
                request_id=str(env.correlation_id),
                message="invalid_workflow_management_payload",
                error_code="invalid_management_payload",
            )
            await _reply_management(env, response)
            return
        response = workflow_schedule_store.apply_management(request)
        await _audit(
            env,
            status="completed" if response.ok else "failed",
            event_id=request.request_id,
            action_name=ACTION_WORKFLOW_MANAGE_V1,
            reason=None if response.ok else response.message,
            extra={"operation": request.operation, "ambiguous": response.ambiguous, "schedule_count": len(response.schedules)},
        )
        await _reply_management(env, response)

    async def handle_envelope(env: BaseEnvelope) -> None:
        kind = str(env.kind or "")
        if kind == WORKFLOW_TRIGGER_KIND:
            await _handle_workflow_schedule(env)
            return
        if kind == WORKFLOW_MANAGE_KIND:
            await _handle_workflow_management(env)
            return
        if kind == "orion.actions.trigger.daily_pulse.v1":
            await _handle_manual_daily(env, action_name=ACTION_DAILY_PULSE_V1)
            return
        if kind == "orion.actions.trigger.daily_metacog.v1":
            await _handle_manual_daily(env, action_name=ACTION_DAILY_METACOG_V1)
            return
        if kind == "orion.actions.trigger.journal.v1":
            await _handle_journal_manual(env)
            return
        if kind == "notify.notification.request.v1":
            await _handle_journal_notify(env)
            return
        if kind == "orion.metacog.trigger.v1":
            await _handle_journal_metacog(env)
            return
        if kind == "collapse.mirror.stored.v1":
            if await _handle_journal_collapse_stored(env):
                return

        # fallback by payload shape/channel drift: collapse flow stays default
        await _handle_collapse(env)

    async def _dispatch_scheduled_skill(parent: BaseEnvelope, *, verb_name: str, wait_for_result: bool, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        env = build_skill_cortex_orch_envelope(
            parent,
            source=src,
            verb=verb_name,
            session_id=settings.actions_session_id,
            user_id=settings.actions_recipient_group,
            metadata=metadata or {},
            options={"timeout_sec": float(settings.actions_exec_timeout_seconds)},
            recall_enabled=False,
        )
        await _audit(parent, status="dispatched", event_id=str(parent.correlation_id), action_name=verb_name, extra={"channel": settings.cortex_request_channel})
        if not wait_for_result:
            await dispatch_cortex_request(bus=hunter.bus, channel=settings.cortex_request_channel, envelope=env)
            return {}
        reply_channel = new_reply_channel("orion:cortex:result")
        rpc_env = env.model_copy(update={"reply_to": reply_channel})
        msg = await hunter.bus.rpc_request(
            settings.cortex_request_channel,
            rpc_env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.actions_exec_timeout_seconds),
        )
        decoded = hunter.bus.codec.decode(msg.get("data"))
        if not decoded.ok or decoded.envelope is None:
            raise RuntimeError(f"cortex_orch_decode_failed:{decoded.error}")
        orch_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        return _extract_skill_result_from_orch(orch_payload)

    async def _scheduler_loop() -> None:
        nonlocal last_skill_run_monotonic, last_journal_run
        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                forced_date = settings.actions_daily_run_once_date
                now_monotonic = time.monotonic()
                if settings.actions_skills_scheduler_enabled and should_run_interval(
                    now_monotonic=now_monotonic,
                    last_run_monotonic=last_skill_run_monotonic,
                    interval_seconds=settings.actions_skills_interval_seconds,
                    run_on_startup=settings.actions_skills_run_on_startup,
                ):
                    skill_parent = BaseEnvelope(kind="orion.actions.trigger.skills.v1", source=src, payload={"scheduled": True})
                    wait_for_result = bool(settings.actions_skills_notify_enabled)
                    biometrics_result = await _dispatch_scheduled_skill(skill_parent, verb_name=SKILL_BIOMETRICS_SNAPSHOT_V1, wait_for_result=wait_for_result, metadata={"schedule": "periodic_skills"})
                    gpu_result = await _dispatch_scheduled_skill(skill_parent, verb_name=SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1, wait_for_result=wait_for_result, metadata={"schedule": "periodic_skills"})
                    if wait_for_result:
                        findings = _scheduler_threshold_findings(biometrics_snapshot=biometrics_result, gpu_snapshot=gpu_result)
                        if findings:
                            await _audit(skill_parent, status="threshold_detected", event_id=str(skill_parent.correlation_id), action_name="skills.periodic.thresholds", extra={"findings": findings})
                            notify_parent = BaseEnvelope(kind="orion.actions.trigger.skills.notify.v1", source=src, payload={"findings": findings})
                            await _dispatch_scheduled_skill(
                                notify_parent,
                                verb_name=SKILL_NOTIFY_CHAT_MESSAGE_V1,
                                wait_for_result=False,
                                metadata={"skill_args": _threshold_notify_skill_args(findings=findings, correlation_id=str(notify_parent.correlation_id))},
                            )
                    last_skill_run_monotonic = now_monotonic

                pulse_should_run, local_date = should_run_daily(
                    now_utc=now_utc,
                    tz_name=settings.actions_daily_timezone,
                    hour_local=settings.actions_daily_pulse_hour_local,
                    minute_local=settings.actions_daily_pulse_minute_local,
                    last_ran_date=last_daily_run.get(ACTION_DAILY_PULSE_V1),
                )
                if settings.actions_daily_pulse_enabled and (pulse_should_run or (settings.actions_daily_run_on_startup and ACTION_DAILY_PULSE_V1 not in last_daily_run)):
                    window = build_daily_window(now_utc=now_utc, tz_name=settings.actions_daily_timezone, override_date=forced_date)
                    key = _daily_pulse_dedupe_key(window)
                    env = BaseEnvelope(kind="orion.actions.trigger.daily_pulse.v1", source=src, payload={"date": window.request_date})
                    await _execute_daily(env, action_name=ACTION_DAILY_PULSE_V1, window=window, dedupe_key=key)
                    last_daily_run[ACTION_DAILY_PULSE_V1] = local_date

                world_pulse_should_run, world_pulse_date = should_run_daily(
                    now_utc=now_utc,
                    tz_name=settings.actions_daily_timezone,
                    hour_local=settings.actions_world_pulse_hour_local,
                    minute_local=settings.actions_world_pulse_minute_local,
                    last_ran_date=last_daily_run.get("world_pulse"),
                )
                if settings.actions_world_pulse_enabled and (
                    world_pulse_should_run or (settings.actions_daily_run_on_startup and "world_pulse" not in last_daily_run)
                ):
                    window = build_daily_window(now_utc=now_utc, tz_name=settings.actions_daily_timezone, override_date=forced_date)
                    dedupe_key = _world_pulse_daily_dedupe_key(window)
                    if not deduper.try_acquire(dedupe_key):
                        logger.info("world_pulse_scheduler_trigger_result date=%s status=deduped", window.request_date)
                    else:
                        ok = await asyncio.to_thread(_trigger_world_pulse_run, date=window.request_date, requested_by="scheduler")
                        logger.info("world_pulse_scheduler_trigger_result date=%s ok=%s", window.request_date, ok)
                        if ok:
                            deduper.mark_done(dedupe_key)
                            last_daily_run["world_pulse"] = world_pulse_date
                        else:
                            deduper.release(dedupe_key)

                meta_should_run, meta_local_date = should_run_daily(
                    now_utc=now_utc,
                    tz_name=settings.actions_daily_timezone,
                    hour_local=settings.actions_daily_metacog_hour_local,
                    minute_local=settings.actions_daily_metacog_minute_local,
                    last_ran_date=last_daily_run.get(ACTION_DAILY_METACOG_V1),
                )
                if settings.actions_daily_metacog_enabled and (meta_should_run or (settings.actions_daily_run_on_startup and ACTION_DAILY_METACOG_V1 not in last_daily_run)):
                    window = build_daily_window(now_utc=now_utc, tz_name=settings.actions_daily_timezone, override_date=forced_date)
                    key = _daily_metacog_dedupe_key(window)
                    env = BaseEnvelope(kind="orion.actions.trigger.daily_metacog.v1", source=src, payload={"date": window.request_date})
                    await _execute_daily(env, action_name=ACTION_DAILY_METACOG_V1, window=window, dedupe_key=key)
                    last_daily_run[ACTION_DAILY_METACOG_V1] = meta_local_date

                journal_should_run, journal_local_date = should_run_daily(
                    now_utc=now_utc,
                    tz_name=settings.actions_daily_timezone,
                    hour_local=settings.actions_daily_pulse_hour_local,
                    minute_local=settings.actions_daily_pulse_minute_local,
                    last_ran_date=last_journal_run,
                )
                if settings.actions_journaling_enabled and settings.actions_journaling_daily_enabled and journal_should_run:
                    window = build_daily_window(now_utc=now_utc, tz_name=settings.actions_daily_timezone, override_date=forced_date)
                    # Keep timezone in scheduler logic, but avoid injecting it into journal prompts
                    # where location inferences (e.g., "Denver") can be overfit by the model.
                    journal_seed = json.dumps(
                        {
                            "request_date": window.request_date,
                            "window_start_utc": window.window_start_utc,
                            "window_end_utc": window.window_end_utc,
                        },
                        sort_keys=True,
                    )
                    trigger = build_scheduler_trigger(
                        summary=f"Daily journal cadence for {window.request_date}.",
                        prompt_seed=journal_seed,
                        source_ref=window.request_date,
                    )
                    env = BaseEnvelope(kind="orion.actions.trigger.journal.v1", source=src, payload=trigger.model_dump(mode="json"))
                    await _dispatch_journal(
                        env,
                        trigger=trigger,
                        audit_action="journal.daily_summary",
                        dedupe_key=_journal_daily_dedupe_key(window),
                    )
                    last_journal_run = journal_local_date

                for claimed in workflow_schedule_store.claim_due(now_utc=now_utc, limit=settings.actions_workflow_schedule_claim_batch_size):
                    dispatch_env = BaseEnvelope(kind=WORKFLOW_TRIGGER_KIND, source=src, correlation_id=str(uuid4()), payload={})
                    try:
                        await _dispatch_scheduled_workflow(claimed)
                        await _audit(
                            dispatch_env,
                            status="dispatched",
                            event_id=claimed.schedule.request_id,
                            action_name="workflow.dispatch.v1",
                            extra={
                                "schedule_id": claimed.schedule.schedule_id,
                                "workflow_id": claimed.schedule.workflow_id,
                                "notify_on": claimed.schedule.execution_policy.notify_on,
                                "next_run_utc": claimed.schedule.next_run_at.isoformat() if claimed.schedule.next_run_at else None,
                            },
                        )
                    except Exception as exc:
                        workflow_schedule_store.mark_dispatch_failed(run_id=claimed.run.run_id, schedule_id=claimed.schedule.schedule_id, error=str(exc), now_utc=now_utc)
                        raise
                signals = workflow_schedule_store.evaluate_attention_signals(
                    now_utc=now_utc,
                    overdue_min_seconds=settings.actions_workflow_attention_overdue_min_seconds,
                    reminder_cooldown_seconds=settings.actions_workflow_attention_reminder_cooldown_seconds,
                )
                for signal in signals:
                    await _publish_workflow_attention_signal(signal=signal, notify=notify)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("daily scheduler iteration failed")

            await asyncio.sleep(45)

    patterns = settings.subscribe_patterns()
    if WORKFLOW_TRIGGER_CHANNEL not in patterns:
        patterns.append(WORKFLOW_TRIGGER_CHANNEL)
    if WORKFLOW_MANAGE_CHANNEL not in patterns:
        patterns.append(WORKFLOW_MANAGE_CHANNEL)
    hunter = Hunter(_cfg(), patterns=patterns, handler=handle_envelope)

    logger.info(
        "Starting orion-actions Hunter channels=%s bus=%s cortex_request=%s",
        patterns,
        settings.orion_bus_url,
        settings.cortex_request_channel,
    )
    logger.info("actions_runtime_identity %s", json.dumps(_runtime_identity(), sort_keys=True))

    hunter_task = asyncio.create_task(hunter.start(), name="orion-actions-hunter")
    scheduler_task = asyncio.create_task(_scheduler_loop(), name="orion-actions-scheduler")

    yield

    for t in (hunter_task, scheduler_task):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.service_name,
        "version": settings.service_version,
        "node": settings.node_name,
    }


@app.get("/info")
async def info() -> dict:
    return _runtime_identity()


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO))
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
