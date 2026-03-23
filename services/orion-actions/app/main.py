from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import uuid4
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import FastAPI

from orion.cognition.plan_loader import build_plan_for_verb
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
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

logger = logging.getLogger("orion-actions")

ACTION_DAILY_PULSE_V1 = "daily_pulse_v1"
ACTION_DAILY_METACOG_V1 = "daily_metacog_v1"


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
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("llm output must be a JSON object")
    return data


def _daily_pulse_dedupe_key(window: DailyWindow) -> str:
    return f"actions:daily_pulse:{window.request_date}:{settings.node_name}:{settings.actions_recipient_group}"


def _daily_metacog_dedupe_key(window: DailyWindow) -> str:
    return f"actions:daily_metacog:{window.request_date}:{settings.node_name}"


def _daily_notify_request(*, event_kind: str, title: str, dedupe_key: str, correlation_id: str, payload: dict[str, Any]) -> NotificationRequest:
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    preview = "; ".join([f"{k}: {v}" for k, v in list(payload.items())[:3]])[:280]
    return NotificationRequest(
        source_service=settings.service_name,
        event_kind=event_kind,
        severity="info",
        title=title,
        body_text=preview,
        body_md=f"## {title}\n\n```json\n{pretty}\n```\n",
        recipient_group=settings.actions_recipient_group,
        session_id=settings.actions_session_id,
        correlation_id=correlation_id,
        dedupe_key=dedupe_key,
        dedupe_window_seconds=int(settings.actions_notify_dedupe_window_seconds),
        tags=["actions", "daily", "json"],
        context={"payload": payload, "preview_text": preview},
    )


def _journal_daily_dedupe_key(window: DailyWindow) -> str:
    return f"actions:journal:daily:{window.request_date}:{settings.node_name}"


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
        req = PlanExecutionRequest(
            plan=plan,
            args=PlanExecutionArgs(
                request_id=str(parent.correlation_id),
                trigger_source=settings.service_name,
                user_id=settings.actions_recipient_group,
                extra={
                    "mode": "brain",
                    "session_id": settings.actions_session_id,
                    "verb": verb_name,
                    "trace_id": str(parent.correlation_id),
                },
            ),
            context=context,
        )
        reply_channel = new_reply_channel("orion:exec:result")
        req_env = parent.derive_child(kind=req.kind, source=src, payload=req, reply_to=reply_channel)
        msg = await hunter.bus.rpc_request(
            settings.cortex_exec_request_channel,
            req_env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.actions_exec_timeout_seconds),
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
        req = build_compose_request(
            trigger,
            session_id=settings.actions_journal_session_id,
            user_id=settings.actions_recipient_group,
            trace_id=str(parent.correlation_id),
            recall_profile=settings.actions_recall_profile,
            options={"timeout_sec": float(settings.actions_exec_timeout_seconds)},
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
            final_text, plan_result_payload = await _run_plan(parent, verb_name=action_name, context=context)
            parsed = _json_loads_strict(final_text)

            if action_name == ACTION_DAILY_PULSE_V1:
                model = DailyPulseV1.model_validate(parsed)
                event_kind = "orion.daily.pulse"
                title = "Orion — Daily Pulse"
            else:
                model = DailyMetacogV1.model_validate(parsed)
                event_kind = "orion.daily.metacog"
                title = "Orion — Daily Metacog"

            notify_req = _daily_notify_request(
                event_kind=event_kind,
                title=title,
                dedupe_key=dedupe_key,
                correlation_id=str(parent.correlation_id),
                payload=model.model_dump(mode="json"),
            )
            accepted = await asyncio.to_thread(notify.send, notify_req)
            dt_ms = int((time.monotonic() - t0) * 1000)
            extra = {
                "duration_ms": dt_ms,
                "notify_ok": accepted.ok,
                "notify_status": accepted.status,
                "notification_id": str(accepted.notification_id) if accepted.notification_id else None,
                "plan_result_status": (plan_result_payload.get("result") or {}).get("status")
                if isinstance(plan_result_payload, dict)
                else None,
            }

            if accepted.ok:
                deduper.mark_done(dedupe_key)
                await _audit(parent, status="completed", event_id=dedupe_key, action_name=action_name, extra=extra)
            else:
                await _audit(parent, status="failed", event_id=dedupe_key, action_name=action_name, reason=accepted.detail, extra=extra)

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

    async def handle_envelope(env: BaseEnvelope) -> None:
        kind = str(env.kind or "")
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
                    trigger = build_scheduler_trigger(
                        summary=f"Daily journal cadence for {window.request_date} in {window.timezone}.",
                        prompt_seed=json.dumps(window.__dict__, sort_keys=True),
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
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("daily scheduler iteration failed")

            await asyncio.sleep(45)

    hunter = Hunter(_cfg(), patterns=settings.subscribe_patterns(), handler=handle_envelope)

    logger.info(
        "Starting orion-actions Hunter channels=%s bus=%s cortex_request=%s",
        settings.subscribe_patterns(),
        settings.orion_bus_url,
        settings.cortex_request_channel,
    )

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


if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO))
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
