"""World-pulse run result -> the world-news journal, composed as an admitted durable run.

Before 2026-09-30 this composed the journal in-process with one direct cortex call; a busy fast GPU
lane at 06:00 local (``gpu_pool_unavailable:deadline``) failed it and nothing retried, so the daily
world-news email stopped after 2026-09-24. Now orion-actions only *submits* a ``journal.compose``
durable run (orion/schemas/journal_compose_run.py) through cortex-orch's durable ingress. The run
holds a GPU pool hold, so a busy pool is a wait (bounded by ``deadline_at`` = next local midnight,
so it can never spend tomorrow's one world_pulse email), and orion-durable-runs publishes the
journal write itself. The existing post-persist email path here is unchanged.

Idempotent by construction: run_id, correlation_id and entry_id are derived from the world-pulse
run_id, and the brief is a pure function of the run result, so a redelivered run result resubmits
the identical request -- durable-runs' store answers it with the existing run (ON CONFLICT), never
a second run or a second journal entry.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, time, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.journaler import JournalTriggerV1, build_world_pulse_reflective_trigger, world_pulse_curiosity_appendix
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.journal_compose_run import JOURNAL_COMPOSE_WORKFLOW, JournalComposeRunBriefV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.world_pulse import WorldPulseRunResultV1

from .settings import Settings

logger = logging.getLogger("orion-actions.world_pulse_journal")

AuditFn = Callable[..., Awaitable[None]]
# (request) -> None when durable-runs confirmed THIS run is registered, else why not.
SubmitFn = Callable[[DurableRunRequestV1], Awaitable[str | None]]
AUDIT_ACTION = "journal.world_pulse_digest"
RUN_ID_PREFIX = "world-pulse-journal:"
# Submission is one short receipt RPC (no GPU). Tries span ~6.5 minutes so a cortex-orch or
# durable-runs rebuild does not lose the day's journal. The run result arrives over pub/sub and is
# never redelivered, so if every try fails the journal for that run is lost -- audited as
# status=failed reason=durable_submit_failed:<why> and logged at ERROR, never silent.
SUBMIT_BACKOFF_SEC: tuple[float, ...] = (0.0, 10.0, 30.0, 90.0, 270.0)


def world_pulse_journal_skip_reason(
    result: WorldPulseRunResultV1,
    *,
    enabled: bool,
    allow_dry_run: bool = False,
) -> str | None:
    """Return skip reason, or None when journal dispatch should proceed."""
    if not enabled:
        return "world_pulse_journal_disabled"
    run = result.run
    if run.dry_run and not allow_dry_run:
        return "world_pulse_dry_run"
    if run.status not in {"completed", "partial"}:
        return f"world_pulse_run_status_{run.status}"
    if result.digest is None:
        return "world_pulse_missing_digest"
    return None


def build_world_pulse_journal_trigger(result: WorldPulseRunResultV1) -> JournalTriggerV1:
    return build_world_pulse_reflective_trigger(result)


def world_pulse_journal_run_id(world_pulse_run_id: str) -> str:
    return f"{RUN_ID_PREFIX}{world_pulse_run_id}"


def next_local_midnight(now: datetime, tz_name: str) -> datetime:
    tz = ZoneInfo(tz_name)
    local = now.astimezone(tz)
    return datetime.combine(local.date() + timedelta(days=1), time(0, 0), tzinfo=tz).astimezone(timezone.utc)


def build_world_pulse_journal_run_request(
    result: WorldPulseRunResultV1,
    *,
    settings: Settings,
    llm_route: str | None,
    now: datetime,
) -> DurableRunRequestV1:
    wp_run_id = result.run.run_id
    trigger = build_world_pulse_journal_trigger(result)
    route = llm_route or "quick_background"
    appendix, markers = world_pulse_curiosity_appendix(result)
    brief = JournalComposeRunBriefV1(
        trigger=trigger,
        entry_id=str(uuid5(NAMESPACE_URL, f"orion:journal:world_pulse_digest:{wp_run_id}")),
        author=settings.actions_journal_author,
        session_id=settings.actions_journal_session_id,
        user_id=settings.actions_recipient_group,
        recall_profile=settings.actions_journal_world_pulse_recall_profile,
        llm_route=route,
        timeout_sec=float(settings.actions_exec_timeout_seconds),
        body_appendix=appendix,
        body_appendix_markers=markers,
    )
    return DurableRunRequestV1(
        run_id=world_pulse_journal_run_id(wp_run_id),
        workflow=JOURNAL_COMPOSE_WORKFLOW,
        correlation_id=str(uuid5(NAMESPACE_URL, f"orion:world_pulse_journal_run:{wp_run_id}")),
        brief=brief,
        # The pool places it by route (gpu_pool.yaml routes: quick_background -> class fast,
        # on_unavailable: wait). Background priority; the deadline keeps it inside today's
        # email slot (durable-runs keeps the FIRST submission's deadline on a resubmit).
        admission=ResourceRequirementV1(
            resource=f"llm.route.{route}",
            preferred_lane=route,
            priority="background",
            deadline_at=next_local_midnight(now, settings.actions_daily_timezone),
        ),
    )


async def submit_durable_run_via_cortex(
    *,
    bus: Any,
    source: ServiceRef,
    request: DurableRunRequestV1,
    request_channel: str,
    timeout_sec: float = 20.0,
) -> str | None:
    """One kickoff RPC through cortex-orch's durable ingress (same shape as cortex-exec's
    durable_kickoff.submit_durable_run). None only when the receipt names THIS run, workflow and
    resource; otherwise the reason. Never raises."""
    reply_channel = f"orion:cortex:result:world-pulse-journal:{uuid4()}"
    payload = {
        "mode": "brain",
        "context": {
            "messages": [{"role": "user", "content": "Compose the world-pulse journal."}],
            "user_message": "Compose the world-pulse journal.",
            "session_id": request.brief.session_id,
            "metadata": {"durable_run": request.model_dump(mode="json", exclude_none=True)},
        },
    }
    envelope = BaseEnvelope(kind="cortex.orch.request", source=source, correlation_id=request.correlation_id,
                            reply_to=reply_channel, payload=payload)
    try:
        raw = await bus.rpc_request(request_channel, envelope, reply_channel=reply_channel, timeout_sec=timeout_sec)
        decoded = bus.codec.decode(raw.get("data") if isinstance(raw, dict) else raw)
        result = decoded.envelope.payload if decoded.ok and decoded.envelope is not None else None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:300]
    if not isinstance(result, dict):
        return "undecodable_reply"
    if result.get("status") != "accepted":
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        return f"not_accepted:{result.get('status') or 'no_status'}:{error.get('message') or error.get('type') or ''}"[:300]
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    receipt = metadata.get("durable_run") if isinstance(metadata.get("durable_run"), dict) else {}
    if receipt.get("run_id") != request.run_id:
        return "receipt_run_id_mismatch"
    if receipt.get("workflow_kind", receipt.get("workflow")) != request.workflow:
        return "receipt_workflow_mismatch"
    if request.admission is not None and receipt.get("requested_resource") != request.admission.resource:
        return "receipt_resource_mismatch"
    return None


async def handle_world_pulse_run_result_journal(
    env: BaseEnvelope,
    *,
    settings: Settings,
    submit: SubmitFn,
    audit: AuditFn,
    llm_route: str | None,
    now_fn: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> bool:
    try:
        result = WorldPulseRunResultV1.model_validate(env.payload)
    except Exception:
        await audit(
            env,
            status="skipped",
            event_id=str(env.correlation_id),
            action_name=AUDIT_ACTION,
            reason="invalid_run_result_payload",
        )
        return True

    skip = world_pulse_journal_skip_reason(
        result,
        enabled=settings.actions_world_pulse_journal_enabled,
        allow_dry_run=settings.actions_world_pulse_journal_allow_dry_run,
    )
    if skip == "world_pulse_journal_disabled":
        return True
    if skip is None and not settings.actions_journaling_enabled:
        skip = "journaling_disabled"
    if skip is not None:
        await audit(
            env,
            status="skipped",
            event_id=result.run.run_id,
            action_name=AUDIT_ACTION,
            reason=skip,
        )
        return True

    now = (now_fn or (lambda: datetime.now(timezone.utc)))()
    try:
        request = build_world_pulse_journal_run_request(result, settings=settings, llm_route=llm_route, now=now)
    except Exception as exc:  # noqa: BLE001 -- a bad request must still leave an audit row
        logger.exception("world_pulse_journal_request_build_failed run_id=%s", result.run.run_id)
        await audit(env, status="failed", event_id=world_pulse_journal_run_id(result.run.run_id),
                    action_name=AUDIT_ACTION, reason=f"request_build_failed:{type(exc).__name__}: {exc}"[:500])
        return True
    error: str | None = None
    for delay in SUBMIT_BACKOFF_SEC:
        if delay:
            await sleep(delay)
        error = await submit(request)
        if error is None:
            break
    if error is None:
        logger.info("world_pulse_journal_durable_submitted run_id=%s entry_id=%s deadline_at=%s",
                    request.run_id, request.brief.entry_id, request.admission.deadline_at)
        await audit(env, status="submitted", event_id=request.run_id, action_name=AUDIT_ACTION,
                    extra={"durable_run_id": request.run_id, "entry_id": request.brief.entry_id,
                           "deadline_at": request.admission.deadline_at.isoformat()})
    else:
        logger.error("world_pulse_journal_durable_submit_failed run_id=%s error=%s", request.run_id, error)
        await audit(env, status="failed", event_id=request.run_id, action_name=AUDIT_ACTION,
                    reason=f"durable_submit_failed:{error}")
    return True
