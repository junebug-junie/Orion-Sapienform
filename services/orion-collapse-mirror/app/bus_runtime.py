from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import logging
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, Rabbit
from orion.schemas.collapse_mirror import CollapseMirrorEntry

from .settings import settings

logger = logging.getLogger("orion.collapse.mirror")


def _service_ref() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME or None,
    )


def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME or None,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=float(settings.HEARTBEAT_INTERVAL_SEC),
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=float(settings.SHUTDOWN_GRACE_SEC),
    )


def _wrap_entry_for_triage(entry: CollapseMirrorEntry) -> Dict[str, Any]:
    collapse_id = f"collapse_{uuid4().hex}"
    return {
        "id": collapse_id,
        "service_name": settings.SERVICE_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry.model_dump(mode="json"),
    }


async def handle_intake(env: BaseEnvelope) -> None:
    """Consumes collapse mirror intake messages and republishes normalized entries to triage."""
    # legacy senders: env.payload is the raw dict they published
    payload = env.payload
    if not isinstance(payload, dict):
        logger.warning("[collapse-mirror] intake payload is not a dict (kind=%s)", env.kind)
        return

    # allow pings
    if payload.get("kind") == "ping":
        return

    try:
        entry = CollapseMirrorEntry.model_validate(payload)
    except ValidationError as ve:
        logger.warning("[collapse-mirror] intake schema failed; dropping: %s", ve)
        return

    # Gate NOOPs
    if str(getattr(entry, "type", "")).strip().lower() == "noop":
        return

    enriched = _wrap_entry_for_triage(entry)
    # publish as legacy dict for now (downstream still expects it)
    await intake_hunter.bus.publish(settings.CHANNEL_COLLAPSE_TRIAGE, enriched)


async def handle_exec_step(env: BaseEnvelope) -> BaseEnvelope:
    """Handles exec-step calls from cortex-exec so CollapseMirrorService can be a plan step."""
    sref = _service_ref()
    started = datetime.now(timezone.utc)

    # env.payload is expected to be a dict shaped like the cortex exec_step payload.
    exec_payload = env.payload if isinstance(env.payload, dict) else {}

    # direct call path: context.collapse_entry
    ctx = exec_payload.get("context") if isinstance(exec_payload, dict) else None
    candidate: Optional[dict] = None
    reason: str = "unknown"

    if isinstance(ctx, dict) and isinstance(ctx.get("collapse_entry"), dict):
        candidate = ctx["collapse_entry"]
        reason = "context.collapse_entry"

    if not candidate:
        prior = exec_payload.get("prior_step_results")
        if isinstance(prior, list):
            for step in reversed(prior):
                if not isinstance(step, dict):
                    continue
                for svc in step.get("services", []) or []:
                    if not isinstance(svc, dict):
                        continue
                    svc_payload = svc.get("payload")
                    if isinstance(svc_payload, dict) and isinstance(svc_payload.get("collapse_entry"), dict):
                        candidate = svc_payload["collapse_entry"]
                        reason = "prior_step_results.collapse_entry"
                        break
                if candidate:
                    break

    if not candidate:
        return BaseEnvelope(
            kind="collapse_mirror.exec_step.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={
                "ok": False,
                "published": False,
                "reason": "no_candidate",
                "note": "Provide context.collapse_entry or set a prior step to output collapse_entry.",
            },
        )

    # Hydrate NOOPs to satisfy Strict Schema
    if str(candidate.get("type", "")).strip().lower() == "noop":
        candidate.setdefault("observer", "Orion")
        candidate.setdefault("trigger", "noop")
        candidate.setdefault("observer_state", ["idle"])
        candidate.setdefault("field_resonance", "none")
        candidate.setdefault("emergent_entity", "none")
        candidate.setdefault("summary", "No collapse detected.")
        candidate.setdefault("mantra", "void")

    try:
        entry = CollapseMirrorEntry.model_validate(candidate)
    except Exception as e:
        return BaseEnvelope(
            kind="collapse_mirror.exec_step.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={
                "ok": False,
                "published": False,
                "reason": f"schema_validation_failed:{type(e).__name__}",
                "details": str(e),
            },
        )

    if str(getattr(entry, "type", "")).strip().lower() == "noop":
        return BaseEnvelope(
            kind="collapse_mirror.exec_step.result",
            source=sref,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={
                "ok": True,
                "published": False,
                "reason": "noop",
                "entry": entry.model_dump(mode="json"),
            },
        )

    # Publish to intake (canonical path)
    await exec_rabbit.bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode="json"))

    elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    return BaseEnvelope(
        kind="collapse_mirror.exec_step.result",
        source=sref,
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        payload={
            "ok": True,
            "published": True,
            "published_to": settings.CHANNEL_COLLAPSE_INTAKE,
            "elapsed_ms": elapsed_ms,
            "reason": reason,
            "entry": entry.model_dump(mode="json"),
        },
    )


# Instances (wired during FastAPI lifespan / main())
intake_hunter: Hunter
exec_rabbit: Rabbit


def build_services() -> tuple[Rabbit, Hunter]:
    global intake_hunter, exec_rabbit

    cfg = _cfg()
    exec_channel = f"{settings.EXEC_REQUEST_PREFIX}:CollapseMirrorService"

    exec_rabbit = Rabbit(cfg, request_channel=exec_channel, handler=handle_exec_step)
    intake_hunter = Hunter(cfg, patterns=[settings.CHANNEL_COLLAPSE_INTAKE], handler=handle_intake)
    return exec_rabbit, intake_hunter


async def start_services(stop_event: asyncio.Event) -> tuple[Rabbit, Hunter]:
    rabbit, hunter = build_services()
    await rabbit.start_background(stop_event)
    await hunter.start_background(stop_event)
    return rabbit, hunter
