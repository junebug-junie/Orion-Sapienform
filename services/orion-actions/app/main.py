from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.notify.client import NotifyClient

from .settings import settings
from .logic import (
    ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
    ActionDedupe,
    build_audit_envelope,
    build_llm_envelope,
    build_notify_request,
    build_recall_envelope,
    decode_llm_result,
    decode_recall_reply,
    dedupe_key_for,
    extract_message_sections,
    new_reply_channel,
    should_trigger,
)

logger = logging.getLogger("orion-actions")


def _ensure_logging() -> None:
    """Ensure app logs show up under uvicorn.

    Uvicorn config often does not configure the root logger level/handlers
    for non-uvicorn namespaces, so we make sure our logger propagates
    to stdout at the configured level.
    """
    level_name = (settings.log_level or os.getenv('LOG_LEVEL') or 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(name)s %(levelname)s - %(message)s',
        )
    else:
        root.setLevel(level)
    logger.setLevel(level)


_ensure_logging()


def _cfg() -> ChassisConfig:
    # Ensure catalog enforcement is applied for the chassis-created bus.
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    deduper = ActionDedupe(ttl_seconds=settings.actions_dedupe_ttl_seconds)
    sem = asyncio.Semaphore(max(1, int(settings.actions_max_concurrency)))
    notify = NotifyClient(
        base_url=settings.notify_url,
        api_token=settings.notify_api_token,
        timeout=10,
    )

    src = _source_ref()

    async def _audit(parent: BaseEnvelope, *, status: str, event_id: str, reason: str | None = None, extra: dict | None = None):
        try:
            env = build_audit_envelope(
                parent,
                source=src,
                status=status,
                action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
                event_id=event_id,
                reason=reason,
                extra=extra,
            )
            await hunter.bus.publish(settings.actions_audit_channel, env)
        except Exception:
            # audit must never take down the pipeline
            logger.debug("audit publish failed", exc_info=True)

    async def handle_envelope(env: BaseEnvelope) -> None:
        # Kind drift tolerant: subscribe by channel and validate payload.
        try:
            entry = CollapseMirrorEntryV2.model_validate(env.payload)
        except Exception as exc:
            logger.debug("Ignoring non-collapse payload: %s", exc)
            return

        event_id = dedupe_key_for(entry, env)

        logger.info("Collapse triage received event_id=%s observer=%s corr=%s", event_id, getattr(entry, "observer", None), env.correlation_id)

        if not should_trigger(entry):
            logger.info("Skip event_id=%s reason=observer_not_juniper observer=%s", event_id, getattr(entry, "observer", None))
            await _audit(env, status="skipped", event_id=event_id, reason="observer_not_juniper")
            return

        if not deduper.try_acquire(event_id):
            logger.info("Deduped event_id=%s (ttl=%ss)", event_id, settings.actions_dedupe_ttl_seconds)
            await _audit(env, status="skipped", event_id=event_id, reason="deduped")
            return

        acquired = False
        t0 = time.monotonic()
        try:
            await sem.acquire()
            acquired = True

            await _audit(env, status="started", event_id=event_id)

            # ── Recall RPC ──────────────────────────────────────────────
            recall_reply = new_reply_channel("orion:exec:result:RecallService")
            recall_env = build_recall_envelope(
                env,
                source=src,
                entry=entry,
                reply_to=recall_reply,
                profile=settings.actions_recall_profile,
                session_id=settings.actions_session_id,
                node_id=settings.node_name,
            )
            recall_msg = await hunter.bus.rpc_request(
                settings.recall_rpc_channel,
                recall_env,
                reply_channel=recall_reply,
                timeout_sec=float(settings.actions_recall_timeout_seconds),
            )
            recall_decoded = hunter.bus.codec.decode(recall_msg.get("data"))
            if not recall_decoded.ok or recall_decoded.envelope is None:
                raise RuntimeError(f"recall_decode_failed:{recall_decoded.error}")
            recall_payload = recall_decoded.envelope.payload if isinstance(recall_decoded.envelope.payload, dict) else {}
            if recall_payload.get("error"):
                logger.warning("Recall returned error: %s", recall_payload.get("error"))
            memory_rendered = decode_recall_reply(recall_payload)

            # ── LLM RPC ────────────────────────────────────────────────
            llm_reply = new_reply_channel("orion:exec:result:LLMGatewayService")
            llm_env = build_llm_envelope(
                env,
                source=src,
                entry=entry,
                memory_rendered=memory_rendered,
                reply_to=llm_reply,
                route=settings.actions_llm_route,
            )
            llm_msg = await hunter.bus.rpc_request(
                settings.llm_rpc_channel,
                llm_env,
                reply_channel=llm_reply,
                timeout_sec=float(settings.actions_llm_timeout_seconds),
            )
            llm_decoded = hunter.bus.codec.decode(llm_msg.get("data"))
            if not llm_decoded.ok or llm_decoded.envelope is None:
                raise RuntimeError(f"llm_decode_failed:{llm_decoded.error}")
            llm_payload = llm_decoded.envelope.payload if isinstance(llm_decoded.envelope.payload, dict) else {}
            if llm_payload.get("error"):
                raise RuntimeError(f"llm_error:{llm_payload.get('error')}")
            llm_text = decode_llm_result(llm_payload)
            introspect_text, message_text = extract_message_sections(llm_text)

            # ── Notify actuation (via /notify) ─────────────────────────
            notify_req = build_notify_request(
                source_service=settings.service_name,
                recipient_group=settings.actions_recipient_group,
                session_id=settings.actions_session_id,
                correlation_id=str(env.correlation_id),
                dedupe_key=event_id,
                dedupe_window_seconds=settings.actions_notify_dedupe_window_seconds,
                entry=entry,
                action_name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
                introspect_text=introspect_text,
                message_text=message_text,
            )

            logger.info("Notify send event_id=%s url=%s", event_id, settings.notify_url)
            accepted = await asyncio.to_thread(notify.send, notify_req)
            logger.info("Notify result event_id=%s ok=%s status=%s id=%s", event_id, accepted.ok, accepted.status, accepted.notification_id)

            dt_ms = int((time.monotonic() - t0) * 1000)
            extra = {
                "duration_ms": dt_ms,
                "notify_ok": accepted.ok,
                "notify_status": accepted.status,
                "notification_id": str(accepted.notification_id) if accepted.notification_id else None,
            }

            if accepted.ok:
                await _audit(env, status="completed", event_id=event_id, extra=extra)
                deduper.mark_done(event_id)
            else:
                await _audit(env, status="failed", event_id=event_id, reason=accepted.detail, extra=extra)

        except Exception as exc:
            dt_ms = int((time.monotonic() - t0) * 1000)
            logger.exception("Action failed event_id=%s corr=%s", event_id, env.correlation_id)
            await _audit(env, status="failed", event_id=event_id, reason=str(exc), extra={"duration_ms": dt_ms})
        finally:
            if acquired:
                sem.release()
            # If not marked done, release inflight so a future replay can retry.
            deduper.release(event_id)

    hunter = Hunter(
        _cfg(),
        pattern=settings.actions_subscribe_channel,
        handler=handle_envelope,
    )

    logger.info(
        "Starting orion-actions Hunter channel=%s bus=%s notify=%s",
        settings.actions_subscribe_channel,
        settings.orion_bus_url,
        settings.notify_url,
    )

    task = asyncio.create_task(hunter.start(), name="orion-actions-hunter")

    def _done(t: asyncio.Task) -> None:
        try:
            t.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Hunter task crashed")

    task.add_done_callback(_done)

    yield

    task.cancel()
    try:
        await task
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
