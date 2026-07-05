from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.harness_finalize import HarnessDraftMoleculeV1, SubstrateFinalizeAppraisalV1
from orion.substrate.appraisal.finalize_draft_v1 import (
    FinalizeDraftAppraisalError,
    appraise_draft_molecule,
)

from .settings import Settings, get_settings

logger = logging.getLogger("orion.substrate.runtime.finalize_appraisal")

_REQUEST_KIND = "harness.draft.molecule.v1"
_RESULT_KIND = "substrate.finalize.appraisal.v1"


def _source(settings: Settings) -> ServiceRef:
    return ServiceRef(
        name=settings.service_name,
        node=settings.node_name,
        version=settings.service_version,
    )


def _envelope_correlation_id(raw: str | UUID | None) -> UUID:
    if raw:
        try:
            return UUID(str(raw))
        except ValueError:
            pass
    return uuid4()


def _result_channel(
    settings: Settings,
    *,
    reply_to: str | None,
    correlation_id: str,
) -> str:
    if reply_to:
        return reply_to
    return f"{settings.channel_finalize_appraisal_result_prefix}{correlation_id}"


async def handle_finalize_appraisal_request(
    bus: OrionBusAsync,
    request: HarnessDraftMoleculeV1,
    *,
    reply_to: str,
    correlation_id: str | None = None,
    causality_chain: list[Any] | None = None,
    settings: Settings | None = None,
    parent_envelope: BaseEnvelope | None = None,
) -> SubstrateFinalizeAppraisalV1:
    """Appraise a draft molecule and publish SubstrateFinalizeAppraisalV1 on reply_to."""
    s = settings or get_settings()
    corr = correlation_id or request.correlation_id or str(uuid4())
    appraisal = appraise_draft_molecule(request)

    if parent_envelope is not None:
        envelope = parent_envelope.derive_child(
            kind=_RESULT_KIND,
            source=_source(s),
            payload=appraisal,
            reply_to=None,
        )
    else:
        envelope = BaseEnvelope(
            kind=_RESULT_KIND,
            source=_source(s),
            correlation_id=_envelope_correlation_id(corr),
            causality_chain=list(causality_chain or []),
            payload=appraisal.model_dump(mode="json"),
        )

    await bus.publish(reply_to, envelope)
    logger.info(
        "finalize_appraisal complete corr=%s reply=%s draft_hash=%s surprise=%.3f",
        corr,
        reply_to,
        appraisal.draft_hash,
        appraisal.surprise_level,
    )
    return appraisal


async def _handle_bus_message(bus: OrionBusAsync, raw_msg: dict[str, Any], settings: Settings) -> None:
    decoded = bus.codec.decode(raw_msg.get("data"))
    if not decoded.ok:
        logger.warning("finalize_appraisal decode failed: %s", decoded.error)
        return

    env = decoded.envelope
    reply_channel = env.reply_to or _result_channel(
        settings,
        reply_to=None,
        correlation_id=str(env.correlation_id or uuid4()),
    )
    corr = str(env.correlation_id or uuid4())
    causality = list(env.causality_chain or [])

    kind = env.kind or ""
    if kind not in (_REQUEST_KIND, "legacy.message"):
        logger.warning("finalize_appraisal unsupported kind=%s corr=%s", kind, corr)
        return

    try:
        request = HarnessDraftMoleculeV1.model_validate(env.payload or {})
        await handle_finalize_appraisal_request(
            bus,
            request,
            reply_to=reply_channel,
            correlation_id=corr,
            causality_chain=causality,
            settings=settings,
            parent_envelope=env,
        )
    except (FinalizeDraftAppraisalError, ValueError) as exc:
        logger.error("finalize_appraisal error corr=%s err=%s", corr, exc)
        err_envelope = env.derive_child(
            kind="system.error",
            source=_source(settings),
            payload={
                "error": "finalize_appraisal_failed",
                "details": str(exc),
                "correlation_id": corr,
            },
            reply_to=None,
        )
        await bus.publish(reply_channel, err_envelope)
    except Exception as exc:
        logger.exception("finalize_appraisal unhandled corr=%s err=%s", corr, exc)
        err_envelope = env.derive_child(
            kind="system.error",
            source=_source(settings),
            payload={
                "error": "finalize_appraisal_failed",
                "details": str(exc),
                "correlation_id": corr,
            },
            reply_to=None,
        )
        await bus.publish(reply_channel, err_envelope)


async def run_finalize_appraisal_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event | None = None,
    *,
    settings: Settings | None = None,
) -> None:
    """Subscribe to finalize_appraisal request channel and reply with substrate appraisal."""
    s = settings or get_settings()
    if not s.orion_bus_enabled:
        logger.info("Bus disabled; finalize_appraisal listener not started")
        return

    channel = s.channel_finalize_appraisal_request
    logger.info("finalize_appraisal listener subscribing channel=%s", channel)

    try:
        async with bus.subscribe(channel) as pubsub:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=1.2,
                    )
                except asyncio.TimeoutError:
                    continue
                if not msg or msg.get("type") not in ("message", "pmessage"):
                    continue
                try:
                    await _handle_bus_message(bus, msg, s)
                except Exception:
                    logger.exception("finalize_appraisal unhandled bus worker error")
    except asyncio.CancelledError:
        raise
    finally:
        logger.info("finalize_appraisal listener stopped channel=%s", channel)


async def start_finalize_appraisal_listener(
    bus: OrionBusAsync,
    stop_event: asyncio.Event,
    *,
    settings: Settings | None = None,
) -> asyncio.Task[None]:
    task = asyncio.create_task(
        run_finalize_appraisal_listener(bus, stop_event, settings=settings),
        name="substrate-finalize-appraisal-listener",
    )
    return task


async def stop_finalize_appraisal_listener(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
