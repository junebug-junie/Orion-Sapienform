"""Bus consumer for ``orion:curiosity:self_inquiry:request`` (2026-10-30).

cortex-orch's ``self_review`` workflow ("reflect on yourself") publishes one
``CuriositySelfInquiryRequestV1`` here. Hub answers by taking one self-inquiry turn,
``CuriosityInvestigation.tick_self_inquiry(force=True)`` -- the same call as
``POST /api/self-inquiry/run-now``. The force skips only the line's cooldown and daily cap;
``enabled`` and every health gate still apply, and the run counts against the budget.

This replaced the self-study reflection loop the phrase used to run (retired with the
fold of self-study into curiosity). Invalid payloads are logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.schemas.curiosity_self_inquiry_request import (
    SELF_INQUIRY_REQUEST_CHANNEL,
    CuriositySelfInquiryRequestV1,
)

logger = logging.getLogger("orion-hub.curiosity_self_inquiry_request")

RESUBSCRIBE_DELAY_SEC = 5.0


async def handle_self_inquiry_request(bus: Any, investigation: Any, msg: dict[str, Any]) -> str | None:
    """Returns the tick's block reason (None when it ran), or "invalid" for a bad payload."""
    try:
        decoded = bus.codec.decode(msg.get("data"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_inquiry_request_invalid err=undecodable: %s", exc)
        return "invalid"
    if not decoded.ok:
        logger.warning("self_inquiry_request_invalid err=undecodable: %s", getattr(decoded, "error", ""))
        return "invalid"
    try:
        request = CuriositySelfInquiryRequestV1.model_validate(decoded.envelope.payload or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_inquiry_request_invalid err=%s", str(exc)[:500])
        return "invalid"
    logger.warning(
        "curiosity_self_inquiry_requested_by_bus requested_by=%s corr=%s",
        request.requested_by,
        request.correlation_id,
    )
    reason = await investigation.tick_self_inquiry(force=True)
    logger.info(
        "self_inquiry_request_handled requested_by=%s corr=%s blocked=%s",
        request.requested_by,
        request.correlation_id,
        reason,
    )
    return reason


async def _serve(bus: Any, investigation: Any, msg: dict[str, Any]) -> None:
    try:
        await handle_self_inquiry_request(bus, investigation, msg)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("self_inquiry_request_failed")


async def self_inquiry_request_loop(bus: Any, investigation: Any) -> None:
    """A self-inquiry turn runs ~20 minutes, so each request is served as its own task;
    resubscribe after a bus failure rather than going deaf."""
    tasks: set[asyncio.Task] = set()
    try:
        while True:
            try:
                async with bus.subscribe(SELF_INQUIRY_REQUEST_CHANNEL) as pubsub:
                    logger.info("self_inquiry_request_loop subscribed channel=%s", SELF_INQUIRY_REQUEST_CHANNEL)
                    async for msg in bus.iter_messages(pubsub):
                        task = asyncio.create_task(_serve(bus, investigation, msg))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("self_inquiry_request_loop_failed; resubscribing in %ss", RESUBSCRIBE_DELAY_SEC)
            await asyncio.sleep(RESUBSCRIBE_DELAY_SEC)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
