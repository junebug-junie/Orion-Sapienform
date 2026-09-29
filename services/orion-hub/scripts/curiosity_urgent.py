"""Bus consumer for urgent curiosity requests (``orion:curiosity:urgent:request``).

The Hub "Run urgent" button and Plan 4's hardware watcher both publish a
``CuriosityUrgentRequestV1`` here, so there is one path into
``CuriosityInvestigation.start_urgent``. Invalid payloads are logged and dropped;
one that still carries a well-formed ``incident_id`` also gets a failed report.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, CuriosityUrgentRequestV1, CuriosityUrgentSeedV1
from scripts.curiosity_investigation import urgent_open_key

logger = logging.getLogger("orion-hub.curiosity_urgent")

RESUBSCRIBE_DELAY_SEC = 5.0

_INCIDENT_ID = re.compile(r"^[0-9a-f]{12,32}$")


def _text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


async def _report_invalid(investigation: Any, payload: Any, exc: Exception) -> None:
    """An invalid request that still names an incident gets a failed report, so a
    requester (the hardware watcher) is never left believing it started. One with
    no usable incident id has nobody to answer to and is only logged."""
    if not isinstance(payload, dict):
        return
    incident_id = payload.get("incident_id")
    if not isinstance(incident_id, str) or not _INCIDENT_ID.match(incident_id):
        return
    reporter = getattr(investigation, "urgent_reporter", None)
    if reporter is None:
        logger.warning("urgent_reporter_missing incident_id=%s", incident_id)
        return
    redis = getattr(getattr(investigation, "_bus", None), "redis", None)
    if redis is not None:
        try:
            held = await redis.get(urgent_open_key(incident_id))
        except Exception:  # noqa: BLE001 -- unsure: still report
            held = None
        if held is not None:
            # A run is still open for this incident and will report; "failed" would be false.
            logger.info("urgent_request_invalid_not_reported incident_id=%s reason=run_open", incident_id)
            return
    stub = {
        "incident_id": incident_id,
        "run_id": "",
        "question": _text(payload.get("question"), 2000),
        "trigger": _text(payload.get("trigger"), 32) or "unknown",
        "subject": _text(payload.get("subject"), 120),
        "requested_at": _text(payload.get("requested_at"), 64),
        "requested_by": _text(payload.get("requested_by"), 64),
        "evidence": None,
        "status": "refused:invalid_request",
    }
    reason = " ".join(str(exc).split())[:300]
    try:
        await reporter.dispatch_failed(stub, f"refused: invalid_request: {reason}")
    except Exception:  # noqa: BLE001
        logger.exception("urgent_reporter_dispatch_failed_error incident_id=%s", incident_id)


async def handle_urgent_request(bus: Any, investigation: Any, msg: dict[str, Any]) -> None:
    try:
        decoded = bus.codec.decode(msg.get("data"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("urgent_request_invalid err=undecodable: %s", exc)
        return
    if not decoded.ok:
        logger.warning("urgent_request_invalid err=undecodable: %s", getattr(decoded, "error", ""))
        return
    payload = decoded.envelope.payload or {}
    try:
        request = CuriosityUrgentRequestV1.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("urgent_request_invalid err=%s", str(exc)[:500])
        await _report_invalid(investigation, payload, exc)
        return
    seed = CuriosityUrgentSeedV1.model_validate(request.model_dump())
    result = await investigation.start_urgent(seed)
    logger.info(
        "urgent_request_handled incident_id=%s ok=%s run=%s reason=%s",
        seed.incident_id, result.get("ok"), result.get("run_id"), result.get("reason"),
    )


async def _serve(bus: Any, investigation: Any, msg: dict[str, Any]) -> None:
    try:
        await handle_urgent_request(bus, investigation, msg)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("urgent_request_failed")


async def urgent_request_loop(bus: Any, investigation: Any) -> None:
    """Serve requests concurrently (a dispatch waits up to 20 s on cortex);
    resubscribe after a bus failure rather than going deaf."""
    tasks: set[asyncio.Task] = set()
    try:
        while True:
            try:
                async with bus.subscribe(URGENT_REQUEST_CHANNEL) as pubsub:
                    logger.info("urgent_request_loop subscribed channel=%s", URGENT_REQUEST_CHANNEL)
                    async for msg in bus.iter_messages(pubsub):
                        task = asyncio.create_task(_serve(bus, investigation, msg))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("urgent_request_loop_failed; resubscribing in %ss", RESUBSCRIBE_DELAY_SEC)
            await asyncio.sleep(RESUBSCRIBE_DELAY_SEC)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
