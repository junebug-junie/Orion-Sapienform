"""Bus consumer for urgent curiosity requests (``orion:curiosity:urgent:request``).

The Hub "Run urgent" button and Plan 4's hardware watcher both publish a
``CuriosityUrgentRequestV1`` here, so there is one path into
``CuriosityInvestigation.start_urgent``. Invalid payloads are logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orion.schemas.curiosity_urgent import URGENT_REQUEST_CHANNEL, CuriosityUrgentRequestV1, CuriosityUrgentSeedV1

logger = logging.getLogger("orion-hub.curiosity_urgent")

RESUBSCRIBE_DELAY_SEC = 5.0


async def handle_urgent_request(bus: Any, investigation: Any, msg: dict[str, Any]) -> None:
    try:
        decoded = bus.codec.decode(msg.get("data"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("urgent_request_invalid err=undecodable: %s", exc)
        return
    if not decoded.ok:
        logger.warning("urgent_request_invalid err=undecodable: %s", getattr(decoded, "error", ""))
        return
    try:
        request = CuriosityUrgentRequestV1.model_validate(decoded.envelope.payload or {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("urgent_request_invalid err=%s", str(exc)[:500])
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
