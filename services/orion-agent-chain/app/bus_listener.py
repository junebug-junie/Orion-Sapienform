# services/orion-planner-react/app/bus_listener.py

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict

from orion.core.bus.service import OrionBus

from .settings import settings
from .api import PlannerRequest, PlannerResponse, run_react_loop

logger = logging.getLogger("planner-react.bus")


def start_planner_bus_listener_background() -> None:
    """
    Start a background thread that listens on `PLANNER_REQUEST_CHANNEL`
    and answers on per-request `PLANNER_RESULT_PREFIX:{trace_id}` channels.
    """
    if not settings.orion_bus_enabled:
        logger.warning(
            "[planner-react] OrionBus disabled; not starting planner bus listener."
        )
        return

    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        logger.error(
            "[planner-react] OrionBus not enabled/connected; cannot start listener."
        )
        return

    request_channel = settings.planner_request_channel
    logger.info(
        "[planner-react] Starting planner bus listener on %s", request_channel
    )

    def _worker() -> None:
        """
        Blocking worker that consumes messages from `request_channel`
        and responds with PlannerResponse on the indicated reply_channel.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Blocking generator
        for msg in bus.subscribe(request_channel):
            data: Dict[str, Any] = msg.get("data") or {}

            reply_channel = data.get("reply_channel")
            payload = data.get("payload") or {}
            trace_id = data.get("trace_id")

            if not reply_channel:
                logger.warning(
                    "[planner-react] bus message missing reply_channel: %r", data
                )
                continue

            try:
                planner_req = PlannerRequest(**payload)
            except Exception as e:
                logger.error(
                    "[planner-react] invalid PlannerRequest payload on %s: %s",
                    request_channel,
                    e,
                    exc_info=True,
                )
                error_resp = PlannerResponse(
                    request_id=payload.get("request_id"),
                    status="error",
                    error={"message": f"invalid PlannerRequest payload: {e}"},
                    final_answer=None,
                    trace=[],
                    usage=None,
                )
                bus.publish(reply_channel, error_resp.model_dump())
                continue

            # Run the ReAct loop in this worker's event loop
            try:
                resp: PlannerResponse = loop.run_until_complete(
                    run_react_loop(planner_req)
                )
            except Exception as e:
                logger.error(
                    "[planner-react] run_react_loop failed (trace_id=%s): %s",
                    trace_id,
                    e,
                    exc_info=True,
                )
                resp = PlannerResponse(
                    request_id=planner_req.request_id,
                    status="error",
                    error={"message": str(e)},
                    final_answer=None,
                    trace=[],
                    usage=None,
                )

            # Make sure request_id is preserved if caller set one
            if resp.request_id is None:
                resp.request_id = planner_req.request_id

            bus.publish(reply_channel, resp.model_dump())
            logger.info(
                "[planner-react] replied on %s (trace_id=%s, status=%s)",
                reply_channel,
                trace_id,
                resp.status,
            )

    thread = threading.Thread(
        target=_worker,
        name="planner-react-bus-listener",
        daemon=True,
    )
    thread.start()
