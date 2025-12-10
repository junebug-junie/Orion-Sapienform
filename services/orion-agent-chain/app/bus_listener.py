# services/orion-agent-chain/app/bus_listener.py

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict

from orion.core.bus.service import OrionBus

from .settings import settings
from .api import AgentChainRequest, execute_agent_chain

logger = logging.getLogger("agent-chain.bus")


def start_agent_chain_bus_listener() -> None:
    """
    Start a background thread that listens on `AGENT_CHAIN_REQUEST_CHANNEL`
    (from UI/Hub) and answers on the ephemeral reply channel.
    """
    if not settings.orion_bus_enabled:
        logger.warning(
            "[agent-chain] OrionBus disabled; not starting bus listener."
        )
        return

    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )
    if not bus.enabled:
        logger.error(
            "[agent-chain] OrionBus not enabled/connected; cannot start listener."
        )
        return

    request_channel = settings.agent_chain_request_channel
    logger.info(
        "[agent-chain] Starting upstream bus listener on %s", request_channel
    )

    def _worker() -> None:
        """
        Blocking worker loop for UI requests.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Subscribe to requests (e.g. from UI)
        for msg in bus.subscribe(request_channel):
            data: Dict[str, Any] = msg.get("data") or {}

            reply_channel = data.get("reply_channel")
            payload = data.get("payload") or {}
            trace_id = data.get("trace_id")

            if not reply_channel:
                logger.warning(
                    "[agent-chain] bus message missing reply_channel: %r", data
                )
                continue

            # 1. Parse Request
            try:
                req = AgentChainRequest(**payload)
            except Exception as e:
                logger.error(
                    "[agent-chain] invalid request payload on %s: %s",
                    request_channel,
                    e,
                    exc_info=True,
                )
                # Return standard error envelope
                bus.publish(reply_channel, {
                    "status": "error",
                    "error": f"Invalid payload: {e}"
                })
                continue

            # 2. Execute Logic (Calls Planner via downstream bus)
            try:
                result = loop.run_until_complete(
                    execute_agent_chain(req)
                )
                
                # Success Response
                response_payload = {
                    "status": "ok",
                    "data": result.model_dump()
                }
                
            except Exception as e:
                logger.error(
                    "[agent-chain] execution failed (trace_id=%s): %s",
                    trace_id,
                    e,
                    exc_info=True,
                )
                response_payload = {
                    "status": "error",
                    "error": str(e)
                }

            # 3. Reply
            bus.publish(reply_channel, response_payload)
            logger.info(
                "[agent-chain] replied on %s (trace_id=%s, status=%s)",
                reply_channel,
                trace_id,
                response_payload["status"],
            )

    thread = threading.Thread(
        target=_worker,
        name="agent-chain-bus-listener",
        daemon=True,
    )
    thread.start()
