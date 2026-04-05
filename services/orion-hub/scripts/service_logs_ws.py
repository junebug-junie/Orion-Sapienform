from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from .service_logs import ServiceLogSession

logger = logging.getLogger("orion-hub.service-logs")


async def service_logs_websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = ServiceLogSession()
    sender_task: asyncio.Task | None = None

    async def _sender() -> None:
        while True:
            event = await session.next_event()
            await websocket.send_text(json.dumps({"type": "log_line", **event}))

    try:
        sender_task = asyncio.create_task(_sender())
        await websocket.send_text(
            json.dumps(
                {
                    "type": "service_inventory",
                    "services": session.available_services,
                }
            )
        )

        while True:
            payload: Any = await websocket.receive_json()
            action = str(payload.get("action") or "").strip().lower()
            if action != "subscribe":
                await websocket.send_text(json.dumps({"type": "error", "error": f"Unknown action: {action}"}))
                continue

            selected = payload.get("services") or []
            if not isinstance(selected, list):
                await websocket.send_text(json.dumps({"type": "error", "error": "services must be a list"}))
                continue

            normalized = [str(item).strip() for item in selected if str(item).strip()]
            active = await session.set_selected_services(normalized)
            await websocket.send_text(json.dumps({"type": "selection_updated", "services": active}))
    except WebSocketDisconnect:
        logger.info("Service log websocket disconnected")
    except Exception as exc:
        logger.warning("Service log websocket error: %s", exc, exc_info=True)
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}))
    finally:
        if sender_task:
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task
        await session.close()

