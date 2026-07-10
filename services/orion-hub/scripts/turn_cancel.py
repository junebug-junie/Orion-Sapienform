from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, TypeVar

from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocket, WebSocketState

logger = logging.getLogger("hub.turn_cancel")

T = TypeVar("T")


async def publish_harness_run_cancel(
    bus: Any,
    *,
    correlation_id: str,
    reason: str = "client_disconnect",
) -> None:
    if bus is None or not correlation_id:
        return
    try:
        from scripts.harness_governor_client import HarnessGovernorClient

        await HarnessGovernorClient(bus).cancel(
            correlation_id=str(correlation_id),
            reason=str(reason or "client_disconnect"),
        )
    except Exception:
        logger.warning(
            "publish_harness_run_cancel failed corr=%s",
            correlation_id,
            exc_info=True,
        )


async def cancel_agent_claude_turn(correlation_id: str) -> bool:
    try:
        from scripts.fcc_claude_bridge import cancel_turn

        return bool(await cancel_turn(str(correlation_id)))
    except Exception:
        logger.warning(
            "cancel_agent_claude_turn failed corr=%s",
            correlation_id,
            exc_info=True,
        )
        return False


async def cancel_in_flight_turn(
    *,
    bus: Any,
    correlation_id: str,
    kind: str,
    reason: str = "client_disconnect",
) -> None:
    """Cancel whichever motor owns this correlation_id."""
    corr = str(correlation_id or "").strip()
    if not corr:
        return
    mode = str(kind or "").strip().lower()
    if mode in {"agent-claude", "agent_claude"}:
        await cancel_agent_claude_turn(corr)
        return
    # Default: Orion unified harness motor (and any other FCC-via-governor path).
    await publish_harness_run_cancel(bus, correlation_id=corr, reason=reason)


async def run_awaitable_cancel_on_ws_disconnect(
    websocket: WebSocket,
    awaitable: Awaitable[T],
    *,
    bus: Any,
    correlation_id: str,
    kind: str,
    poll_sec: float = 0.5,
) -> T:
    """
    Await a turn while polling websocket.client_state.
    On disconnect (poll or WebSocketDisconnect from send), cancel the in-flight
    FCC/harness motor so it does not orphan.
    """
    cancelled = False

    async def _cancel_once() -> None:
        nonlocal cancelled
        if cancelled:
            return
        cancelled = True
        logger.info(
            "ws_disconnect_cancel corr=%s kind=%s state=%s",
            correlation_id,
            kind,
            getattr(websocket, "client_state", None),
        )
        await cancel_in_flight_turn(
            bus=bus,
            correlation_id=correlation_id,
            kind=kind,
            reason="client_disconnect",
        )

    async def _watch() -> None:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                await _cancel_once()
                return
            await asyncio.sleep(poll_sec)

    watcher = asyncio.create_task(_watch(), name=f"ws-cancel-{correlation_id}")
    try:
        return await awaitable
    except WebSocketDisconnect:
        await _cancel_once()
        raise
    finally:
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass
