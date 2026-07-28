"""WebSocket fan-out for the OrionTissue read model.

Relocated 2026-07-28 from services/orion-spark-introspector/app/conn_manager.py
as part of the spark-introspector retirement (docs/superpowers/specs/2026-07-28-
spark-introspector-retirement-and-honest-substrate-convergence.md). Trivial
broadcast helper -- no behavior change from the original.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger("orion-vector-host.tissue")


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                logger.debug("tissue ws broadcast failed, dropping connection: %s", exc)
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()
