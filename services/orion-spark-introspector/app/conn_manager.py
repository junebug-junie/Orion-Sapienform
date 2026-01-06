from typing import List, Optional, Dict, Any
from fastapi import WebSocket
import logging
import json
import asyncio

logger = logging.getLogger("orion-spark-introspector")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_tissue_payload: Optional[Dict[str, Any]] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if self.last_tissue_payload:
            await websocket.send_json(self.last_tissue_payload)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        self.last_tissue_payload = message
        # Copy list to avoid modification during iteration issues
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                self.disconnect(connection)

manager = ConnectionManager()
