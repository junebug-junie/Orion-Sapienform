"""
In-memory store for the latest social-room routing_debug per room.

Hub writes a snapshot here after every social-room chat turn (bridge or
direct).  The UI polls GET /api/social-room/inspection/latest so the Social
Inspection panel can render even when the turn was routed through the bridge
rather than the Hub WebSocket.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("orion-hub.social-room-inspection")

# room_id → { routing_debug, stored_at, room_id }
_latest: Dict[str, Dict[str, Any]] = {}


def store(room_id: str, routing_debug: Dict[str, Any]) -> None:
    """Record the latest routing_debug for a completed social-room turn."""
    if not room_id or not routing_debug:
        return
    _latest[room_id] = {
        "routing_debug": routing_debug,
        "stored_at": datetime.now(timezone.utc).isoformat(),
        "room_id": room_id,
    }
    logger.info(
        "social_room_inspection_stored room_id=%s social_inspection_present=%s",
        room_id,
        bool(routing_debug.get("social_inspection")),
    )


def get(room_id: str) -> Optional[Dict[str, Any]]:
    """Return the latest snapshot for a specific room, or None."""
    return _latest.get(room_id)


def get_latest() -> Optional[Dict[str, Any]]:
    """Return the most recently stored snapshot across all rooms."""
    if not _latest:
        return None
    return max(_latest.values(), key=lambda v: v["stored_at"])
