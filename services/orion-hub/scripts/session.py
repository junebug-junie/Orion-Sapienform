# services/orion-hub/scripts/session.py
from __future__ import annotations

import logging
from typing import Optional

from .warm_start import warm_start_session

logger = logging.getLogger("orion-hub.session")


async def ensure_session(session_id: Optional[str], bus) -> str:
    """
    Ensure a session exists and is warm-started.

    - If no session_id: create + warm-start.
    - If session_id exists: check Redis via bus.client (if available)
      for `warm_started`. If missing, warm-start and mark it.
    """
    # If bus is missing/disabled, just delegate to warm_start_session without Redis bookkeeping
    if not bus or not getattr(bus, "enabled", False):
        logger.warning(
            "ensure_session called but OrionBus is disabled; returning raw session id."
        )
        if session_id is None:
            return await warm_start_session(None, bus=None)
        return session_id

    # No session id → new + warm start
    if session_id is None:
        return await warm_start_session(None, bus)

    client = getattr(bus, "client", None)
    if client is None:
        logger.info(
            "OrionBus has no `client` attribute; "
            "treating session %s as already warm-started.",
            session_id,
        )
        return session_id

    key = f"orion:hub:session:{session_id}:state"

    try:
        state = client.hgetall(key)
    except Exception as e:
        logger.warning(
            "Failed to read warm-start state from Redis for %s: %s",
            session_id,
            e,
        )
        return session_id

    if not state or state.get("warm_started") != "1":
        # Session exists but not warm-started — fix it
        return await warm_start_session(session_id, bus)

    return session_id
