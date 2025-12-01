# services/orion-hub/app/session.py

from __future__ import annotations

import logging
from typing import Optional

from .warm_start import warm_start_session

logger = logging.getLogger("orion-hub.session")


async def ensure_session(session_id: Optional[str], bus) -> str:
    """
    Single source of truth for Hub session handling.

    - If no session_id: create + warm-start via warm_start_session().
    - If session_id exists:
        - Try to read warm-start state from Redis via bus.client.
        - If missing or not flagged, warm-start it.
        - If bus/client missing, just treat as "already warm" and return id.

    Returns the active session_id (always a string).
    """
    # No bus → we can't do actual Redis bookkeeping, but we still allow
    # warm_start_session to run once so identity is "loaded" logically.
    if not bus or not getattr(bus, "enabled", False):
        logger.warning(
            "ensure_session called but OrionBus is disabled; "
            "delegating to warm_start_session without Redis state."
        )
        if session_id is None:
            # Let warm_start_session generate the id, but we won't persist.
            return await warm_start_session(None, bus=None)
        # We already have an id; just return it.
        return session_id

    # No session id → new + warm start
    if session_id is None:
        return await warm_start_session(None, bus)

    # Bus exists; see if we can inspect Redis state.
    client = getattr(bus, "client", None)
    if client is None:
        logger.info(
            "OrionBus has no 'client' attribute; "
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
