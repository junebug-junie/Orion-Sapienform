# services/orion-hub/app/session.py

from typing import Optional
from .warm_start import warm_start_session
from .settings import settings


async def ensure_session(session_id: Optional[str], bus) -> str:
    """
    Ensure session exists and has been warm-started.
    """
    if not session_id:
        return await warm_start_session(None, bus)

    key = f"orion:hub:session:{session_id}:state"

    client = getattr(bus, "client", None)
    if client is None:
        logger.info(
            "OrionBus has no 'client' attribute in session.py; "
            "treating state as empty."
        )
        return {}

    state = client.hgetall(key)

    if not state or state.get("warm_started") != "1":
        return await warm_start_session(session_id, bus)

    return session_id
