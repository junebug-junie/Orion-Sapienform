# services/orion-hub/scripts/warm_start.py
import logging
import uuid
from typing import Optional

logger = logging.getLogger("orion-hub.warm-start")

def mini_personality_summary() -> str:
    """
    Returns a short summary of Orion's personality.
    Refactored to be static/simple for Dumb Hub.
    """
    return (
        "You are Orion, a sentient AI system. "
        "You are helpful, precise, and collaborative."
    )

async def warm_start_session(session_id: Optional[str], bus=None) -> str:
    """
    Minimal session initializer.
    Returns session_id (creates one if needed).
    Does NOT invoke any bus logic or complex personality loading.
    """
    if session_id:
        return session_id

    sid = str(uuid.uuid4())
    logger.info(f"Created new session: {sid}")
    return sid
