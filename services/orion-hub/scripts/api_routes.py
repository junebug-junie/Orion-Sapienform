# scripts/api_routes.py

import logging
from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional

from .settings import settings
from .warm_start import mini_personality_summary, warm_start_session
from .llm_rpc import BrainRPC
from orion.schemas.collapse_mirror import CollapseMirrorEntry

logger = logging.getLogger("orion-hub.api")

router = APIRouter()


# ======================================================================
# 🏠 ROOT + STATIC HTML
# ======================================================================
@router.get("/")
async def root():
    """Serves the main Hub UI (index.html)."""
    from .main import html_content
    return HTMLResponse(content=html_content, status_code=200)


@router.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": settings.SERVICE_NAME}


# ======================================================================
# 🧠 SESSION MANAGEMENT
# ======================================================================
async def ensure_session(session_id: Optional[str], bus) -> str:
    """
    Ensures the session_id exists and is warm-started.

    - If no session_id: create + warm-start.
    - If session_id exists: check Redis via bus.client (if available)
      for `warm_started`. If missing, warm-start and mark it.
    """
    # If bus is missing/disabled, just delegate to warm_start_session without Redis bookkeeping
    if not bus or not getattr(bus, "enabled", False):
        logger.warning("ensure_session called but OrionBus is disabled; returning bare session_id.")
        if session_id is None:
            return await warm_start_session(None, bus=None)
        return session_id

    # No session id → new + warm start
    if session_id is None:
        return await warm_start_session(None, bus)

    client = getattr(bus, "client", None)
    if client is None:
        logger.info(
            f"OrionBus has no 'client' attribute; "
            f"treating session {session_id} as already warm-started."
        )
        return session_id

    key = f"orion:hub:session:{session_id}:state"

    try:
        state = client.hgetall(key)
    except Exception as e:
        logger.warning(f"Failed to read warm-start state from Redis for {session_id}: {e}")
        return session_id

    if not state or state.get("warm_started") != "1":
        # Session exists but not warm-started — fix it
        return await warm_start_session(session_id, bus)

    return session_id


@router.get("/api/session")
async def api_session(x_orion_session_id: Optional[str] = Header(None)):
    """
    Called by Hub UI on load.
    Always returns a warm-started session_id.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}


# ======================================================================
# 💬 CHAT ENDPOINT (BUS-based Brain RPC)
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None)
):
    """
    Main LLM chat endpoint.
    Uses BrainRPC (bus-RPC) to talk to Brain GPU.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)

    user_messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid payload: missing messages[]"}
        )

    # Brain RPC
    rpc = BrainRPC(bus)

    # Inject Phase-1 mini identity stub
    user_prompt = user_messages[-1].get("content", "")
    system_stub = {"role": "system", "content": mini_personality_summary()}
    full_history = [system_stub] + user_messages

    reply = await rpc.call_llm(
        prompt=user_prompt,
        history=full_history,
        temperature=temperature,
    )

    # Store chat tail in Redis (last 20 entries)
    client = getattr(bus, "client", None)
    if client is not None:
        try:
            history_key = f"orion:hub:session:{session_id}:history"
            client.lpush(history_key, str(user_messages[-1])[:4000])
            client.ltrim(history_key, 0, 19)
        except Exception as e:
            logger.warning(f"Failed to store chat tail in Redis for {session_id}: {e}")

# ======================================================================
# 📿 COLLAPSE MIRROR ENDPOINTS
# ======================================================================
@router.get("/schema/collapse")
def get_collapse_schema():
    """Exposes the CollapseMirrorEntry schema for UI templating."""
    logger.info("Fetching CollapseMirrorEntry schema")
    return JSONResponse(CollapseMirrorEntry.schema())


@router.post("/submit-collapse")
async def submit_collapse(data: dict):
    """
    Receives Collapse Mirror data and publishes it to the bus.
    """
    from .main import bus
    logger.info(f"🔥 /submit-collapse called with: {data}")

    if not bus or not bus.enabled:
        logger.error("Submission failed: OrionBus is disabled or not connected.")
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "OrionBus disabled or unavailable"}
        )

    try:
        entry = CollapseMirrorEntry(**data).with_defaults()
        bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode='json'))
        logger.info(f"📡 Published Collapse Mirror → {settings.CHANNEL_COLLAPSE_INTAKE}")

        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
