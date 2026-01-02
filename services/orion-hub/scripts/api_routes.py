from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Any, List, Dict

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from .session import ensure_session
from .bus_clients.cortex_client import CortexClient
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
@router.get("/api/session")
async def api_session(x_orion_session_id: Optional[str] = Header(None)):
    """
    Called by Hub UI on load.
    Always returns a warm-started session_id.
    """
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}


# ======================================================================
# 💬 CHAT ENDPOINT
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    Main Chat Endpoint (Dumb Hub).
    Delegates everything to Cortex Gateway via Bus RPC.
    """
    from .main import bus
    if not bus:
        raise HTTPException(status_code=503, detail="OrionBus not initialized.")

    # 1. Ensure Session
    session_id = await ensure_session(x_orion_session_id, bus)

    # 2. Extract Data
    messages = payload.get("messages", [])
    if not messages:
         return {"error": "No messages provided"}

    prompt = messages[-1].get("content", "")
    mode = payload.get("mode", "brain")

    # Extract recall/options/packs if present
    # Hub UI might send `use_recall` bool, we can map it to recall directive if needed
    # but CortexChatRequest.recall expects a dict or None.
    # For now, we support passing raw `recall` dict from UI if advanced,
    # or we can infer it. Let's pass what we have.

    recall_opts = payload.get("recall")
    if not recall_opts and payload.get("use_recall"):
         recall_opts = {"enabled": True, "mode": "hybrid"}

    options = payload.get("options", {})
    # Map temperature if present
    if "temperature" in payload:
        options["temperature"] = payload["temperature"]

    # 3. RPC to Gateway
    client = CortexClient(bus)
    result = await client.send_chat_request(
        prompt=prompt,
        mode=mode,
        session_id=session_id,
        user_id=None, # HTTP auth not yet impl
        packs=payload.get("packs"),
        options=options,
        recall=recall_opts,
    )

    # 4. Format Response for Hub UI
    # Hub UI expects:
    # {
    #   "session_id": ...,
    #   "text": ...,
    #   "mode": ...,
    #   "recall_debug": ...,
    #   "raw": ...
    # }

    # CortexClientResult (dict) has:
    # final_text, steps, recall_debug, status, etc.

    text = result.get("final_text") or ""

    return {
        "session_id": session_id,
        "mode": result.get("mode", mode),
        "text": text,
        "recall_debug": result.get("recall_debug", {}),
        "raw": result, # send full dump for debug UI
    }


# ======================================================================
# 🔍 RECALL / RAG ENDPOINT
# ======================================================================
@router.post("/api/recall")
async def api_recall(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    Legacy Recall explorer endpoint.
    Since Hub is dumb, we probably shouldn't do direct recall queries here unless
    Gateway supports a "recall only" verb.
    For now, return a stub or NotImplemented, or implement via Gateway if needed.
    The instructions say: "Hub should no longer contain recall routing... Hub doesn’t decide anything."
    But keeping endpoints stable is a goal.

    We will stub this out or remove it?
    "Remove recall-related endpoint behavior or internal routing."
    So we remove the behavior.
    """
    return {
        "error": "Direct recall query not supported in this version of Hub. Use chat.",
        "session_id": x_orion_session_id
    }


# ======================================================================
# 📿 COLLAPSE MIRROR ENDPOINTS
# ======================================================================
@router.get("/schema/collapse")
def get_collapse_schema():
    """Exposes the CollapseMirrorEntry schema for UI templating."""
    return JSONResponse(CollapseMirrorEntry.schema())


@router.post("/submit-collapse")
async def submit_collapse(data: dict):
    """
    Receives Collapse Mirror data and publishes it to the bus.
    """
    from .main import bus
    if not bus or not bus.enabled:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "OrionBus disabled or unavailable"},
        )

    try:
        entry = CollapseMirrorEntry(**data).with_defaults()
        bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode="json"))
        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Hub publish error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )

@router.get("/api/debug/spark-last")
async def api_debug_spark_last(
    x_orion_session_id: Optional[str] = Header(None),
):
    # Stubbed out for now as Redis usage in Hub is minimized
    return {"session_id": x_orion_session_id, "spark_meta": None}
