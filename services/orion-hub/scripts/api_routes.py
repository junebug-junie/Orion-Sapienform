from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Any, List, Dict, Tuple

from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from .session import ensure_session
from .library import scan_cognition_library
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult

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
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}

# ======================================================================
# 📚 COGNITION LIBRARY (Verbs & Packs)
# ======================================================================
@router.get("/api/cognition/library")
def get_cognition_library():
    """
    Returns the scanned list of Packs and Verbs available in the system.
    Used by the UI to populate dropdowns and filters.
    """
    return scan_cognition_library()


# ======================================================================
# 💬 SHARED CHAT CORE (HTTP + WS)
# ======================================================================

async def handle_chat_request(
    cortex_client,
    payload: dict,
    session_id: str,
) -> Dict[str, Any]:
    """
    Core chat handler used by both HTTP /api/chat and (optionally) WebSocket.
    Delegate strict typed requests to orion-cortex-gateway via Bus.
    """
    user_messages = payload.get("messages", [])
    mode = payload.get("mode", "brain")

    use_recall = bool(payload.get("use_recall", False))
    logger.warning("Force-disabling recall in Hub due to known service outage.")
    packs = payload.get("packs")
    user_id = payload.get("user_id")

    # Handle Verbs override (multi-select from UI)
    ui_verbs = payload.get("verbs")

    verb_override = None
    options = payload.get("options") or {}

    if isinstance(ui_verbs, list) and len(ui_verbs) > 0:
        if len(ui_verbs) == 1:
             # Single verb -> override entry point
             verb_override = ui_verbs[0]
        else:
             # Multiple verbs -> pass as allowed tools/verbs in options
             options["allowed_verbs"] = ui_verbs

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    # Build the Request
    req = CortexChatRequest(
        prompt=user_prompt,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        packs=packs,
        verb=verb_override,
        options=options if options else None,
        recall={"enabled": use_recall} if use_recall else None,
        metadata={"source": "hub_http"},
    )

    try:
        # Call Bus RPC
        resp: CortexChatResult = await cortex_client.chat(req)

        # Extract Text
        text = resp.final_text or ""

        # Map raw result for UI debug
        raw_result = resp.cortex_result.model_dump(mode="json")

        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "text": text,
            "tokens": len(text.split()), # simple approx
            "raw": raw_result,
            "recall_debug": resp.cortex_result.recall_debug,
            "spark_meta": None,
        }

    except Exception as e:
        logger.error(f"Chat RPC failed: {e}", exc_info=True)
        return {"error": str(e)}


# ======================================================================
# 💬 CHAT ENDPOINT (HTTP wrapper around core)
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    Main LLM chat endpoint.
    Delegates to Cortex Gateway via Bus RPC.
    """
    from .main import bus, cortex_client
    if not bus or not cortex_client:
        raise RuntimeError("OrionBus/Client not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)

    # Core chat handling
    result = await handle_chat_request(cortex_client, payload, session_id)

    # ─────────────────────────────────────────────
    # 📡 Publish HTTP chat → chat history log
    # (restores legacy Brain→SQL behavior via CHANNEL_CHAT_LOG)
    # ─────────────────────────────────────────────
    text = result.get("text")
    if text and getattr(bus, "enabled", False):
        try:
            user_messages = payload.get("messages", [])
            latest_user_prompt = ""
            if isinstance(user_messages, list) and user_messages:
                latest_user_prompt = user_messages[-1].get("content", "") or ""

            use_recall = bool(payload.get("use_recall", False))

            chat_log_payload = {
                "trace_id": str(uuid4()),
                "source": settings.SERVICE_NAME,
                "prompt": latest_user_prompt,
                "response": text,
                "session_id": session_id,
                "mode": result.get("mode", "brain"),
                #"recall": use_recall,
                "user_id": None,
                "spark_meta": None,
            }

            bus.publish(
                settings.CHANNEL_CHAT_HISTORY_LOG,
                chat_log_payload,
            )
        except Exception as e:
            logger.warning(
                "Failed to publish HTTP chat to chat history log: %s",
                e,
                exc_info=True,
            )

    return result


# ======================================================================
# 🔍 RECALL / RAG ENDPOINT (Retired)
# ======================================================================
@router.post("/api/recall")
async def api_recall(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    DEPRECATED: Hub no longer performs direct recall.
    This logic is now handled by Cortex Gateway / Orchestrator.
    """
    return {"error": "Endpoint deprecated. Recall is handled by Cortex Stack."}


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
            content={"success": False, "error": "OrionBus disabled or unavailable"},
        )

    try:
        entry = CollapseMirrorEntry(**data).with_defaults()
        bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode="json"))
        logger.info(
            "📡 Published Collapse Mirror → %s",
            settings.CHANNEL_COLLAPSE_INTAKE,
        )

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
    # Legacy debug endpoint - likely broken but kept safe
    return JSONResponse(
        {
            "session_id": x_orion_session_id,
            "spark_meta": None,
            "note": "Spark debug deprecated in dumb hub.",
        }
    )
