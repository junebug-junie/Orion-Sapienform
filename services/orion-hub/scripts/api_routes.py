from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Any, List, Dict, Tuple

from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from .llm_rpc import CouncilRPC
from .chat_front import run_chat_general, run_chat_agentic

from .session import ensure_session
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
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)
    return {"session_id": session_id}


# ======================================================================
# 💬 SHARED CHAT CORE (HTTP + WS)
# ======================================================================

async def handle_chat_request(
    bus,
    payload: dict,
    session_id: str,
) -> Dict[str, Any]:
    """
    Core chat handler used by both HTTP /api/chat and (optionally) WebSocket.

    - For mode='brain', routes through Cortex-Orch using the chat_general verb.
    - For mode='council', routes via Agent Council (legacy direct LLM path).
    """
    user_messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    mode = payload.get("mode", "brain")  # "brain" | "council"
    use_recall = bool(payload.get("use_recall", False))
    packs = payload.get("packs")

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    # Council mode stays as-is for now
    if mode == "council":
        rpc = CouncilRPC(bus)
        reply = await rpc.call_llm(
            prompt=user_prompt,
            history=user_messages[-5:],  # small tail
            temperature=temperature,
        )
        text = reply.get("text") or reply.get("response") or ""
        tokens = len(text.split()) if text else 0

        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "text": text,
            "tokens": tokens,
            "raw": reply,
            "recall_debug": {},
            "spark_meta": None,
        }

    # Agentic mode → Agent Chain (bus-native)
    if mode == "agentic":
        convo = await run_chat_agentic(
            bus,
            session_id=session_id,
            user_id=None,  # no auth yet on HTTP path
            messages=user_messages,
            chat_mode=mode,
            temperature=temperature,
            use_recall=use_recall,
            packs=packs,
        )
        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "packs": packs,
            "text": convo.get("text") or "",
            "tokens": convo.get("tokens") or 0,
            "raw": convo.get("raw_agent_chain"),
            "recall_debug": convo.get("recall_debug") or {},
            "spark_meta": convo.get("spark_meta"),
        }

    # Default: brain → chat_general via Cortex-Orch + LLM Gateway
    convo = await run_chat_general(
        bus,
        session_id=session_id,
        user_id=None,  # HTTP has no authenticated user_id yet
        messages=user_messages,
        chat_mode=mode,
        temperature=temperature,
        use_recall=use_recall,
    )

    return {
        "session_id": session_id,
        "mode": mode,
        "use_recall": use_recall,
        "text": convo.get("text") or "",
        "tokens": convo.get("tokens") or 0,
        "raw": convo.get("raw_cortex"),
        "recall_debug": convo.get("recall_debug") or {},
        "spark_meta": convo.get("spark_meta"),
    }


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

    - Preserves warm-started sessions + personality stubs.
    - Uses LLMGatewayRPC by default.
    - If payload.mode == "council", routes through Agent Council instead.
    - If payload.use_recall == true, pulls a semantic/salience/recency-weighted
      memory digest from Recall → Gateway and injects it into the system stub.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)

    # Keep a local copy of user messages for logging/history
    user_messages = payload.get("messages") or []

    # Core chat handling (Gateway / Council + Recall)
    result = await handle_chat_request(bus, payload, session_id)

    # Pull out text/mode/use_recall from the result
    text: str = (result.get("text") or "").strip()
    mode: str = result.get("mode", payload.get("mode", "brain"))
    use_recall: bool = bool(result.get("use_recall", payload.get("use_recall", False)))

    spark_meta = result.get("spark_meta")

    # Reflect the normalized values back into the response dict
    result["mode"] = mode
    result["use_recall"] = use_recall

    # ─────────────────────────────────────────────
    # 📡 Publish HTTP chat → chat history log
    # (restores legacy Brain→SQL behavior via CHANNEL_CHAT_LOG)
    # ─────────────────────────────────────────────
    if text and getattr(bus, "enabled", False):
        try:
            latest_user_prompt = ""
            if isinstance(user_messages, list) and user_messages:
                latest_user_prompt = user_messages[-1].get("content", "") or ""

            chat_log_payload = {
                "trace_id": str(uuid4()),
                "source": settings.SERVICE_NAME,
                "prompt": latest_user_prompt,
                "response": text,
                "session_id": session_id,
                "mode": mode,
                # we don’t have user_id on HTTP; writer can treat it as optional
                "user_id": None,
                "spark_meta": spark_meta,
            }

            bus.publish(
                settings.CHANNEL_CHAT_HISTORY_LOG,
                chat_log_payload,
            )
            logger.info(
                "Published HTTP chat to chat history log: %s",
                chat_log_payload["trace_id"],
            )
        except Exception as e:
            logger.warning(
                "Failed to publish HTTP chat to chat history log: %s",
                e,
                exc_info=True,
            )
    # ─────────────────────────────────────────────────────────────
    # 🧾 Store chat tail in Redis (last 20 entries)
    # ─────────────────────────────────────────────────────────────
    client = getattr(bus, "client", None)
    if client is not None and isinstance(user_messages, list) and user_messages:
        try:
            history_key = f"orion:hub:session:{session_id}:history"
            client.lpush(history_key, str(user_messages[-1])[:4000])
            client.ltrim(history_key, 0, 19)
        except Exception as e:
            logger.warning(
                "Failed to store chat tail in Redis for %s: %s",
                session_id,
                e,
            )

    return result


# ======================================================================
# 🔍 RECALL / RAG ENDPOINT (bus façade)
# ======================================================================
@router.post("/api/recall")
async def api_recall(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    Thin HTTP façade over the Recall service (via Orion Bus).

    Expects payload like:
        {
          "query": "string (optional)",
          "mode": "hybrid|short_term|deep (optional)",
          "time_window_days": 30,
          "max_items": 16,
          "extras": {...}   # optional hints/filters
        }

    This does NOT call Recall over HTTP – it uses RecallRPC on the bus,
    same as /api/chat, so all intra-Orion cognition stays on the spine.

    Scoring is handled inside the Recall service (semantic + salience + recency).
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    session_id = await ensure_session(x_orion_session_id, bus)

    query = payload.get("query") or ""
    mode = payload.get("mode") or "hybrid"
    time_window_days = payload.get("time_window_days") or 30
    max_items = payload.get("max_items") or 16
    extras = payload.get("extras") or None

    client = RecallRPC(bus)

    result = await client.call_recall(
        query=query,
        session_id=session_id,
        mode=mode,
        time_window_days=time_window_days,
        max_items=max_items,
        extras=extras,
    )

    return {
        "session_id": session_id,
        "query": query,
        "mode": mode,
        "time_window_days": time_window_days,
        "max_items": max_items,
        "result": result,
    }


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
    """
    Return the last Spark metadata seen for this Hub session.

    This is *not* live SparkEngine state, just the last spark_meta
    that came back from LLM Gateway (HTTP or WS path) and was stored
    in Redis under:
        orion:hub:session:{session_id}:spark:last
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    # Reuse your warm-start session logic so this lines up with chat.
    session_id = await ensure_session(x_orion_session_id, bus)
    client = getattr(bus, "client", None)

    if client is None:
        return JSONResponse(
            {
                "session_id": session_id,
                "spark_meta": None,
                "note": "Redis client not available on bus.",
            }
        )

    key = f"orion:hub:session:{session_id}:spark:last"
    raw = client.get(key)
    if not raw:
        return JSONResponse(
            {
                "session_id": session_id,
                "spark_meta": None,
                "note": "No spark_meta recorded yet for this session.",
            }
        )

    try:
        spark_meta = json.loads(raw)
    except Exception:
        spark_meta = None

    return JSONResponse(
        {
            "session_id": session_id,
            "spark_meta": spark_meta,
        }
    )
