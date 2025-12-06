from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Any, List, Dict, Tuple

from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from .warm_start import mini_personality_summary
from .llm_rpc import CouncilRPC, LLMGatewayRPC
from .recall_rpc import RecallRPC
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
# 🔎 RECALL → MEMORY DIGEST HELPERS
# ======================================================================

def _format_fragments_for_digest(
    fragments: List[Dict[str, Any]],
    limit: int = 8,
) -> str:
    """
    Turn raw recall fragments into a compact bullet list for an internal
    'memory digest' LLM call.
    """
    lines: List[str] = []
    for f in fragments[:limit]:
        kind = f.get("kind", "unknown")
        source = f.get("source", "unknown")
        text = (f.get("text") or "").replace("\n", " ").strip()

        meta = f.get("meta") or {}
        observer = meta.get("observer")
        field_resonance = meta.get("field_resonance")

        extras: List[str] = []
        if observer:
            extras.append(f"observer={observer}")
        if field_resonance:
            extras.append(f"field_resonance={field_resonance}")

        suffix = f" [{' | '.join(extras)}]" if extras else ""
        lines.append(f"- [{kind}/{source}] {text}{suffix}")

    return "\n".join(lines)


async def build_memory_digest(
    bus,
    session_id: str,
    user_prompt: str,
    chat_mode: str = "brain",
    max_items: int = 12,
) -> Tuple[str, Dict[str, Any]]:
    """
    1) Call Recall over the Orion bus.
    2) Ask LLM Gateway to condense fragments into 3–5 bullets.
    3) Return (digest_text, recall_debug) for use as internal context.

    This leans on the Recall service's semantic+salience+recency scoring;
    hub stays intentionally dumb and only does LLM summarization.
    """
    # Choose recall mode/window based on chat mode
    if chat_mode == "council":
        recall_mode = "deep"
        time_window_days = 90
    else:
        recall_mode = "hybrid"
        time_window_days = 30

    recall_client = RecallRPC(bus)
    recall_result = await recall_client.call_recall(
        query=user_prompt,
        session_id=session_id,
        mode=recall_mode,
        time_window_days=time_window_days,
        max_items=max_items,
        extras=None,
    )

    fragments = recall_result.get("fragments") or []
    debug = recall_result.get("debug") or {}

    if not fragments:
        logger.info("build_memory_digest: no fragments returned from recall.")
        return "", {
            "total_fragments": 0,
            "mode": recall_mode,
            "time_window_days": time_window_days,
        }

    fragments_block = _format_fragments_for_digest(fragments)
    if not fragments_block:
        return "", {
            "total_fragments": len(fragments),
            "mode": recall_mode,
            "time_window_days": time_window_days,
        }

    # 2) Memory digest via LLM Gateway (bus-native)
    system = (
        "You are Oríon, Juniper's collaborative AI co-journeyer.\n"
        "You will receive:\n"
        "1) The user's current message.\n"
        "2) A small list of past events and dialogues ('fragments').\n\n"
        "Your job:\n"
        "- Identify ONLY the 3–5 most relevant threads for understanding and responding to the current message.\n"
        "- Return them as short bullet points.\n"
        "- Each bullet should be one sentence.\n"
        "- Do not include anything unrelated.\n"
        "- This is internal memory context; the user will not see this directly.\n"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Current message: {user_prompt}"},
        {
            "role": "user",
            "content": "Relevant memory fragments:\n" + fragments_block,
        },
    ]

    gateway = LLMGatewayRPC(bus)
    result = await gateway.call_chat(
        prompt=user_prompt,
        history=messages,
        temperature=0.0,
        source="hub-memory-digest",
        session_id=session_id,
    )

    text = (result.get("text") or "").strip()
    logger.info("build_memory_digest: got digest length=%d", len(text))

    # merge recall debug with basic info
    debug_out = {
        "total_fragments": len(fragments),
        "mode": debug.get("mode", recall_mode),
        "time_window_days": time_window_days,
        "max_items": max_items,
        "note": debug.get("note", "semantic+salience+recency scoring"),
    }

    return text, debug_out


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

    - Handles Brain vs Council routing (via LLM Gateway & Council).
    - Optionally pulls a Recall → memory digest when payload.use_recall is true.
    - Injects mini_personality_summary + digest as a single system stub.
    """
    user_messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    mode = payload.get("mode", "brain")  # "brain" | "council"
    use_recall = bool(payload.get("use_recall", False))

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    # Optional: Recall → Digest
    memory_digest = ""
    recall_debug: Dict[str, Any] = {}
    if use_recall:
        try:
            memory_digest, recall_debug = await build_memory_digest(
                bus=bus,
                session_id=session_id,
                user_prompt=user_prompt,
                chat_mode=mode,
                max_items=12,
            )
        except Exception as e:
            logger.warning("Memory digest failed: %s", e, exc_info=True)
            memory_digest = ""
            recall_debug = {"error": str(e)}

    # Build system stub = personality + optional memory digest
    system_content = mini_personality_summary()
    if memory_digest:
        system_content += "\n\nInternal memory context:\n" + memory_digest

    system_stub = {"role": "system", "content": system_content}
    full_history = [system_stub] + user_messages

    # Choose backend: Council vs LLM Gateway
    if mode == "council":
        rpc = CouncilRPC(bus)
        reply = await rpc.call_llm(
            prompt=user_prompt,
            history=full_history,
            temperature=temperature,
        )
    else:
        gateway = LLMGatewayRPC(bus)
        reply = await gateway.call_chat(
            prompt=user_prompt,
            history=full_history,
            temperature=temperature,
            source="hub-http",
            session_id=session_id,
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
        "recall_debug": recall_debug,
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
