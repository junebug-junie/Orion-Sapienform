# services/orion-hub/scripts/api_routes.py
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header
from fastapi.responses import HTMLResponse, JSONResponse

from .settings import settings
from .warm_start import mini_personality_summary
from .llm_rpc import BrainRPC, CouncilRPC
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
# 💬 CHAT ENDPOINT (Brain vs Council + Recall)
# ======================================================================
@router.post("/api/chat")
async def api_chat(
    payload: dict,
    x_orion_session_id: Optional[str] = Header(None),
):
    """
    Main LLM chat endpoint.

    - Preserves warm-started sessions + personality stubs.
    - Uses BrainRPC by default.
    - If payload.mode == "council", routes through Agent Council instead.
    - Uses RecallRPC over Orion Bus to fetch recent, relevant memories
      and injects them as a system message.
    """
    from .main import bus
    if not bus:
        raise RuntimeError("OrionBus not initialized.")

    # Ensure warm-started session
    session_id = await ensure_session(x_orion_session_id, bus)

    user_messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    mode = payload.get("mode", "brain")  # "brain" | "council"

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid payload: missing messages[]"},
        )

    user_prompt = user_messages[-1].get("content", "")

    # ─────────────────────────────────────────────────────────────
    # 🧠 Phase 1: Personality stub
    # ─────────────────────────────────────────────────────────────
    system_stub = {"role": "system", "content": mini_personality_summary()}

    # ─────────────────────────────────────────────────────────────
    # 🔍 Phase 2: Recall over bus (RecallRPC)
    # ─────────────────────────────────────────────────────────────
    memory_snippets: list[str] = []
    recall_debug: dict[str, Any] | None = None

    try:
        from .recall_rpc import RecallRPC  # local import to avoid cycles
        recall_client = RecallRPC(bus)

        # You can tune these knobs as you like
        recall_result = await recall_client.call_recall(
            query=user_prompt,
            session_id=session_id,
            mode="hybrid",
            time_window_days=14,
            max_items=12,
            extras=None,
        )

        recall_debug = {
            "total_fragments": len(recall_result.get("fragments", [])),
            "mode": recall_result.get("debug", {}).get("mode"),
        }

        for frag in recall_result.get("fragments", []):
            kind = frag.get("kind")
            text = (frag.get("text") or "").strip()
            if not text:
                continue

            # Optionally skip pure enrichment/tag clouds in the injected block
            if kind == "enrichment":
                continue

            memory_snippets.append(f"[{kind}] {text[:260]}")
    except Exception as e:
        logger.warning(f"RecallRPC lookup failed in /api/chat: {e}", exc_info=True)

    memory_block = ""
    if memory_snippets:
        memory_block = (
            "Relevant past memories about Juniper, Orion, and recent context. "
            "Use ONLY the events listed below as factual memory. "
            "If Juniper asks whether you remember something that is not mentioned "
            "here or in the recent dialogue history, explicitly say that you do not recall "
            "instead of guessing. Do NOT invent specific cities, people, dates, or events.\n"
            + "\n".join(f"- {s}" for s in memory_snippets[:6])
        )

    memory_msg = {"role": "system", "content": memory_block} if memory_block else None

    # ─────────────────────────────────────────────────────────────
    # 🧮 Phase 3: Build full history with memories inline
    # ─────────────────────────────────────────────────────────────
    # Build full_history = [system_stub] + memory_msg? + user_messages ...
    full_history = [system_stub]
    if memory_msg:
        full_history.append(memory_msg)
    full_history.extend(user_messages)

    rendered_prompt = render_history_to_prompt(
        full_history,
        user_prompt.strip(),
    )

    reply = await rpc.call_llm(
        prompt=rendered_prompt,
        history=[],  # or omit
        temperature=temperature,
    )

    # ─────────────────────────────────────────────────────────────
    # 🧑‍⚖️ Phase 4: Choose backend (Brain vs Council)
    # ─────────────────────────────────────────────────────────────
    if mode == "council":
        from .council_rpc import CouncilRPC  # assuming you already have this
        rpc = CouncilRPC(bus)
    else:
        rpc = BrainRPC(bus)

    reply = await rpc.call_llm(
        prompt=user_prompt,
        history=full_history,
        temperature=temperature,
    )

    # Normalize text + token count
    text = reply.get("text") or reply.get("response") or ""
    tokens = len(text.split()) if text else 0

    # ─────────────────────────────────────────────────────────────
    # 🧾 Phase 5: Store chat tail in Redis (last 20 entries)
    # ─────────────────────────────────────────────────────────────
    client = getattr(bus, "client", None)
    if client is not None:
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

    return {
        "session_id": session_id,
        "mode": mode,
        "text": text,
        "tokens": tokens,
        "raw": reply,
        "recall_debug": recall_debug,
    }


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

    from .recall_rpc import RecallRPC
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
