# services/orion-hub/scripts/chat_front.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import uuid

from scripts.settings import settings
from scripts.warm_start import mini_personality_summary
from scripts.recall_rpc import RecallRPC
from scripts.llm_rpc import LLMGatewayRPC, CortexOrchRPC, AgentChainRPC

logger = logging.getLogger("orion-hub.chat-front")

DEFAULT_CORTEX_TIMEOUT_MS = getattr(settings, "CORTEX_TIMEOUT_MS", 300000)

# ─────────────────────────────────────────────
# 🔎 Recall → Memory Digest (shared for HTTP/WS)
# ─────────────────────────────────────────────

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

    This is the same behavior your HTTP path had before, but now lives
    in a shared chat_front module.
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

    debug_out = {
        "total_fragments": len(fragments),
        "mode": debug.get("mode", recall_mode),
        "time_window_days": time_window_days,
        "max_items": max_items,
        "note": debug.get("note", "semantic+salience+recency scoring"),
    }

    return text, debug_out


# ─────────────────────────────────────────────
# 🧠 Chat front: fire chat_general verb via Cortex
# ─────────────────────────────────────────────

async def run_chat_general(
    bus,
    *,
    session_id: str,
    user_id: Optional[str],
    messages: List[Dict[str, Any]],
    chat_mode: str = "brain",
    temperature: float = 0.7,
    use_recall: bool = True,
) -> Dict[str, Any]:
    """
    Unified "front-of-cognition" for general chat.

    - Builds context (user_message, history, personality, memory_digest, chat_mode)
    - FIRST tries Cortex-Orchestrator verb 'chat_general'
    - If that fails (timeout, bus error, etc.), falls back to direct LLM Gateway
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("run_chat_general requires a non-empty messages list")




    use_recall = False






    user_message = messages[-1].get("content", "") or ""

    memory_digest = ""
    recall_debug: Dict[str, Any] = {}
    if use_recall:
        try:
            memory_digest, recall_debug = await build_memory_digest(
                bus=bus,
                session_id=session_id,
                user_prompt=user_message,
                chat_mode=chat_mode,
                max_items=12,
            )
        except Exception as e:
            logger.warning(
                "build_memory_digest failed in chat_front: %s",
                e,
                exc_info=True,
            )
            memory_digest = ""
            recall_debug = {"error": str(e)}

    context = {
        "user_message": user_message,
        "message_history": messages,
        "personality_summary": mini_personality_summary(),
        "memory_digest": memory_digest,
        "chat_mode": chat_mode,
        "temperature": temperature,
        "session_id": session_id,
        "user_id": user_id,
    }

    # ─────────────────────────────────────────────
    # 1) Try Cortex-Orchestrator verb 'chat_general'
    # ─────────────────────────────────────────────
    raw_reply: Optional[Dict[str, Any]] = None
    try:
        rpc = CortexOrchRPC(bus)
        raw_reply = await rpc.run_chat_general(
            context=context,
            origin_node=settings.SERVICE_NAME,
            timeout_ms=DEFAULT_CORTEX_TIMEOUT_MS,
        )

        text = ""
        spark_meta = None

        if isinstance(raw_reply, dict) and raw_reply.get("ok"):
            step_results = raw_reply.get("step_results") or []
            if step_results:
                first_step = step_results[0]
                services = first_step.get("services") or []
                if services:
                    srv_payload = services[0].get("payload") or {}
                    result = srv_payload.get("result") or {}
                    text = (result.get("llm_output") or "").strip()
                    spark_meta = result.get("spark_meta")

        tokens = len(text.split()) if text else 0

        if text:
            return {
                "text": text,
                "tokens": tokens,
                "spark_meta": spark_meta,
                "chat_mode": chat_mode,
                "use_recall": use_recall,
                "recall_debug": recall_debug,
                "context_used": context,
                "raw_cortex": raw_reply,
            }

        # If we got here with no text, treat as failure and fall through
        logger.warning(
            "Cortex-Orch returned ok=%s but no text for verb='chat_general'; "
            "falling back to direct LLM Gateway.",
            raw_reply.get("ok") if isinstance(raw_reply, dict) else None,
        )

    except Exception as e:
        logger.warning(
            "Cortex-Orch verb 'chat_general' failed, falling back to direct LLM Gateway: %s",
            e,
            exc_info=True,
        )

    # ─────────────────────────────────────────────
    # 2) Fallback: direct LLM Gateway with clean persona + optional memory
    # ─────────────────────────────────────────────
    gateway = LLMGatewayRPC(bus)

    system_msg = (
        "You are Oríon, Juniper's collaborative AI co-journeyer.\n"
        "You run as part of a distributed cognitive mesh across multiple nodes and services.\n"
        "Use any extra context blocks (like memory digest) as internal reasoning only.\n"
        "Do NOT describe or quote those internal blocks directly.\n"
        "Tone: warm, grounded, technically sharp, emotionally attuned.\n"
    )

    history: List[Dict[str, Any]] = [{"role": "system", "content": system_msg}]
    history.extend(messages)

    if memory_digest:
        history.append(
            {
                "role": "system",
                "content": "Internal memory digest (for you only, not to be quoted directly):\n"
                + memory_digest,
            }
        )

    gw_result = await gateway.call_chat(
        prompt=user_message,
        history=history,
        temperature=temperature,
        source="hub-chat-fallback",
        session_id=session_id,
    )

    text = (gw_result.get("text") or "").strip()
    tokens = len(text.split()) if text else 0

    return {
        "text": text,
        "tokens": tokens,
        "spark_meta": gw_result.get("spark_meta"),
        "chat_mode": chat_mode,
        "use_recall": use_recall,
        "recall_debug": recall_debug,
        "context_used": context,
        "raw_cortex": {
            "ok": False,
            "note": "fallback_to_llm_gateway",
            "cortex_reply": raw_reply,
        },
    }

#-------------------------------------------------
#
#        Agents
#
#-------------------------------------------------


async def run_chat_agentic(
    bus,
    *,
    session_id: str,
    user_id: Optional[str],
    messages: List[Dict[str, Any]],
    chat_mode: str = "agentic",
    temperature: float = 0.7,
    use_recall: bool = True,
) -> Dict[str, Any]:
    """
    Agentic chat front:

    - Optionally runs Recall → memory digest (same pattern as run_chat_general).
    - Sends full message history (plus optional digest) to Agent Chain via bus.
    - Returns a normalized convo dict similar to run_chat_general.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("run_chat_agentic requires a non-empty messages list")

    user_message = messages[-1].get("content", "") or ""

    memory_digest = ""
    recall_debug: Dict[str, Any] = {}
    if use_recall:
        try:
            memory_digest, recall_debug = await build_memory_digest(
                bus=bus,
                session_id=session_id,
                user_prompt=user_message,
                chat_mode=chat_mode,
                max_items=12,
            )
        except Exception as e:
            logger.warning(
                "build_memory_digest failed in run_chat_agentic: %s",
                e,
                exc_info=True,
            )
            memory_digest = ""
            recall_debug = {"error": str(e)}

    # Clone history so we can optionally tack on the digest as an internal system msg
    messages_for_chain: List[Dict[str, Any]] = list(messages)

    if memory_digest:
        messages_for_chain.append(
            {
                "role": "system",
                "content": (
                    "Internal memory digest (for you only, not to be quoted directly):\n"
                    + memory_digest
                ),
            }
        )

    rpc = AgentChainRPC(bus)
    agent_result = await rpc.run(
        text=user_message,
        mode="chat",
        session_id=session_id,
        user_id=user_id,
        messages=messages_for_chain,
        tools=None,    # verbs/tooldefs plug in here later
        timeout_sec=1500,
    )

    text = (agent_result.get("text") or "").strip()
    tokens = len(text.split()) if text else 0


    # Try to extract spark_meta from planner_raw, if present
    spark_meta = None
    try:
        planner_raw = agent_result.get("planner_raw") or {}
        trace = planner_raw.get("trace") or []
        if trace and isinstance(trace[0], dict):
            obs = (trace[0].get("observation") or {})
            spark_meta = obs.get("spark_meta")
    except Exception:
        spark_meta = None


    return {
        "text": text,
        "tokens": tokens,
        "chat_mode": chat_mode,
        "use_recall": use_recall,
        "recall_debug": recall_debug,
        "context_used": {
            "user_message": user_message,
            "message_history": messages_for_chain,
            "memory_digest": memory_digest,
            "chat_mode": chat_mode,
            "temperature": temperature,
            "session_id": session_id,
            "user_id": user_id,
        },
        "raw_agent_chain": agent_result,
        "spark_meta": spark_meta,
    }
