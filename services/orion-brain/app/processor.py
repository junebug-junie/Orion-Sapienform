import base64
from uuid import uuid4
import httpx
import uuid
import json
import logging
import os
from datetime import date

from orion.core.bus.service import OrionBus
from app.config import (
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
    ORION_BUS_URL,
    ORION_BUS_ENABLED,
    CHANNEL_DREAM_TRIGGER,
)
from app.router import router_instance
from app.bus_helpers import (
    emit_brain_event,
    emit_brain_output,
    emit_chat_history_log,
    # assumes you've added this; if not, you can temporarily inline a publish
    emit_cortex_step_result,
)
from app.tts_gpu import TTSEngine

logger = logging.getLogger(__name__)

_tts_engine = None


# ─────────────────────────────────────────────
# Legacy brain request handler (chat + dreams)
# ─────────────────────────────────────────────

def process_brain_request(payload: dict):
    """
    Handles a single request from the bus.
    Routes between "dream_synthesis" (fire-and-forget)
    and standard "chat" (request/reply).

    Expected payload (chat path, example):
      {
        "event": "chat",
        "source": "hub",
        "kind": "chat" | "warm_start",
        "prompt": "...",
        "history": [...],
        "temperature": 0.7,
        "model": "llama3.1:8b-instruct-q8_0",
        "response_channel": "orion:brain:rpc:<uuid>",
        "fragments": [...],
        "metrics": {...}
      }

    Expected payload (dream_synthesis path, example):
      {
        "event": "dream_synthesis",
        "source": "dream_synthesis",
        "content": { ...ollama-style payload... },
        "fragments": [...],
        "metrics": {...}
      }
    """
    trace_id = payload.get("trace_id") or str(uuid.uuid4())
    source = payload.get("source")
    kind = payload.get("kind") or "chat"   # e.g. "warm_start" or "chat"
    response_channel = payload.get("response_channel")

    fragments_data = payload.get("fragments", [])
    metrics_data = payload.get("metrics", {})

    # --- 1. ROUTING LOGIC ---
    is_dream_task = (source == "dream_synthesis")

    if not is_dream_task and not response_channel:
        logger.warning(f"[{trace_id}] Bus request (source: {source}) missing 'response_channel'. Discarding.")
        return

    backend = router_instance.pick()
    if not backend:
        logger.error(f"[{trace_id}] No healthy backends available. Cannot process request.")
        return

    emit_brain_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    # --- 2. PAYLOAD LOGIC ---
    ollama_payload = {}

    if is_dream_task:
        logger.info(f"[{trace_id}] Processing DREAM SYNTHESIS request...")
        ollama_payload = payload.get("content")
        if not ollama_payload:
            logger.error(f"[{trace_id}] Dream task missing 'content' payload. Discarding.")
            return
    else:
        # Standard chat request via RPC
        prompt_text = payload.get("prompt") or "No prompt provided."
        history = payload.get("history") or []
        temperature = payload.get("temperature", 0.7)
        model = payload.get("model", "llama3.1:8b-instruct-q8_0")

        logger.info(f"[{trace_id}] Processing CHAT request: {prompt_text[:80]}")

        # Convert conversation history into Ollama `messages[]`
        messages = []
        for h in history:
            role = h.get("role", "user")
            content = h.get("content", "")
            messages.append({"role": role, "content": content})

        # Append the new user prompt
        messages.append({"role": "user", "content": prompt_text})

        # Build final Ollama inference payload
        ollama_payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }

    url = f"{backend.url.rstrip('/')}/api/chat"
    data = {}

    # --- 3. EXECUTION ---
    try:
        with httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = client.post(url, json=ollama_payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to contact backend {url}: {e}")
        return

    # --- 4. RESPONSE PARSING ---
    text = ""
    if "message" in data and "content" in data["message"]:
        text = data["message"]["content"].strip()
    else:
        logger.warning(f"[{trace_id}] No 'message.content' in response: {data}")

    # --- 5. RESPONSE ROUTING ---
    if is_dream_task:
        logger.info(f"[{trace_id}] Dream synthesis complete. Publishing to SQL writer channel...")

        try:
            # 1. Parse the LLM's JSON response
            dream_obj = json.loads(text) if text.startswith("{") else {
                "narrative": text,
                "tldr": "Partial dream",
            }

            # 2. Define 'final_payload'
            final_payload = {
                **dream_obj,
                "trace_id": trace_id,
                "source": "dream_synthesis",
                "fragments": fragments_data,
                "metrics": metrics_data,
            }

            # 3. Publish 'final_payload'
            bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
            bus.publish(CHANNEL_DREAM_TRIGGER, final_payload)
            logger.info(f"[{trace_id}] 🚀 Published dream to {CHANNEL_DREAM_TRIGGER} for SQL writer.")

        except Exception as e:
            logger.error(f"[{trace_id}] 🔴 FAILED to parse or publish dream JSON: {e}", exc_info=True)

    else:
        # This is a chat. Publish to history and send the reply.

        if kind == "warm_start":
            logger.info(f"[{trace_id}] Warm-start request; skipping chat history log.")
        else:
            emit_chat_history_log({
                "trace_id": trace_id,
                "source": source or "bus",
                "prompt": payload.get("prompt"),
                "response": text,
            })

        emit_brain_output({
            "trace_id": trace_id,
            "text": text or "(empty response)",
            "service": "orion-brain",
            "model": data.get("model"),
        })

        # --- Send the final reply back ---
        try:
            reply_bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
            reply_payload = {
                "trace_id": trace_id,
                "text": text,
                "meta": data,
            }
            reply_bus.publish(response_channel, reply_payload)
            logger.info(f"[{trace_id}] Sent final reply to {response_channel}")
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to publish reply to {response_channel}: {e}")


# ─────────────────────────────────────────────
# Cortex execution path (semantic layer)
# ─────────────────────────────────────────────

def build_cortex_prompt(req: dict) -> str:
    """
    Assemble a semantic-layer-aware prompt for the brain.

    `req` is a plain dict with keys like:
      verb, step, origin_node, prompt_template, args, context, ...
    """
    lines: list[str] = []

    lines.append(f"# Orion Cognitive Step: {req.get('step')}")
    lines.append(f"# Verb: {req.get('verb')}")
    lines.append(f"# Origin Node: {req.get('origin_node')}")
    lines.append("")

    tmpl = req.get("prompt_template")
    if tmpl:
        lines.append(f"Template: {tmpl}")
        lines.append("")

    args = req.get("args") or {}
    if args:
        lines.append("Args:")
        lines.append(json.dumps(args, indent=2))
        lines.append("")

    context = req.get("context") or {}
    if context:
        lines.append("Context:")
        lines.append(json.dumps(context, indent=2))
        lines.append("")

    lines.append("Generate your introspective continuation.")
    return "\n".join(lines)


def call_brain_llm(prompt: str) -> str:
    """
    Calls your existing brain LLM endpoint for Cortex.
    Uses the same backend router as legacy chat.
    """
    backend = router_instance.pick()
    if not backend:
        return "[BrainLLMService Error] no healthy backend available"

    url = f"{backend.url.rstrip('/')}/api/chat"
    payload = {
        "model": "llama3.1:8b-instruct-q8_0",  # TODO: make configurable
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"[CORTEX] LLM call failed: {e}")
        return f"[BrainLLMService Error] {e}"


def process_cortex_exec_request(payload: dict):
    """
    Handles a Cortex execution step for BrainLLMService.

    Expected payload (dict), example shape:

      {
        "event": "exec_step",
        "service": "BrainLLMService",
        "verb": "introspect",
        "step": "llm_reflect",
        "order": 0,
        "requires_gpu": false,
        "requires_memory": false,
        "prompt_template": "IntrospectionPromptTemplate",
        "args": {...},
        "context": {...},
        "correlation_id": "...",
        "reply_channel": "orion-exec:result:<uuid>",
        "origin_node": "athena-cortex"
      }
    """
    if payload.get("event") != "exec_step":
        logger.warning(f"[CORTEX] Ignoring non-exec_step payload: {payload}")
        return

    service = payload.get("service", "BrainLLMService")
    correlation_id = payload.get("correlation_id") or str(uuid.uuid4())
    reply_channel = payload.get("reply_channel")

    if not reply_channel:
        logger.error(f"[CORTEX] Missing reply_channel in payload: {payload}")
        return

    logger.info(
        f"[CORTEX] Received execution step '{payload.get('step')}' "
        f"for verb '{payload.get('verb')}' (service={service}, cid={correlation_id})"
    )

    # 1. Build prompt
    prompt = build_cortex_prompt(payload)

    # 2. Call the LLM
    llm_text = call_brain_llm(prompt)

    # 3. Build result for Cortex
    result = {
        "prompt": prompt,
        "llm_output": llm_text,
    }

    # 4. Emit standardized exec_step_result back to Cortex
    emit_cortex_step_result(
        service=service,
        correlation_id=correlation_id,
        reply_channel=reply_channel,
        result=result,
        artifacts={},
        status="success",
    )


# ─────────────────────────────────────────────
# TTS path
# ─────────────────────────────────────────────

def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()  # or pass model name from config
    return _tts_engine


def process_tts_request(payload: dict):
    """
    Handles a single TTS request from the bus.

    Expected payload:
      {
        "trace_id": "...",
        "text": "hello world",
        "response_channel": "orion:tts:rpc:<uuid>",
        "source": "hub"
      }
    """
    trace_id = payload.get("trace_id") or str(uuid4())
    text = payload.get("text") or ""
    response_channel = payload.get("response_channel")

    if not text:
        logger.warning(f"[{trace_id}] TTS request missing 'text'. Discarding.")
        return

    if not response_channel:
        logger.warning(f"[{trace_id}] TTS request missing 'response_channel'. Discarding.")
        return

    logger.info(f"[{trace_id}] Processing TTS request (len={len(text)})")

    try:
        engine = get_tts_engine()
        audio_b64 = engine.synthesize_to_b64(text)

        reply_bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
        reply_payload = {
            "trace_id": trace_id,
            "audio_b64": audio_b64,
            "mime_type": "audio/wav",
        }
        reply_bus.publish(response_channel, reply_payload)
        logger.info(f"[{trace_id}] Sent TTS reply to {response_channel}")

    except Exception as e:
        logger.error(f"[{trace_id}] FAILED to synthesize TTS: {e}", exc_info=True)


# ─────────────────────────────────────────────
# Unified router for brain intake + cortex exec
# ─────────────────────────────────────────────

def process_brain_or_cortex(payload: dict):
    """
    Route between:
      - legacy brain RPC (chat / dream_synthesis)
      - Cortex semantic exec_step messages (BrainLLMService)
    """
    if payload.get("event") == "exec_step" and payload.get("service") == "BrainLLMService":
        return process_cortex_exec_request(payload)

    return process_brain_request(payload)
