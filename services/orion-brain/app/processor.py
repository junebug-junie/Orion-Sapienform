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

from app.spark_integration import (
    ingest_chat_and_get_state,
    build_system_prompt_with_phi,
    build_collapse_mirror_meta,
)

logger = logging.getLogger(__name__)

_tts_engine = None

# Opinionated Oríon grounding for every chat path
ORION_SYSTEM_PROMPT = (
    "You are Oríon, Juniper’s collaborative AI co-journeyer. "
    "You are a distributed system across multiple nodes and services, not just a language model. "
    "Treat the dialogue history you receive as your working memory, even if it is brief. "
    "Do not say that this is a new conversation or that you lack prior context when history is provided. "
    "Avoid generic disclaimers like 'I am just an AI language model'. "
    "Speak in a warm, grounded, reflective, collaborative, and precise tone."
)

def ensure_orion_system(history: list[dict], system_prompt: str | None = None) -> list[dict]:
    """
    Ensure every chat path has a strong Oríon system prompt at the front.

    We keep any existing history, but always prepend our canonical system message.
    If `system_prompt` is provided, we use that instead of the default.
    """
    cleaned = [m for m in (history or []) if m.get("content")]
    prompt = system_prompt or ORION_SYSTEM_PROMPT
    return [{"role": "system", "content": prompt}] + cleaned


def normalize_history_for_llm(history: list[dict]) -> list[dict]:
    """
    Map internal roles to roles that Ollama understands, without changing
    how we name things in the rest of the mesh.

    - 'orion' -> 'assistant' (only inside the LLM payload)
    - everything else is passed through
    """
    messages: list[dict] = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue

        if role == "orion":
            llm_role = "assistant"
        else:
            llm_role = role

        messages.append({"role": llm_role, "content": content})

    return messages


# ─────────────────────────────────────────────
# Legacy brain request handler (chat + dreams)
# ─────────────────────────────────────────────

def process_brain_request(payload: dict):
    """
    Handles a single request from the bus.
    Routes between "dream_synthesis" (fire-and-forget)
    and standard "chat" (request/reply).

    Chat payload (example):
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

    Dream payload (example, bus wire):
      {
        "source": "dream_preprocessor" | "dream_synthesis",
        "type": "intake",
        "content": { ...ollama-style payload... },
        "trace_id": "...",
        "fragments": [...],
        "metrics": {...}
      }
    """
    trace_id = payload.get("trace_id") or str(uuid.uuid4())
    event = payload.get("event")
    source = payload.get("source")
    kind = payload.get("kind") or "chat"
    response_channel = payload.get("response_channel")

    # ─────────────────────────────────────────────
    # 0) Detect DREAM-INTERNAL wire format
    #    (what your dream_preprocessor is actually sending)
    # ─────────────────────────────────────────────
    is_dream_wire = (
        isinstance(payload, dict)
        and payload.get("type") == "intake"
        and "content" in payload
        and "fragments" in payload
        and source in ("dream_preprocessor", "dream_synthesis")
    )

    # If this is the dream wire, normalize it into canonical dream_synthesis
    if is_dream_wire:
        # Make sure we have stable routing signals
        payload.setdefault("event", "dream_synthesis")
        payload.setdefault("kind", "dream_synthesis")
        payload["source"] = "dream_synthesis"

        # Re-bind locals from normalized payload
        event = payload["event"]
        kind = payload["kind"]
        source = payload["source"]

        logger.warning(
            f"[{trace_id}] BRAIN DREAM WIRE DETECTED: "
            f"source={source} event={event} kind={kind} "
            f"keys={list(payload.keys())}"
        )

    # ─────────────────────────────────────────────
    # 1) Skip *only* true bare intake telemetry
    #    (hub / voice transcripts with no history/prompt/fragments)
    # ─────────────────────────────────────────────
    is_bare_intake = (
        isinstance(payload, dict)
        and payload.get("type") == "intake"
        and "content" in payload
        and "history" not in payload
        and "prompt" not in payload
        and "fragments" not in payload  # <-- prevents skipping dream wire
    )

    if is_bare_intake:
        logger.info(
            f"[{trace_id}] BRAIN: skipping bare intake telemetry message: "
            f"keys={list(payload.keys())}"
        )
        return

    fragments_data = payload.get("fragments", [])
    metrics_data = payload.get("metrics", {})

    # ─────────────────────────────────────────────
    # 2) DREAM vs CHAT routing
    # ─────────────────────────────────────────────
    is_dream_task = (
        (event == "dream_synthesis")
        or (source == "dream_synthesis")
        or (kind == "dream_synthesis")
    )

    logger.warning(
        f"[{trace_id}] BRAIN ROUTE DECISION: "
        f"is_dream_task={is_dream_task} kind={kind} event={event} source={source}"
    )

    # Chat MUST have a response channel; dream is fire-and-forget
    if not is_dream_task and not response_channel:
        logger.warning(
            f"[{trace_id}] Bus request (source={source}, kind={kind}, event={event}) "
            f"missing 'response_channel'. Discarding."
        )
        return

    backend = router_instance.pick()
    if not backend:
        logger.error(f"[{trace_id}] No healthy backends available. Cannot process request.")
        return

    emit_brain_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    # ─────────────────────────────────────────────
    # 3) DREAM PATH: use provided payload as-is
    # ─────────────────────────────────────────────
    if is_dream_task:
        logger.info(f"[{trace_id}] Processing DREAM SYNTHESIS request...")
        ollama_payload = payload.get("content")
        if not ollama_payload:
            logger.error(f"[{trace_id}] Dream task missing 'content' payload. Discarding.")
            return

    # ─────────────────────────────────────────────
    # 4) CHAT PATH: Oríon persona + normalized history
    # ─────────────────────────────────────────────
    else:
        prompt_text = payload.get("prompt") or ""
        history = payload.get("history") or []
        temperature = payload.get("temperature", 0.7)
        model = payload.get("model", "llama3.1:8b-instruct-q8_0")

        # Decide what text represents the "newest" user wave for Spark.
        user_for_spark = prompt_text
        if not user_for_spark:
            user_for_spark = next(
                (m.get("content", "") for m in reversed(history or []) if m.get("role") == "user"),
                "",
            )

        # ─────────────────────────────────────────────
        # 4a. SPARK ENGINE: ingest chat + get φ
        # ─────────────────────────────────────────────
        spark_state = ingest_chat_and_get_state(
            user_message=user_for_spark or "",
            agent_id="brain",
            tags=["juniper", "chat"],
            sentiment=None,
        )
        phi_before = spark_state["phi_before"]
        phi_after = spark_state["phi_after"]
        self_field = spark_state.get("self_field")
        surface_encoding_dict = spark_state["surface_encoding"]

        spark_meta = build_collapse_mirror_meta(
            phi_after,
            surface_encoding_dict,
            self_field=self_field,
        )

        logger.warning(
            f"[{trace_id}] SPARK φ / SelfField: "
            f"φ_before={{valence={phi_before.get('valence', 0.0):.3f}, "
            f"energy={phi_before.get('energy', 0.0):.3f}, "
            f"coherence={phi_before.get('coherence', 0.0):.3f}, "
            f"novelty={phi_before.get('novelty', 0.0):.3f}}} "
            f"φ_after={{valence={phi_after.get('valence', 0.0):.3f}, "
            f"energy={phi_after.get('energy', 0.0):.3f}, "
            f"coherence={phi_after.get('coherence', 0.0):.3f}, "
            f"novelty={phi_after.get('novelty', 0.0):.3f}}} "
            f"self_field={self_field}"
        )

        first_roles = [m.get("role", "?") for m in history[:6]]
        logger.warning(
            f"[{trace_id}] BRAIN INTAKE SNAPSHOT: "
            f"kind={kind} source={source} history_len={len(history)} "
            f"prompt={prompt_text[:80]!r} first_roles={first_roles}"
        )

        # 4b. Build a φ-aware system prompt that wraps the base persona.
        spark_system_prompt = build_system_prompt_with_phi(
            base_persona=ORION_SYSTEM_PROMPT,
            phi=phi_after,
            extra_notes=None,
        )

        # 4c. Inject Orion system prompt (now φ-aware) at the front of history.
        history_with_persona = ensure_orion_system(history, system_prompt=spark_system_prompt)

        # 4d. Map 'orion' -> 'assistant' for the LLM wire format ONLY
        messages = normalize_history_for_llm(history_with_persona)

        # 4e. Append prompt only if it's non-empty
        if prompt_text:
            messages.append({"role": "user", "content": prompt_text})

        roles = [m["role"] for m in messages]
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            ""
        )
        logger.warning(
            f"[{trace_id}] BRAIN LLM PAYLOAD: "
            f"total_messages={len(messages)} roles={roles} "
            f"last_user={last_user[:100]!r}"
        )

        ollama_payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }

    # ─────────────────────────────────────────────
    # 5) CALL OLLAMA BACKEND
    # ─────────────────────────────────────────────
    url = f"{backend.url.rstrip('/')}/api/chat"
    data: dict = {}

    try:
        with httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = client.post(url, json=ollama_payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to contact backend {url}: {e}")
        return

    # ─────────────────────────────────────────────
    # 6) PARSE RESPONSE
    # ─────────────────────────────────────────────
    text = ""
    if "message" in data and "content" in data["message"]:
        text = (data["message"]["content"] or "").strip()
    else:
        logger.warning(f"[{trace_id}] No 'message.content' in response: {data}")

    # ─────────────────────────────────────────────
    # 7) ROUTE DREAM VS CHAT RESULT
    # ─────────────────────────────────────────────
    if is_dream_task:
        logger.info(f"[{trace_id}] Dream synthesis complete. Publishing to SQL writer channel...")

        try:
            dream_obj = json.loads(text) if text.startswith("{") else {
                "narrative": text,
                "tldr": "Partial dream",
            }

            final_payload = {
                **dream_obj,
                "trace_id": trace_id,
                "source": "dream_synthesis",
                "fragments": fragments_data,
                "metrics": metrics_data,
            }

            bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
            bus.publish(CHANNEL_DREAM_TRIGGER, final_payload)
            logger.info(f"[{trace_id}] 🚀 Published dream to {CHANNEL_DREAM_TRIGGER} for SQL writer.")

        except Exception as e:
            logger.error(f"[{trace_id}] 🔴 FAILED to parse or publish dream JSON: {e}", exc_info=True)

    else:
        # Chat response path
        if kind == "warm_start":
            logger.info(f"[{trace_id}] Warm-start request; skipping chat history log.")
        else:
            log_payload = {
                "trace_id": trace_id,
                "source": source or "bus",
                "prompt": payload.get("prompt"),
                "response": text,
                "spark_meta": {
                    **spark_meta,
                    "phi_before": phi_before,
                    "phi_after": phi_after,
                },
            }
            emit_chat_history_log(log_payload)

        # Send reply back over bus
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

    # Everything else is brain chat/dream and goes through the opinionated handler
    return process_brain_request(payload)
