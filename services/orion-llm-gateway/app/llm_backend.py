from typing import Any, Dict, Optional, List
import logging
import time
import json

import httpx
from httpx import HTTPStatusError

from orion.core.bus.service import OrionBus

from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .settings import settings
from .profiles import LLMProfileRegistry, LLMProfile

from orion.spark.integration import (
    ingest_chat_and_get_state,
    build_collapse_mirror_meta,
)

logger = logging.getLogger("orion-llm-gateway.backend")

# ─────────────────────────────────────────────
# 1. Configuration & Profile Loading
# ─────────────────────────────────────────────

_profile_registry: LLMProfileRegistry = settings.load_profile_registry()


def _common_http_client() -> httpx.Client:
    """
    Returns an HTTP client using the configured timeouts from settings.
    """
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=getattr(settings, "connect_timeout_sec", 10.0),
            read=getattr(settings, "read_timeout_sec", 60.0),
            write=10.0,
            pool=10.0,
        )
    )


# ─────────────────────────────────────────────
# 2. Helpers (Text Extraction, Profiles)
# ─────────────────────────────────────────────

def _extract_text_from_openai_response(data: Dict[str, Any]) -> str:
    """
    Generic extractor for OpenAI-compatible responses (vLLM & llama.cpp).
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""

        first = choices[0] or {}

        # 1. Chat Completion
        msg = first.get("message")
        if isinstance(msg, dict) and msg.get("content") is not None:
            return str(msg["content"]).strip()

        # 2. Text Completion
        if "text" in first:
            return str(first["text"]).strip()

    except Exception:
        pass

    logger.warning(f"[LLM-GW] Response format not understood: {str(data)[:200]}...")
    return ""


def _extract_vector_from_openai_response(data: Dict[str, Any]) -> Optional[List[float]]:
    """
    Best-effort extraction of an embedding/state vector from OpenAI-compatible
    response payloads.

    We support multiple possible key names because Orion may run mixed gateways:
    - A "neural" llama.cpp host that returns an internal state vector
    - A standard host that returns only text

    Preferred key order:
      1) spark_vector
      2) state_embedding / state_vector
      3) embedding / embeds / vector

    Also checks inside choices[0] for the same keys.
    """

    def _maybe_vec(obj: Any) -> Optional[List[float]]:
        if isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
            return [float(x) for x in obj]
        return None

    # Top-level keys
    for k in ("spark_vector", "state_embedding", "state_vector", "embedding", "embeds", "vector"):
        v = _maybe_vec(data.get(k))
        if v is not None:
            return v

    # Common OpenAI-compatible location: choices[0]
    try:
        choices = data.get("choices") or []
        first = (choices[0] or {}) if choices else {}
        for k in ("spark_vector", "state_embedding", "state_vector", "embedding", "embeds", "vector"):
            v = _maybe_vec(first.get(k))
            if v is not None:
                return v
    except Exception:
        pass

    return None


def _select_profile(profile_name: str | None) -> LLMProfile | None:
    if not _profile_registry.profiles:
        return None

    # 1. Explicit
    if profile_name:
        try:
            return _profile_registry.get(profile_name)
        except KeyError:
            logger.warning(f"[LLM-GW] Profile '{profile_name}' not found.")

    # 2. Default
    if settings.llm_default_profile_name:
        try:
            return _profile_registry.get(settings.llm_default_profile_name)
        except KeyError:
            pass

    return None


def _pick_backend(options: Dict[str, Any] | None, profile: LLMProfile | None) -> str:
    opts = options or {}
    backend = opts.get("backend")
    if not backend and profile:
        backend = profile.backend

    backend = (backend or settings.default_backend or "vllm").lower()

    if backend not in ("vllm", "llamacpp"):
        logger.warning(f"[LLM-GW] Unknown backend '{backend}'; defaulting to vllm")
        return "vllm"
    return backend


def _resolve_model(body_model: str | None, profile: LLMProfile | None) -> str:
    if body_model:
        return body_model
    if profile:
        return profile.model_id

    # Fallback logic
    fallback = getattr(settings, "default_model", None)
    if not fallback:
        # Try env var scan
        for attr in ("ORION_DEFAULT_LLM_MODEL", "orion_default_llm_model"):
            val = getattr(settings, attr, None)
            if val:
                fallback = val
                break

    return fallback or "Active-GGUF-Model"


def _normalize_model_for_vllm(model: str) -> str:
    """Resolve aliases like 'mistral' -> 'mistralai/Mistral-7B...' via registry."""
    if not model or "/" in model:
        return model
    try:
        return _profile_registry.get(model).model_id
    except Exception:
        return model


# ─────────────────────────────────────────────
# 3. Spark Integration
# ─────────────────────────────────────────────

def _get_raw_user_text(body: ChatBody) -> Optional[str]:
    """
    Best-effort extraction of a canonical 'raw user text' that should drive Spark.

    Why: some verbs/packs (e.g., plan_goal / exec_step) build a mega-prompt that
    includes templates + context. If Spark encodes that scaffold, metrics will
    swing dramatically across verbs and habituate incorrectly.

    Preferred location (non-breaking): body.options["raw_user_text"].
    Also supports a couple aliases to keep it flexible.
    """
    try:
        if getattr(body, "raw_user_text", None):
            raw = str(body.raw_user_text or "").strip()
            if raw:
                return raw
        opts = body.options or {}
        raw = (
            opts.get("raw_user_text")
            or opts.get("spark_raw_user_text")
            or opts.get("user_text")
        )
        if raw is None:
            return None
        raw = str(raw).strip()
        return raw or None
    except Exception:
        return None


def _derive_mode_tags(verb: str, text: str | None) -> List[str]:
    verb_l = (verb or "").lower()
    text_l = (text or "").lower()
    tags: List[str] = []

    if any(k in verb_l or k in text_l for k in ("summary", "summarize", "tl;dr")):
        tags.append("mode:summarize")
    if any(k in verb_l or k in text_l for k in ("analy", "analysis", "inspect", "review")):
        tags.append("mode:analyze")
    if any(k in verb_l or k in text_l for k in ("debug", "traceback", "stack", "error")):
        tags.append("mode:debug")
    if any(k in verb_l or k in text_l for k in ("plan", "goal", "roadmap", "exec", "step")):
        tags.append("mode:plan")
    if any(k in verb_l or k in text_l for k in ("build", "code", "implement", "write")):
        tags.append("mode:build")
    if not tags:
        tags.append("mode:chat")
    return tags


def _spark_ingest_text(*, text: str, agent_id: str, tags: List[str], spark_vector: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Thin wrapper so we can reuse the ingest pathway consistently.
    """
    state = ingest_chat_and_get_state(
        user_message=str(text).strip(),
        agent_id=agent_id,
        tags=tags,
        sentiment=None,
        spark_vector=spark_vector,
    )
    return state


def _spark_ingest_for_body(body: ChatBody) -> Dict[str, Any]:
    try:
        messages = body.messages or []
        if not messages:
            return {}

        source = getattr(body, "source", "llm-gateway")
        verb = getattr(body, "verb", None) or "unknown"

        # Prefer canonical raw user text if provided (prevents encoding mega-prompts).
        raw_user_text = _get_raw_user_text(body)
        latest_user = raw_user_text

        # Fallback: Find last user message in message list
        if not latest_user:
            for m in reversed(messages):
                if (m.get("role") or "").lower() == "user":
                    latest_user = m.get("content")
                    break

        # Fallback: last message content
        if not latest_user:
            latest_user = messages[-1].get("content")

        if not latest_user:
            return {}

        # Debug: prove exactly what Spark is encoding pre-LLM
        logger.warning(
            "[SPARK_DEBUG] phase=pre verb=%s source=%s using_raw=%s latest_user_200=%r",
            verb,
            source,
            bool(raw_user_text),
            str(latest_user)[:200],
        )

        tags = ["juniper", "chat", source, f"verb:{verb}", "phase:pre", *(_derive_mode_tags(verb, latest_user))]
        spark_vector: Optional[List[float]] = None
        if settings.include_embeddings:
            try:
                spark_vector = _fetch_embedding_internal(str(latest_user))
            except Exception as embed_err:
                logger.warning(f"[LLM-GW Spark] Pre-ingest embedding failed: {embed_err}")

        state = _spark_ingest_text(text=str(latest_user), agent_id=source, tags=tags, spark_vector=spark_vector)

        meta = build_collapse_mirror_meta(
            state["phi_after"],
            state["surface_encoding"],
            self_field=state.get("self_field"),
        )
        meta.update(
            {
                "phi_before": state["phi_before"],
                "phi_after": state["phi_after"],
                "latest_user_message": str(latest_user),
                "trace_verb": verb,
                "spark_phase": "pre",
                "spark_used_raw_user_text": bool(raw_user_text),
            }
        )
        return meta
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Ingestion failed: {e}")
        return {}


def _maybe_publish_spark_introspect(body: ChatBody, spark_meta: Dict, response_text: str):
    try:
        if not spark_meta:
            return

        phi_before = spark_meta.get("phi_before", {})
        phi_after = spark_meta.get("phi_after", {})
        self_field = spark_meta.get("spark_self_field", {})

        delta = abs((phi_after.get("valence", 0) - phi_before.get("valence", 0)))

        if delta > 0.05 or self_field.get("uncertainty", 0) > 0.3:
            bus = OrionBus(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
            if bus.enabled:
                bus.publish(
                    settings.channel_spark_introspect_candidate,
                    {
                        "event": "spark_introspect_candidate",
                        "trace_id": body.trace_id or "gw",
                        "source": getattr(body, "source", "gw"),
                        "kind": "chat",
                        "prompt": spark_meta.get("latest_user_message"),
                        "response": response_text,
                        "spark_meta": spark_meta,
                    },
                )
    except Exception as e:
        logger.error(f"[LLM-GW Spark] Publish failed: {e}")


def _spark_post_ingest_for_reply(body: ChatBody, spark_meta: Dict[str, Any], response_text: str) -> None:
    """
    Post-LLM Spark ingest: encode the assistant reply too.

    Why: encoding only the user's input often under-represents semantic change.
    Assistant responses are frequently the "semantic bulk" (summaries, plans, code).

    This intentionally does NOT replace phi_before/phi_after from the pre-ingest.
    It writes separate keys so downstream analysis can compare phases cleanly.
    """
    try:
        if not spark_meta:
            return
        if not response_text:
            return

        source = getattr(body, "source", "llm-gateway")
        verb = getattr(body, "verb", None) or "unknown"

        logger.warning(
            "[SPARK_DEBUG] phase=post verb=%s source=%s reply_200=%r",
            verb,
            source,
            str(response_text)[:200],
        )

        tags = ["juniper", "chat", source, f"verb:{verb}", "assistant_reply", "phase:post", *(_derive_mode_tags(verb, response_text))]
        spark_vector: Optional[List[float]] = None
        if settings.include_embeddings:
            try:
                spark_vector = _fetch_embedding_internal(str(response_text))
            except Exception as embed_err:
                logger.warning(f"[LLM-GW Spark] Post-ingest embedding failed: {embed_err}")

        post_state = _spark_ingest_text(text=str(response_text), agent_id=source, tags=tags, spark_vector=spark_vector)

        # Bound what we store; keep full text out of telemetry by default.
        spark_meta["latest_assistant_message"] = str(response_text)[:2000]
        spark_meta["phi_post_before"] = post_state.get("phi_before")
        spark_meta["phi_post_after"] = post_state.get("phi_after")
        spark_meta["spark_phase_post"] = True
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Post-ingest failed: {e}")


# ─────────────────────────────────────────────
# 4. Core HTTP Execution (Unified)
# ─────────────────────────────────────────────

def _fetch_embedding_internal(text: str) -> Optional[List[float]]:
    """
    Internal helper to fetch embeddings for generated text.
    Uses llamacpp embedding lobe by default if configured, otherwise falls back to vLLM.
    """
    # Prefer dedicated embedding lobe
    url = None
    if settings.llamacpp_embedding_url:
        url = f"{settings.llamacpp_embedding_url.rstrip('/')}/v1/embeddings"
    elif settings.llamacpp_url:
        # Fallback to standard host for embeddings if neural host not configured
        url = f"{settings.llamacpp_url.rstrip('/')}/v1/embeddings"
    elif settings.vllm_url:
        url = f"{settings.vllm_url.rstrip('/')}/v1/embeddings"

    if not url:
        return None

    try:
        # Use a generic model name or specific if known.
        # For llama-server embeddings, model name is often ignored or can be anything.
        payload = {"model": "default", "input": text}

        with _common_http_client() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # Standard OpenAI format: data: [{embedding: [...]}]
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"[LLM-GW] Embedding fetch failed: {e}")

    return None


def _execute_openai_chat(
    body: ChatBody,
    model: str,
    base_url: str,
    backend_name: str,
) -> Dict[str, Any]:
    """
    Unified logic for both vLLM and llama.cpp since they share the OpenAI API shape.
    """
    if not base_url:
        err = f"{backend_name} URL not configured"
        logger.error(f"[LLM-GW] {err}")
        return {"text": f"[Error: {err}]", "spark_meta": {}, "raw": {}}

    # 1. Spark Ingestion (pre-LLM)
    spark_meta = _spark_ingest_for_body(body)

    # 2. Prepare Payload
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    opts = body.options or {}

    payload = {
        "model": model,
        "messages": body.messages,
        "stream": False,
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
        "max_tokens": opts.get("max_tokens"),
        "stop": opts.get("stop"),
        "presence_penalty": opts.get("presence_penalty"),
        "frequency_penalty": opts.get("frequency_penalty"),
    }
    # Clean None values
    payload = {k: v for k, v in payload.items() if v is not None}

    # 3. Execute
    logger.info(f"[LLM-GW] {backend_name} req model={model} msgs={len(body.messages or [])} url={url}")

    try:
        with _common_http_client() as client:
            r = client.post(url, json=payload)

            if r.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} 404 Not Found at {url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }

            r.raise_for_status()
            raw_data = r.json()
            text = _extract_text_from_openai_response(raw_data)

            # 3b. Spark Post-Ingest (assistant reply)
            _spark_post_ingest_for_reply(body, spark_meta, text)

            # Post-processing: embed/state vector (if present)
            spark_vector = _extract_vector_from_openai_response(raw_data)

            # Case B: Standard Host (Reflective)
            # If the backend didn't include a vector, optionally do a secondary
            # embedding call (useful for a *separate* "embeds-on" gateway instance).
            if settings.include_embeddings and (not spark_vector) and text:
                spark_vector = _fetch_embedding_internal(text)

            # Post-processing: Spark Introspect
            _maybe_publish_spark_introspect(body, spark_meta, text)

            return {
                "text": text,
                "spark_meta": spark_meta,
                "spark_vector": spark_vector,
                "raw": raw_data,
            }

    except httpx.TimeoutException:
        logger.error(f"[LLM-GW] {backend_name} TIMEOUT on {url}")
        return {
            "text": f"[Error: {backend_name} timed out after waiting]",
            "spark_meta": spark_meta,
            "raw": {},
        }
    except Exception as e:
        logger.error(f"[LLM-GW] {backend_name} error: {e}", exc_info=True)
        return {
            "text": f"[Error: {backend_name} failed: {str(e)}]",
            "spark_meta": spark_meta,
            "raw": {},
        }


# ─────────────────────────────────────────────
# 5. Public Entrypoints
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody) -> Dict[str, Any]:
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    # Normalize aliases if vLLM, harmless if llama.cpp
    if backend == "vllm":
        model = _normalize_model_for_vllm(model)
        base_url = settings.vllm_url
    else:
        base_url = settings.llamacpp_url

    return _execute_openai_chat(body, model, base_url, backend)


def run_llm_generate(body: GenerateBody) -> str:
    """Wrapper to make Generate look like Chat"""
    chat_body = ChatBody(
        messages=[{"role": "user", "content": body.prompt}],
        options=body.options,
        trace_id=body.trace_id,
        user_id=body.user_id,
        session_id=body.session_id,
        source=body.source,
        verb=body.verb,
        profile_name=body.profile_name,
        # Pass through model implicitly via main logic
        model=body.model,
    )
    result = run_llm_chat(chat_body)
    return result.get("text") or ""


def run_llm_embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    # Embeddings logic is distinct enough to keep separate for now
    url = None
    if settings.llamacpp_embedding_url:
        url = f"{settings.llamacpp_embedding_url.rstrip('/')}/v1/embeddings"
    elif settings.vllm_url:
        url = f"{settings.vllm_url.rstrip('/')}/v1/embeddings"

    if not url:
        raise RuntimeError("No embedding URL configured (vLLM or LlamaCpp Embedding Lobe)")

    profile = _select_profile(body.profile_name)
    model = _resolve_model(body.model, profile)
    model = _normalize_model_for_vllm(model)

    payload = {"model": model, "input": body.input}
    if body.options:
        payload.update(body.options)

    with _common_http_client() as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def run_llm_exec_step(body: ExecStepPayload) -> Dict[str, Any]:
    t0 = time.time()

    # 1. Build Prompt
    if body.prompt:
        final_prompt = body.prompt
    else:
        ctx_json = json.dumps(body.context or {}, indent=2, ensure_ascii=False)
        prior_json = json.dumps(body.prior_step_results or [], indent=2, ensure_ascii=False)
        final_prompt = f"{body.prompt_template or ''}\n\n# Context\n{ctx_json}\n\n# Prior Results\n{prior_json}\n"

    # 2. Resolve Config
    profile = _select_profile(getattr(body, "profile_name", None))
    backend = _pick_backend({}, profile)
    model = _resolve_model(None, profile)
    if backend == "vllm":
        model = _normalize_model_for_vllm(model)

    # 3. Execute via Chat Interface
    chat_body = ChatBody(
        model=model,
        messages=[{"role": "user", "content": final_prompt}],
        raw_user_text=body.raw_user_text or (body.context.get("user_message") if isinstance(body.context, dict) else None),
        options={},
        trace_id=body.origin_node,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=getattr(body, "profile_name", None),
    )

    if backend == "llamacpp":
        result = _execute_openai_chat(chat_body, model, settings.llamacpp_url, "llamacpp")
    else:
        result = _execute_openai_chat(chat_body, model, settings.vllm_url, "vllm")

    elapsed_ms = int((time.time() - t0) * 1000)

    # 4. Log (WITH FIXED ARGUMENTS)
    logger.info(
        "[LLM-GW] exec_step verb=%s step=%s service=%s elapsed_ms=%d backend=%s model=%s",
        body.verb,
        body.step,
        body.service,
        elapsed_ms,
        backend,
        model,
    )

    return {
        "prompt": final_prompt,
        "llm_output": result.get("text") or "",
        "spark_meta": result.get("spark_meta"),
        "spark_vector": result.get("spark_vector"),
        "raw_llm": result.get("raw_llm") or result.get("raw"),
    }
