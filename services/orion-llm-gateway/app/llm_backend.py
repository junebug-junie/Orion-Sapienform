from typing import Any, Dict, Optional
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
            pool=10.0
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

def _spark_ingest_for_body(body: ChatBody) -> Dict[str, Any]:
    try:
        messages = body.messages or []
        if not messages:
            return {}
        
        # Find last user message
        latest_user = None
        for m in reversed(messages):
            if (m.get("role") or "").lower() == "user":
                latest_user = m.get("content")
                break
        
        if not latest_user:
            latest_user = messages[-1].get("content")

        if not latest_user:
            return {}

        source = getattr(body, "source", "llm-gateway")
        state = ingest_chat_and_get_state(
            user_message=str(latest_user).strip(),
            agent_id=source,
            tags=["juniper", "chat", source],
            sentiment=None
        )
        
        meta = build_collapse_mirror_meta(
            state["phi_after"], 
            state["surface_encoding"], 
            self_field=state.get("self_field")
        )
        meta.update({
            "phi_before": state["phi_before"],
            "phi_after": state["phi_after"],
            "latest_user_message": latest_user
        })
        return meta
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Ingestion failed: {e}")
        return {}

def _maybe_publish_spark_introspect(body: ChatBody, spark_meta: Dict, response_text: str):
    try:
        if not spark_meta: return
        
        phi_before = spark_meta.get("phi_before", {})
        phi_after = spark_meta.get("phi_after", {})
        self_field = spark_meta.get("spark_self_field", {})
        
        delta = abs((phi_after.get("valence", 0) - phi_before.get("valence", 0)))
        
        if delta > 0.05 or self_field.get("uncertainty", 0) > 0.3:
            bus = OrionBus(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
            if bus.enabled:
                bus.publish(settings.channel_spark_introspect_candidate, {
                    "event": "spark_introspect_candidate",
                    "trace_id": body.trace_id or "gw",
                    "source": getattr(body, "source", "gw"),
                    "kind": "chat",
                    "prompt": spark_meta.get("latest_user_message"),
                    "response": response_text,
                    "spark_meta": spark_meta
                })
    except Exception as e:
        logger.error(f"[LLM-GW Spark] Publish failed: {e}")

# ─────────────────────────────────────────────
# 4. Core HTTP Execution (Unified)
# ─────────────────────────────────────────────

def _execute_openai_chat(
    body: ChatBody, 
    model: str, 
    base_url: str, 
    backend_name: str
) -> Dict[str, Any]:
    """
    Unified logic for both vLLM and llama.cpp since they share the OpenAI API shape.
    """
    if not base_url:
        err = f"{backend_name} URL not configured"
        logger.error(f"[LLM-GW] {err}")
        return {"text": f"[Error: {err}]", "spark_meta": {}, "raw": {}}

    # 1. Spark Ingestion
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
                    "raw": {}
                }
            
            r.raise_for_status()
            raw_data = r.json()
            text = _extract_text_from_openai_response(raw_data)
            
            # Post-processing
            _maybe_publish_spark_introspect(body, spark_meta, text)
            
            return {
                "text": text,
                "spark_meta": spark_meta,
                "raw": raw_data
            }

    except httpx.TimeoutException:
        logger.error(f"[LLM-GW] {backend_name} TIMEOUT on {url}")
        return {
            "text": f"[Error: {backend_name} timed out after waiting]",
            "spark_meta": spark_meta,
            "raw": {}
        }
    except Exception as e:
        logger.error(f"[LLM-GW] {backend_name} error: {e}", exc_info=True)
        return {
            "text": f"[Error: {backend_name} failed: {str(e)}]",
            "spark_meta": spark_meta,
            "raw": {}
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
        model=body.model
    )
    result = run_llm_chat(chat_body)
    return result.get("text") or ""


def run_llm_embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    # Embeddings logic is distinct enough to keep separate for now
    if not settings.vllm_url:
        raise RuntimeError("vLLM URL not configured for embeddings")

    profile = _select_profile(body.profile_name)
    model = _resolve_model(body.model, profile)
    model = _normalize_model_for_vllm(model)

    url = f"{settings.vllm_url.rstrip('/')}/v1/embeddings"
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
        options={},
        trace_id=body.origin_node,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=getattr(body, "profile_name", None)
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
        "raw_llm": result.get("raw_llm") or result.get("raw"),
    }
