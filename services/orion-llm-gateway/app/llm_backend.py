from typing import Any, Dict
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
# Profile loading
# ─────────────────────────────────────────────

_profile_registry: LLMProfileRegistry = settings.load_profile_registry()


def _extract_text_from_vllm_completion(data: Dict[str, Any]) -> str:
    """
    For /v1/completions-style responses:
      {
        "choices": [
          { "text": "...", ... }
        ],
        ...
      }
    """
    try:
        choices = data.get("choices") or []
        if choices:
            txt = choices[0].get("text")
            if txt:
                return str(txt).strip()
    except Exception:
        pass
    logger.warning(f"[LLM-GW] vLLM /v1/completions response not understood: {data}")
    return ""


def _flatten_messages_to_prompt(messages: list[Dict[str, Any]]) -> str:
    """
    Fallback: turn OpenAI chat-style messages into a single text prompt
    for /v1/completions-style endpoints.
    """
    if not messages:
        return ""

    lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        # simple conversational flattening
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _select_profile(
    profile_name: str | None,
) -> LLMProfile | None:
    """
    Select explicit profile_name.
    Returns None if profiles are not configured.

    Simple selection:

      1) Explicit profile_name, if provided.
      2) Global default profile from settings.
      3) Otherwise: no profile (use settings.default_model).
    """
    if not _profile_registry.profiles:
        return None

    # 1) Explicit override
    if profile_name:
        try:
            return _profile_registry.get(profile_name)
        except KeyError:
            logger.warning(
                "[LLM-GW] Unknown profile_name '%s'; falling back to default",
                profile_name,
            )

    # 2) Default profile for this gateway
    if settings.llm_default_profile_name:
        try:
            return _profile_registry.get(settings.llm_default_profile_name)
        except KeyError:
            logger.warning(
                "[LLM-GW] Default profile '%s' not found in registry",
                settings.llm_default_profile_name,
            )

    # 3) No profile – gateway will fall back to settings.default_model
    return None


# ─────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────

def _pick_backend(options: Dict[str, Any] | None, profile: LLMProfile | None) -> str:
    opts = options or {}

    backend = opts.get("backend")
    if backend:
        backend = backend.lower()
    elif profile is not None:
        backend = profile.backend
    else:
        backend = (settings.default_backend or "vllm").lower()

    if backend not in ("vllm", "llamacpp"):
        logger.warning(
            "[LLM-GW] Unknown backend '%s'; falling back to '%s'",
            backend,
            settings.default_backend or "vllm",
        )
        return (settings.default_backend or "vllm").lower()

    return backend


def _fallback_default_model() -> str:
    """
    Safe fallback when no model and no profile are provided.

    - First, try Settings.default_model if it exists.
    - Otherwise, fall back to a hard-coded name that aligns with llama.cpp.
    """
    # Try Settings.default_model if present
    try:
        dm = getattr(settings, "default_model")
        if dm:
            return dm
    except AttributeError:
        pass

    # Try any alias that might exist on Settings via env
    for attr in ("ORION_DEFAULT_LLM_MODEL", "orion_default_llm_model"):
        if hasattr(settings, attr):
            val = getattr(settings, attr)
            if isinstance(val, str) and val:
                return val

    # Last-resort fallback
    return "Active-GGUF-Model"


def _resolve_model(body_model: str | None, profile: LLMProfile | None) -> str:
    """
    Priority:
      1) body.model if provided
      2) profile.model_id if profile is selected
      3) fallback default model
    """
    if body_model:
        return body_model
    if profile is not None:
        return profile.model_id

    fallback = _fallback_default_model()
    logger.warning(
        "[LLM-GW] No model specified and no profile resolved; "
        "falling back to '%s'",
        fallback,
    )
    return fallback


def _normalize_model_for_vllm(model: str) -> str:
    """
    Ensure vLLM sees a concrete model id (like 'mistralai/Mistral-7B-Instruct-v0.3').

    If `model` looks like a profile key (no slash), try to resolve it via
    the profile registry and return its `model_id`. Otherwise just pass through.
    """
    if not model:
        return model

    # Already looks like a full HuggingFace / OpenAI id
    if "/" in model:
        return model

    try:
        profile = _profile_registry.get(model)
        resolved = profile.model_id
        logger.info(
            "[LLM-GW] Normalized vLLM model alias '%s' → '%s'",
            model,
            resolved,
        )
        return resolved
    except Exception:
        logger.warning(
            "[LLM-GW] Could not normalize vLLM model '%s'; using as-is",
            model,
        )
        return model


def _extract_text_from_vllm(data: Dict[str, Any]) -> str:
    """
    Handle both OpenAI-style chat completions and plain completions:

      - Chat: choices[0].message.content
      - Text: choices[0].text
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("No choices in vLLM response")

        first = choices[0] or {}

        # Chat-style (what /v1/chat/completions returns)
        msg = first.get("message")
        if isinstance(msg, dict) and msg.get("content") is not None:
            return str(msg["content"]).strip()

        # Text-style (what /v1/completions returns)
        if "text" in first and first["text"] is not None:
            return str(first["text"]).strip()

    except Exception:
        pass

    logger.warning(f"[LLM-GW] vLLM/llamacpp response not understood: {data}")
    return ""


def _extract_text_from_brain_chat(data: Dict[str, Any]) -> str:
    if isinstance(data, dict):
        if data.get("response"):
            return str(data["response"]).strip()
        if data.get("text"):
            return str(data["text"]).strip()
    logger.warning(f"[LLM-GW] Brain /chat response not understood: {data}")
    return ""


def _common_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            settings.connect_timeout_sec,
            read=settings.read_timeout_sec,
        )
    )

# ─────────────────────────────────────────────
# Spark integration helpers
# ─────────────────────────────────────────────

def _spark_ingest_for_body(body: ChatBody) -> Dict[str, Any]:
    """
    Run Spark on the latest user message from a ChatBody.
    """
    try:
        messages = body.messages or []
        if not isinstance(messages, list) or not messages:
            return {}

        # Find the latest explicit user message; fall back to the last message.
        latest_user = None
        for m in reversed(messages):
            if (m.get("role") or "").lower() == "user":
                latest_user = m.get("content") or ""
                break

        if latest_user is None:
            latest_user = messages[-1].get("content") or ""

        latest_user = (latest_user or "").strip()
        if not latest_user:
            return {}

        source = getattr(body, "source", None) or "llm-gateway"
        tags = ["juniper", "chat", source]

        spark_state = ingest_chat_and_get_state(
            user_message=latest_user,
            agent_id=source,
            tags=tags,
            sentiment=None,
        )

        phi_before = spark_state["phi_before"]
        phi_after = spark_state["phi_after"]
        self_field = spark_state.get("self_field")
        surface_encoding = spark_state["surface_encoding"]

        spark_meta = build_collapse_mirror_meta(
            phi_after,
            surface_encoding,
            self_field=self_field,
        )

        spark_meta["phi_before"] = phi_before
        spark_meta["phi_after"] = phi_after
        spark_meta["latest_user_message"] = latest_user

        return spark_meta

    except Exception as e:
        logger.warning("[LLM-GW Spark] ingestion failed: %s", e, exc_info=True)
        return {}


def _maybe_publish_spark_introspect_candidate(
    body: ChatBody,
    spark_meta: Dict[str, Any],
    response_text: str,
) -> None:
    try:
        if not spark_meta:
            return

        phi_before = spark_meta.get("phi_before") or {}
        phi_after = spark_meta.get("phi_after") or {}
        self_field = spark_meta.get("spark_self_field") or {}

        delta_valence = abs(
            (phi_after.get("valence") or 0.0)
            - (phi_before.get("valence") or 0.0)
        )
        uncertainty = self_field.get("uncertainty", 0.0)
        stress_load = self_field.get("stress_load", 0.0)

        should_flag = (
            delta_valence > 0.05
            or uncertainty > 0.3
            or stress_load > 0.4
        )

        if not should_flag:
            return

        bus = OrionBus(
            url=settings.orion_bus_url,
            enabled=settings.orion_bus_enabled,
        )
        if not bus.enabled:
            logger.warning("[LLM-GW Spark] bus disabled; skipping introspect candidate")
            return

        trace_id = body.trace_id or "llm-gateway"
        source = getattr(body, "source", None) or "llm-gateway"
        prompt = spark_meta.get("latest_user_message") or ""

        candidate_payload = {
            "event": "spark_introspect_candidate",
            "trace_id": trace_id,
            "source": source,
            "kind": "chat",
            "prompt": prompt,
            "response": response_text,
            "spark_meta": spark_meta,
        }

        bus.publish(
            settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE,
            candidate_payload,
        )
        logger.info(
            "[%s] [LLM-GW Spark] published introspection candidate to %s",
            trace_id,
            settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE,
        )

    except Exception as e:
        logger.error(
            "[LLM-GW Spark] error while publishing introspection candidate: %s",
            e,
            exc_info=True,
        )


# ─────────────────────────────────────────────
# Backend-specific chat implementations
# ─────────────────────────────────────────────

def _chat_via_vllm(body: ChatBody, model: str) -> Dict[str, Any]:
    """
    Call vLLM via OpenAI-compatible /v1/chat/completions and attach Spark meta.
    """
    logger.info(
        "[LLM-GW] vLLM chat request model=%s messages=%d",
        model,
        len(body.messages or []),
    )

    if not settings.vllm_url:
        logger.error("[LLM-GW] vLLM URL not configured")
        return {
            "text": "[LLM-Gateway Error: vllm] vLLM URL not configured",
            "spark_meta": {},
            "raw": {},
        }

    spark_meta = _spark_ingest_for_body(body)

    base = settings.vllm_url.rstrip("/")
    url = base + "/v1/chat/completions"
    last_error: Exception | None = None

    opts = body.options or {}
    shared_params: Dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
    ):
        if key in opts and opts[key] is not None:
            shared_params[key] = opts[key]

    chat_payload: Dict[str, Any] = {
        "model": model,
        "messages": body.messages,
        "stream": False,
        **shared_params,
    }

    raw_data: Dict[str, Any] = {}
    text: str = ""

    with _common_http_client() as client:
        logger.info("[LLM-GW] Trying vLLM endpoint %s", url)

        try:
            r = client.post(url, json=chat_payload)

            if r.status_code == 404:
                logger.error(
                    "[LLM-GW] vLLM endpoint %s returned 404 (chat/completions missing)",
                    url,
                )
                return {
                    "text": "[LLM-Gateway Error: vllm] /v1/chat/completions not available on vLLM server",
                    "spark_meta": spark_meta,
                    "raw": {},
                }

            r.raise_for_status()
            raw_data = r.json()
            text = _extract_text_from_vllm(raw_data)

            _maybe_publish_spark_introspect_candidate(
                body=body,
                spark_meta=spark_meta,
                response_text=text,
            )

            return {
                "text": text,
                "spark_meta": spark_meta,
                "raw": raw_data,
            }

        except HTTPStatusError as e:
            last_error = e
            try:
                body_text = r.text
            except Exception:
                body_text = "<no body>"
            logger.error(
                "[LLM-GW] vLLM HTTP error on %s: %s | body=%s",
                url,
                e,
                body_text,
                exc_info=True,
            )
        except Exception as e:
            last_error = e
            logger.error("[LLM-GW] vLLM unknown error on %s: %s", url, e, exc_info=True)

    if last_error is not None:
        logger.error(
            "[LLM-GW] No working vLLM chat endpoint; last_error=%r", last_error
        )
        return {
            "text": f"[LLM-Gateway Error: vllm] No working chat endpoint; last_error={last_error!r}",
            "spark_meta": spark_meta,
            "raw": {},
        }

    logger.error("[LLM-GW] No working vLLM chat endpoint; unknown cause")
    return {
        "text": "[LLM-Gateway Error: vllm] No working chat endpoint; unknown cause",
        "spark_meta": spark_meta,
        "raw": {},
    }


def _chat_via_llamacpp(body: ChatBody, model: str) -> Dict[str, Any]:
    """
    Call llama.cpp via OpenAI-compatible /v1/chat/completions endpoint.

    Returns dict:
      {
        "text": "<assistant text or error>",
        "spark_meta": { ... },  # we can still do Spark here
        "raw": { ...full JSON... }
      }
    """
    logger.info(
        "[LLM-GW] llama.cpp chat request model=%s messages=%d",
        model,
        len(body.messages or []),
    )

    if not settings.llamacpp_url:
        logger.error("[LLM-GW] llamacpp URL not configured")
        return {
            "text": "[LLM-Gateway Error: llamacpp] llamacpp URL not configured",
            "spark_meta": {},
            "raw": {},
        }

    # Run Spark ingestion exactly like vLLM path
    spark_meta = _spark_ingest_for_body(body)

    base = settings.llamacpp_url.rstrip("/")
    # llama.cpp server uses /v1/chat/completions
    url = base + "/v1/chat/completions"
    last_error: Exception | None = None

    opts = body.options or {}
    shared_params: Dict[str, Any] = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
    ):
        if key in opts and opts[key] is not None:
            shared_params[key] = opts[key]

    payload: Dict[str, Any] = {
        "model": model,              # llama.cpp mostly ignores this but spec requires it
        "messages": body.messages,
        "stream": False,
        **shared_params,
    }

    raw_data: Dict[str, Any] = {}
    text: str = ""

    with _common_http_client() as client:
        logger.info("[LLM-GW] Trying llama.cpp endpoint %s", url)
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()
            raw_data = r.json()

            # Response is OpenAI-style
            try:
                choices = raw_data.get("choices") or []
                msg = (choices[0] or {}).get("message") or {}
                text = str(msg.get("content") or "").strip()
            except Exception:
                logger.warning("[LLM-GW] llamacpp response not understood: %s", raw_data)
                text = ""

            _maybe_publish_spark_introspect_candidate(
                body=body,
                spark_meta=spark_meta,
                response_text=text,
            )

            return {
                "text": text,
                "spark_meta": spark_meta,
                "raw": raw_data,
            }

        except HTTPStatusError as e:
            last_error = e
            try:
                body_text = r.text
            except Exception:
                body_text = "<no body>"
            logger.error(
                "[LLM-GW] llamacpp HTTP error on %s: %s | body=%s",
                url,
                e,
                body_text,
                exc_info=True,
            )
        except Exception as e:
            last_error = e
            logger.error(
                "[LLM-GW] llamacpp unknown error on %s: %s",
                url,
                e,
                exc_info=True,
            )

    if last_error is not None:
        return {
            "text": f"[LLM-Gateway Error: llamacpp] last_error={last_error!r}",
            "spark_meta": spark_meta,
            "raw": {},
        }

    return {
        "text": "[LLM-Gateway Error: llamacpp] unknown error",
        "spark_meta": spark_meta,
        "raw": {},
    }

def _generate_via_vllm(body: GenerateBody, model: str) -> str:
    chat_body = ChatBody(
        model=model,
        messages=[{"role": "user", "content": body.prompt}],
        options=body.options,
        stream=body.stream,
        return_json=body.return_json,
        trace_id=body.trace_id,
        user_id=body.user_id,
        session_id=body.session_id,
        source=body.source,
        verb=body.verb,
        profile_name=body.profile_name,
    )
    result = _chat_via_vllm(chat_body, model=model)
    return result.get("text") or ""


def _generate_via_llamacpp(body: GenerateBody, model: str) -> str:
    chat_body = ChatBody(
        model=model,
        messages=[{"role": "user", "content": body.prompt}],
        options=body.options,
        stream=body.stream,
        return_json=body.return_json,
        trace_id=body.trace_id,
        user_id=body.user_id,
        session_id=body.session_id,
        source=body.source,
        verb=body.verb,
        profile_name=body.profile_name,
    )
    result = _chat_via_llamacpp(chat_body, model=model)
    return result.get("text") or ""


# ─────────────────────────────────────────────
# Embeddings implementations
# ─────────────────────────────────────────────

def _embeddings_via_vllm(body: EmbeddingsBody, model: str) -> Dict[str, Any]:
    if not settings.vllm_url:
        raise RuntimeError("vLLM URL not configured for embeddings")

    url = f"{settings.vllm_url.rstrip('/')}/v1/embeddings"

    payload: Dict[str, Any] = {
        "model": model,
        "input": body.input,
    }

    opts = body.options or {}
    payload.update({k: v for k, v in opts.items() if v is not None})

    logger.info(f"[LLM-GW] vLLM embeddings model={model} inputs={len(body.input)}")

    with _common_http_client() as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ─────────────────────────────────────────────
# Public entrypoints used by main.py
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody):
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)
    model = _normalize_model_for_vllm(model)  # harmless for llamacpp; keeps profile-alias behavior

    logger.info("[LLM-GW] run_llm_chat backend=%s model=%s", backend, model)

    if backend == "llamacpp":
        return _chat_via_llamacpp(body, model=model)

    # default: vllm
    return _chat_via_vllm(body, model=model)


def run_llm_generate(body: GenerateBody) -> str:
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)
    model_for_vllm = _normalize_model_for_vllm(model) if backend == "vllm" else model

    logger.info("[LLM-GW] run_llm_generate backend=%s model=%s", backend, model_for_vllm)

    if backend == "llamacpp":
        return _generate_via_llamacpp(body, model=model_for_vllm or "Active-GGUF-Model")

    return _generate_via_vllm(body, model=model_for_vllm)


def run_llm_embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    """
    Embeddings entrypoint — currently vLLM-only.
    """
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)
    model_for_vllm = _normalize_model_for_vllm(model) if backend == "vllm" else model

    if backend == "vllm":
        return _embeddings_via_vllm(body, model=model_for_vllm)

    raise RuntimeError(f"Unknown backend for embeddings: {backend}")


def run_llm_exec_step(body: ExecStepPayload) -> Dict[str, Any]:
    """
    Execute a Cortex exec_step via the selected LLM backend.
    """
    t0 = time.time()

    if body.prompt:
        final_prompt = body.prompt
    else:
        header = body.prompt_template or ""
        ctx = body.context or {}
        prior = body.prior_step_results or []
        final_prompt = (
            f"{header}\n\n"
            "# Context (JSON)\n"
            f"{json.dumps(ctx, indent=2, ensure_ascii=False)}\n\n"
            "# Prior Step Results (JSON)\n"
            f"{json.dumps(prior, indent=2, ensure_ascii=False)}\n"
        )

    profile = _select_profile(getattr(body, "profile_name", None))
    backend = _pick_backend({}, profile)
    model = _resolve_model(None, profile)
    model = _normalize_model_for_vllm(model)

    logger.info(
        "[LLM-GW] exec_step backend=%s model=%s verb=%s step=%s service=%s",
        backend, model, body.verb, body.step, body.service,
    )

    chat_body = ChatBody(
        model=model,
        messages=[{"role": "user", "content": final_prompt}],
        options={},
        stream=False,
        return_json=False,
        trace_id=body.origin_node,
        user_id=None,
        session_id=None,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=getattr(body, "profile_name", None),
    )

    if backend == "llamacpp":
        result = _chat_via_llamacpp(chat_body, model=model)
    else:
        result = _chat_via_vllm(chat_body, model=model)


    text = result.get("text") or ""
    elapsed_ms = int((time.time() - t0) * 1000)

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
        "llm_output": text,
        "spark_meta": result.get("spark_meta"),
        "raw_llm": result.get("raw"),
    }
