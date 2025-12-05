from typing import Any, Dict
import logging
import time
import json

import httpx

from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .settings import settings
from .profiles import LLMProfileRegistry, LLMProfile

logger = logging.getLogger("orion-llm-gateway.backend")

# ─────────────────────────────────────────────
# Profile loading
# ─────────────────────────────────────────────

_profile_registry: LLMProfileRegistry = settings.load_profile_registry()


def _select_profile(
    verb: str | None,
    profile_name: str | None,
) -> LLMProfile | None:
    """
    Try to select a profile given verb + explicit profile_name.
    Returns None if profiles are not configured.
    """
    if not _profile_registry.profiles:
        return None

    # Explicit profile beats verb-based
    if profile_name:
        try:
            return _profile_registry.get(profile_name)
        except KeyError:
            logger.warning(f"[LLM-GW] Unknown profile_name '{profile_name}'")
            # fall through

    if verb:
        default = settings.llm_default_profile_name
        prof = _profile_registry.for_verb(verb, default=default)
        if prof:
            return prof

    # Last resort: default profile name if configured
    if settings.llm_default_profile_name:
        try:
            return _profile_registry.get(settings.llm_default_profile_name)
        except KeyError:
            logger.warning(
                "[LLM-GW] Default profile '%s' not found",
                settings.llm_default_profile_name,
            )

    return None


# ─────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────

def _pick_backend(options: Dict[str, Any] | None, profile: LLMProfile | None) -> str:
    """
    Decide which backend to use for this request.

    Priority:
      1) options["backend"] if present
      2) profile.backend (if profile is selected)
      3) settings.default_backend
    """
    opts = options or {}

    backend = opts.get("backend")
    if backend:
        backend = backend.lower()
    elif profile is not None:
        backend = profile.backend
    else:
        backend = (settings.default_backend or "ollama").lower()

    if backend not in ("ollama", "vllm", "brain"):
        logger.warning(f"[LLM-GW] Unknown backend '{backend}', falling back to 'ollama'")
        return "ollama"
    return backend


def _resolve_model(body_model: str | None, profile: LLMProfile | None) -> str:
    """
    Decide which model string to send to the backend.

    Priority:
      1) body.model if provided
      2) profile.model_id if profile is selected
      3) settings.default_model
    """
    if body_model:
        return body_model
    if profile is not None:
        return profile.model_id
    return settings.default_model


def _extract_text_from_ollama(data: Dict[str, Any]) -> str:
    if isinstance(data, dict):
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if content:
                return str(content).strip()

        if data.get("response"):
            return str(data["response"]).strip()

        if data.get("text"):
            return str(data["text"]).strip()

    logger.warning(f"[LLM-GW] Ollama response missing 'message.content': {data}")
    return ""


def _extract_text_from_vllm(data: Dict[str, Any]) -> str:
    try:
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if content:
                return str(content).strip()
    except Exception:
        pass
    logger.warning(f"[LLM-GW] vLLM response not understood: {data}")
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
# Backend-specific chat implementations
# ─────────────────────────────────────────────

def _chat_via_ollama(body: ChatBody, model: str) -> str:
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": body.messages,
        "options": body.options or {},
        "stream": False,
    }

    logger.info(
        f"[LLM-GW] Ollama chat model={payload['model']} messages={len(payload['messages'])}"
    )

    try:
        with _common_http_client() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return _extract_text_from_ollama(data)
    except Exception as e:
        logger.error(f"[LLM-GW] Ollama chat failed: {e}", exc_info=True)
        return f"[LLM-Gateway Error: ollama] {e}"


def _chat_via_vllm(body: ChatBody, model: str) -> str:
    """
    vLLM assumed to expose an OpenAI-compatible /v1/chat/completions.
    """
    if not settings.vllm_url:
        logger.warning("[LLM-GW] vLLM URL not configured; falling back to Ollama")
        return _chat_via_ollama(body, model=model)

    url = f"{settings.vllm_url.rstrip('/')}/v1/chat/completions"

    opts = body.options or {}
    params: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"):
        if key in opts and opts[key] is not None:
            params[key] = opts[key]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": body.messages,
        "stream": False,
        **params,
    }

    logger.info(
        f"[LLM-GW] vLLM chat model={payload['model']} messages={len(payload['messages'])}"
    )

    try:
        with _common_http_client() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return _extract_text_from_vllm(data)
    except Exception as e:
        logger.error(f"[LLM-GW] vLLM chat failed: {e}", exc_info=True)
        return f"[LLM-Gateway Error: vllm] {e}"


def _chat_via_brain_http(body: ChatBody, model: str) -> str:
    if not settings.brain_url:
        logger.warning("[LLM-GW] Brain URL not configured; falling back to Ollama")
        return _chat_via_ollama(body, model=model)

    url = f"{settings.brain_url.rstrip('/')}/chat"
    payload = body.model_dump(mode="json")
    payload["model"] = model

    logger.info("[LLM-GW] Brain /chat proxy")

    try:
        with _common_http_client() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return _extract_text_from_brain_chat(data)
    except Exception as e:
        logger.error(f"[LLM-GW] Brain /chat failed: {e}", exc_info=True)
        return f"[LLM-Gateway Error: brain] {e}"


# ─────────────────────────────────────────────
# Backend-specific generate implementations
# ─────────────────────────────────────────────

def _generate_via_ollama(body: GenerateBody, model: str) -> str:
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
    return _chat_via_ollama(chat_body, model=model)


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
    return _chat_via_vllm(chat_body, model=model)


def _generate_via_brain_http(body: GenerateBody, model: str) -> str:
    if not settings.brain_url:
        logger.warning("[LLM-GW] Brain URL not configured; falling back to Ollama")
        return _generate_via_ollama(body, model=model)

    url = f"{settings.brain_url.rstrip('/')}/chat"
    payload = body.model_dump(mode="json")
    payload["model"] = model

    logger.info("[LLM-GW] Brain /chat generate proxy")

    try:
        with _common_http_client() as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return _extract_text_from_brain_chat(data)
    except Exception as e:
        logger.error(f"[LLM-GW] Brain generate failed: {e}", exc_info=True)
        return f"[LLM-Gateway Error: brain-generate] {e}"


# ─────────────────────────────────────────────
# Embeddings implementations
# ─────────────────────────────────────────────

def _embeddings_via_vllm(body: EmbeddingsBody, model: str) -> Dict[str, Any]:
    if not settings.vllm_url:
        raise RuntimeError("vLLM URL not configured for embeddings")

    url = f"{settings.vllm_url.rstrip('/')}/v1/embeddings"

    payload: Dict[str, Any] = {
        "model": model,
        "input": body.inputs,
    }

    opts = body.options or {}
    payload.update({k: v for k, v in opts.items() if v is not None})

    logger.info(f"[LLM-GW] vLLM embeddings model={model} inputs={len(body.inputs)}")

    with _common_http_client() as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _embeddings_via_ollama(body: EmbeddingsBody, model: str) -> Dict[str, Any]:
    """
    If you later expose embeddings via Ollama/brain, hook it up here.
    For now, we raise to make it explicit.
    """
    raise NotImplementedError("Embeddings via ollama/brain not implemented")


# ─────────────────────────────────────────────
# Public entrypoints used by main.py
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody) -> str:
    profile = _select_profile(body.verb, body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _chat_via_vllm(body, model=model)
    elif backend == "brain":
        return _chat_via_brain_http(body, model=model)
    return _chat_via_ollama(body, model=model)


def run_llm_generate(body: GenerateBody) -> str:
    profile = _select_profile(body.verb, body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _generate_via_vllm(body, model=model)
    elif backend == "brain":
        return _generate_via_brain_http(body, model=model)
    return _generate_via_ollama(body, model=model)


def run_llm_embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    """
    Embeddings entrypoint — currently vLLM-only.
    """
    profile = _select_profile(body.verb, body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _embeddings_via_vllm(body, model=model)
    elif backend in ("ollama", "brain"):
        return _embeddings_via_ollama(body, model=model)
    raise RuntimeError(f"Unknown backend for embeddings: {backend}")


def run_llm_exec_step(body: ExecStepPayload) -> Dict[str, Any]:
    """
    Execute a Cortex exec_step via the selected LLM backend.

    Returns a dict suitable for 'result' in emit_cortex_step_result:
      {
        "prompt": "<final prompt used>",
        "llm_output": "<model text>",
      }
    """
    t0 = time.time()

    # Prefer the fully-built prompt if provided; otherwise fall back to template + context.
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

    gen_body = GenerateBody(
        model=body.model,
        prompt=final_prompt,
        options={},  # you can forward richer options if needed
        stream=False,
        return_json=False,
        trace_id=body.origin_node,
        user_id=None,
        session_id=None,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=body.profile_name,
    )

    text = run_llm_generate(gen_body)
    elapsed_ms = int((time.time() - t0) * 1000)

    logger.info(
        "[LLM-GW] exec_step verb=%s step=%s service=%s elapsed_ms=%d",
        body.verb,
        body.step,
        body.service,
        elapsed_ms,
    )

    return {
        "prompt": final_prompt,
        "llm_output": text,
    }
