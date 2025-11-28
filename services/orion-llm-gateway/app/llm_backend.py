# services/orion-llm-gateway/app/llm_backend.py

from typing import Any, Dict
import logging

import httpx

from .models import ChatBody, GenerateBody
from .settings import settings

logger = logging.getLogger("orion-llm-gateway.backend")


# ─────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────

def _pick_backend(options: Dict[str, Any] | None) -> str:
    """
    Decide which backend to use for this request.

    Priority:
      1) body.options["backend"] if present
      2) settings.default_backend
    """
    opts = options or {}
    backend = (opts.get("backend") or settings.default_backend or "ollama").lower()
    if backend not in ("ollama", "vllm", "brain"):
        logger.warning(f"[LLM-GW] Unknown backend '{backend}', falling back to 'ollama'")
        return "ollama"
    return backend


def _extract_text_from_ollama(data: Dict[str, Any]) -> str:
    """
    Match your Brain expectations:
      - prefer data["message"]["content"]
      - fallback to "response" or "text"
    """
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
    """
    vLLM typically speaks OpenAI-style:
      { "choices": [ { "message": { "content": "..." } } ] }
    """
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
    """
    If we ever call Brain's /chat directly, align to its response:
      { "trace_id": "...", "backend": "...", "response": "..." }
    """
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

def _chat_via_ollama(body: ChatBody) -> str:
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"

    payload = {
        "model": body.model or settings.default_model,
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


def _chat_via_vllm(body: ChatBody) -> str:
    """
    vLLM assumed to expose an OpenAI-compatible /v1/chat/completions.

    You may need to tweak this once you actually bring vLLM online,
    but callers don't have to change.
    """
    if not settings.vllm_url:
        logger.warning("[LLM-GW] vLLM URL not configured; falling back to Ollama")
        return _chat_via_ollama(body)

    url = f"{settings.vllm_url.rstrip('/')}/v1/chat/completions"

    # Only forward a subset of options that vLLM/OpenAI understand
    opts = body.options or {}
    params: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"):
        if key in opts and opts[key] is not None:
            params[key] = opts[key]

    payload: Dict[str, Any] = {
        "model": body.model or settings.default_model,
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


def _chat_via_brain_http(body: ChatBody) -> str:
    """
    Optional path: call Brain's /chat HTTP endpoint directly.

    This keeps its Spark / φ / collapse wiring if you ever want that
    from the gateway, but it's not the default.
    """
    if not settings.brain_url:
        logger.warning("[LLM-GW] Brain URL not configured; falling back to Ollama")
        return _chat_via_ollama(body)

    url = f"{settings.brain_url.rstrip('/')}/chat"
    payload = body.model_dump(mode="json")

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
# (generate -> single user message into /api/chat or /v1/chat/completions)
# ─────────────────────────────────────────────

def _generate_via_ollama(body: GenerateBody) -> str:
    chat_body = ChatBody(
        model=body.model or settings.default_model,
        messages=[{"role": "user", "content": body.prompt}],
        options=body.options,
        stream=body.stream,
        return_json=body.return_json,
        trace_id=body.trace_id,
        user_id=body.user_id,
        session_id=body.session_id,
        source=body.source,
    )
    return _chat_via_ollama(chat_body)


def _generate_via_vllm(body: GenerateBody) -> str:
    chat_body = ChatBody(
        model=body.model or settings.default_model,
        messages=[{"role": "user", "content": body.prompt}],
        options=body.options,
        stream=body.stream,
        return_json=body.return_json,
        trace_id=body.trace_id,
        user_id=body.user_id,
        session_id=body.session_id,
        source=body.source,
    )
    return _chat_via_vllm(chat_body)


def _generate_via_brain_http(body: GenerateBody) -> str:
    """
    Brain's /chat endpoint treats 'messages' vs 'prompt' differently.
    This path is optional and may not be needed often.
    """
    if not settings.brain_url:
        logger.warning("[LLM-GW] Brain URL not configured; falling back to Ollama")
        return _generate_via_ollama(body)

    url = f"{settings.brain_url.rstrip('/')}/chat"
    payload = body.model_dump(mode="json")

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
# Public entrypoints used by main.py
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody) -> str:
    backend = _pick_backend(body.options)
    if backend == "vllm":
        return _chat_via_vllm(body)
    elif backend == "brain":
        return _chat_via_brain_http(body)
    # default
    return _chat_via_ollama(body)


def run_llm_generate(body: GenerateBody) -> str:
    backend = _pick_backend(body.options)
    if backend == "vllm":
        return _generate_via_vllm(body)
    elif backend == "brain":
        return _generate_via_brain_http(body)
    # default
    return _generate_via_ollama(body)
