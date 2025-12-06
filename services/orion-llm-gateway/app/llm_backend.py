from typing import Any, Dict
import logging
import time
import json

import httpx
from httpx import HTTPStatusError

from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .settings import settings
from .profiles import LLMProfileRegistry, LLMProfile

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
        backend = (settings.default_backend or "vllm").lower()

    if backend != "vllm":
        logger.warning(
            f"[LLM-GW] Unknown backend '{backend}', falling back to 'vllm'"
        )
        return "vllm"

    return backend


def _resolve_model(body_model: str | None, profile: LLMProfile | None) -> str:
    """
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

from httpx import HTTPStatusError

def _chat_via_vllm(body: ChatBody, model: str) -> str:
    """
    vLLM assumed to expose OpenAI-compatible endpoints.

    We try, in order:
      1) /v1/chat/completions  (chat-style: messages[])
      2) /v1/completions       (text-style: prompt)

    If both fail, we return a readable error string instead of raising.
    """
    if not settings.vllm_url:
        logger.warning("[LLM-GW] vLLM URL not configured; falling back to Ollama")
        return _chat_via_ollama(body, model=model)

    base = settings.vllm_url.rstrip("/")
    candidate_paths = ["/v1/chat/completions", "/v1/completions"]
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

    # --- Chat payload (for /v1/chat/completions) ---
    chat_payload: Dict[str, Any] = {
        "model": model,
        "messages": body.messages,
        "stream": False,
        **shared_params,
    }

    # --- Text payload (for /v1/completions) ---
    # Use last user message as the prompt, or last message if no explicit user role.
    if body.messages:
        user_msgs = [m for m in body.messages if m.get("role") == "user"]
        if user_msgs:
            last_user = user_msgs[-1].get("content", "")
        else:
            last_user = body.messages[-1].get("content", "")
        prompt = last_user or ""
    else:
        prompt = ""

    text_payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **shared_params,
    }

    with _common_http_client() as client:
        for path in candidate_paths:
            url = base + path
            if path.endswith("/chat/completions"):
                payload = chat_payload
            else:
                payload = text_payload

            logger.info("[LLM-GW] Trying vLLM endpoint %s", url)

            try:
                r = client.post(url, json=payload)

                # If this endpoint simply doesn't exist, try the next one.
                if r.status_code == 404:
                    logger.warning(
                        "[LLM-GW] vLLM endpoint %s returned 404; trying next candidate",
                        url,
                    )
                    continue

                r.raise_for_status()
                data = r.json()
                return _extract_text_from_vllm(data)

            except HTTPStatusError as e:
                last_error = e
                logger.error("[LLM-GW] vLLM HTTP error on %s: %s", url, e, exc_info=True)
            except Exception as e:
                last_error = e
                logger.error("[LLM-GW] vLLM unknown error on %s: %s", url, e, exc_info=True)

    # If we got here, nothing worked; don't crash the gateway, just surface a readable error.
    if last_error is not None:
        logger.error(
            "[LLM-GW] No working vLLM chat endpoint; last_error=%r", last_error
        )
        return f"[LLM-Gateway Error: vllm] No working chat endpoint; last_error={last_error!r}"

    logger.error("[LLM-GW] No working vLLM chat endpoint; unknown cause")
    return "[LLM-Gateway Error: vllm] No working chat endpoint; unknown cause"


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

# ─────────────────────────────────────────────
# Public entrypoints used by main.py
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody) -> str:
    # 🔧 profile is decided purely by explicit profile_name or gateway default
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _chat_via_vllm(body, model=model)

    # Optional: if you truly never want ollama here, you can raise instead.
    return _chat_via_ollama(body, model=model)


def run_llm_generate(body: GenerateBody) -> str:
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _generate_via_vllm(body, model=model)

    return _generate_via_ollama(body, model=model)


def run_llm_embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    """
    Embeddings entrypoint — currently vLLM-only.
    """
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _embeddings_via_vllm(body, model=model)

    raise RuntimeError(f"Unknown backend for embeddings: {backend}")


def run_llm_exec_step(body: ExecStepPayload) -> Dict[str, Any]:
    """
    Execute a Cortex exec_step via the selected LLM backend.

    Returns a dict suitable for 'result' in emit_cortex_step_result:
      {
        "prompt": "<final prompt used>",
        "llm_output": "<model text>",
      }

    NOTE: exec_step is profile/verb-driven; it no longer relies on a 'model'
    field on ExecStepPayload. Model resolution is handled in run_llm_generate()
    via profiles + defaults.
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
        prompt=final_prompt,
        options={},  # you can later pipe through richer options if ExecStepPayload supports them
        stream=False,
        return_json=False,
        trace_id=body.origin_node,
        user_id=None,
        session_id=None,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=getattr(body, "profile_name", None),
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
