from typing import Any, Dict, List
import logging
import time
import json

import httpx
from httpx import HTTPStatusError

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
    logger.warning("[LLM-GW] vLLM /v1/completions response not understood: %s", data)
    return ""


def _flatten_messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Fallback: turn OpenAI chat-style messages into a single text prompt
    for /v1/completions-style endpoints.
    """
    if not messages:
        return ""

    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _select_profile(
    profile_name: str | None,
) -> LLMProfile | None:
    """
    Select explicit profile_name.
    Returns None if profiles are not configured.

    Priority:
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


def _pick_backend(options: Dict[str, Any] | None, profile: LLMProfile | None) -> str:
    """
    Decide which backend to use for this request.

    Priority:
      1) options["backend"] if present
      2) profile.backend (if profile is selected)
      3) settings.default_backend

    Right now, only 'vllm' is supported; everything else is coerced to 'vllm'.
    """
    opts = options or {}

    backend = opts.get("backend")
    if backend:
        backend = backend.lower()
    elif profile is not None:
        backend = (profile.backend or "vllm").lower()
    else:
        backend = (settings.default_backend or "vllm").lower()

    if backend != "vllm":
        logger.warning(
            "[LLM-GW] Unknown backend '%s', falling back to 'vllm'",
            backend,
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

    logger.warning("[LLM-GW] vLLM response not understood: %s", data)
    return ""


def _extract_text_from_brain_chat(data: Dict[str, Any]) -> str:
    if isinstance(data, dict):
        if data.get("response"):
            return str(data["response"]).strip()
        if data.get("text"):
            return str(data["text"]).strip()
    logger.warning("[LLM-GW] Brain /chat response not understood: %s", data)
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

    Returns a spark_meta dict (possibly empty) that can be attached
    to the LLM-Gateway reply payload.

    This NEVER raises – failures are logged and ignored so LLM calls
    still succeed even if Spark misbehaves.
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

        # Use the body.source as an agent_id tag if available.
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

        # Attach φ before/after so downstream can see deltas.
        spark_meta["phi_before"] = phi_before
        spark_meta["phi_after"] = phi_after

        return spark_meta

    except Exception as e:
        logger.warning("[LLM-GW Spark] ingestion failed: %s", e, exc_info=True)
        return {}


# ─────────────────────────────────────────────
# Message role normalization
# ─────────────────────────────────────────────

_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function"}


def _normalize_roles_for_llm(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure all roles are valid for OpenAI / vLLM chat:

      - 'orion' -> 'assistant'
      - unknown roles -> 'user' (except system/tool/function)
    """
    out: List[Dict[str, Any]] = []

    for m in messages or []:
        content = m.get("content")
        if content is None:
            continue

        role_raw = m.get("role") or "user"
        role = str(role_raw).lower()

        if role == "orion":
            role = "assistant"
        elif role not in _ALLOWED_ROLES:
            # keep system if it was exactly system; otherwise treat as user
            if role != "system":
                role = "user"

        out.append(
            {
                "role": role,
                "content": content,
            }
        )

    return out


# ─────────────────────────────────────────────
# Backend-specific chat implementations (vLLM-only)
# ─────────────────────────────────────────────

def _chat_via_vllm(body: ChatBody, model: str) -> Dict[str, Any]:
    """
    Call vLLM via OpenAI-compatible endpoints and attach Spark meta.

    Returns a dict:

        {
          "text": "<assistant text or error>",
          "spark_meta": { ... },  # may be empty
          "raw": { ...full vLLM JSON if available... }
        }

    We try, in order:
      1) /v1/chat/completions  (chat-style: messages[])
      2) /v1/completions       (text-style: prompt)

    If both fail, we still return a readable error string in "text"
    instead of raising.
    """
    if not settings.vllm_url:
        logger.error("[LLM-GW] vLLM URL not configured")
        return {
            "text": "[LLM-Gateway Error: vllm] vLLM URL not configured",
            "spark_meta": {},
            "raw": {},
        }

    # Run Spark ingestion *before* we call vLLM.
    spark_meta = _spark_ingest_for_body(body)

    base = settings.vllm_url.rstrip("/")
    candidate_paths = ["/v1/chat/completions", "/v1/completions"]
    last_error: Exception | None = None

    # Normalize roles so vLLM (Mistral) never sees 'orion'.
    normalized_messages = _normalize_roles_for_llm(body.messages or [])

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
        "messages": normalized_messages,
        "stream": False,
        **shared_params,
    }

    # --- Text payload (for /v1/completions) ---
    if normalized_messages:
        user_msgs = [m for m in normalized_messages if m["role"] == "user"]
        if user_msgs:
            last_user = user_msgs[-1].get("content", "")
        else:
            last_user = normalized_messages[-1].get("content", "")
        prompt = str(last_user or "")
    else:
        prompt = ""

    text_payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **shared_params,
    }

    raw_data: Dict[str, Any] = {}

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
                raw_data = r.json()
                text = _extract_text_from_vllm(raw_data)

                return {
                    "text": text,
                    "spark_meta": spark_meta,
                    "raw": raw_data,
                }

            except HTTPStatusError as e:
                last_error = e
                logger.error("[LLM-GW] vLLM HTTP error on %s: %s", url, e, exc_info=True)
            except Exception as e:
                last_error = e
                logger.error("[LLM-GW] vLLM unknown error on %s: %s", url, e, exc_info=True)

    # If we got here, nothing worked; don't crash the gateway.
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


def _generate_via_vllm(body: GenerateBody, model: str) -> str:
    """
    Generate-style call implemented in terms of the chat path
    so we still get Spark + profiles.
    """
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
    if isinstance(result, dict):
        return (result.get("text") or "").strip()
    return str(result or "")


# ─────────────────────────────────────────────
# Embeddings implementations (vLLM-only)
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

    logger.info("[LLM-GW] vLLM embeddings model=%s inputs=%d", model, len(body.inputs))

    with _common_http_client() as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────
# Public entrypoints used by main.py
# ─────────────────────────────────────────────

def run_llm_chat(body: ChatBody):
    """
    Top-level chat router for the LLM Gateway.

    - Chooses backend (currently vLLM-only) based on settings / options / profiles.
    - Returns a dict with text + spark_meta + raw.
    """
    profile = _select_profile(getattr(body, "profile_name", None))
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _chat_via_vllm(body, model=model)

    # Should not happen, _pick_backend coerces unknown backends to vLLM.
    raise RuntimeError(f"Unknown backend: {backend}")


def run_llm_generate(body: GenerateBody) -> str:
    profile = _select_profile(body.profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if backend == "vllm":
        return _generate_via_vllm(body, model=model)

    # Should not happen, _pick_backend coerces unknown backends to vLLM.
    raise RuntimeError(f"Unknown backend for generate: {backend}")


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
        options={},  # can later pass richer options from ExecStepPayload
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
