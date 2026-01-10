from typing import Any, Dict, Optional, List, Coroutine
import asyncio
import logging
import time
import json

import httpx
from httpx import HTTPStatusError

from orion.core.bus.async_service import OrionBusAsync

from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .settings import settings
from .profiles import LLMProfileRegistry, LLMProfile

from orion.spark.integration import (
    ingest_chat_and_get_state,
    build_collapse_mirror_meta,
)

logger = logging.getLogger("orion-llm-gateway.backend")


def _run_async(coro: asyncio.Future | Coroutine[Any, Any, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)

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
        verb = getattr(body, "verb", None) or "unknown"
        source = getattr(body, "source", "juniper") or "juniper"

        raw_user_text = _get_raw_user_text(body)
        latest_user: Optional[str] = None

        if raw_user_text:
            latest_user = raw_user_text
        else:
            # Infer last user message from message list
            for m in reversed(messages):
                if (m.get("role") or "").lower() == "user":
                    latest_user = m.get("content")
                    break
            if not latest_user and messages:
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
            # IMPORTANT: Titanium boundary.
            # This payload will be wrapped in a Titanium envelope in _publish_spark_introspect().
            payload = {
                "trace_id": body.trace_id or "gw",
                "source": getattr(body, "source", "gw"),
                "prompt": spark_meta.get("latest_user_message") or "",
                "response": response_text,
                "spark_meta": spark_meta,
            }
            _run_async(_publish_spark_introspect(payload))
    except Exception as e:
        logger.error(f"[LLM-GW Spark] Publish failed: {e}")


async def _publish_spark_introspect(payload: Dict[str, Any]) -> None:
    """Publish Spark introspection candidates as Titanium envelopes.

    Prior behavior published a raw dict on the channel, which bypassed strict
    payload validation (because the bus validator only inspects BaseEnvelope payloads).
    This function enforces the Titanium boundary by:
      1) validating payload as SparkCandidateV1 (schema registry)
      2) wrapping in BaseEnvelope (orion.envelope)
      3) publishing to the cataloged channel
    """
    import uuid

    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.telemetry.spark_candidate import SparkCandidateV1

    # Validate + normalize to the schema declared in orion/bus/channels.yaml
    candidate = SparkCandidateV1.model_validate(payload)

    # correlation_id: prefer trace_id if it's a UUID; otherwise generate a new one
    try:
        corr_id = uuid.UUID(str(candidate.trace_id))
    except Exception:
        corr_id = uuid.uuid4()

    env = BaseEnvelope(
        kind="spark.candidate",
        source=ServiceRef(
            name=settings.service_name,
            node=settings.node_name,
            version=settings.service_version,
            instance=None,
        ),
        correlation_id=corr_id,
        payload=candidate.model_dump(mode="json"),
    )

    bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    await bus.connect()
    try:
        await bus.publish(settings.channel_spark_introspect_candidate, env)
    finally:
        await bus.close()


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

        state = _spark_ingest_text(text=str(response_text), agent_id=source, tags=tags, spark_vector=spark_vector)

        # add post-phase meta keys (do not overwrite pre phase)
        spark_meta.update(
            {
                "phi_post_before": state.get("phi_before"),
                "phi_post_after": state.get("phi_after"),
                "spark_phase_post": True,
            }
        )
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Post ingestion failed: {e}")


# ─────────────────────────────────────────────
# 4. Embeddings Endpoint (Optional)
# ─────────────────────────────────────────────

def _fetch_embedding_internal(text: str) -> Optional[List[float]]:
    """
    If include_embeddings is enabled, fetch an embedding/state vector for `text`.
    Uses llama.cpp embedding endpoint if configured.

    NOTE: This is best-effort and must never break core chat functionality.
    """
    url = settings.llamacpp_embedding_url or settings.llamacpp_url
    if not url:
        return None

    req = {"input": text}
    try:
        with _common_http_client() as client:
            r = client.post(f"{url.rstrip('/')}/v1/embeddings", json=req)
            r.raise_for_status()
            data = r.json()
            # OpenAI style: {"data":[{"embedding":[...]}]}
            d = data.get("data") or []
            if d and isinstance(d[0], dict) and isinstance(d[0].get("embedding"), list):
                emb = d[0]["embedding"]
                if emb and all(isinstance(x, (int, float)) for x in emb):
                    return [float(x) for x in emb]
            # fallback: sometimes server returns "embedding" directly
            emb = data.get("embedding")
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]
    except Exception:
        return None

    return None


# ─────────────────────────────────────────────
# 5. Core LLM Methods
# ─────────────────────────────────────────────

def chat(body: ChatBody) -> Dict[str, Any]:
    """
    Main chat entrypoint: routes to configured backend, returns OpenAI-compatible payload.
    Adds Spark meta and publishes introspect candidates when thresholds are crossed.
    """
    profile = _select_profile(body.profile)
    backend = _pick_backend(body.options, profile)

    # Spark pre-ingest (user side)
    spark_meta: Dict[str, Any] = _spark_ingest_for_body(body)

    model = _resolve_model(body.model, profile)
    if backend == "vllm":
        model = _normalize_model_for_vllm(model)

    # Build request for backend
    options = body.options or {}
    payload = {
        "model": model,
        "messages": body.messages,
        "temperature": float(options.get("temperature", getattr(profile, "temperature", 0.7) if profile else 0.7)),
        "max_tokens": int(options.get("max_tokens", getattr(profile, "max_tokens", 1024) if profile else 1024)),
        "stream": False,
    }

    # Route
    if backend == "vllm":
        url = settings.vllm_url
        if not url:
            raise RuntimeError("vLLM backend selected but ORION_LLM_VLLM_URL is not set.")
        endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    else:
        url = settings.llamacpp_url
        if not url:
            raise RuntimeError("llama.cpp backend selected but ORION_LLM_LLAMACPP_URL is not set.")
        endpoint = f"{url.rstrip('/')}/v1/chat/completions"

    # Execute
    t0 = time.time()
    raw_llm: Dict[str, Any] = {}
    try:
        with _common_http_client() as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            raw_llm = resp.json()
    except HTTPStatusError as http_err:
        logger.error(f"[LLM-GW] HTTP error calling backend: {http_err}")
        raise
    except Exception as err:
        logger.error(f"[LLM-GW] Backend call failed: {err}")
        raise

    # Normalize output
    text = _extract_text_from_openai_response(raw_llm)
    duration_ms = int((time.time() - t0) * 1000)

    # Try to extract vector from LLM response; if not present, optionally fetch
    spark_vector = _extract_vector_from_openai_response(raw_llm)
    if spark_vector is None and settings.include_embeddings:
        try:
            spark_vector = _fetch_embedding_internal(text)
        except Exception:
            spark_vector = None

    # Spark post-ingest (assistant reply side)
    _spark_post_ingest_for_reply(body, spark_meta, text)

    # Maybe publish introspection candidate
    if text:
        _maybe_publish_spark_introspect(body, spark_meta, text)

    # Emit result
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "model": model,
        "usage": {"duration_ms": duration_ms},
        "spark_meta": spark_meta,
        "spark_vector": spark_vector,
        "raw_llm": raw_llm,
    }


def generate(body: GenerateBody) -> Dict[str, Any]:
    """
    Text-completion style generation. Kept for compatibility.
    """
    profile = _select_profile(body.profile)
    backend = _pick_backend(body.options, profile)

    model = _resolve_model(body.model, profile)
    if backend == "vllm":
        model = _normalize_model_for_vllm(model)

    prompt = body.prompt or ""
    options = body.options or {}

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": float(options.get("temperature", getattr(profile, "temperature", 0.7) if profile else 0.7)),
        "max_tokens": int(options.get("max_tokens", getattr(profile, "max_tokens", 1024) if profile else 1024)),
        "stream": False,
    }

    if backend == "vllm":
        url = settings.vllm_url
        if not url:
            raise RuntimeError("vLLM backend selected but ORION_LLM_VLLM_URL is not set.")
        endpoint = f"{url.rstrip('/')}/v1/completions"
    else:
        url = settings.llamacpp_url
        if not url:
            raise RuntimeError("llama.cpp backend selected but ORION_LLM_LLAMACPP_URL is not set.")
        endpoint = f"{url.rstrip('/')}/v1/completions"

    with _common_http_client() as client:
        r = client.post(endpoint, json=payload)
        r.raise_for_status()
        return r.json()


def exec_step(payload: ExecStepPayload) -> Dict[str, Any]:
    """
    Execute a single step from cortex-exec that targets the LLM gateway.
    """
    body = ChatBody(
        model=payload.model,
        profile=payload.profile,
        messages=payload.messages,
        options=payload.options or {},
        trace_id=payload.trace_id,
        source=payload.source or "cortex-exec",
        verb=payload.verb or "exec_step",
        raw_user_text=payload.raw_user_text,
    )
    return chat(body)


def embeddings(body: EmbeddingsBody) -> Dict[str, Any]:
    """
    Expose embeddings endpoint for callers that want explicit vectors.
    """
    vec = _fetch_embedding_internal(body.input)
    return {"embedding": vec or []}
