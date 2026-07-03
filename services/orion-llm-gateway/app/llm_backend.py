from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, List, Coroutine, Sequence, Tuple
import asyncio
import logging
import os
import time
import json
import re

import httpx

from orion.llm.openai_message_content import join_openai_message_content
from orion.core.bus.async_service import OrionBusAsync

from .models import ChatBody, ChatMessage, GenerateBody, ExecStepPayload
from .settings import settings
from .profiles import LLMProfileRegistry, LLMProfile
from .lane_routes import resolve_llm_lane_route
from .structured_output import apply_structured_output_to_payload
from .llm_uncertainty import (
    extract_llm_uncertainty_from_native_completion,
    extract_llm_uncertainty_from_openai_response,
)

logger = logging.getLogger("orion-llm-gateway.backend")


def _thought_debug_enabled() -> bool:
    return str(os.getenv("DEBUG_THOUGHT_PROCESS", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _debug_len(value: Any) -> int:
    return len(str(value or ""))


def _debug_snippet(value: Any, max_len: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…"


def _preview_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    return repr(value[:limit])


@dataclass(frozen=True)
class RouteTarget:
    url: str
    backend: Optional[str] = None
    served_by: Optional[str] = None


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


@lru_cache
def _load_route_targets() -> Dict[str, RouteTarget]:
    route_table: Dict[str, RouteTarget] = {}
    raw_json = settings.llm_route_table_json
    if raw_json:
        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error("[LLM-GW] Invalid LLM_GATEWAY_ROUTE_TABLE_JSON: %s", exc)
            return {}

        if not isinstance(raw, dict):
            logger.error("[LLM-GW] Route table JSON must be a dict, got %s", type(raw))
            return {}

        for route, value in raw.items():
            if isinstance(value, str):
                route_table[str(route)] = RouteTarget(url=value)
                continue
            if isinstance(value, dict):
                url = value.get("url") or value.get("base_url")
                if not url:
                    logger.warning("[LLM-GW] Route '%s' missing url/base_url", route)
                    continue
                route_table[str(route)] = RouteTarget(
                    url=url,
                    backend=value.get("backend"),
                    served_by=value.get("served_by"),
                )
                continue
            logger.warning("[LLM-GW] Route '%s' ignored (invalid value type %s)", route, type(value))
    else:
        for route, url in {
            "chat": settings.llm_route_chat_url,
            "metacog": settings.llm_route_metacog_url,
            "latents": settings.llm_route_latents_url,
            "specialist": settings.llm_route_specialist_url,
        }.items():
            if url:
                route_table[route] = RouteTarget(url=url)

    served_by_defaults = {
        "chat": settings.llm_route_chat_served_by or "atlas-worker-1",
        "metacog": settings.llm_route_metacog_served_by or settings.atlas_metacog_service_name or "atlas-worker-2",
        "latents": settings.llm_route_latents_served_by or "atlas-worker-2",
        "specialist": settings.llm_route_specialist_served_by or "atlas-worker-3",
    }
    for route, target in list(route_table.items()):
        if target.served_by or route not in served_by_defaults:
            continue
        route_table[route] = RouteTarget(
            url=target.url,
            backend=target.backend,
            served_by=served_by_defaults[route],
        )
    return route_table


def get_route_targets() -> Dict[str, RouteTarget]:
    return _load_route_targets()


def _resolve_route(body: ChatBody) -> Tuple[str, Optional[RouteTarget], bool, str]:
    opts = body.options or {}
    route = body.route or opts.get("route") or opts.get("routing_key")
    if body.route:
        route_source = "payload.route"
    elif opts.get("route"):
        route_source = "options.route"
    elif opts.get("routing_key"):
        route_source = "options.routing_key"
    else:
        route_source = "default"
    route_table = get_route_targets()

    if route_table:
        resolved_route = str(route or settings.llm_route_default or "chat")
        return resolved_route, route_table.get(resolved_route), True, route_source

    resolved_route = str(route or settings.llm_route_default or "chat")
    return resolved_route, None, False, route_source


def _timeout_summary(read_sec: Optional[float] = None) -> str:
    r = read_sec if read_sec is not None else float(getattr(settings, "read_timeout_sec", 60.0) or 60.0)
    return (
        f"connect:{getattr(settings, 'connect_timeout_sec', 10.0)} "
        f"read:{r}"
    )


def _resolve_http_read_timeout_sec(body: Optional[ChatBody]) -> float:
    """
    HTTP read timeout for upstream OpenAI-compatible / Ollama calls.

    Callers (e.g. cortex-exec) may set options['gateway_read_timeout_sec'] so the
    gateway does not abort before the bus RPC timeout that wraps this request.
    """
    default = float(getattr(settings, "read_timeout_sec", 60.0) or 60.0)
    if body is None:
        return max(30.0, min(default, 900.0))
    opts = getattr(body, "options", None) or {}
    raw = opts.get("gateway_read_timeout_sec")
    if raw is None:
        return max(30.0, min(default, 900.0))
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return max(30.0, min(default, 900.0))
    return max(30.0, min(v, 900.0))


def _common_http_client(body: Optional[ChatBody] = None) -> httpx.Client:
    """
    Returns an HTTP client using configured timeouts; optional ChatBody can
    override read timeout via options['gateway_read_timeout_sec'].
    """
    read_sec = _resolve_http_read_timeout_sec(body)
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=getattr(settings, "connect_timeout_sec", 10.0),
            read=read_sec,
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
            return join_openai_message_content(msg.get("content"))

        # 2. Text Completion
        if "text" in first:
            return str(first["text"]).strip()

    except Exception:
        pass

    logger.warning(f"[LLM-GW] Response format not understood: {str(data)[:200]}...")
    return ""


def _extract_reasoning_from_openai_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract structured reasoning emitted by provider/OpenAI-compatible payloads only."""
    try:
        choices = data.get("choices") or []
        if not choices:
            return None
        first = choices[0] or {}
        msg = first.get("message")
        candidates: List[Any] = []
        if isinstance(msg, dict):
            candidates.extend(
                [
                    msg.get("reasoning_content"),
                    msg.get("reasoning"),
                    msg.get("reasoning_text"),
                ]
            )
            content_parts = msg.get("content")
            if isinstance(content_parts, list):
                for part in content_parts:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in {"reasoning", "reasoning_text", "thinking", "analysis"}:
                        candidates.extend(
                            [
                                part.get("text"),
                                part.get("content"),
                                part.get("reasoning"),
                            ]
                        )
        candidates.extend(
            [
                first.get("reasoning_content"),
                first.get("reasoning"),
                first.get("reasoning_text"),
            ]
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    except Exception:
        return None
    return None


_THINK_CLOSE_ONLY_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)


def _split_think_blocks(text: str) -> Tuple[str, Optional[str]]:
    raw = str(text or "")
    if "<think>" not in raw:
        close_match = _THINK_CLOSE_ONLY_RE.search(raw)
        if close_match:
            hidden = raw[: close_match.start()].strip()
            visible = raw[close_match.end() :].strip()
            return visible, (hidden or None)
        return raw.strip(), None
    visible = raw
    traces: List[str] = []
    while "<think>" in visible:
        start = visible.find("<think>")
        if start < 0:
            break
        end = visible.find("</think>", start)
        close_len = len("</think>") if end >= 0 else 0
        block = visible[start + len("<think>") : (end if end >= 0 else len(visible))].strip()
        if block:
            traces.append(block)
        if end >= 0:
            visible = (visible[:start] + visible[end + close_len :]).strip()
        else:
            # Unclosed think block: strip from open tag to end to avoid leaking reasoning.
            visible = visible[:start].strip()
            break
    reasoning = "\n\n".join(traces).strip() if traces else None
    return visible.strip(), (reasoning or None)


def _debug_think_capture(raw_text: str, label: str) -> None:
    if not raw_text:
        logger.warning("think_debug label=%s raw_text empty", label)
        return
    start = raw_text.find("<think>")
    end = raw_text.find("</think>")
    excerpt_start = max(0, (start if start >= 0 else 0) - 200)
    excerpt_end = min(len(raw_text), ((end + len("</think>")) if end >= 0 else min(len(raw_text), 400)) + 200)
    excerpt = raw_text[excerpt_start:excerpt_end]
    logger.warning(
        "think_debug label=%s has_open=%s has_close=%s start=%s end=%s excerpt=%r raw_len=%s",
        label,
        start >= 0,
        end >= 0,
        start,
        end,
        excerpt,
        len(raw_text),
    )


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
        if isinstance(obj, list) and obj:
            if all(isinstance(x, (int, float)) for x in obj):
                return [float(x) for x in obj]
            if isinstance(obj[0], list) and obj[0] and all(isinstance(x, (int, float)) for x in obj[0]):
                return [float(x) for x in obj[0]]
        return None

    # Top-level keys
    for k in ("spark_vector", "state_embedding", "state_vector", "embedding", "embeds", "vector", "action_indices"):
        v = _maybe_vec(data.get(k))
        if v is not None:
            return v

    # Common OpenAI-compatible location: choices[0]
    try:
        choices = data.get("choices") or []
        first = (choices[0] or {}) if choices else {}
        for k in ("spark_vector", "state_embedding", "state_vector", "embedding", "embeds", "vector", "action_indices"):
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


def _normalize_backend_name(backend: str) -> str:
    normalized = backend.replace("_", "-").lower()
    if normalized == "llama-cpp":
        return "llamacpp"
    return normalized


def _pick_backend(options: Dict[str, Any] | None, profile: LLMProfile | None) -> str:
    opts = options or {}
    backend = opts.get("backend")
    if not backend and profile:
        backend = profile.backend

    backend = _normalize_backend_name(backend or settings.default_backend or "vllm")

    if backend not in ("vllm", "llamacpp", "ollama", "llama-cola"):
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


def _serialize_messages(messages: Sequence[ChatMessage] | Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    serialized: List[Dict[str, str]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            dumped = msg.model_dump()
            if dumped.get("content") is not None:
                serialized.append({"role": dumped.get("role", "user"), "content": str(dumped["content"])})
        elif isinstance(msg, dict):
            content = msg.get("content")
            if content is not None:
                serialized.append({"role": str(msg.get("role", "user")), "content": str(content)})
    return serialized


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


def _spark_ingest_for_body(body: ChatBody) -> Dict[str, Any]:
    try:
        messages = _serialize_messages(body.messages or [])
        if not messages:
            return {}

        source = getattr(body, "source", "llm-gateway")
        verb = getattr(body, "verb", None) or "unknown"

        raw_user_text = _get_raw_user_text(body)
        latest_user = raw_user_text

        if not latest_user:
            for m in reversed(messages):
                if (m.get("role") or "").lower() == "user":
                    latest_user = m.get("content")
                    break

        if not latest_user:
            latest_user = messages[-1].get("content")

        if not latest_user:
            return {}

        return {
            "latest_user_message": str(latest_user),
            "trace_verb": verb,
            "spark_phase": "pre",
            "spark_used_raw_user_text": bool(raw_user_text),
        }
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Ingestion failed: {e}")
        return {}


_SKIP_SPARK_CANDIDATE_PURPOSES = frozenset({"introspect", "classify"})
_INTROSPECT_SPARK_VERB = "introspect_spark"


def _should_publish_spark_candidate(body: ChatBody, spark_meta: Dict) -> bool:
    """
    Chat-turn candidates feed spark-introspector; internal RPCs (introspect_spark,
    memory classify) must not republish or they create self-sustaining heavy loops.
    """
    if not spark_meta or not spark_meta.get("latest_user_message"):
        return False

    opts = body.options or {}
    if opts.get("skip_spark_candidate_publish"):
        return False

    purpose = str(opts.get("purpose") or "").strip().lower()
    if purpose in _SKIP_SPARK_CANDIDATE_PURPOSES:
        return False

    verb = str(
        getattr(body, "verb", None)
        or opts.get("verb")
        or spark_meta.get("trace_verb")
        or ""
    ).strip().lower()
    if verb == _INTROSPECT_SPARK_VERB:
        return False

    if bool(opts.get("post_turn")) and bool(opts.get("skip_chat_stance_inputs")):
        return False

    return True


def _maybe_publish_spark_introspect(body: ChatBody, spark_meta: Dict, response_text: str):
    try:
        if not _should_publish_spark_candidate(body, spark_meta):
            return

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
    """
    Option 1 fix: publish SparkCandidate as a Titanium envelope.

    Old behavior published a raw dict, which breaks strict consumers (Hunter/codec)
    and bypasses channel catalog ↔ schema registry enforcement.
    """
    import uuid
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.schemas.telemetry.spark_candidate import SparkCandidateV1

    # Validate against schema declared in channels.yaml for this channel
    candidate = SparkCandidateV1.model_validate(payload)

    # correlation_id: if trace_id is a UUID, use it for joinability; else generate one
    try:
        corr_id = uuid.UUID(str(candidate.trace_id))
    except Exception:
        corr_id = uuid.uuid4()

    env = BaseEnvelope(
        kind="spark.candidate",  # keep the existing kind string already seen on-bus
        source=ServiceRef(
            name=settings.service_name,
            node=settings.node_name,
            version=settings.service_version,
            instance=None,
        ),
        correlation_id=corr_id,
        payload=candidate.model_dump(mode="json"),
    )

    bus = OrionBusAsync(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
        enforce_catalog=settings.orion_bus_enforce_catalog,
    )
    await bus.connect()
    try:
        await bus.publish(settings.channel_spark_introspect_candidate, env)
    finally:
        await bus.close()


def _spark_post_ingest_for_reply(body: ChatBody, spark_meta: Dict[str, Any], response_text: str) -> None:
    try:
        if not spark_meta:
            return
        if not response_text:
            return
        spark_meta["latest_assistant_message"] = str(response_text)[:2000]
    except Exception as e:
        logger.warning(f"[LLM-GW Spark] Post-ingest failed: {e}")


# ─────────────────────────────────────────────
# 4. Core HTTP Execution (Unified)
# ─────────────────────────────────────────────



def _extract_text_from_ollama_response(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    message = data.get("message") or {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"]).strip()
    response = data.get("response")
    if response is not None:
        return str(response).strip()
    return ""


def _build_ollama_payload(body: ChatBody, model: str) -> Dict[str, Any]:
    opts = body.options or {}
    options_payload: Dict[str, Any] = {}

    mapped = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_tokens": "num_predict",
        "num_predict": "num_predict",
        "repeat_penalty": "repeat_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "num_ctx": "num_ctx",
        "stop": "stop",
    }

    for source_key, target_key in mapped.items():
        val = opts.get(source_key)
        if val is not None:
            options_payload[target_key] = val

    return {
        "model": model,
        "messages": _serialize_messages(body.messages or []),
        "stream": False,
        "options": options_payload,
    }


def _execute_ollama_chat(
    body: ChatBody,
    model: str,
    base_url: str,
    route: Optional[str] = None,
    served_by: Optional[str] = None,
) -> Dict[str, Any]:
    if not base_url:
        err = "ollama URL not configured"
        logger.error(f"[LLM-GW] {err}")
        return {"text": f"[Error: {err}]", "spark_meta": {}, "raw": {}}

    spark_meta = _spark_ingest_for_body(body)
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = _build_ollama_payload(body, model)

    logger.info(f"[LLM-GW] ollama req model={model} msgs={len(body.messages or [])} url={url}")

    try:
        with _common_http_client(body) as client:
            r = client.post(url, json=payload)
            if r.status_code == 404:
                return {
                    "text": f"[Error: ollama 404 Not Found at {url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }
            r.raise_for_status()
            raw_data = r.json()
            text = _extract_text_from_ollama_response(raw_data)

            _spark_post_ingest_for_reply(body, spark_meta, text)
            _maybe_publish_spark_introspect(body, spark_meta, text)

            return {
                "text": text,
                "spark_meta": spark_meta,
                "spark_vector": None,
                "raw": raw_data,
            }
    except httpx.TimeoutException:
        logger.error(
            "[LLM-GW] ollama TIMEOUT route=%s served_by=%s url=%s corr=%s timeouts=%s",
            route,
            served_by,
            url,
            body.trace_id,
            _timeout_summary(_resolve_http_read_timeout_sec(body)),
        )
        return {
            "text": "[Error: ollama timed out after waiting]",
            "spark_meta": spark_meta,
            "raw": {},
        }
    except Exception as e:
        logger.error(f"[LLM-GW] ollama error: {e}", exc_info=True)
        return {
            "text": f"[Error: ollama failed: {str(e)}]",
            "spark_meta": spark_meta,
            "raw": {},
        }


def _should_use_native_llamacpp_completion(body: ChatBody, backend: str) -> bool:
    opts = body.options or {}
    if opts.get("response_format"):
        return False
    return (
        backend == "llamacpp"
        and bool(opts.get("return_logprobs"))
        and bool(getattr(settings, "llm_logprob_summary_enabled", False))
        and bool(getattr(settings, "llm_logprob_native_completion_enabled", False))
        and str(opts.get("logprob_probe_mode") or "").strip().lower() == "native_completion"
    )


def _execute_llamacpp_native_completion(
    body: ChatBody,
    model: str,
    base_url: str,
    backend_name: str,
    route: Optional[str] = None,
    served_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Aligned generation via llama.cpp /apply-template + /completion with n_probs."""
    if not base_url:
        err = f"{backend_name} URL not configured"
        logger.error(f"[LLM-GW] {err}")
        return {"text": f"[Error: {err}]", "spark_meta": {}, "raw": {}}

    spark_meta = _spark_ingest_for_body(body)
    opts = body.options or {}
    apply_url = f"{base_url.rstrip('/')}/apply-template"
    completion_url = f"{base_url.rstrip('/')}/completion"

    top_k = int(opts.get("logprobs_top_k") or getattr(settings, "llm_logprob_top_k_default", 5))
    n_probs = max(1, min(top_k, 20))
    max_tokens = opts.get("max_tokens")
    if max_tokens is None:
        max_tokens = int(getattr(settings, "llm_logprob_native_completion_max_tokens", 256))
    max_tokens = int(max_tokens)

    apply_payload: Dict[str, Any] = {
        "model": model,
        "messages": _serialize_messages(body.messages or []),
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
    }
    ctk = opts.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk:
        apply_payload["chat_template_kwargs"] = ctk
    apply_payload = {k: v for k, v in apply_payload.items() if v is not None}

    completion_payload: Dict[str, Any] = {
        "n_predict": max_tokens,
        "n_probs": n_probs,
        "stream": False,
        "post_sampling_probs": False,
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
        "stop": opts.get("stop"),
    }
    completion_payload = {k: v for k, v in completion_payload.items() if v is not None}

    logger.info(
        "[LLM-GW] %s native completion corr=%s route=%s served_by=%s n_probs=%s n_predict=%s",
        backend_name,
        body.trace_id,
        route,
        served_by,
        n_probs,
        max_tokens,
    )

    try:
        with _common_http_client(body) as client:
            apply_resp = client.post(apply_url, json=apply_payload)
            if apply_resp.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} /apply-template 404 at {apply_url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }
            apply_resp.raise_for_status()
            apply_data = apply_resp.json()
            prompt = apply_data.get("prompt") if isinstance(apply_data, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                return {
                    "text": f"[Error: {backend_name} /apply-template returned empty prompt]",
                    "spark_meta": spark_meta,
                    "raw": apply_data if isinstance(apply_data, dict) else {},
                }

            completion_payload["prompt"] = prompt
            r = client.post(completion_url, json=completion_payload)
            if r.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} /completion 404 at {completion_url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }
            r.raise_for_status()
            raw_data = r.json()
            if not isinstance(raw_data, dict):
                raw_data = {}

        text = str(raw_data.get("content") or "")
        llm_uncertainty = None
        if bool(getattr(settings, "llm_logprob_summary_enabled", False)):
            llm_uncertainty = extract_llm_uncertainty_from_native_completion(raw_data)

        text, think_reasoning = _split_think_blocks(text)
        _spark_post_ingest_for_reply(body, spark_meta, text)

        raw_out = dict(raw_data)
        if opts.get("logprob_summary_only", True):
            if isinstance(raw_out.get("probs"), list) or isinstance(
                raw_out.get("completion_probabilities"), list
            ):
                raw_out = {
                    k: v
                    for k, v in raw_out.items()
                    if k not in ("probs", "completion_probabilities")
                }

        return {
            "text": text,
            "spark_meta": spark_meta,
            "spark_vector": None,
            "raw": raw_out,
            "reasoning_content": None,
            "reasoning_trace": None,
            "inline_think_content": think_reasoning or None,
            "structured_output_diagnostics": None,
            "llm_uncertainty": llm_uncertainty,
        }
    except httpx.TimeoutException:
        logger.error("[LLM-GW] %s native completion TIMEOUT corr=%s", backend_name, body.trace_id)
        return {"text": f"[Error: {backend_name} timed out]", "spark_meta": spark_meta, "raw": {}}
    except Exception as e:
        logger.error(f"[LLM-GW] {backend_name} native completion error: {e}", exc_info=True)
        return {"text": f"[Error: {backend_name} failed: {str(e)}]", "spark_meta": spark_meta, "raw": {}}


def _execute_openai_chat(
    body: ChatBody,
    model: str,
    base_url: str,
    backend_name: str,
    route: Optional[str] = None,
    served_by: Optional[str] = None,
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

    structured_diag: Dict[str, Any] = {}
    if backend_name in ("vllm", "llamacpp", "llama-cola"):
        opts, structured_diag = apply_structured_output_to_payload(
            opts,
            backend_name=backend_name,
            env_default=str(getattr(settings, "llm_structured_output_method", None) or "none"),
        )
        structured_diag["route"] = route
        structured_diag["served_by"] = served_by

    payload = {
        "model": model,
        "messages": _serialize_messages(body.messages or []),
        "stream": False,
        "temperature": opts.get("temperature"),
        "top_p": opts.get("top_p"),
        "max_tokens": opts.get("max_tokens"),
        "stop": opts.get("stop"),
        "presence_penalty": opts.get("presence_penalty"),
        "frequency_penalty": opts.get("frequency_penalty"),
    }
    response_format = opts.get("response_format")
    if response_format and backend_name in ("vllm", "llamacpp", "llama-cola"):
        payload["response_format"] = response_format
    elif opts.get("return_json") and backend_name in ("vllm", "llamacpp", "llama-cola"):
        payload["response_format"] = {"type": "json_object"}
    # llama.cpp OpenAI server: per-request template kwargs (Qwen3 enable_thinking, etc.) — no worker restart.
    if backend_name in ("llamacpp", "llama-cola"):
        ctk = opts.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk:
            payload["chat_template_kwargs"] = ctk

    if structured_diag.get("structured_output_requested"):
        logger.info(
            "[LLM-GW] structured_output corr=%s method=%s shape=%s schema_name=%s required_keys=%s thinking_policy=%s",
            body.trace_id,
            structured_diag.get("structured_output_method"),
            structured_diag.get("response_format_shape"),
            structured_diag.get("structured_output_schema_name"),
            structured_diag.get("structured_output_schema_required_keys"),
            structured_diag.get("thinking_policy"),
        )
    return_logprobs = bool(opts.get("return_logprobs"))
    logprob_summary_enabled = bool(getattr(settings, "llm_logprob_summary_enabled", False))
    if return_logprobs and logprob_summary_enabled and backend_name in ("vllm", "llamacpp", "llama-cola"):
        top_k = int(opts.get("logprobs_top_k") or getattr(settings, "llm_logprob_top_k_default", 5))
        payload["logprobs"] = True
        payload["top_logprobs"] = max(1, min(top_k, 20))
    # Clean None values
    payload = {k: v for k, v in payload.items() if v is not None}

    # 3. Execute
    logger.info(f"[LLM-GW] {backend_name} req model={model} msgs={len(body.messages or [])} url={url}")
    logger.info(
        "[LLM-GW] llm_gateway_payload_budget corr=%s route=%s served_by=%s payload_max_tokens=%s",
        body.trace_id,
        route,
        served_by,
        payload.get("max_tokens"),
    )

    try:
        with _common_http_client(body) as client:
            r = client.post(url, json=payload)

            if r.status_code == 404:
                return {
                    "text": f"[Error: {backend_name} 404 Not Found at {url}]",
                    "spark_meta": spark_meta,
                    "raw": {},
                }

            r.raise_for_status()
            raw_data = r.json()
            raw_choices = raw_data.get("choices") if isinstance(raw_data, dict) else []
            raw_first = raw_choices[0] if isinstance(raw_choices, list) and raw_choices else {}
            raw_msg = raw_first.get("message") if isinstance(raw_first, dict) else {}
            raw_reasoning_fields = {
                "message.reasoning_content": bool(str((raw_msg.get("reasoning_content") if isinstance(raw_msg, dict) else None) or "").strip()),
                "message.reasoning": bool(str((raw_msg.get("reasoning") if isinstance(raw_msg, dict) else None) or "").strip()),
                "message.reasoning_text": bool(str((raw_msg.get("reasoning_text") if isinstance(raw_msg, dict) else None) or "").strip()),
                "choice.reasoning_content": bool(str((raw_first.get("reasoning_content") if isinstance(raw_first, dict) else None) or "").strip()),
                "choice.reasoning": bool(str((raw_first.get("reasoning") if isinstance(raw_first, dict) else None) or "").strip()),
                "choice.reasoning_text": bool(str((raw_first.get("reasoning_text") if isinstance(raw_first, dict) else None) or "").strip()),
            }
            raw_message_content = raw_msg.get("content") if isinstance(raw_msg, dict) else None
            raw_reasoning_parts = 0
            if isinstance(raw_message_content, list):
                for part in raw_message_content:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in {"reasoning", "reasoning_text", "thinking", "analysis"}:
                        part_text = str(part.get("text") or part.get("content") or part.get("reasoning") or "").strip()
                        if part_text:
                            raw_reasoning_parts += 1
            raw_reasoning_fields["message.content.reasoning_parts"] = raw_reasoning_parts > 0
            print(
                "===THINK_HOP=== hop=llm_gateway_raw "
                f"corr={getattr(body, 'trace_id', None)} "
                f"keys={sorted(list(raw_data.keys())) if isinstance(raw_data, dict) else []} "
                f"reasoning_fields={raw_reasoning_fields}",
                flush=True,
            )
            if _thought_debug_enabled():
                raw_reasoning = _extract_reasoning_from_openai_response(raw_data)
                logger.info(
                    "THOUGHT_DEBUG_LLM stage=raw_response corr=%s model=%s keys=%s raw_reasoning_exists=%s raw_reasoning_len=%s raw_reasoning_snippet=%r raw_content_len=%s raw_content_snippet=%r",
                    getattr(body, "trace_id", None),
                    model,
                    sorted(list(raw_data.keys())) if isinstance(raw_data, dict) else [],
                    bool(str(raw_reasoning or "").strip()),
                    _debug_len(raw_reasoning),
                    _debug_snippet(raw_reasoning),
                    _debug_len((raw_msg or {}).get("content") if isinstance(raw_msg, dict) else None),
                    _debug_snippet((raw_msg or {}).get("content") if isinstance(raw_msg, dict) else None),
                )
            text = _extract_text_from_openai_response(raw_data)
            llm_uncertainty = None
            if return_logprobs and logprob_summary_enabled:
                source_label = f"{backend_name}_openai_chat"
                llm_uncertainty = extract_llm_uncertainty_from_openai_response(
                    raw_data, source=source_label
                )
            reasoning_content = _extract_reasoning_from_openai_response(raw_data)
            raw_usage = raw_data.get("usage") if isinstance(raw_data.get("usage"), dict) else {}
            raw_choices = raw_data.get("choices") if isinstance(raw_data.get("choices"), list) else []
            first_choice = raw_choices[0] if raw_choices and isinstance(raw_choices[0], dict) else {}
            finish_reason = first_choice.get("finish_reason")
            completion_tokens = raw_usage.get("completion_tokens")
            logger.info(
                "[LLM-GW] llm_gateway_provider_result corr=%s route=%s served_by=%s completion_tokens=%s finish_reason=%s",
                body.trace_id,
                route,
                served_by,
                completion_tokens,
                finish_reason,
            )
            if _thought_debug_enabled():
                _debug_think_capture(str(text or ""), "raw_provider_text")
                logger.warning(
                    "think_debug label=pre_split structured_reasoning_exists=%s structured_reasoning_len=%s",
                    bool(str(reasoning_content or "").strip()),
                    _debug_len(reasoning_content),
                )
            think_blocks_found = "<think>" in str(text or "") and "</think>" in str(text or "")
            text, think_reasoning = _split_think_blocks(text)
            if _thought_debug_enabled():
                logger.warning(
                    "think_debug label=post_split visible_len=%s extracted_think_len=%s extracted_think_snippet=%r",
                    _debug_len(text),
                    _debug_len(think_reasoning),
                    _debug_snippet(think_reasoning),
                )
            structured_reasoning = reasoning_content
            if _thought_debug_enabled():
                logger.info(
                    "THOUGHT_DEBUG_LLM stage=extracted corr=%s model=%s structured_reasoning_exists=%s structured_reasoning_len=%s inline_think_len=%s visible_len=%s think_blocks_found=%s structured_reasoning_snippet=%r inline_think_snippet=%r visible_snippet=%r",
                    getattr(body, "trace_id", None),
                    model,
                    bool(str(structured_reasoning or "").strip()),
                    _debug_len(structured_reasoning),
                    _debug_len(think_reasoning),
                    _debug_len(text),
                    think_blocks_found,
                    _debug_snippet(structured_reasoning),
                    _debug_snippet(think_reasoning),
                    _debug_snippet(text),
                )
            if structured_reasoning:
                logger.info(
                    "[LLM-GW] reasoning_trace_extracted corr=%s model=%s chars=%s",
                    getattr(body, "trace_id", None),
                    model,
                    len(structured_reasoning.strip()),
                )
            else:
                logger.info(
                    "[LLM-GW] reasoning_trace_extracted corr=%s model=%s chars=0",
                    getattr(body, "trace_id", None),
                    model,
                )

            # 3b. Spark Post-Ingest (assistant reply)
            _spark_post_ingest_for_reply(body, spark_meta, text)

            # Post-processing: embed/state vector (if present)
            spark_vector = _extract_vector_from_openai_response(raw_data)

            # Post-processing: Spark Introspect
            _maybe_publish_spark_introspect(body, spark_meta, text)

            reasoning_trace = (
                {
                    "role": "reasoning",
                    "stage": "post_answer",
                    "content": reasoning_content,
                }
                if reasoning_content
                else None
            )
            trace_content = reasoning_trace.get("content") if isinstance(reasoning_trace, dict) else None
            print(
                "===THINK_HOP=== hop=llm_gateway_out "
                f"corr={getattr(body, 'trace_id', None)} "
                f"has_reasoning_content={bool(structured_reasoning)} "
                f"reasoning_len={len(structured_reasoning) if structured_reasoning else 0} "
                f"trace_len={len(trace_content) if trace_content else 0} "
                f"inline_think_len={len(think_reasoning) if think_reasoning else 0} "
                f"reasoning_origin={'structured_provider' if structured_reasoning else 'none'} "
                f"preview={_preview_text(structured_reasoning or think_reasoning)}",
                flush=True,
            )
            raw_out = raw_data if isinstance(raw_data, dict) else {}
            if (
                return_logprobs
                and logprob_summary_enabled
                and opts.get("logprob_summary_only", True)
                and isinstance(raw_out.get("choices"), list)
            ):
                stripped_choices = []
                for choice in raw_out["choices"]:
                    if isinstance(choice, dict):
                        stripped_choices.append({k: v for k, v in choice.items() if k != "logprobs"})
                    else:
                        stripped_choices.append(choice)
                raw_out = {**raw_out, "choices": stripped_choices}
            if structured_diag:
                raw_out = {**raw_out, "structured_output_diagnostics": structured_diag}
            return {
                "text": text,
                "spark_meta": spark_meta,
                "spark_vector": spark_vector,
                "raw": raw_out,
                "reasoning_content": structured_reasoning,
                "reasoning_trace": reasoning_trace,
                "inline_think_content": think_reasoning or None,
                "structured_output_diagnostics": structured_diag or None,
                "llm_uncertainty": llm_uncertainty,
            }

    except httpx.TimeoutException:
        logger.error(
            "[LLM-GW] %s TIMEOUT route=%s served_by=%s url=%s corr=%s timeouts=%s",
            backend_name,
            route,
            served_by,
            url,
            body.trace_id,
            _timeout_summary(_resolve_http_read_timeout_sec(body)),
        )
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
    route_table = get_route_targets()
    lane_routing = bool(getattr(settings, "llm_lane_routing_enabled", False)) and bool(route_table)
    if lane_routing:
        decision = resolve_llm_lane_route(
            body.options,
            body.route,
            llm_lane_default=str(getattr(settings, "llm_lane_default", "chat") or "chat"),
            llm_route_default=str(getattr(settings, "llm_route_default", "chat") or "chat"),
            llm_allow_background_to_chat_fallback=bool(
                getattr(settings, "llm_allow_background_to_chat_fallback", False)
            ),
            llm_route_spark_served_by=getattr(settings, "llm_route_spark_served_by", None),
            llm_route_background_served_by=getattr(settings, "llm_route_background_served_by", None),
            llm_route_agent_served_by=getattr(settings, "llm_route_agent_served_by", None),
            route_table_keys=set(route_table.keys()),
            route_served_by={k: t.served_by for k, t in route_table.items()},
        )
        corr = getattr(body, "trace_id", None)
        logger.info(
            "llm_gateway_lane_route corr=%s trace_id=%s requested_lane=%s resolved_lane=%s served_by=%s status=%s reason=%s fallback_used=%s",
            corr,
            corr,
            decision.requested_llm_lane,
            decision.resolved_llm_lane,
            decision.served_by,
            decision.route_status,
            decision.reason,
            decision.fallback_used,
        )
        if decision.fallback_used and "emergency_chat_fallback" in decision.reason:
            logger.warning(
                "LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK emergency path corr=%s trace_id=%s reason=%s served_by=%s",
                corr,
                corr,
                decision.reason,
                decision.served_by,
            )
        if decision.route_table_key is None:
            logger.info(
                "llm_gateway_lane_rejected corr=%s trace_id=%s requested_lane=%s status=%s reason=%s",
                corr,
                corr,
                decision.requested_llm_lane,
                decision.route_status,
                decision.reason,
            )
            return {
                "text": "",
                "content": "",
                "spark_meta": {},
                "raw": {
                    "error": "llm_route_unavailable",
                    "details": {
                        "llm_lane": decision.resolved_llm_lane,
                        "route_status": decision.route_status,
                        "reason": decision.reason,
                        "client_route": body.route,
                        "chat_fallback_allowed": bool(
                            getattr(settings, "llm_allow_background_to_chat_fallback", False)
                            and (body.options or {}).get("allow_chat_fallback")
                        ),
                    },
                },
                "route": body.route,
                "served_by": None,
            }
        body = body.model_copy(update={"route": decision.route_table_key})

    route, route_target, has_route_table, route_source = _resolve_route(body)
    effective_profile_name = body.profile_name
    if (
        route == "metacog"
        and not effective_profile_name
        and settings.atlas_metacog_profile_name
    ):
        # Keep metacog lane deterministic even when callers omit profile.
        effective_profile_name = settings.atlas_metacog_profile_name
    profile = _select_profile(effective_profile_name)
    backend = _pick_backend(body.options, profile)
    model = _resolve_model(body.model, profile)

    if has_route_table and not route_target:
        logger.error("[LLM-GW] Route '%s' not configured in route table", route)
        return {
            "text": f"[Error: route '{route}' not configured]",
            "spark_meta": {},
            "raw": {"error": "route_not_configured", "route": route},
            "route": route,
        }

    route_url: Optional[str] = None
    served_by: Optional[str] = None
    if route_target:
        backend = _normalize_backend_name(route_target.backend or "llamacpp")
        base_url = route_target.url
        route_url = base_url
        served_by = route_target.served_by
        if backend == "vllm":
            model = _normalize_model_for_vllm(model)
        if backend == "ollama":
            if settings.ollama_use_openai_compat:
                logger.info(
                    "[LLM-GW] route=%s route_source=%s backend=%s served_by=%s url=%s model=%s corr=%s timeouts=%s",
                    route,
                    route_source,
                    backend,
                    served_by,
                    base_url,
                    model,
                    body.trace_id,
                    _timeout_summary(_resolve_http_read_timeout_sec(body)),
                )
                result = _execute_openai_chat(
                    body,
                    model,
                    base_url,
                    "ollama",
                    route=route,
                    served_by=served_by,
                )
                if isinstance(result, dict):
                    result["backend"] = backend
                    result["model"] = model
                    result["route"] = route
                    result["served_by"] = served_by
                return result
            logger.info(
                "[LLM-GW] route=%s route_source=%s backend=%s served_by=%s url=%s model=%s corr=%s timeouts=%s",
                route,
                route_source,
                backend,
                served_by,
                base_url,
                model,
                body.trace_id,
                _timeout_summary(_resolve_http_read_timeout_sec(body)),
            )
            result = _execute_ollama_chat(
                body,
                model,
                base_url,
                route=route,
                served_by=served_by,
            )
            if isinstance(result, dict):
                result["backend"] = backend
                result["model"] = model
                result["route"] = route
                result["served_by"] = served_by
            return result
    else:
        # Normalize aliases if vLLM, harmless if llama.cpp
        if backend == "vllm":
            model = _normalize_model_for_vllm(model)
            base_url = settings.vllm_url
            route_url = base_url

        elif backend == "ollama":
            base_url = settings.ollama_url
            route_url = base_url
            if settings.ollama_use_openai_compat:
                logger.info(
                    "[LLM-GW] route=%s route_source=%s backend=%s served_by=%s url=%s model=%s corr=%s timeouts=%s",
                    route,
                    route_source,
                    backend,
                    served_by,
                    base_url,
                    model,
                    body.trace_id,
                    _timeout_summary(_resolve_http_read_timeout_sec(body)),
                )
                result = _execute_openai_chat(
                    body,
                    model,
                    base_url,
                    "ollama",
                    route=route,
                    served_by=served_by,
                )
                if isinstance(result, dict):
                    result["backend"] = backend
                    result["model"] = model
                    result["route"] = route
                    result["served_by"] = served_by
                return result
            logger.info(
                "[LLM-GW] route=%s route_source=%s backend=%s served_by=%s url=%s model=%s corr=%s timeouts=%s",
                route,
                route_source,
                backend,
                served_by,
                base_url,
                model,
                body.trace_id,
                _timeout_summary(_resolve_http_read_timeout_sec(body)),
            )
            result = _execute_ollama_chat(
                body,
                model,
                base_url,
                route=route,
                served_by=served_by,
            )
            if isinstance(result, dict):
                result["backend"] = backend
                result["model"] = model
                result["route"] = route
                result["served_by"] = served_by
            return result

        elif backend == "llama-cola":
            base_url = settings.llama_cola_url
            route_url = base_url

        else:
            base_url = settings.llamacpp_url
            route_url = base_url

    logger.info(
        "[LLM-GW] route=%s route_source=%s backend=%s served_by=%s url=%s model=%s corr=%s timeouts=%s",
        route,
        route_source,
        backend,
        served_by,
        route_url,
        model,
        body.trace_id,
        _timeout_summary(_resolve_http_read_timeout_sec(body)),
    )
    if _should_use_native_llamacpp_completion(body, backend):
        result = _execute_llamacpp_native_completion(
            body,
            model,
            base_url,
            backend,
            route=route,
            served_by=served_by,
        )
    else:
        result = _execute_openai_chat(
            body,
            model,
            base_url,
            backend,
            route=route,
            served_by=served_by,
        )
    if isinstance(result, dict):
        result["backend"] = backend
        result["model"] = model
        result["route"] = route
        result["served_by"] = served_by
    return result


def run_llm_generate(body: GenerateBody) -> str:
    """Wrapper to make Generate look like Chat"""
    chat_body = ChatBody(
        messages=[ChatMessage(role="user", content=body.prompt)],
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
        messages=[ChatMessage(role="user", content=final_prompt)],
        raw_user_text=body.raw_user_text or (body.context.get("user_message") if isinstance(body.context, dict) else None),
        options={},
        trace_id=body.origin_node,
        source=f"cortex:{body.service}",
        verb=body.verb,
        profile_name=getattr(body, "profile_name", None),
    )

    if backend == "ollama":
        result = _execute_ollama_chat(chat_body, model, settings.ollama_url)

    elif backend == "llamacpp":
        result = _execute_openai_chat(chat_body, model, settings.llamacpp_url, "llamacpp")

    elif backend == "llama-cola":
        result = _execute_openai_chat(chat_body, model, settings.llama_cola_url, "llama-cola")

    else:
        result = _execute_openai_chat(chat_body, model, settings.vllm_url, "vllm")

    elapsed_ms = int((time.time() - t0) * 1000)

    # 4. Log
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
