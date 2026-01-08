from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from app.settings import settings
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.collapse import create_entry_from_v2

logger = logging.getLogger("orion-collapse-mirror.exec-worker")


def _extract_text(obj: Any) -> Optional[str]:
    """Recursively attempts to find a text string in likely LLM output structures."""
    if isinstance(obj, str) and obj.strip():
        return obj.strip()

    if not isinstance(obj, dict):
        return None

    # common direct keys
    for k in ("text", "final_text", "output", "content", "llm_output"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # common nesting: result -> llm_output/text
    res = obj.get("result")
    if isinstance(res, dict):
        for k in ("text", "final_text", "output", "content", "llm_output"):
            v = res.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # common nesting: payload -> body -> text
    nested = obj.get("payload")
    if isinstance(nested, dict):
        for k in ("text", "final_text", "output", "content", "llm_output"):
            v = nested.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        body = nested.get("body")
        if isinstance(body, dict):
            for k in ("text", "final_text", "output", "content", "llm_output"):
                v = body.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    return None


def _extract_candidate_from_prior(prior_step_results: Any) -> Tuple[Optional[dict], str]:
    """
    prior_step_results is a list of StepExecutionResult dicts.
    We walk backward, find text, and attempt to parse JSON (even if embedded in chatty text).
    """
    if not isinstance(prior_step_results, list) or not prior_step_results:
        return None, "no_prior_step_results"

    for step in reversed(prior_step_results):
        if not isinstance(step, dict):
            continue

        services = step.get("services")
        if not isinstance(services, list) or not services:
            continue

        for svc in services:
            if not isinstance(svc, dict):
                continue

            svc_payload = svc.get("payload") or {}
            if not isinstance(svc_payload, dict):
                continue

            # Attempt extraction from common locations
            text = _extract_text(svc_payload)
            if not text and isinstance(svc_payload.get("payload"), dict):
                text = _extract_text(svc_payload.get("payload"))

            if not text:
                continue

            # --- STRATEGY 1: Clean Parse ---
            try:
                d = json.loads(text)
                if isinstance(d, dict):
                    return d, "ok"
            except Exception:
                pass

            # --- STRATEGY 2: Fuzzy Extraction (Find { ... }) ---
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = text[start : end + 1]
                    d = json.loads(candidate)
                    if isinstance(d, dict):
                        return d, "fuzzy_extracted"
            except Exception:
                pass

    return None, "no_json_found"


def _fill_defaults(entry: CollapseMirrorEntryV2) -> CollapseMirrorEntryV2:
    return entry.with_defaults()


def _extract_candidate(exec_payload: Dict[str, Any]) -> Tuple[Optional[dict], str]:
    """
    Supports two sources:
      A) context.collapse_entry or context.collapse_json (direct calls)
      B) prior_step_results (normal two-step verb via cortex-orch)
    """
    ctx = exec_payload.get("context") or {}
    if isinstance(ctx, dict):
        direct = ctx.get("collapse_entry")
        if isinstance(direct, dict):
            return direct, "context.collapse_entry"

        direct_json = ctx.get("collapse_json")
        if isinstance(direct_json, str) and direct_json.strip():
            try:
                d = json.loads(direct_json)
                if isinstance(d, dict):
                    return d, "context.collapse_json"
            except Exception:
                pass

    prior = exec_payload.get("prior_step_results") or []
    return _extract_candidate_from_prior(prior)


def _handle_exec_step(bus, envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles cortex-orch style envelopes:
      { event, service, correlation_id, reply_channel, payload: {...} }
    """
    started = time.time()

    event = envelope.get("event")
    if event != "exec_step":
        return {}

    correlation_id = envelope.get("correlation_id") or envelope.get("trace_id") or ""
    
    # cortex-orch: nested payload
    exec_payload = envelope.get("payload")
    if not isinstance(exec_payload, dict):
        # fallback: allow direct-style payload (rare)
        exec_payload = envelope if isinstance(envelope, dict) else {}

    candidate, reason = _extract_candidate(exec_payload)
    if not candidate:
        return {
            "trace_id": correlation_id,
            "correlation_id": correlation_id,
            "service": "CollapseMirrorService",
            "ok": False,
            "elapsed_ms": int((time.time() - started) * 1000),
            "published": False,
            "reason": reason,
        }

    # ─────────────────────────────────────────────────────────────
    # Hydrate NOOPs to satisfy Strict Schema
    # If type="noop", we provide dummy values for mandatory fields.
    # ─────────────────────────────────────────────────────────────
    if str(candidate.get("type", "")).strip().lower() == "noop":
        defaults = {
            "observer": "Orion",
            "trigger": "noop",
            "observer_state": ["idle"],
            "field_resonance": "none",
            "emergent_entity": "none",
            "summary": "No collapse detected.",
            "mantra": "void",
            "causal_echo": None,
            "timestamp": None,
            "environment": None
        }
        for k, v in defaults.items():
            if k not in candidate:
                candidate[k] = v

    try:
        entry = create_entry_from_v2(candidate, source_service=settings.SERVICE_NAME, source_node=settings.NODE_NAME)
        entry = _fill_defaults(entry)
    except Exception as e:
        return {
            "trace_id": correlation_id,
            "correlation_id": correlation_id,
            "service": "CollapseMirrorService",
            "ok": False,
            "elapsed_ms": int((time.time() - started) * 1000),
            "published": False,
            "reason": f"schema_validation_failed:{type(e).__name__}",
        }

    # Gate noop (do not publish to downstream)
    if (entry.type or "").strip().lower() == "noop":
        return {
            "trace_id": correlation_id,
            "correlation_id": correlation_id,
            "service": "CollapseMirrorService",
            "ok": True,
            "elapsed_ms": int((time.time() - started) * 1000),
            "published": False,
            "reason": "noop",
            "entry": entry.model_dump(mode="json"),
        }

    # Publish to intake (this service owns the channel knowledge)
    bus.publish(settings.CHANNEL_COLLAPSE_INTAKE, entry.model_dump(mode="json"))

    return {
        "trace_id": correlation_id,
        "correlation_id": correlation_id,
        "service": "CollapseMirrorService",
        "ok": True,
        "elapsed_ms": int((time.time() - started) * 1000),
        "published": True,
        "published_to": settings.CHANNEL_COLLAPSE_INTAKE,
        "reason": "ok",
        "entry": entry.model_dump(mode="json"),
    }


def start_collapse_mirror_exec_worker(bus) -> None:
    """
    Threaded worker: blocking subscribe to EXEC_REQUEST_PREFIX:CollapseMirrorService.
    Must not run on the FastAPI event loop.
    """
    exec_request_prefix = getattr(settings, "EXEC_REQUEST_PREFIX", None) or "orion-exec:request"
    listen_channel = f"{exec_request_prefix}:CollapseMirrorService"

    def _loop():
        logger.info("[CollapseMirrorService] Subscribing to %s", listen_channel)
        
        for msg in bus.raw_subscribe(listen_channel):
            raw_data = msg.get("data")
            
            # Redis sends integers for subscription events; skip them.
            if isinstance(raw_data, int) or raw_data is None:
                continue

            # Decode JSON string/bytes to Dict
            envelope = raw_data
            if isinstance(envelope, (str, bytes)):
                try:
                    envelope = json.loads(envelope)
                except Exception:
                    logger.warning("[CollapseMirrorService] Ignored non-JSON payload")
                    continue

            if not isinstance(envelope, dict):
                continue

            # Only handle exec_step
            if envelope.get("event") != "exec_step":
                continue

            reply_channel = envelope.get("reply_channel") or ""
            correlation_id = envelope.get("correlation_id") or envelope.get("trace_id") or ""

            if not reply_channel:
                logger.warning("[CollapseMirrorService] Missing reply_channel (corr_id=%s)", correlation_id)
                continue

            try:
                result = _handle_exec_step(bus, envelope)
                if not result:
                    continue
                
                # Ensure envelope fields are mirrored back if missing
                if "trace_id" not in result:
                    result["trace_id"] = correlation_id
                
            except Exception as e:
                logger.exception("[CollapseMirrorService] Handler error (corr_id=%s): %s", correlation_id, e)
                result = {
                    "trace_id": correlation_id,
                    "correlation_id": correlation_id,
                    "service": "CollapseMirrorService",
                    "ok": False,
                    "elapsed_ms": 0,
                    "published": False,
                    "reason": f"exception:{type(e).__name__}",
                }

            bus.publish(reply_channel, result)

    t = threading.Thread(target=_loop, daemon=True, name="CollapseMirrorServiceWorker")
    t.start()
