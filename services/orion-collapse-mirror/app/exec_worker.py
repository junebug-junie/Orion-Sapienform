from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.settings import settings
from orion.schemas.collapse_mirror import CollapseMirrorEntry

logger = logging.getLogger("orion-collapse-mirror.exec-worker")


def _extract_text(obj: Any) -> Optional[str]:
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
    prior_step_results is a list of StepExecutionResult dicts (from cortex-orch).
    We walk backward and look for a JSON string inside services[].payload.
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

            text = _extract_text(svc_payload)
            if not text and isinstance(svc_payload.get("payload"), dict):
                text = _extract_text(svc_payload.get("payload"))

            if not text:
                continue

            try:
                d = json.loads(text)
                if isinstance(d, dict):
                    return d, "ok"
            except Exception:
                continue

    return None, "no_json_found"


def _fill_defaults(entry: CollapseMirrorEntry) -> CollapseMirrorEntry:
    """
    Prefer schema helper if present; else fill the same defaults as Hub:
      - timestamp: ISO8601 UTC
      - environment: CHRONICLE_ENVIRONMENT (default dev)
    """
    if hasattr(entry, "with_defaults") and callable(getattr(entry, "with_defaults")):
        return entry.with_defaults()

    payload = entry.model_dump(mode="json")
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    if not payload.get("environment"):
        import os
        payload["environment"] = os.getenv("CHRONICLE_ENVIRONMENT", "dev")
    return CollapseMirrorEntry.model_validate(payload)


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

    Also tolerates direct-style payloads (no nested "payload") by treating
    the top-level dict as exec payload if needed.
    """
    started = time.time()

    event = envelope.get("event")
    if event != "exec_step":
        return {}

    correlation_id = envelope.get("correlation_id") or envelope.get("trace_id") or ""
    reply_channel = envelope.get("reply_channel") or ""

    # cortex-orch: nested payload
    exec_payload = envelope.get("payload")
    if not isinstance(exec_payload, dict):
        # fallback: allow direct-style payload (rare)
        exec_payload = envelope if isinstance(envelope, dict) else {}

    candidate, reason = _extract_candidate(exec_payload)
    if not candidate:
        return {
            # cortex-orch fan-in matches on "trace_id"
            "trace_id": correlation_id,
            "correlation_id": correlation_id,
            "service": "CollapseMirrorService",
            "ok": False,
            "elapsed_ms": int((time.time() - started) * 1000),
            "published": False,
            "reason": reason,
        }

    try:
        entry = CollapseMirrorEntry.model_validate(candidate)
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

    # Gate noop
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
            envelope = msg.get("data") or {}
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
