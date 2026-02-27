from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.cognition.verb_activation import is_active

from .settings import settings


def default_mode() -> str:
    return "auto" if settings.HUB_AUTO_DEFAULT_ENABLED else "brain"


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"brain", "agent", "council", "auto"}:
        return mode
    return default_mode()


def build_cortex_chat_request(
    *,
    prompt: str,
    payload: Dict[str, Any],
    session_id: Optional[str],
    user_id: Optional[str],
    trace_id: Optional[str],
    source_label: str,
) -> Tuple[CortexChatRequest, Dict[str, Any], bool]:
    mode = _normalize_mode(payload.get("mode"))

    raw_recall = payload.get("use_recall", None)
    if raw_recall is None:
        use_recall = True
    elif isinstance(raw_recall, bool):
        use_recall = raw_recall
    elif isinstance(raw_recall, (int, float)):
        use_recall = bool(raw_recall)
    else:
        lowered = str(raw_recall).strip().lower()
        use_recall = lowered in {"1", "true", "yes", "y", "on"}

    recall_payload: Dict[str, Any] = {"enabled": use_recall}
    if payload.get("recall_mode"):
        recall_payload["mode"] = payload.get("recall_mode")
    if payload.get("recall_profile"):
        recall_payload["profile"] = payload.get("recall_profile")
    if payload.get("recall_required"):
        recall_payload["required"] = True
    if use_recall and "profile" not in recall_payload:
        recall_payload["profile"] = "reflect.v1"

    options = payload.get("options")
    options = dict(options) if isinstance(options, dict) else {}

    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]
    verb_override = None
    if len(selected_verbs) == 1:
        verb_override = selected_verbs[0]
        options.pop("allowed_verbs", None)
        if mode == "auto":
            if verb_override == "agent_runtime":
                mode = "agent"
            elif verb_override == "council_runtime":
                mode = "council"
            else:
                mode = "brain"
    elif len(selected_verbs) > 1:
        options["allowed_verbs"] = selected_verbs

    if mode == "auto":
        options["route_intent"] = "auto"
    else:
        options["route_intent"] = "none"

    req = CortexChatRequest(
        prompt=prompt,
        mode=mode,
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        packs=payload.get("packs"),
        verb=verb_override,
        options=options,
        recall=recall_payload,
        metadata={"source": source_label},
    )

    debug = {
        "mode": req.mode,
        "verb": req.verb,
        "route_intent": (req.options or {}).get("route_intent"),
        "allowed_verbs_count": len(((req.options or {}).get("allowed_verbs") or [])),
        "packs": req.packs or [],
        "recall_enabled": bool((req.recall or {}).get("enabled", True)),
        "recall_required": bool((req.recall or {}).get("required", False)),
        "recall_profile": (req.recall or {}).get("profile"),
    }
    return req, debug, use_recall


def validate_single_verb_override(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]
    if len(selected_verbs) != 1:
        return None
    verb = selected_verbs[0]
    if is_active(verb, node_name=settings.NODE_NAME):
        return None
    return {
        "error": f"inactive_verb:{verb}",
        "message": f"Verb '{verb}' is inactive on node {settings.NODE_NAME}.",
        "verb": verb,
        "node": settings.NODE_NAME,
    }
