from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from orion.cognition.verb_activation import is_active
from orion.schemas.cortex.contracts import CortexChatRequest


def _normalize_mode(value: Any, *, default_mode: str, auto_default_enabled: bool) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"brain", "agent", "council", "auto"}:
        return mode
    return "auto" if auto_default_enabled else default_mode


def build_cortex_chat_request(
    *,
    payload: Dict[str, Any],
    session_id: str | None,
    user_id: str | None,
    trace_id: str | None,
    default_mode: str,
    auto_default_enabled: bool,
    source_label: str,
    prompt: str,
) -> Tuple[CortexChatRequest, Dict[str, Any], bool]:
    mode = _normalize_mode(payload.get("mode"), default_mode=default_mode, auto_default_enabled=auto_default_enabled)

    raw_recall = payload.get("use_recall", None)
    if raw_recall is None:
        use_recall = True
    elif isinstance(raw_recall, bool):
        use_recall = raw_recall
    elif isinstance(raw_recall, (int, float)):
        use_recall = bool(raw_recall)
    else:
        use_recall = str(raw_recall).strip().lower() in {"1", "true", "yes", "y", "on"}

    recall_payload: Dict[str, Any] = {"enabled": use_recall}
    if payload.get("recall_mode"):
        recall_payload["mode"] = payload.get("recall_mode")
    if payload.get("recall_profile"):
        recall_payload["profile"] = payload.get("recall_profile")
    if payload.get("recall_required"):
        recall_payload["required"] = True
    if use_recall and "profile" not in recall_payload:
        recall_payload["profile"] = "reflect.v1"

    options = dict(payload.get("options") or {}) if isinstance(payload.get("options"), dict) else {}
    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]

    verb_override: str | None = None
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

    route_intent = "none"
    if mode == "auto" and not verb_override and len(selected_verbs) == 0:
        route_intent = "auto"
        options["route_intent"] = "auto"
    else:
        options.pop("route_intent", None)

    req = CortexChatRequest(
        prompt=prompt,
        mode=mode,
        route_intent=route_intent,
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
        "route_intent": (req.options or {}).get("route_intent") or "none",
        "allowed_verbs_count": len(((req.options or {}).get("allowed_verbs") or [])),
        "packs": req.packs or [],
        "recall_enabled": bool((req.recall or {}).get("enabled", True)),
        "recall_required": bool((req.recall or {}).get("required", False)),
        "recall_profile": (req.recall or {}).get("profile"),
    }
    return req, debug, use_recall


def validate_single_verb_override(payload: Dict[str, Any], *, node_name: str) -> Optional[Dict[str, Any]]:
    selected_verbs = [str(v).strip() for v in (payload.get("verbs") or []) if str(v).strip()]
    if len(selected_verbs) != 1:
        return None
    verb = selected_verbs[0]
    if is_active(verb, node_name=node_name):
        return None
    return {
        "error": f"inactive_verb:{verb}",
        "message": f"Verb '{verb}' is inactive on node {node_name}.",
        "verb": verb,
        "node": node_name,
    }


def build_chat_request(
    *,
    payload: Dict[str, Any],
    session_id: str | None,
    user_id: str | None,
    trace_id: str | None,
    default_mode: str,
    auto_default_enabled: bool,
    source_label: str,
    prompt: str,
) -> Tuple[CortexChatRequest, Dict[str, Any], bool]:
    """Compatibility wrapper: canonical Hub chat request builder for HTTP + WS."""
    return build_cortex_chat_request(
        payload=payload,
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        default_mode=default_mode,
        auto_default_enabled=auto_default_enabled,
        source_label=source_label,
        prompt=prompt,
    )
