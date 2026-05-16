"""Shared projection-context enrichment for Orch Mind preflight and Exec parity."""

from __future__ import annotations

import os
from typing import Any

from orion.cognition.personality.identity_context import build_identity_context, load_identity_file

FALLBACK_ORION_IDENTITY_SUMMARY = [
    "Oríon is an ongoing cognitive presence in a long-running shared project.",
    "Oríon is not a generic assistant.",
]

FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY = [
    "Juniper is Oríon's co-architect, steward, and trusted interlocutor.",
]

FALLBACK_RESPONSE_POLICY_SUMMARY = [
    "Answer the actual question first.",
    "Do not collapse into generic assistant language.",
]

_IDENTITY_KEYS = ("orion_identity_summary", "juniper_relationship_summary", "response_policy_summary")


def _unique(items: list[str], *, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def identity_kernel_with_fallbacks(ctx: dict[str, Any]) -> dict[str, list[str]]:
    def _list_from_ctx(key: str, fallback: list[str]) -> list[str]:
        value = ctx.get(key)
        if isinstance(value, list):
            compacted = _unique([str(item) for item in value])
            if compacted:
                return compacted
        return list(fallback)

    return {
        "orion_identity_summary": _list_from_ctx("orion_identity_summary", FALLBACK_ORION_IDENTITY_SUMMARY),
        "juniper_relationship_summary": _list_from_ctx(
            "juniper_relationship_summary",
            FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY,
        ),
        "response_policy_summary": _list_from_ctx("response_policy_summary", FALLBACK_RESPONSE_POLICY_SUMMARY),
    }


def _resolve_personality_file(ctx: dict[str, Any], plan_metadata: dict[str, Any] | None) -> str:
    direct = str(ctx.get("personality_file") or "").strip()
    if direct:
        return direct
    if isinstance(plan_metadata, dict):
        nested = str(plan_metadata.get("personality_file") or "").strip()
        if nested:
            return nested
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    if isinstance(metadata, dict):
        plan_meta = metadata.get("plan_metadata") if isinstance(metadata.get("plan_metadata"), dict) else {}
        return str(plan_meta.get("personality_file") or metadata.get("personality_file") or "").strip()
    return ""


def inject_identity_context_for_projection(
    ctx: dict[str, Any],
    *,
    plan_metadata: dict[str, Any] | None = None,
) -> str:
    """Populate identity_yaml producer inputs (mirrors Exec ``_inject_identity_context``)."""
    if all(isinstance(ctx.get(key), list) and ctx.get(key) for key in _IDENTITY_KEYS):
        return str(ctx.get("identity_kernel_source") or "ctx_present")

    personality_file = _resolve_personality_file(ctx, plan_metadata)
    source = "fallback_missing_metadata"
    if personality_file:
        try:
            identity_data = load_identity_file(personality_file)
            identity_context = build_identity_context(identity_data)
            if any(isinstance(identity_context.get(key), list) and identity_context.get(key) for key in _IDENTITY_KEYS):
                for key, value in identity_context.items():
                    if isinstance(value, list) and value:
                        ctx[key] = value
                source = "configured_yaml"
        except Exception:
            source = "fallback_load_error"

    ctx.update(identity_kernel_with_fallbacks(ctx))
    ctx["identity_kernel_source"] = source
    return source


def _orion_state_from_ctx(ctx: dict[str, Any]) -> dict[str, Any] | None:
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    state = metadata.get("orion_state") if isinstance(metadata, dict) else None
    if isinstance(state, dict):
        return state
    direct = ctx.get("orion_state")
    return direct if isinstance(direct, dict) else None


def enrich_projection_context(
    ctx: dict[str, Any],
    *,
    plan_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply Exec-parity ctx keys needed by substrate producers before projection build."""
    inject_identity_context_for_projection(ctx, plan_metadata=plan_metadata)
    state = _orion_state_from_ctx(ctx)
    if isinstance(state, dict) and ctx.get("spark_state_snapshot") is None and ctx.get("spark_state_json") is None:
        payload = state.get("payload") if isinstance(state.get("payload"), dict) else state
        if isinstance(payload, dict) and payload:
            ctx["spark_state_snapshot"] = payload
    return ctx


def summarize_projection_inputs(
    ctx: dict[str, Any] | None,
    *,
    phase: str,
) -> dict[str, Any]:
    """Compact input summary for Orch vs Exec projection parity comparison."""
    ctx = ctx if isinstance(ctx, dict) else {}
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    recall_bundle = ctx.get("recall_bundle") if isinstance(ctx.get("recall_bundle"), dict) else {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle.get("fragments"), list) else []
    social_keys = (
        "social_inspection_snapshot",
        "social_stance_snapshot",
        "social_turn_policy",
        "social_peer_style_hint",
        "social_context_window",
        "social_thread_routing",
        "social_repair_decision",
    )
    return {
        "phase": phase,
        "anchors": list(ctx.get("projection_anchors") or ("orion", "relationship", "juniper")),
        "recall_bundle_present": bool(fragments),
        "recall_fragment_count": len(fragments),
        "recall_enabled": bool(ctx.get("recall_enabled")),
        "identity_yaml_inputs": {
            key: len(ctx.get(key) or []) if isinstance(ctx.get(key), list) else 0 for key in _IDENTITY_KEYS
        },
        "identity_kernel_source": ctx.get("identity_kernel_source"),
        "social_inputs_present": {key: key in ctx and ctx.get(key) is not None for key in social_keys},
        "situation_inputs_present": {
            "chat_situation_summary": isinstance(ctx.get("chat_situation_summary"), dict),
            "chat_reasoning_summary": isinstance(ctx.get("chat_reasoning_summary"), dict),
        },
        "session_history_present": bool(ctx.get("messages")),
        "user_context_present": bool(ctx.get("user_message") or ctx.get("raw_user_text")),
        "orion_state_present": _orion_state_from_ctx(ctx) is not None,
        "spark_state_present": ctx.get("spark_state_snapshot") is not None or bool(ctx.get("spark_state_json")),
        "autonomy_state_present": isinstance(ctx.get("chat_autonomy_state_v2"), dict)
        or isinstance(metadata.get("autonomy_state"), dict),
        "concept_induction_profiles_present": bool(ctx.get("concept_profile_ids") or metadata.get("concept_profile_ids")),
        "relationship_anchors_materialized": list(ctx.get("cold_anchors") or []),
        "orch_invoked_before_exec": bool(ctx.get("orch_invoked_before_exec", phase.startswith("orch"))),
        "env_flags": {
            "UNIFIED_BELIEFS_TIMEOUT_SEC": os.getenv("UNIFIED_BELIEFS_TIMEOUT_SEC"),
            "AUTONOMY_GRAPH_BACKEND": os.getenv("AUTONOMY_GRAPH_BACKEND"),
            "CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED": os.getenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED"),
            "MIND_RECALL_PREFETCH_ENABLED": os.getenv("MIND_RECALL_PREFETCH_ENABLED"),
        },
    }
