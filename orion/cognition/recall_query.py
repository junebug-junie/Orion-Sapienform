"""Shared recall query construction and reply → ctx normalization (Exec + Orch)."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1

# Catalog-registered reply prefix (orion/bus/channels.yaml); Exec and Orch Mind preflight share this.
DEFAULT_RECALL_REPLY_PREFIX = "orion:exec:result:RecallService"

_PROFILES_DIR = Path(__file__).resolve().parents[1] / "recall" / "profiles"


@lru_cache(maxsize=32)
def recall_profile_prompt_flags(profile_name: str | None) -> dict[str, Any]:
    """Load opt-in render/prompt flags from recall profile YAML."""
    if not profile_name:
        return {}
    path = _PROFILES_DIR / f"{profile_name}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {
        "strict_prompt_budget": bool(data.get("strict_prompt_budget")),
        "render_char_budget": int(data.get("render_char_budget") or 0),
        "max_render_snippet_chars": int(data.get("max_render_snippet_chars") or 0),
        "prompt_safe_ctx": bool(data.get("prompt_safe_ctx")),
    }



def last_user_message_from_ctx(ctx: dict[str, Any]) -> str:
    """Best-effort latest user utterance from plan/exec ctx."""
    for key in ("raw_user_text", "user_message"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = ctx.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue
            content = message.get("content") or message.get("text") or ""
            if isinstance(content, str) and content.strip():
                return content.strip()[:10_000]
    return ""


def active_turn_ids_for_recall(ctx: dict[str, Any], correlation_id: str) -> list[str]:
    """Exec-parity active-turn exclusion ids."""
    trace_val = ctx.get("trace_id") or correlation_id
    ids: list[str] = []
    for candidate in (
        correlation_id,
        trace_val,
        ctx.get("chat_correlation_id"),
        ctx.get("trigger_correlation_id"),
        ctx.get("request_id"),
        ctx.get("session_id"),
    ):
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value and value not in ids:
            ids.append(value)
    return ids


def build_recall_query_v1(
    ctx: dict[str, Any],
    *,
    correlation_id: str,
    recall_profile: str | None = None,
    recall_cfg: dict[str, Any] | None = None,
    reply_to: str | None = None,
) -> RecallQueryV1 | None:
    """Build RecallQueryV1 using the same fields Exec ``run_recall_step`` sends."""
    recall_cfg = recall_cfg if isinstance(recall_cfg, dict) else {}
    fragment_text = last_user_message_from_ctx(ctx)
    if not fragment_text:
        return None
    lane_val = recall_cfg.get("lane")
    if lane_val is not None:
        lane_val = str(lane_val).strip() or None
    profile_explicit = bool(recall_cfg.get("profile_explicit"))
    return RecallQueryV1(
        fragment=fragment_text,
        verb=str(ctx.get("verb") or recall_cfg.get("verb") or "unknown"),
        intent=ctx.get("intent"),
        session_id=ctx.get("session_id"),
        node_id=ctx.get("node_id"),
        profile=(recall_profile or recall_cfg.get("profile") or "reflect.v1").strip() or "reflect.v1",
        lane=lane_val,
        profile_explicit=profile_explicit,
        exclude={
            "active_turn_ids": active_turn_ids_for_recall(ctx, correlation_id),
            "active_turn_text": fragment_text,
            "active_turn_ts": time.time(),
        },
        reply_to=reply_to,
    )


def recall_ctx_merge_from_reply(
    reply: RecallReplyV1,
    *,
    prompt_safe_ctx: bool = False,
) -> dict[str, Any]:
    """Normalize RecallReplyV1 into ctx keys expected by projection recall producer."""
    recall_fragments: list[dict[str, Any]] = []
    recall_citations: list[dict[str, Any]] = []
    for item in reply.bundle.items[:12]:
        snippet = str(getattr(item, "snippet", "") or "")[:480]
        recall_fragment = {
            "id": str(getattr(item, "id", "") or ""),
            "snippet": snippet,
            "score": float(getattr(item, "score", 0.0) or 0.0),
            "tags": [str(tag) for tag in (getattr(item, "tags", []) or []) if tag],
            "source": str(getattr(item, "source", "") or ""),
            "source_ref": getattr(item, "source_ref", None),
            "uri": getattr(item, "uri", None),
        }
        recall_fragments.append(recall_fragment)
        recall_citations.append(
            {
                "id": recall_fragment["id"],
                "source": recall_fragment["source"],
                "source_ref": recall_fragment["source_ref"],
                "uri": recall_fragment["uri"],
            }
        )
    memory_digest = reply.bundle.rendered if hasattr(reply.bundle, "rendered") else ""
    merge: dict[str, Any] = {
        "recall_bundle": {
            "fragments": recall_fragments,
            "citations": recall_citations,
            "rendered": memory_digest,
        },
        "memory_digest": memory_digest,
        "recall_fragments": recall_fragments,
        "memory_used": bool(recall_fragments),
    }
    if prompt_safe_ctx:
        merge["memory_bundle"] = {"rendered": memory_digest}
        merge["recall_memory_bundle_debug"] = reply.bundle.model_dump(mode="json")
        merge["recall_prompt_safe_ctx"] = True
    else:
        merge["memory_bundle"] = reply.bundle.model_dump(mode="json")
    return merge


def recall_cfg_from_recall_directive(directive: Any) -> dict[str, Any]:
    """Map CortexClientRequest.recall → recall_cfg dict for ``build_recall_query_v1``."""
    if directive is None:
        return {}
    return {
        "enabled": bool(getattr(directive, "enabled", True)),
        "profile": getattr(directive, "profile", None),
        "lane": getattr(directive, "lane", None),
        "profile_explicit": bool(getattr(directive, "profile_explicit", False)),
    }
