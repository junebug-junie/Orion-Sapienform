"""Runtime installer for the shared chat stance cognitive spine.

This is the safe Phase-5 cut: make ``app.chat_stance`` use the shared
``orion.cognition.projection_builder`` unified-beliefs path without rewriting the
large service module through a full-file GitHub update.

The local duplicated registry in ``chat_stance.py`` remains in the file for now,
but it is no longer the runtime path when this installer is active.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from orion.cognition.projection import project_unified_beliefs_for_mind
from orion.cognition.projection_builder import unified_beliefs_for_chat_stance
from orion.substrate.relational import UnifiedRelationalBeliefSetV1

logger = logging.getLogger("orion.cortex.exec.chat_stance_shared_spine")

_INSTALLED = False
_ORIGINAL = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _record_shared_spine_marker(
    ctx: dict[str, Any],
    *,
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    error: str | None = None,
) -> None:
    marker = {
        "enabled": True,
        "adapter": "services/orion-cortex-exec/app/chat_stance_shared_spine.py",
        "builder": "orion.cognition.projection_builder.unified_beliefs_for_chat_stance",
        "beliefs_present": beliefs is not None,
        "cold_anchors": list(getattr(beliefs, "cold_anchors", []) or []) if beliefs is not None else [],
        "degraded_producers": list(getattr(beliefs, "degraded_producers", []) or []) if beliefs is not None else [],
        "lineage": list(getattr(beliefs, "lineage", []) or []) if beliefs is not None else [],
        "error": error,
    }
    ctx["chat_stance_shared_projection_spine"] = marker
    metadata = ctx.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["chat_stance_shared_projection_spine"] = marker


def _record_projection_snapshot(ctx: dict[str, Any], beliefs: UnifiedRelationalBeliefSetV1 | None) -> None:
    if beliefs is None:
        ctx["chat_cognitive_projection"] = None
        ctx["chat_cognitive_projection_debug"] = {"present": False, "reason": "beliefs_absent"}
        return
    try:
        projection = project_unified_beliefs_for_mind(beliefs)
    except Exception as exc:
        logger.warning("chat_stance_projection_snapshot_failed error=%s", exc)
        ctx["chat_cognitive_projection"] = None
        ctx["chat_cognitive_projection_debug"] = {
            "present": False,
            "reason": "projection_failed",
            "error": str(exc),
        }
        return
    if projection is None:
        ctx["chat_cognitive_projection"] = None
        ctx["chat_cognitive_projection_debug"] = {"present": False, "reason": "projection_none"}
        return
    payload = projection.model_dump(mode="json")
    ctx["chat_cognitive_projection"] = payload
    ctx["chat_cognitive_projection_debug"] = {
        "present": True,
        "projection_id": payload.get("projection_id"),
        "item_count": payload.get("item_count"),
        "anchor_count": len(payload.get("anchors") or {}),
        "cold_anchors": list(payload.get("cold_anchors") or []),
        "degraded_producers": list(payload.get("degraded_producers") or []),
        "notes": list(payload.get("notes") or []),
    }


def shared_unified_beliefs_for_stance(ctx: dict[str, Any]) -> UnifiedRelationalBeliefSetV1 | None:
    """Exec chat-stance adapter into the shared cognitive projection builder.

    Besides returning beliefs to legacy chat stance, this records a compact marker
    and projection snapshot in ``ctx`` so Inspect/debug surfaces can prove which
    cognitive spine served the turn.
    """
    try:
        beliefs = unified_beliefs_for_chat_stance(
            ctx,
            timeout_sec=_env_float("UNIFIED_BELIEFS_TIMEOUT_SEC", 5.0),
        )
    except Exception as exc:
        _record_shared_spine_marker(ctx, beliefs=None, error=str(exc))
        logger.warning("chat_stance_shared_projection_spine_failed error=%s", exc)
        raise
    _record_shared_spine_marker(ctx, beliefs=beliefs)
    _record_projection_snapshot(ctx, beliefs)
    logger.info(
        "chat_stance_shared_projection_spine_used beliefs_present=%s projection_present=%s item_count=%s cold_anchors=%s degraded=%s",
        beliefs is not None,
        bool((ctx.get("chat_cognitive_projection_debug") or {}).get("present")) if isinstance(ctx.get("chat_cognitive_projection_debug"), dict) else False,
        (ctx.get("chat_cognitive_projection_debug") or {}).get("item_count") if isinstance(ctx.get("chat_cognitive_projection_debug"), dict) else None,
        list(getattr(beliefs, "cold_anchors", []) or []) if beliefs is not None else [],
        list(getattr(beliefs, "degraded_producers", []) or []) if beliefs is not None else [],
    )
    return beliefs


def install_chat_stance_shared_spine(*, force: bool = False) -> bool:
    """Patch ``app.chat_stance`` to use the shared unified-beliefs builder.

    Returns True when the shared spine is installed or already installed.
    Returns False only when explicitly disabled by env.
    """
    global _INSTALLED, _ORIGINAL

    disabled = (os.getenv("CHAT_STANCE_SHARED_PROJECTION_SPINE_DISABLED") or "").strip().lower()
    if disabled in {"1", "true", "yes", "on"} and not force:
        logger.info("chat_stance_shared_projection_spine_disabled")
        return False
    if _INSTALLED and not force:
        return True

    from . import chat_stance

    current = getattr(chat_stance, "_unified_beliefs_for_stance", None)
    if current is not shared_unified_beliefs_for_stance:
        _ORIGINAL = current
        setattr(chat_stance, "_unified_beliefs_for_stance", shared_unified_beliefs_for_stance)
    setattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE", True)
    _INSTALLED = True
    logger.info("chat_stance_shared_projection_spine_installed")
    return True


def restore_chat_stance_shared_spine_for_tests() -> None:
    """Restore the original local path in tests only."""
    global _INSTALLED, _ORIGINAL
    if _ORIGINAL is None:
        _INSTALLED = False
        return
    from . import chat_stance

    setattr(chat_stance, "_unified_beliefs_for_stance", _ORIGINAL)
    setattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE", False)
    _INSTALLED = False
