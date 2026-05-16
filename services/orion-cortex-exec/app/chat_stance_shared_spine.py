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
from typing import Any, Callable

from orion.cognition.projection import project_unified_beliefs_for_mind
from orion.cognition.projection_builder import unified_beliefs_for_chat_stance
from orion.substrate.relational import UnifiedRelationalBeliefSetV1

logger = logging.getLogger("orion.cortex.exec.chat_stance_shared_spine")

_INSTALLED = False
_ORIGINAL = None
_ORIGINAL_DEBUG_BUILDER: Callable[..., dict[str, Any]] | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _projection_item_count(projection: dict[str, Any] | None) -> int:
    if not isinstance(projection, dict):
        return 0
    try:
        return int(projection.get("item_count") or 0)
    except Exception:
        return 0


def _inline_projection_from_metadata(ctx: dict[str, Any]) -> dict[str, Any] | None:
    metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
    for key in ("cognitive_projection_facet", "cognitive_projection"):
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if isinstance(value, dict):
            return value
    return None


def _record_shared_spine_marker(
    ctx: dict[str, Any],
    *,
    beliefs: UnifiedRelationalBeliefSetV1 | None,
    error: str | None = None,
    inline_projection: dict[str, Any] | None = None,
) -> None:
    marker = {
        "enabled": True,
        "adapter": "services/orion-cortex-exec/app/chat_stance_shared_spine.py",
        "builder": "orion.cognition.projection_builder.unified_beliefs_for_chat_stance",
        "beliefs_present": beliefs is not None or isinstance(inline_projection, dict),
        "cold_anchors": list(getattr(beliefs, "cold_anchors", []) or []) if beliefs is not None else list((inline_projection or {}).get("cold_anchors") or []),
        "degraded_producers": list(getattr(beliefs, "degraded_producers", []) or []) if beliefs is not None else list((inline_projection or {}).get("degraded_producers") or []),
        "lineage": list(getattr(beliefs, "lineage", []) or []) if beliefs is not None else list((inline_projection or {}).get("lineage") or []),
        "error": error,
    }
    if isinstance(inline_projection, dict):
        marker["projection_source"] = "orion_cortex_orch_mind_runtime"
        marker["projection_reused_from_metadata"] = True
    ctx["chat_stance_shared_projection_spine"] = marker
    metadata = ctx.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["chat_stance_shared_projection_spine"] = marker


def _record_projection_payload(ctx: dict[str, Any], payload: dict[str, Any], *, source_label: str) -> None:
    ctx["chat_cognitive_projection"] = payload
    ctx["chat_cognitive_projection_debug"] = {
        "present": True,
        "projection_id": payload.get("projection_id"),
        "item_count": payload.get("item_count"),
        "anchor_count": len(payload.get("anchors") or {}),
        "cold_anchors": list(payload.get("cold_anchors") or []),
        "degraded_producers": list(payload.get("degraded_producers") or []),
        "notes": list(payload.get("notes") or []),
        "source": source_label,
    }


def _record_projection_snapshot(ctx: dict[str, Any], beliefs: UnifiedRelationalBeliefSetV1 | None) -> None:
    inline_projection = _inline_projection_from_metadata(ctx)
    if isinstance(inline_projection, dict) and _projection_item_count(inline_projection) > 0:
        _record_projection_payload(ctx, inline_projection, source_label="orion_cortex_orch_mind_runtime")
        return
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
    _record_projection_payload(ctx, projection.model_dump(mode="json"), source_label="exec_shared_spine")


def _projection_debug_bundle(ctx: dict[str, Any]) -> dict[str, Any]:
    marker = ctx.get("chat_stance_shared_projection_spine")
    if not isinstance(marker, dict):
        metadata = ctx.get("metadata") if isinstance(ctx.get("metadata"), dict) else {}
        marker = metadata.get("chat_stance_shared_projection_spine") if isinstance(metadata, dict) else None
    projection_debug = ctx.get("chat_cognitive_projection_debug")
    projection = ctx.get("chat_cognitive_projection")
    return {
        "shared_spine": marker if isinstance(marker, dict) else {"enabled": False, "reason": "marker_absent"},
        "projection_debug": projection_debug if isinstance(projection_debug, dict) else {"present": False, "reason": "debug_absent"},
        "projection": projection if isinstance(projection, dict) else None,
    }


def _inject_projection_debug(debug_payload: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    bundle = _projection_debug_bundle(ctx)
    debug_payload["cognitive_projection"] = bundle

    lineage = debug_payload.get("lineage_summary")
    if isinstance(lineage, list):
        projection_debug = bundle.get("projection_debug") if isinstance(bundle.get("projection_debug"), dict) else {}
        shared_spine = bundle.get("shared_spine") if isinstance(bundle.get("shared_spine"), dict) else {}
        lineage.append(
            "shared projection spine used: "
            + ("yes" if bool(shared_spine.get("enabled")) and bool(shared_spine.get("beliefs_present")) else "no")
        )
        lineage.append(
            f"cognitive projection items: {projection_debug.get('item_count') if projection_debug.get('present') else 0}"
        )

    raw = debug_payload.setdefault("raw", {})
    if isinstance(raw, dict):
        raw["cognitive_projection"] = bundle
    return debug_payload


def shared_build_chat_stance_debug_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Decorate legacy ChatStanceDebug with shared-spine/projection details."""
    if _ORIGINAL_DEBUG_BUILDER is None:
        raise RuntimeError("chat_stance_debug_builder_not_installed")
    debug_payload = _ORIGINAL_DEBUG_BUILDER(*args, **kwargs)
    ctx = kwargs.get("ctx")
    if not isinstance(ctx, dict) and args:
        maybe_ctx = args[0]
        ctx = maybe_ctx if isinstance(maybe_ctx, dict) else None
    if isinstance(debug_payload, dict) and isinstance(ctx, dict):
        return _inject_projection_debug(debug_payload, ctx)
    return debug_payload


def shared_unified_beliefs_for_stance(ctx: dict[str, Any]) -> UnifiedRelationalBeliefSetV1 | None:
    """Exec chat-stance adapter into the shared cognitive projection builder.

    Besides returning beliefs to legacy chat stance, this records a compact marker
    and projection snapshot in ``ctx`` so Inspect/debug surfaces can prove which
    cognitive spine served the turn.
    """
    inline_projection = _inline_projection_from_metadata(ctx)
    if isinstance(inline_projection, dict) and _projection_item_count(inline_projection) > 0:
        _record_shared_spine_marker(ctx, beliefs=None, inline_projection=inline_projection)
        _record_projection_snapshot(ctx, None)
        logger.info(
            "chat_stance_shared_projection_spine_reused projection_id=%s item_count=%s",
            inline_projection.get("projection_id"),
            inline_projection.get("item_count"),
        )
        return None
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
    """Patch ``app.chat_stance`` to use the shared unified-beliefs/debug builders.

    Returns True when the shared spine is installed or already installed.
    Returns False only when explicitly disabled by env.
    """
    global _INSTALLED, _ORIGINAL, _ORIGINAL_DEBUG_BUILDER

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

    current_debug = getattr(chat_stance, "build_chat_stance_debug_payload", None)
    if current_debug is not shared_build_chat_stance_debug_payload:
        _ORIGINAL_DEBUG_BUILDER = current_debug
        setattr(chat_stance, "build_chat_stance_debug_payload", shared_build_chat_stance_debug_payload)

    setattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE", True)
    _INSTALLED = True
    logger.info("chat_stance_shared_projection_spine_installed")
    return True


def restore_chat_stance_shared_spine_for_tests() -> None:
    """Restore the original local path in tests only."""
    global _INSTALLED, _ORIGINAL, _ORIGINAL_DEBUG_BUILDER
    if _ORIGINAL is None and _ORIGINAL_DEBUG_BUILDER is None:
        _INSTALLED = False
        return
    from . import chat_stance

    if _ORIGINAL is not None:
        setattr(chat_stance, "_unified_beliefs_for_stance", _ORIGINAL)
    if _ORIGINAL_DEBUG_BUILDER is not None:
        setattr(chat_stance, "build_chat_stance_debug_payload", _ORIGINAL_DEBUG_BUILDER)
    setattr(chat_stance, "_CHAT_STANCE_SHARED_PROJECTION_SPINE", False)
    _INSTALLED = False
