"""Shared recall source policy — vector removal diagnostics (vector no longer fetched)."""

from __future__ import annotations

from typing import Any

from .settings import settings


def _profile_vector_top_k(profile: dict[str, Any]) -> int:
    try:
        return int(profile.get("vector_top_k", settings.RECALL_DEFAULT_MAX_ITEMS))
    except Exception:
        return 0


def recall_vector_allowed(
    profile: dict[str, Any],
    settings_obj: Any | None = None,
    *,
    path: str,
) -> tuple[bool, dict[str, Any]]:
    """Vector retrieval was removed from orion-recall; always disabled (diagnostics only)."""
    s = settings_obj or settings
    profile_name = str(profile.get("profile") or "")
    vector_top_k = _profile_vector_top_k(profile)
    return False, {
        "allowed": False,
        "reason": "removed_from_orion_recall",
        "profile": profile_name,
        "vector_top_k": vector_top_k,
        "RECALL_ENABLE_VECTOR": bool(getattr(s, "RECALL_ENABLE_VECTOR", False)),
        "path": path,
    }


def build_vector_policy(
    profile: dict[str, Any],
    settings_obj: Any | None = None,
    *,
    paths: tuple[str, ...] = (
        "main",
        "anchor",
        "graphtri",
        "collectors",
        "v2_shadow_exact",
        "v2_shadow_semantic",
    ),
) -> dict[str, dict[str, Any]]:
    """Build path-keyed vector policy diagnostics for recall_debug."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        allowed, detail = recall_vector_allowed(profile, settings_obj, path=path)
        out[path] = {
            "allowed": allowed,
            "reason": detail["reason"],
            "profile": detail["profile"],
            "vector_top_k": detail["vector_top_k"],
            "RECALL_ENABLE_VECTOR": detail["RECALL_ENABLE_VECTOR"],
        }
    return out

