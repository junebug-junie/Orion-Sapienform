"""Shared recall source policy — vector gating with path-specific diagnostics."""

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
    """Return whether vector fetch is allowed for this profile/path and why."""
    s = settings_obj or settings
    profile_name = str(profile.get("profile") or "")
    vector_top_k = _profile_vector_top_k(profile)
    enable_vector = profile.get("enable_vector")
    if enable_vector is False:
        return False, {
            "allowed": False,
            "reason": "disabled_profile_enable_vector_false",
            "profile": profile_name,
            "vector_top_k": vector_top_k,
            "RECALL_ENABLE_VECTOR": bool(getattr(s, "RECALL_ENABLE_VECTOR", True)),
            "path": path,
        }
    if not bool(getattr(s, "RECALL_ENABLE_VECTOR", True)):
        return False, {
            "allowed": False,
            "reason": "disabled_global",
            "profile": profile_name,
            "vector_top_k": vector_top_k,
            "RECALL_ENABLE_VECTOR": False,
            "path": path,
        }
    if vector_top_k <= 0:
        return False, {
            "allowed": False,
            "reason": "disabled_profile_vector_top_k_zero",
            "profile": profile_name,
            "vector_top_k": vector_top_k,
            "RECALL_ENABLE_VECTOR": bool(getattr(s, "RECALL_ENABLE_VECTOR", True)),
            "path": path,
        }
    return True, {
        "allowed": True,
        "reason": "enabled",
        "profile": profile_name,
        "vector_top_k": vector_top_k,
        "RECALL_ENABLE_VECTOR": bool(getattr(s, "RECALL_ENABLE_VECTOR", True)),
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


def log_vector_skipped(path: str, detail: dict[str, Any], logger: Any) -> None:
    logger.info(
        "recall vector skipped path=%s reason=%s profile=%s vector_top_k=%s global_vector_enabled=%s",
        path,
        detail.get("reason"),
        detail.get("profile"),
        detail.get("vector_top_k"),
        detail.get("RECALL_ENABLE_VECTOR"),
    )
