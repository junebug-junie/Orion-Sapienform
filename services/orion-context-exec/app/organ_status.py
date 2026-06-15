"""Per-run organ fetch status (enabled vs attempted vs ok)."""

from __future__ import annotations

from typing import Any

from orion.schemas.context_exec import ContextExecPermissionV1

from .settings import ContextExecSettings


def _empty_entry(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "attempted": False,
        "ok": False,
        "hit_count": 0,
        "error": None,
    }


def initial_organ_status(
    permissions: ContextExecPermissionV1,
    cfg: ContextExecSettings,
) -> dict[str, dict[str, Any]]:
    return {
        "recall": _empty_entry(
            enabled=bool(cfg.context_exec_real_recall_enabled and permissions.read_recall)
        ),
        "trace": _empty_entry(
            enabled=bool(cfg.context_exec_real_trace_enabled and permissions.read_redis_traces)
        ),
        "repo": _empty_entry(
            enabled=bool(cfg.context_exec_real_repo_enabled and permissions.read_repo)
        ),
    }


def record_recall(status: dict[str, dict[str, Any]], result: dict[str, Any]) -> None:
    entry = status.get("recall")
    if not entry or not entry.get("enabled"):
        return
    entry["attempted"] = True
    hits = result.get("hits")
    entry["hit_count"] = len(hits) if isinstance(hits, list) else 0
    error = result.get("error")
    entry["error"] = str(error) if error else None
    entry["ok"] = entry["error"] is None


def record_trace(
    status: dict[str, dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    error: str | None = None,
) -> None:
    entry = status.get("trace")
    if not entry or not entry.get("enabled"):
        return
    entry["attempted"] = True
    entry["hit_count"] = len(hits)
    entry["error"] = error
    entry["ok"] = error is None


def record_repo(
    status: dict[str, dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    error: str | None = None,
) -> None:
    entry = status.get("repo")
    if not entry or not entry.get("enabled"):
        return
    entry["attempted"] = True
    entry["hit_count"] = len(hits)
    entry["error"] = error
    entry["ok"] = error is None
