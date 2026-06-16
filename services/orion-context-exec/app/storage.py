from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .settings import settings

logger = logging.getLogger("orion-context-exec.storage")


def configured_storage_paths() -> dict[str, Path]:
    return {
        "root": Path(settings.context_exec_storage_root),
        "runs": Path(settings.context_exec_run_root),
        "artifacts": Path(settings.context_exec_artifact_root),
        "ledger": Path(settings.context_exec_ledger_root),
        "workspaces": Path(settings.context_exec_workspace_root),
        "cache": Path(settings.context_exec_cache_root),
        "tmp": Path(settings.context_exec_tmp_root),
    }


def ensure_storage_dirs() -> None:
    for name, path in configured_storage_paths().items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("storage dir unavailable name=%s path=%s error=%s", name, path, exc)


def _path_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    writable = bool(exists and is_dir and os.access(path, os.W_OK))
    return {
        "path": str(path),
        "present": exists,
        "is_dir": is_dir,
        "writable": writable,
    }


def storage_health_block() -> dict[str, Any]:
    paths = configured_storage_paths()
    dirs = {name: _path_status(path) for name, path in paths.items()}

    configured = bool(str(settings.context_exec_storage_root).strip())
    ok = configured and all(
        block["present"] and block["is_dir"] and block["writable"]
        for block in dirs.values()
    )

    error = None
    if not configured:
        error = "CONTEXT_EXEC_STORAGE_ROOT is not configured"
    elif not ok:
        bad = [
            name
            for name, block in dirs.items()
            if not (block["present"] and block["is_dir"] and block["writable"])
        ]
        error = f"storage dirs unavailable or not writable: {', '.join(bad)}"

    return {
        "configured": configured,
        "ok": ok,
        "root": str(paths["root"]),
        "run_ledger_enabled": settings.context_exec_run_ledger_enabled,
        "dirs": dirs,
        "error": error,
    }
