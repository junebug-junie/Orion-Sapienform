from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import settings
from .storage import write_json_atomic

logger = logging.getLogger("orion-context-exec.workspace")

_WORKSPACE_SCHEMA = "orion.context_exec.workspace.v1"

_SKIP_COPY_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".orion-smoke-logs",
    }
)


@dataclass(frozen=True)
class ContextExecWorkspace:
    run_id: str
    root: Path
    scratch_dir: Path
    outputs_dir: Path
    patches_dir: Path
    repo_dir: Path
    manifest_path: Path


def workspace_dir(run_id: str) -> Path:
    if not run_id or run_id != Path(run_id).name:
        raise ValueError(f"invalid workspace run_id: {run_id!r}")
    root = Path(settings.context_exec_workspace_root).resolve()
    target = (root / run_id).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"workspace path escapes root: {run_id!r}")
    return target


def _load_existing_workspace(run_id: str, root: Path) -> ContextExecWorkspace | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    scratch = root / "scratch"
    outputs = root / "outputs"
    patches = root / "patches"
    repo = root / "repo"
    if not all(p.is_dir() for p in (scratch, outputs, patches, repo)):
        return None
    return ContextExecWorkspace(
        run_id=run_id,
        root=root,
        scratch_dir=scratch,
        outputs_dir=outputs,
        patches_dir=patches,
        repo_dir=repo,
        manifest_path=manifest_path,
    )


def _build_manifest(
    *,
    run_id: str,
    root: Path,
    scratch_dir: Path,
    outputs_dir: Path,
    patches_dir: Path,
    repo_dir: Path,
    canonical_repo_root: str,
    repo_materialized: bool,
    repo_materialize_mode: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": _WORKSPACE_SCHEMA,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(root),
        "scratch_dir": str(scratch_dir),
        "outputs_dir": str(outputs_dir),
        "patches_dir": str(patches_dir),
        "repo_dir": str(repo_dir),
        "canonical_repo_root": canonical_repo_root,
        "repo_materialized": repo_materialized,
        "repo_materialize_mode": repo_materialize_mode,
        "canonical_repo_write_allowed": False,
        "workspace_write_allowed": True,
        "notes": [
            "Canonical repo remains read-only.",
            "Patch-producing agents must write diffs under patches/.",
        ],
    }
    if warnings:
        manifest["warnings"] = warnings
    return manifest


def _materialize_repo(
    *,
    canonical_root: Path,
    target_repo: Path,
    max_bytes: int,
) -> tuple[bool, str, list[str]]:
    """Conservative copy from canonical repo into workspace/repo. Fail-open."""
    warnings: list[str] = []
    if not canonical_root.is_dir():
        warnings.append(f"canonical repo not present: {canonical_root}")
        return False, "none", warnings

    total_bytes = 0
    try:
        for src in canonical_root.rglob("*"):
            if not src.is_file():
                continue
            rel_parts = src.relative_to(canonical_root).parts
            if any(part in _SKIP_COPY_DIRS for part in rel_parts):
                continue
            try:
                size = src.stat().st_size
            except OSError as exc:
                warnings.append(f"stat failed for {src}: {exc}")
                continue
            total_bytes += size
            if total_bytes > max_bytes:
                warnings.append(
                    f"repo copy exceeded max bytes ({total_bytes} > {max_bytes}); copy aborted"
                )
                return False, "aborted_over_limit", warnings

        for src in canonical_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(canonical_root)
            if any(part in _SKIP_COPY_DIRS for part in rel.parts):
                continue
            dest = target_repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

        return True, "copy", warnings
    except Exception as exc:
        warnings.append(f"repo materialization failed: {exc}")
        logger.warning("workspace repo materialization failed error=%s", exc)
        return False, "failed", warnings


def allocate_workspace(
    run_id: str,
    *,
    request: Any | None = None,
    repo_root: str | None = None,
) -> ContextExecWorkspace:
    """Allocate a per-run workspace under workspaces/{run_id}/.

    Idempotent for the same run_id when a valid workspace already exists.
    """
    del request  # reserved for future request-scoped workspace options

    root = workspace_dir(run_id)
    existing = _load_existing_workspace(run_id, root)
    if existing is not None:
        return existing

    scratch_dir = root / "scratch"
    outputs_dir = root / "outputs"
    patches_dir = root / "patches"
    repo_dir = root / "repo"
    manifest_path = root / "manifest.json"

    for path in (scratch_dir, outputs_dir, patches_dir, repo_dir):
        path.mkdir(parents=True, exist_ok=True)

    canonical_repo_root = str(repo_root or settings.context_exec_repo_root)
    repo_materialized = False
    repo_materialize_mode = "none"
    manifest_warnings: list[str] = []

    if settings.context_exec_workspace_materialize_repo:
        materialized, mode, copy_warnings = _materialize_repo(
            canonical_root=Path(canonical_repo_root),
            target_repo=repo_dir,
            max_bytes=settings.context_exec_workspace_copy_max_bytes,
        )
        manifest_warnings.extend(copy_warnings)
        repo_materialize_mode = mode
        if materialized:
            repo_materialized = True
        else:
            # Remove partial copy on failure.
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
            repo_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(
        run_id=run_id,
        root=root,
        scratch_dir=scratch_dir,
        outputs_dir=outputs_dir,
        patches_dir=patches_dir,
        repo_dir=repo_dir,
        canonical_repo_root=canonical_repo_root,
        repo_materialized=repo_materialized,
        repo_materialize_mode=repo_materialize_mode,
        warnings=manifest_warnings or None,
    )
    write_json_atomic(manifest_path, manifest)

    return ContextExecWorkspace(
        run_id=run_id,
        root=root,
        scratch_dir=scratch_dir,
        outputs_dir=outputs_dir,
        patches_dir=patches_dir,
        repo_dir=repo_dir,
        manifest_path=manifest_path,
    )


def workspace_health_block() -> dict[str, Any]:
    root = Path(settings.context_exec_workspace_root)
    exists = root.exists()
    is_dir = root.is_dir() if exists else False
    writable = bool(exists and is_dir and os.access(root, os.W_OK))
    return {
        "enabled": settings.context_exec_workspace_enabled,
        "root": str(root),
        "present": exists,
        "writable": writable,
        "materialize_repo": settings.context_exec_workspace_materialize_repo,
    }
