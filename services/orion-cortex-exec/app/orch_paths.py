"""Repo root and orch round-trip helpers for verb_adapters.

These lived in ``app/self_study.py`` until the Self Atlas retirement (2026-11,
docs/superpowers/specs/2026-10-30-self-atlas-replacement-proposal.md, Patch B)
deleted the daily self-knowledge scan. verb_adapters still needs them, so they
moved here unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

logger = logging.getLogger("orion.cortex.exec.orch_paths")


def _resolve_repo_root(module_path: str | Path | None = None) -> Path:
    env_root = os.getenv("ORION_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    resolved_path = Path(module_path or __file__).resolve()
    search_roots = (
        (resolved_path.parent, *resolved_path.parents)
        if resolved_path.suffix
        else (resolved_path, *resolved_path.parents)
    )
    for candidate in search_roots:
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "services").is_dir() and (
            (candidate / "orion").is_dir() or (candidate / "config").is_dir()
        ):
            return candidate

    fallback_root = resolved_path.parent.parent if resolved_path.suffix else resolved_path.parent
    logger.warning(
        "orch_paths_repo_root_fallback module_path=%s fallback=%s",
        resolved_path,
        fallback_root,
    )
    return fallback_root


REPO_ROOT = _resolve_repo_root()
# Public cortex intake channel. verb_adapters.py round-trips its own orch calls through it.
CORTEX_ORCH_REQUEST_CHANNEL = os.getenv("CORTEX_REQUEST_CHANNEL", "orion:cortex:request")


def _as_envelope_correlation_id(raw: str) -> str:
    try:
        return str(UUID(str(raw)))
    except Exception:
        return str(uuid5(NAMESPACE_URL, str(raw)))
