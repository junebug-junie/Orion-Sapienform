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


def shared_unified_beliefs_for_stance(ctx: dict[str, Any]) -> UnifiedRelationalBeliefSetV1 | None:
    """Exec chat-stance adapter into the shared cognitive projection builder."""
    return unified_beliefs_for_chat_stance(
        ctx,
        timeout_sec=_env_float("UNIFIED_BELIEFS_TIMEOUT_SEC", 5.0),
    )


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
