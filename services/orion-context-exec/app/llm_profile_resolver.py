"""Resolve and validate context-exec llm_profile → gateway route binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orion.schemas.context_exec import ALLOWED_CONTEXT_EXEC_LLM_PROFILES

from .settings import ContextExecSettings


@dataclass(frozen=True)
class LLMProfileSelection:
    requested: str | None
    selected: str
    route_used: str
    fallback_used: bool = False
    fallback_reason: str | None = None


class LLMProfileValidationError(ValueError):
    """Invalid llm_profile id (not in allowed route set)."""


def _settings(cfg: ContextExecSettings | None = None) -> ContextExecSettings:
    if cfg is not None:
        return cfg
    from .settings import settings as live

    return live


def normalize_llm_profile(raw: str | None) -> str | None:
    if raw is None:
        return None
    norm = str(raw).strip().lower()
    if not norm:
        return None
    if norm not in ALLOWED_CONTEXT_EXEC_LLM_PROFILES:
        raise LLMProfileValidationError(
            f"llm_profile must be one of {sorted(ALLOWED_CONTEXT_EXEC_LLM_PROFILES)}; got {raw!r}"
        )
    return norm


def resolve_llm_profile_default(
    requested: str | None,
    cfg: ContextExecSettings | None = None,
) -> LLMProfileSelection:
    """Pick effective profile without gateway health probe (sync)."""
    cfg = _settings(cfg)
    norm_requested = normalize_llm_profile(requested)
    default = normalize_llm_profile(cfg.context_exec_default_llm_profile) or "chat"
    selected = norm_requested if norm_requested is not None else default
    return LLMProfileSelection(
        requested=norm_requested,
        selected=selected,
        route_used=selected,
    )


async def resolve_llm_profile(
    requested: str | None,
    cfg: ContextExecSettings | None = None,
) -> LLMProfileSelection:
    """Resolve llm_profile to its gateway route. No pre-call health read.

    This used to read the gateway's ``GET /routes`` and fail closed (or fall back to the default
    profile) when the route looked down. GPU pool stage 6.3 removed that read: ``/routes`` is a
    compatibility view being retired, and under the pool a route is never "down" ahead of a call
    -- the pool queues the call or refuses it, and the gateway's refusal is the honest answer the
    run records. Its old fallback also treated "gateway unreachable" as "every route available",
    so it never guarded anything a dispatch-time refusal does not.
    """
    return resolve_llm_profile_default(requested, cfg)


def selection_runtime_debug(selection: LLMProfileSelection) -> dict[str, Any]:
    return {
        "llm_profile_requested": selection.requested,
        "llm_profile_selected": selection.selected,
        "route_used": selection.route_used,
        "fallback_used": selection.fallback_used,
        "fallback_reason": selection.fallback_reason,
    }
