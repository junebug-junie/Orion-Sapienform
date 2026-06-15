"""Resolve and validate context-exec llm_profile → gateway route binding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from orion.schemas.context_exec import ALLOWED_CONTEXT_EXEC_LLM_PROFILES

from .settings import ContextExecSettings

logger = logging.getLogger("orion-context-exec.llm_profile_resolver")

_ROUTE_UP_STATUSES = frozenset({"up", "unknown"})


@dataclass(frozen=True)
class LLMProfileSelection:
    requested: str | None
    selected: str
    route_used: str
    fallback_used: bool = False
    fallback_reason: str | None = None


class LLMProfileValidationError(ValueError):
    """Invalid llm_profile id (not in allowed route set)."""


class LLMProfileUnavailableError(ValueError):
    """Selected route is down/not configured and fallback is disabled."""


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


async def fetch_route_status_map(
    gateway_url: str,
    *,
    timeout_sec: float = 1.5,
) -> dict[str, str]:
    base = gateway_url.strip().rstrip("/")
    if not base:
        return {}
    url = f"{base}/routes"
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("llm gateway /routes unreachable url=%s err=%s", url, exc)
        return {}
    if not isinstance(payload, dict):
        return {}
    routes_raw = payload.get("routes") or []
    status_map: dict[str, str] = {}
    if isinstance(routes_raw, list):
        for item in routes_raw:
            if not isinstance(item, dict):
                continue
            route_id = str(item.get("id") or "").strip().lower()
            if route_id in ALLOWED_CONTEXT_EXEC_LLM_PROFILES:
                status_map[route_id] = str(item.get("status") or "unknown")
    return status_map


def _route_is_available(status: str | None) -> bool:
    if status is None:
        return True
    return str(status).strip().lower() in _ROUTE_UP_STATUSES


async def resolve_llm_profile(
    requested: str | None,
    cfg: ContextExecSettings | None = None,
) -> LLMProfileSelection:
    """Resolve llm_profile with optional gateway route health check."""
    cfg = _settings(cfg)
    base = resolve_llm_profile_default(requested, cfg)
    gateway_url = str(cfg.context_exec_llm_gateway_url or "").strip()
    if not gateway_url:
        return base

    status_map = await fetch_route_status_map(
        gateway_url,
        timeout_sec=float(cfg.context_exec_llm_gateway_timeout_sec),
    )
    if not status_map:
        return base

    route_status = status_map.get(base.selected)
    if base.selected not in status_map:
        route_status = "not_configured"
    if _route_is_available(route_status):
        return base

    if not cfg.context_exec_llm_profile_fallback_enabled:
        raise LLMProfileUnavailableError(
            f"llm_profile route {base.selected!r} unavailable (status={route_status!r})"
        )

    fallback = normalize_llm_profile(cfg.context_exec_default_llm_profile) or "chat"
    fallback_status = status_map.get(fallback)
    if fallback not in status_map:
        fallback_status = "not_configured"
    if not _route_is_available(fallback_status):
        raise LLMProfileUnavailableError(
            f"llm_profile route {base.selected!r} unavailable and default "
            f"{fallback!r} also unavailable (status={fallback_status!r})"
        )
    return LLMProfileSelection(
        requested=base.requested,
        selected=fallback,
        route_used=fallback,
        fallback_used=True,
        fallback_reason=f"route_unavailable:{base.selected}:{route_status}",
    )


def selection_runtime_debug(selection: LLMProfileSelection) -> dict[str, Any]:
    return {
        "llm_profile_requested": selection.requested,
        "llm_profile_selected": selection.selected,
        "route_used": selection.route_used,
        "fallback_used": selection.fallback_used,
        "fallback_reason": selection.fallback_reason,
    }
