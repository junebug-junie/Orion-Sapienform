from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

VALID_LLM_LANES = frozenset({"chat", "spark", "background", "agent"})

# Prefer explicit lane-named keys; metacog remains a backward-compatible background-class worker.
_BACKGROUND_ROUTE_KEYS = ("background", "metacog")
_SPARK_ROUTE_KEYS = ("spark",)
_AGENT_ROUTE_KEYS = ("agent",)


@dataclass(frozen=True)
class LlmLaneRouteDecision:
    """Which route NAME a lane resolves to. Which GPU serves it is orion-gpu-pool's decision
    (app/pool_placement.py); there is no gateway-side fallback onto another route any more --
    the pool spills a class across roles itself."""

    requested_llm_lane: str
    resolved_llm_lane: str
    route_table_key: Optional[str]
    route_status: str
    reason: str
    fallback_used: bool = False
    degraded: bool = False


def _norm_lane(raw: Any, default: str) -> str:
    s = str(raw or "").strip().lower()
    return s if s in VALID_LLM_LANES else str(default or "chat").strip().lower()


def _first_route_key(keys: Set[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in keys:
            return c
    return None


def resolve_llm_lane_route(
    options: Optional[Dict[str, Any]],
    body_route: Optional[str],
    *,
    llm_lane_default: str,
    llm_route_default: str,
    route_table_keys: Set[str],
) -> LlmLaneRouteDecision:
    """
    Side-effect free: picks a route name (a `config/gpu_pool.yaml` `routes:` key).

    Chat lane preserves the caller's route key (e.g. quick vs chat) when it is a known route;
    otherwise falls back to llm_route_default / quick.
    """
    opts = options if isinstance(options, dict) else {}
    default_lane = _norm_lane(llm_lane_default, "chat")
    raw_lane = opts.get("llm_lane") or opts.get("execution_lane")
    logical_lane = _norm_lane(raw_lane, default_lane)
    invalid_lane = bool(str(raw_lane or "").strip()) and str(raw_lane).strip().lower() not in VALID_LLM_LANES

    def _ok(
        route_key: str,
        resolved_lane: str,
        reason: str,
        *,
        fallback_used: bool = False,
        status: str = "ok",
        degraded: bool = False,
    ) -> LlmLaneRouteDecision:
        return LlmLaneRouteDecision(
            requested_llm_lane=logical_lane,
            resolved_llm_lane=resolved_lane,
            route_table_key=route_key,
            route_status=status,
            reason=reason,
            fallback_used=fallback_used,
            degraded=degraded,
        )

    def _missing(reason: str) -> LlmLaneRouteDecision:
        return LlmLaneRouteDecision(
            requested_llm_lane=logical_lane,
            resolved_llm_lane=logical_lane,
            route_table_key=None,
            route_status="missing_route",
            reason=reason,
            fallback_used=False,
            degraded=True,
        )

    if logical_lane == "chat":
        br = str(body_route or "").strip() or str(llm_route_default or "chat")
        if br not in route_table_keys:
            # quick first: a misspelled route must not land on Juniper's reserved chat lane
            # (scripts/check_chat_route_poachers.py); chat only if quick is not a known route.
            br = _first_route_key(route_table_keys, ("quick", "chat")) or str(llm_route_default or "quick")
        if br not in route_table_keys:
            return _missing("chat_lane_no_matching_route_table_key")
        st = "invalid_lane" if invalid_lane else "ok"
        return _ok(br, "chat", "verb_chat_lane", status=st, degraded=invalid_lane)

    if logical_lane == "spark":
        rk = _first_route_key(route_table_keys, _SPARK_ROUTE_KEYS)
        if rk:
            return _ok(rk, "spark", "spark_route")
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        if rk:
            return _ok(rk, "background", "spark_missing_used_background_lane", fallback_used=True)
        return _missing("spark_route_missing_background_missing")

    if logical_lane == "background":
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        if rk:
            return _ok(rk, "background", "background_route")
        return _missing("background_route_missing")

    if logical_lane == "agent":
        rk = _first_route_key(route_table_keys, _AGENT_ROUTE_KEYS)
        if rk:
            return _ok(rk, "agent", "agent_route")
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        if rk:
            return _ok(rk, "background", "agent_missing_used_background_lane", fallback_used=True)
        return _missing("agent_route_missing_background_missing")

    return _missing("unreachable_lane")
