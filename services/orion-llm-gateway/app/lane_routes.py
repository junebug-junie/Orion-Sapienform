from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set

VALID_LLM_LANES = frozenset({"chat", "spark", "background", "agent"})

# Prefer explicit lane-named keys; metacog remains a backward-compatible background-class worker.
_BACKGROUND_ROUTE_KEYS = ("background", "metacog")
_SPARK_ROUTE_KEYS = ("spark",)
_AGENT_ROUTE_KEYS = ("agent",)


@dataclass(frozen=True)
class LlmLaneRouteDecision:
    requested_llm_lane: str
    resolved_llm_lane: str
    route_table_key: Optional[str]
    served_by: Optional[str]
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


def _match_served_by(route_served_by: Mapping[str, Optional[str]], label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    want = str(label).strip()
    if not want:
        return None
    for rk, sb in route_served_by.items():
        if sb and str(sb).strip() == want:
            return rk
    return None


def resolve_llm_lane_route(
    options: Optional[Dict[str, Any]],
    body_route: Optional[str],
    *,
    llm_lane_default: str,
    llm_route_default: str,
    llm_allow_background_to_chat_fallback: bool,
    llm_route_spark_served_by: Optional[str],
    llm_route_background_served_by: Optional[str],
    llm_route_agent_served_by: Optional[str],
    route_table_keys: Set[str],
    route_served_by: Mapping[str, Optional[str]],
) -> LlmLaneRouteDecision:
    """
    Side-effect free: picks a route-table key for the LLM gateway HTTP hop.

    Chat lane preserves the caller's route key (e.g. quick vs chat) when present
    in the route table; otherwise falls back to llm_route_default / chat.
    """
    opts = options if isinstance(options, dict) else {}
    default_lane = _norm_lane(llm_lane_default, "chat")
    raw_lane = opts.get("llm_lane") or opts.get("execution_lane")
    logical_lane = _norm_lane(raw_lane, default_lane)
    invalid_lane = bool(str(raw_lane or "").strip()) and str(raw_lane).strip().lower() not in VALID_LLM_LANES

    allow_req = bool(opts.get("allow_chat_fallback"))

    def _ok(
        route_key: str,
        resolved_lane: str,
        reason: str,
        *,
        fallback_used: bool = False,
        status: str = "ok",
        degraded: bool = False,
    ) -> LlmLaneRouteDecision:
        sb = route_served_by.get(route_key)
        return LlmLaneRouteDecision(
            requested_llm_lane=logical_lane,
            resolved_llm_lane=resolved_lane,
            route_table_key=route_key,
            served_by=sb,
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
            served_by=None,
            route_status="missing_route",
            reason=reason,
            fallback_used=False,
            degraded=True,
        )

    def _chat_fallback(reason: str) -> LlmLaneRouteDecision:
        if not (llm_allow_background_to_chat_fallback and allow_req):
            return LlmLaneRouteDecision(
                requested_llm_lane=logical_lane,
                resolved_llm_lane=logical_lane,
                route_table_key=None,
                served_by=None,
                route_status="disallowed_chat_fallback",
                reason=reason,
                fallback_used=False,
                degraded=True,
            )
        br = str(body_route or "").strip() or str(llm_route_default or "chat")
        if br not in route_table_keys:
            br = _first_route_key(route_table_keys, ("chat", "quick", "agent")) or str(llm_route_default or "chat")
        if not br or br not in route_table_keys:
            return _missing(reason + "_chat_fallback_disallowed_or_no_chat_key")
        return LlmLaneRouteDecision(
            requested_llm_lane=logical_lane,
            resolved_llm_lane="chat",
            route_table_key=br,
            served_by=route_served_by.get(br),
            route_status="ok",
            reason=reason + "_emergency_chat_fallback",
            fallback_used=True,
            degraded=True,
        )

    if logical_lane == "chat":
        br = str(body_route or "").strip() or str(llm_route_default or "chat")
        if br not in route_table_keys:
            br = _first_route_key(route_table_keys, ("chat", "quick")) or str(llm_route_default or "chat")
        if br not in route_table_keys:
            return _missing("chat_lane_no_matching_route_table_key")
        st = "invalid_lane" if invalid_lane else "ok"
        return _ok(br, "chat", "verb_chat_lane", status=st, degraded=invalid_lane)

    if logical_lane == "spark":
        rk = _first_route_key(route_table_keys, _SPARK_ROUTE_KEYS)
        rk = rk or _match_served_by(route_served_by, llm_route_spark_served_by)
        if rk:
            return _ok(rk, "spark", "spark_route")
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        rk = rk or _match_served_by(route_served_by, llm_route_background_served_by)
        if rk:
            return _ok(rk, "background", "spark_missing_used_background_lane", fallback_used=True)
        return _chat_fallback("spark_route_missing_background_missing")

    if logical_lane == "background":
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        rk = rk or _match_served_by(route_served_by, llm_route_background_served_by)
        if rk:
            return _ok(rk, "background", "background_route")
        return _chat_fallback("background_route_missing")

    if logical_lane == "agent":
        rk = _first_route_key(route_table_keys, _AGENT_ROUTE_KEYS)
        rk = rk or _match_served_by(route_served_by, llm_route_agent_served_by)
        if rk:
            return _ok(rk, "agent", "agent_route")
        rk = _first_route_key(route_table_keys, _BACKGROUND_ROUTE_KEYS)
        rk = rk or _match_served_by(route_served_by, llm_route_background_served_by)
        if rk:
            return _ok(rk, "background", "agent_missing_used_background_lane", fallback_used=True)
        return _chat_fallback("agent_route_missing_background_missing")

    return _missing("unreachable_lane")
