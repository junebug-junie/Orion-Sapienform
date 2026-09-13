"""Pure lane policy over declared capabilities and persisted reservation facts.

Estimates use declared inference budgets, not measured predictions or renewable
lease TTLs. Unknown backend occupancy fails closed. No cold-load penalty is
invented. These estimates never drive cognition.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Mapping


def satisfies(capabilities: Mapping[str, Any], requirements: Mapping[str, Any]) -> bool:
    for key, required in requirements.items():
        actual = capabilities.get(key)
        if key.startswith("minimum_"):
            actual = capabilities.get(key.removeprefix("minimum_"), actual)
            if (isinstance(required, bool) or not isinstance(required, (float, int)) or not isfinite(required)
                or isinstance(actual, bool) or not isinstance(actual, (float, int)) or not isfinite(actual) or actual < required):
                return False
        elif actual != required:
            return False
    return True


@dataclass(frozen=True)
class LaneDecision:
    requested_lane: str
    eligible_lanes: list[str]
    assigned_lane: str | None
    estimates: dict[str, float | None]
    reason: str
    suppressed: dict[str, str]
    threshold_seconds: float
    operator_override: str | None

    def detail(self) -> dict[str, Any]:
        return dict(self.__dict__)


def decide_lane(
    requirement: Mapping[str, Any],
    lanes: Mapping[str, Mapping[str, Any]],
    *,
    waited_seconds: float,
    active_remaining: Mapping[str, float],
    queued_ahead: Mapping[str, float],
    lease_seconds: float,
    widen_after_seconds: float = 1200,
    hysteresis_seconds: float = 120,
    widening_enabled: bool = False,
) -> LaneDecision:
    preferred = str(requirement.get("preferred_lane", "agent"))
    override = requirement.get("operator_override")
    pin = requirement.get("pinned_lane")
    target = str(override or pin or preferred)
    hard = requirement.get("requirements") or {}
    eligible: list[str] = []
    suppressed: dict[str, str] = {}
    estimates: dict[str, float | None] = {}

    def compatible(lane: str) -> bool:
        meta = lanes.get(lane)
        if not meta or not meta.get("configured") or not meta.get("backend_key"):
            suppressed[lane] = "route_not_configured"
            return False
        if meta.get("healthy") is not True:
            suppressed[lane] = "health_unknown_or_unavailable"
            return False
        if not satisfies(meta.get("capabilities") or {}, hard):
            suppressed[lane] = "hard_requirements_incompatible"
            return False
        return True

    if compatible(target):
        eligible.append(target)
    for lane in dict.fromkeys(requirement.get("alternatives") or []):
        if lane == target:
            continue
        if override or pin:
            suppressed[lane] = "operator_override" if override else "experimental_pin"
        elif not widening_enabled:
            suppressed[lane] = "widening_disabled"
        elif waited_seconds < widen_after_seconds:
            suppressed[lane] = "wait_threshold"
        elif preferred not in (lanes.get(lane, {}).get("compatible_with") or []):
            suppressed[lane] = "compatibility_not_declared"
        elif compatible(lane):
            # Quality loss must be explicitly declared in compatibility config.
            meta = lanes[lane]
            drop = meta.get("quality_drop", 0)
            if not isinstance(drop, (int, float)) or drop > meta.get("max_quality_drop", 1):
                suppressed[lane] = "quality_drop"
            else:
                eligible.append(lane)

    for lane in eligible:
        backend = str(lanes[lane]["backend_key"])
        estimates[lane] = max(0, active_remaining.get(backend, 0)) + max(0, queued_ahead.get(backend, 0))
    target_estimate = estimates.get(target, inf)
    allowed: list[str] = []
    for lane in eligible:
        if lane == target:
            allowed.append(lane)
            continue
        cost = max(0, float(lanes[lane].get("switching_cost_seconds", 0)))
        if estimates[lane] + cost < target_estimate - hysteresis_seconds:
            allowed.append(lane)
        else:
            suppressed[lane] = "hysteresis"
    free = [lane for lane in allowed if estimates[lane] == 0 and lanes[lane].get("external_busy", False) is False]
    chosen = min(free, key=lambda lane: (lane != target, lane)) if free else None
    reason = "preferred_available" if chosen == target else "earlier_compatible_start" if chosen else "waiting_capacity"
    if override and chosen:
        reason = "operator_override"
    elif pin and chosen:
        reason = "experimental_pin"
    return LaneDecision(preferred, eligible, chosen, estimates, reason, suppressed, widen_after_seconds, override)
