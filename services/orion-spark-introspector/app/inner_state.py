"""Honest inner-state feature assembly for spark-introspector.

Replaces the geometric-mean φ (which a single saturated infra signal could
floor) with a decontaminated, robust-scaled feature vector plus an arithmetic
cold-start headline and a degeneracy freeze (the GIGO guard).
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import median as _median
from typing import Deque, Dict, List, Optional, Tuple

from orion.schemas.telemetry.inner_state import InnerFeatureV1, InnerStateFeaturesV1

SCALE_CLIP = 4.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = pct * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def robust_scale(value: float, *, median: float, iqr: float) -> float:
    if iqr <= 1e-9:
        return 0.0
    scaled = (float(value) - median) / iqr
    return max(-SCALE_CLIP, min(SCALE_CLIP, scaled))


class RollingRobustScaler:
    """Bounded rolling median/IQR standardizer, one window per feature name."""

    def __init__(self, maxlen: int = 256) -> None:
        self._maxlen = max(2, int(maxlen))
        self._windows: Dict[str, Deque[float]] = {}

    def observe(self, name: str, value: float) -> None:
        w = self._windows.get(name)
        if w is None:
            w = deque(maxlen=self._maxlen)
            self._windows[name] = w
        w.append(float(value))

    def scale(self, name: str, value: float) -> float:
        w = self._windows.get(name)
        if not w or len(w) < 2:
            return 0.0
        vals = list(w)
        med = _median(vals)
        iqr = _percentile(vals, 0.75) - _percentile(vals, 0.25)
        return robust_scale(value, median=med, iqr=iqr)

    def observe_and_scale(self, name: str, value: float) -> float:
        self.observe(name, value)
        return self.scale(name, value)


# Felt + cognitive dimensions φ reads. (execution_pressure / reasoning_pressure
# are the cortex-exec-lane-fed cognitive signal; reliability_pressure is now
# decontaminated at source.) policy_pressure + uncertainty are proven dead and
# are NOT listed here.
FELT_DIMENSIONS: Tuple[str, ...] = (
    "coherence",
    "field_intensity",
    "agency_readiness",
    "execution_pressure",
    "reasoning_pressure",
    "resource_pressure",
    "reliability_pressure",
    "continuity_pressure",
    "social_pressure",
    "introspection_pressure",
)

DROPPED_DIMENSIONS: frozenset[str] = frozenset({"policy_pressure", "uncertainty"})

# Retained for provenance only; φ never reads these.
INFRA_CHANNELS: Tuple[str, ...] = (
    "bus_health",
    "delivery_confidence",
    "transport_integrity",
    "contract_pressure",
    "catalog_drift_pressure",
)


def _dim_score(ss, key: str, default: float = 0.0) -> float:
    dim = getattr(ss, "dimensions", {}).get(key)
    return float(dim.score) if dim is not None else default


def _felt_tuple(ss) -> Tuple[float, ...]:
    return tuple(round(_dim_score(ss, k), 4) for k in FELT_DIMENSIONS)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def honest_headline(raw: Dict[str, float]) -> float:
    """Arithmetic vitality/load aggregate. No geometric collapse: a single
    saturated pressure lowers but cannot floor the headline."""
    def g(k: str, default: float = 0.0) -> float:
        return float(raw.get(k, default))

    vitality = _mean([
        g("coherence"),
        g("field_intensity"),
        g("agency_readiness"),
        1.0 - g("resource_pressure"),
        1.0 - g("execution_pressure"),
    ])
    load = _mean([
        g("reliability_pressure"),
        g("reasoning_pressure"),
        g("social_pressure"),
        g("continuity_pressure"),
    ])
    return _clamp01(0.6 * vitality + 0.4 * (1.0 - load))


def build_inner_state_features(
    ss,
    scaler: RollingRobustScaler,
    *,
    features_version: str,
    grammar_degraded: bool,
    prev_felt: Optional[Tuple[float, ...]],
    prev_headline: Optional[float],
    degenerate_streak: int,
    degenerate_limit: int,
) -> Tuple[InnerStateFeaturesV1, Tuple[float, ...], int]:
    """Assemble one InnerStateFeaturesV1 from a SelfStateV1.

    Returns (payload, current_felt_tuple, new_degenerate_streak).
    """
    felt_tuple = _felt_tuple(ss)

    features: List[InnerFeatureV1] = []
    raw_map: Dict[str, float] = {}
    for key in FELT_DIMENSIONS:
        raw = _dim_score(ss, key)
        raw_map[key] = raw
        features.append(
            InnerFeatureV1(
                name=key,
                raw_value=round(raw, 4),
                scaled_value=round(scaler.observe_and_scale(key, raw), 4),
                source=f"self_state.dimensions.{key}",
            )
        )

    # overall_intensity as an extra felt feature
    intensity = float(getattr(ss, "overall_intensity", 0.0) or 0.0)
    raw_map["overall_intensity"] = intensity
    features.append(
        InnerFeatureV1(
            name="overall_intensity",
            raw_value=round(intensity, 4),
            scaled_value=round(scaler.observe_and_scale("overall_intensity", intensity), 4),
            source="self_state.overall_intensity",
        )
    )

    infra: List[InnerFeatureV1] = []
    dom = getattr(ss, "dominant_field_channels", {}) or {}
    for ch in INFRA_CHANNELS:
        if ch in dom:
            infra.append(
                InnerFeatureV1(
                    name=ch,
                    raw_value=round(float(dom[ch]), 4),
                    scaled_value=0.0,  # infra never scaled/read by φ
                    source=f"self_state.dominant_field_channels.{ch}",
                )
            )

    # --- degeneracy freeze (GIGO guard) ---
    if prev_felt is not None and felt_tuple == prev_felt:
        new_streak = degenerate_streak + 1
    else:
        new_streak = 0

    degenerate = new_streak >= degenerate_limit
    headline = honest_headline(raw_map)
    phi_health = "ok"
    if grammar_degraded or degenerate:
        phi_health = "frozen"
        if prev_headline is not None:
            headline = prev_headline

    gen = getattr(ss, "generated_at", None) or datetime.now(timezone.utc)
    payload = InnerStateFeaturesV1(
        features_version=features_version,
        generated_at=gen,
        self_state_id=getattr(ss, "self_state_id", None),
        features=features,
        infra=infra,
        headline=round(_clamp01(headline), 4),
        headline_source="cold_start_aggregate",
        phi_health=phi_health,
        phi_degenerate_streak=new_streak,
        grammar_truth_degraded=bool(grammar_degraded),
        liveness={"self_state": True, "grammar_truth": not grammar_degraded},
        metadata={
            "overall_condition": getattr(ss, "overall_condition", None),
            "trajectory_condition": getattr(ss, "trajectory_condition", None),
        },
    )
    return payload, felt_tuple, new_streak
