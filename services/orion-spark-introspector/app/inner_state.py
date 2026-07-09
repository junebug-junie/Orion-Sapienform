"""Honest inner-state feature assembly for spark-introspector.

Replaces the geometric-mean φ (which a single saturated infra signal could
floor) with a decontaminated, robust-scaled feature vector plus an arithmetic
cold-start headline and a degeneracy freeze (the GIGO guard).
"""
from __future__ import annotations

import math
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

ENCODER_EXCLUDED_FELT: frozenset[str] = frozenset({
    "field_intensity",
    "resource_pressure",
    "introspection_pressure",
})
INFRA_ONLY_FELT: frozenset[str] = frozenset({"reliability_pressure"})

# Proven-frozen SelfStateV1 dims — excised from seed-v4's trainable set. Still
# recorded in `infra` for provenance (same treatment as INFRA_ONLY_FELT), just
# never scaled/trained on. See docs/superpowers/specs/2026-07-09-phi-seedv4-feature-set-design.md.
SEEDV4_THEATER_FELT: frozenset[str] = frozenset({"coherence", "continuity_pressure", "social_pressure"})

# Retained for provenance only; φ never reads these.
INFRA_CHANNELS: Tuple[str, ...] = (
    "bus_health",
    "delivery_confidence",
    "transport_integrity",
    "contract_pressure",
    "catalog_drift_pressure",
)

COGNITIVE_FEATURE_NAMES: Tuple[str, ...] = (
    "recall_gate_fired",
    "reasoning_present",
    "exec_step_fail_rate",
    "execution_friction",
)

# seed-v4 cognitive slot names: drop the structurally-sparse exec_step_fail_rate
# / execution_friction pair, add token-based execution_load and real
# reasoning_activity-sourced reasoning_load.
SEEDV4_COGNITIVE_FEATURE_NAMES: Tuple[str, ...] = (
    "recall_gate_fired",
    "reasoning_present",
    "execution_load",
    "reasoning_load",
)


def encoder_trainable_feature_names(features_version: str) -> list[str]:
    if features_version == "seed-v4":
        felt = [
            k for k in FELT_DIMENSIONS
            if k not in ENCODER_EXCLUDED_FELT
            and k not in INFRA_ONLY_FELT
            and k not in SEEDV4_THEATER_FELT
        ]
        return felt + ["overall_intensity"] + list(SEEDV4_COGNITIVE_FEATURE_NAMES)
    if features_version == "seed-v3":
        felt = [
            k for k in FELT_DIMENSIONS
            if k not in ENCODER_EXCLUDED_FELT and k not in INFRA_ONLY_FELT
        ]
        return felt + ["overall_intensity"] + list(COGNITIVE_FEATURE_NAMES)
    felt = [k for k in FELT_DIMENSIONS if k not in DROPPED_DIMENSIONS]
    return felt + ["overall_intensity"] + list(COGNITIVE_FEATURE_NAMES)


def _dim_score(ss, key: str, default: float = 0.0) -> float:
    dim = getattr(ss, "dimensions", {}).get(key)
    return float(dim.score) if dim is not None else default


def _felt_tuple(ss) -> Tuple[float, ...]:
    return tuple(round(_dim_score(ss, k), 4) for k in FELT_DIMENSIONS)


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _active_trajectory_runs(
    projection: dict | None,
    *,
    now: datetime,
    max_age_sec: int,
) -> list[dict]:
    """Runs from an execution_trajectory projection whose last_updated_at is
    within max_age_sec of now. Never raises: malformed timestamps/shapes are
    skipped rather than propagated."""
    if not projection or not projection.get("runs"):
        return []
    active: list[dict] = []
    for run in projection["runs"].values():
        try:
            ts = datetime.fromisoformat(str(run["last_updated_at"]).replace("Z", "+00:00"))
        except Exception:
            continue
        if (now - ts).total_seconds() <= max_age_sec:
            active.append(run)
    return active


def _recall_gate_fired(active_runs: list[dict]) -> float:
    return 1.0 if any(r.get("recall_observed") for r in active_runs) else 0.0


def cognitive_features_from_trajectory(
    projection: dict | None,
    *,
    now: datetime,
    max_age_sec: int,
) -> List[InnerFeatureV1]:
    """Return four cognitive InnerFeatureV1 rows (raw only; scaling in build_inner_state_features)."""
    none_source = "execution_trajectory.none"
    if not projection or not projection.get("runs"):
        return [
            InnerFeatureV1(name=n, raw_value=0.0, scaled_value=0.0, source=none_source)
            for n in COGNITIVE_FEATURE_NAMES
        ]
    active = _active_trajectory_runs(projection, now=now, max_age_sec=max_age_sec)
    if not active:
        return [
            InnerFeatureV1(name=n, raw_value=0.0, scaled_value=0.0, source=none_source)
            for n in COGNITIVE_FEATURE_NAMES
        ]
    recall = _recall_gate_fired(active)
    reasoning_frac = sum(1 for r in active if r.get("reasoning_present")) / len(active)
    steps = sum(int(r.get("step_count") or 0) for r in active)
    fails = sum(int(r.get("failed_step_count") or 0) for r in active)
    fail_rate = fails / max(1, steps)
    frictions = [float(r.get("pressure_hints", {}).get("execution_friction", 0.0)) for r in active]
    friction = sum(frictions) / len(frictions)
    raw_by_name = {
        "recall_gate_fired": recall,
        "reasoning_present": reasoning_frac,
        "exec_step_fail_rate": fail_rate,
        "execution_friction": friction,
    }
    return [
        InnerFeatureV1(
            name=name,
            raw_value=round(raw_by_name[name], 4),
            scaled_value=0.0,
            source=f"execution_trajectory.runs.*.{name}",
        )
        for name in COGNITIVE_FEATURE_NAMES
    ]


def cognitive_features_seed_v4(
    trajectory_projection: dict | None,
    reasoning_activity_projection: dict | None,
    *,
    now: datetime,
    exec_trajectory_max_age_sec: int,
) -> List[InnerFeatureV1]:
    """Seed-v4 cognitive slots: recall_gate_fired (execution_trajectory),
    reasoning_present + reasoning_load (reasoning_activity), execution_load
    (reasoning_activity token throughput, execution_trajectory step-count
    fallback when the reasoning_activity projection is dark). Raw only —
    scaling happens uniformly in build_inner_state_features, same contract as
    cognitive_features_from_trajectory. Never raises: absent/malformed inputs
    degrade to a truthful 0.0 with a `.none`-suffixed source, per-feature."""
    active = _active_trajectory_runs(
        trajectory_projection, now=now, max_age_sec=exec_trajectory_max_age_sec
    )

    # recall_gate_fired
    if active:
        recall_raw = _recall_gate_fired(active)
        recall_source = "execution_trajectory.runs.*.recall_gate_fired"
    else:
        recall_raw = 0.0
        recall_source = "execution_trajectory.none"

    ra = reasoning_activity_projection if isinstance(reasoning_activity_projection, dict) else None
    call_count = 0
    if ra is not None:
        try:
            call_count = int(ra.get("call_count") or 0)
        except Exception:
            call_count = 0
    ra_live = ra is not None and call_count > 0

    # reasoning_present: continuous reasoning_present_rate, truthful 0.0 when dark.
    if ra_live:
        try:
            reasoning_present_raw = round(float(ra.get("reasoning_present_rate", 0.0)), 4)
        except Exception:
            reasoning_present_raw = 0.0
        reasoning_present_source = "reasoning_activity.reasoning_present_rate"
    else:
        reasoning_present_raw = 0.0
        reasoning_present_source = "reasoning_activity.none"

    # execution_load: primary = log1p(completion_tokens_sum); fallback =
    # log1p(summed step_count) from the same active trajectory runs; else 0.0.
    completion_tokens_sum = 0
    if ra_live:
        try:
            completion_tokens_sum = int(ra.get("completion_tokens_sum") or 0)
        except Exception:
            completion_tokens_sum = 0
    if ra_live and completion_tokens_sum > 0:
        execution_load_raw = round(math.log1p(float(completion_tokens_sum)), 4)
        execution_load_source = "reasoning_activity.completion_tokens_sum"
    else:
        steps = sum(int(r.get("step_count") or 0) for r in active) if active else 0
        if active and steps > 0:
            execution_load_raw = round(math.log1p(float(steps)), 4)
            execution_load_source = "execution_trajectory.step_count_fallback"
        else:
            execution_load_raw = 0.0
            execution_load_source = "execution_trajectory.none"

    # reasoning_load: log1p(thinking_tokens_sum) only when a real positive int
    # is present. thinking_tokens_sum is None whenever no call in the window had
    # thinking_enabled — today, always. Truthful 0.0, never a fake floor.
    thinking_tokens_sum = None
    if ra_live:
        thinking_tokens_sum = ra.get("thinking_tokens_sum")
    thinking_tokens_valid = (
        isinstance(thinking_tokens_sum, (int, float))
        and not isinstance(thinking_tokens_sum, bool)
        and thinking_tokens_sum > 0
    )
    if ra_live and thinking_tokens_valid:
        reasoning_load_raw = round(math.log1p(float(thinking_tokens_sum)), 4)
        reasoning_load_source = "reasoning_activity.thinking_tokens_sum"
    else:
        reasoning_load_raw = 0.0
        reasoning_load_source = "reasoning_activity.none"

    raw_and_source = {
        "recall_gate_fired": (recall_raw, recall_source),
        "reasoning_present": (reasoning_present_raw, reasoning_present_source),
        "execution_load": (execution_load_raw, execution_load_source),
        "reasoning_load": (reasoning_load_raw, reasoning_load_source),
    }
    return [
        InnerFeatureV1(
            name=name,
            raw_value=raw_and_source[name][0],
            scaled_value=0.0,
            source=raw_and_source[name][1],
        )
        for name in SEEDV4_COGNITIVE_FEATURE_NAMES
    ]


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
    degraded_reasons: Optional[List[str]] = None,
    trajectory_projection: Optional[dict] = None,
    reasoning_activity_projection: Optional[dict] = None,
    exec_trajectory_max_age_sec: int = 120,
    prev_felt: Optional[Tuple[float, ...]] = None,
    prev_headline: Optional[float] = None,
    degenerate_streak: int = 0,
    degenerate_limit: int = 20,
) -> Tuple[InnerStateFeaturesV1, Tuple[float, ...], int]:
    """Assemble one InnerStateFeaturesV1 from a SelfStateV1.

    Returns (payload, current_felt_tuple, new_degenerate_streak).
    """
    felt_tuple = _felt_tuple(ss)

    features: List[InnerFeatureV1] = []
    raw_map: Dict[str, float] = {}
    infra_only_felt: List[InnerFeatureV1] = []
    for key in FELT_DIMENSIONS:
        raw = _dim_score(ss, key)
        raw_map[key] = raw
        if (features_version == "seed-v3" and key in INFRA_ONLY_FELT) or (
            features_version == "seed-v4" and key in (INFRA_ONLY_FELT | SEEDV4_THEATER_FELT)
        ):
            infra_only_felt.append(
                InnerFeatureV1(
                    name=key,
                    raw_value=round(raw, 4),
                    scaled_value=0.0,
                    source=f"self_state.dimensions.{key}",
                )
            )
            continue
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

    # seed-v2/v3 always emit the four cognitive slots so encoder/corpus dims stay
    # stable even when trajectory HTTP fails (zeros + execution_trajectory.none).
    # seed-v1 and other versions only append when a projection was provided.
    include_cognitive = (
        features_version.startswith("seed-v2")
        or features_version == "seed-v3"
        or features_version == "seed-v4"
        or trajectory_projection is not None
    )
    if include_cognitive:
        gen_for_traj = getattr(ss, "generated_at", None) or datetime.now(timezone.utc)
        if features_version == "seed-v4":
            cognitive_feats = cognitive_features_seed_v4(
                trajectory_projection,
                reasoning_activity_projection,
                now=gen_for_traj,
                exec_trajectory_max_age_sec=exec_trajectory_max_age_sec,
            )
        else:
            cognitive_feats = cognitive_features_from_trajectory(
                trajectory_projection,
                now=gen_for_traj,
                max_age_sec=exec_trajectory_max_age_sec,
            )
        for feat in cognitive_feats:
            features.append(
                InnerFeatureV1(
                    name=feat.name,
                    raw_value=feat.raw_value,
                    scaled_value=round(scaler.observe_and_scale(feat.name, feat.raw_value), 4),
                    source=feat.source,
                )
            )

    infra: List[InnerFeatureV1] = list(infra_only_felt)
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
    metadata: Dict[str, object] = {
        "overall_condition": getattr(ss, "overall_condition", None),
        "trajectory_condition": getattr(ss, "trajectory_condition", None),
    }
    if degraded_reasons:
        metadata["degraded_reasons"] = degraded_reasons
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
        metadata=metadata,
    )
    return payload, felt_tuple, new_streak
