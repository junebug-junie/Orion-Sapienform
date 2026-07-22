"""Honest inner-state feature assembly for spark-introspector.

2026-07-22 (SelfStateV1 burn, docs/superpowers/specs/2026-07-22-self-state-
phi-endo-origination-burn-spec.md, decision 3): SelfStateV1 and its 10
FELT_DIMENSIONS are gone -- they were self-state's own hand-tuned, empirically
pinned/flat dimension scores (12/12 pinned in a live 2026-07-22 replay). What
survives is the 4 real cognitive features (recall_gate_fired,
reasoning_present, execution_load, reasoning_load), sourced from
execution_trajectory/reasoning_activity projections -- independently real,
never self-state-derived, unaffected by the burn.

honest_headline() (the old FELT_DIMENSIONS-derived vitality/load aggregate)
is REMOVED, not replaced. No principled non-self-state formula for it exists;
inventing one wasn't asked for and isn't attempted here (same "not designing
a replacement" discipline as the burn spec). headline defaults to an honest
0.0 / "not_computed" now, overridden by the encoder's real phi output when
available (see app/worker.py) -- same override precedent as before.
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

# 2026-07-22 (SelfStateV1 burn): the only trainable feature set the live
# service produces going forward. Kept as its own named tuple (identical
# contents to SEEDV4_COGNITIVE_FEATURE_NAMES today) so a future encoder
# feature-set change doesn't have to disturb the seed-v4 historical name.
SEEDV5_TRAINABLE_FEATURE_NAMES: Tuple[str, ...] = SEEDV4_COGNITIVE_FEATURE_NAMES


def encoder_trainable_feature_names(features_version: str) -> list[str]:
    """2026-07-22 (SelfStateV1 burn): seed-v1/v2/v3/v4's FELT_DIMENSIONS-based
    branches removed -- they depended on SelfStateV1, which no longer exists.
    Offline tooling that needs to interpret historical seed-v2/v3/v4 corpus
    rows (written before this burn) should read those branches from git
    history rather than expect them here; the live service only ever
    produces seed-v5 going forward, and this function reflects that."""
    del features_version
    return list(SEEDV5_TRAINABLE_FEATURE_NAMES)


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
    """Seed-v4 (and seed-v5) cognitive slots: recall_gate_fired
    (execution_trajectory), reasoning_present + reasoning_load
    (reasoning_activity), execution_load (reasoning_activity token
    throughput, execution_trajectory step-count fallback when the
    reasoning_activity projection is dark). Raw only -- scaling happens
    uniformly in build_inner_state_features, same contract as
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
    # thinking_enabled -- today, always. Truthful 0.0, never a fake floor.
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


def build_inner_state_features(
    scaler: RollingRobustScaler,
    *,
    features_version: str,
    grammar_degraded: bool,
    degraded_reasons: Optional[List[str]] = None,
    trajectory_projection: Optional[dict] = None,
    reasoning_activity_projection: Optional[dict] = None,
    exec_trajectory_max_age_sec: int = 120,
    now: Optional[datetime] = None,
    prev_felt: Optional[Tuple[float, ...]] = None,
    degenerate_streak: int = 0,
    degenerate_limit: int = 20,
) -> Tuple[InnerStateFeaturesV1, Tuple[float, ...], int]:
    """Assemble one InnerStateFeaturesV1 from execution_trajectory/
    reasoning_activity projections directly.

    2026-07-22 (SelfStateV1 burn): no longer takes a SelfStateV1 -- the 10
    FELT_DIMENSIONS derived from it are gone, and `now` replaces
    `ss.generated_at` as the tick's own generation time (see app/worker.py's
    poll-loop trigger, which replaced the old substrate.self_state.v1 bus
    subscription).

    Returns (payload, current_feature_tuple, new_degenerate_streak). The
    degeneracy freeze (GIGO guard) is now keyed on the 4 cognitive features'
    raw values instead of self-state's old 10-dimension felt_tuple -- same
    purpose (detect a feature vector that's stopped moving), real surviving
    input.
    """
    gen = now or datetime.now(timezone.utc)

    if features_version == "seed-v4" or features_version == "seed-v5":
        cognitive_feats = cognitive_features_seed_v4(
            trajectory_projection,
            reasoning_activity_projection,
            now=gen,
            exec_trajectory_max_age_sec=exec_trajectory_max_age_sec,
        )
    else:
        cognitive_feats = cognitive_features_from_trajectory(
            trajectory_projection,
            now=gen,
            max_age_sec=exec_trajectory_max_age_sec,
        )

    features: List[InnerFeatureV1] = [
        InnerFeatureV1(
            name=feat.name,
            raw_value=feat.raw_value,
            scaled_value=round(scaler.observe_and_scale(feat.name, feat.raw_value), 4),
            source=feat.source,
        )
        for feat in cognitive_feats
    ]
    feature_tuple = tuple(feat.raw_value for feat in cognitive_feats)

    # --- degeneracy freeze (GIGO guard) ---
    if prev_felt is not None and feature_tuple == prev_felt:
        new_streak = degenerate_streak + 1
    else:
        new_streak = 0

    degenerate = new_streak >= degenerate_limit
    phi_health = "ok"
    if grammar_degraded or degenerate:
        phi_health = "frozen"

    metadata: Dict[str, object] = {}
    if degraded_reasons:
        metadata["degraded_reasons"] = degraded_reasons
    payload = InnerStateFeaturesV1(
        features_version=features_version,
        generated_at=gen,
        # self_state_id intentionally left None -- no self-state tick exists
        # to identify this row against anymore.
        features=features,
        infra=[],
        # No principled non-self-state headline formula exists (see module
        # docstring) -- honest "not computed" rather than a fabricated
        # aggregate. app/worker.py overrides this with the trained encoder's
        # real phi output when the encoder tick succeeds, same as before.
        headline=0.0,
        headline_source="not_computed",
        phi_health=phi_health,
        phi_degenerate_streak=new_streak,
        grammar_truth_degraded=bool(grammar_degraded),
        # self_state liveness is honestly False now -- no self-state feed
        # exists. No live consumer reads this key (checked before removing
        # it); kept present rather than dropped so a strict consumer doesn't
        # KeyError.
        liveness={"self_state": False, "grammar_truth": not grammar_degraded},
        metadata=metadata,
    )
    return payload, feature_tuple, new_streak
