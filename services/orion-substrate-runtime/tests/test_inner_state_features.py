"""Ported from orion-spark-introspector's tests/test_inner_state_features.py
2026-07-28, alongside app/inner_state_features.py's extraction -- see that
module's docstring. Only the pure feature-assembly tests are ported; the
corpus-sink tests from the original file tested functionality deliberately
not brought over (see inner_state_features.py's docstring)."""
from datetime import datetime, timezone

from app.inner_state_features import (
    SCALE_CLIP,
    RollingRobustScaler,
    build_inner_state_features,
    robust_scale,
)


def test_robust_scale_zero_when_iqr_degenerate() -> None:
    assert robust_scale(0.5, median=0.5, iqr=0.0) == 0.0


def test_robust_scale_standardizes() -> None:
    assert robust_scale(2.0, median=1.0, iqr=1.0) == 1.0
    assert robust_scale(0.0, median=1.0, iqr=1.0) == -1.0


def test_rolling_scaler_centers_after_history() -> None:
    s = RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s.observe("cpu", v)
    scaled = s.scale("cpu", 0.5)
    assert abs(scaled) < 1e-6


def test_rolling_scaler_saturated_input_does_not_explode() -> None:
    s = RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.1, 0.2, 0.3, 0.4]:
        s.observe("pressure", v)
    scaled = s.scale("pressure", 1.0)
    assert scaled <= SCALE_CLIP


def test_degeneracy_freeze_after_streak() -> None:
    scaler = RollingRobustScaler(maxlen=64)
    now = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
    prev_felt = None
    streak = 0
    payload = None
    for _ in range(21):  # limit=20 -> frozen on the 21st identical tick
        payload, prev_felt, streak = build_inner_state_features(
            scaler, features_version="seed-v1", grammar_degraded=False,
            trajectory_projection=None, now=now,
            prev_felt=prev_felt, degenerate_streak=streak, degenerate_limit=20,
        )
    assert payload.phi_health == "frozen"
    assert payload.phi_degenerate_streak >= 20


def test_grammar_degraded_forces_frozen() -> None:
    scaler = RollingRobustScaler(maxlen=8)
    payload, _f, _s = build_inner_state_features(
        scaler, features_version="seed-v1", grammar_degraded=True,
        trajectory_projection=None,
        now=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        prev_felt=None, degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.phi_health == "frozen"


def test_seed_v4_execution_load_falls_back_to_step_count() -> None:
    now = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
    trajectory = {
        "runs": {
            "run-1": {
                "last_updated_at": now.isoformat(),
                "recall_observed": False,
                "reasoning_present": False,
                "step_count": 5,
            }
        }
    }
    scaler = RollingRobustScaler(maxlen=8)
    payload, _f, _s = build_inner_state_features(
        scaler, features_version="seed-v5", grammar_degraded=False,
        trajectory_projection=trajectory, reasoning_activity_projection=None,
        now=now, prev_felt=None, degenerate_streak=0, degenerate_limit=20,
    )
    by_name = {f.name: f for f in payload.features}
    assert by_name["execution_load"].source == "execution_trajectory.step_count_fallback"
    assert by_name["execution_load"].raw_value > 0.0
    assert by_name["reasoning_load"].source == "reasoning_activity.none"
    assert by_name["reasoning_load"].raw_value == 0.0
