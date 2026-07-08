from datetime import datetime, timedelta, timezone

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spark_inner_state",
    Path(__file__).resolve().parents[1] / "services" / "orion-spark-introspector" / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inner_state)


def test_execution_trajectory_features_reasoning_present() -> None:
    now = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    projection = {
        "runs": {
            "a": {
                "reasoning_present": True,
                "recall_observed": True,
                "step_count": 10,
                "failed_step_count": 2,
                "pressure_hints": {"execution_friction": 0.4},
                "last_updated_at": now.isoformat(),
            }
        }
    }
    feats = inner_state.cognitive_features_from_trajectory(
        projection, now=now, max_age_sec=120,
    )
    by_name = {f.name: f for f in feats}
    assert by_name["reasoning_present"].raw_value == 1.0
    assert by_name["recall_gate_fired"].raw_value == 1.0
    assert abs(by_name["exec_step_fail_rate"].raw_value - 0.2) < 1e-6


def test_execution_trajectory_stale_runs_excluded() -> None:
    now = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    stale = now - timedelta(seconds=300)
    projection = {
        "runs": {
            "a": {
                "reasoning_present": True,
                "recall_observed": True,
                "step_count": 1,
                "failed_step_count": 0,
                "pressure_hints": {},
                "last_updated_at": stale.isoformat(),
            }
        }
    }
    feats = inner_state.cognitive_features_from_trajectory(
        projection, now=now, max_age_sec=120,
    )
    assert all(f.source == "execution_trajectory.none" for f in feats)
    assert all(f.raw_value == 0.0 for f in feats)


def test_seed_v2_emits_cognitive_slots_when_trajectory_absent() -> None:
    """seed-v2 keeps encoder input dims stable when HTTP traj is dark."""
    from types import SimpleNamespace

    ss = SimpleNamespace(
        generated_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
        overall_intensity=0.4,
        overall_condition="steady",
        trajectory_condition="stable",
        self_state_id="s1",
        dimensions={},
        dominant_field_channels={},
    )
    scaler = inner_state.RollingRobustScaler(maxlen=32)
    payload, _, _ = inner_state.build_inner_state_features(
        ss,
        scaler,
        features_version="seed-v2",
        grammar_degraded=False,
        trajectory_projection=None,
    )
    names = {f.name for f in payload.features}
    for name in inner_state.COGNITIVE_FEATURE_NAMES:
        assert name in names
    cogn = [f for f in payload.features if f.name in inner_state.COGNITIVE_FEATURE_NAMES]
    assert all(f.source == "execution_trajectory.none" for f in cogn)
    assert all(f.raw_value == 0.0 for f in cogn)


def test_seed_v1_omits_cognitive_slots_without_trajectory() -> None:
    from types import SimpleNamespace

    ss = SimpleNamespace(
        generated_at=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
        overall_intensity=0.4,
        overall_condition="steady",
        trajectory_condition="stable",
        self_state_id="s1",
        dimensions={},
        dominant_field_channels={},
    )
    scaler = inner_state.RollingRobustScaler(maxlen=32)
    payload, _, _ = inner_state.build_inner_state_features(
        ss,
        scaler,
        features_version="seed-v1",
        grammar_degraded=False,
        trajectory_projection=None,
    )
    names = {f.name for f in payload.features}
    for name in inner_state.COGNITIVE_FEATURE_NAMES:
        assert name not in names
