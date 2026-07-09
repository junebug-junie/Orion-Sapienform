from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spark_inner_state_seed_v3",
    Path(__file__).resolve().parents[1] / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inner_state)


class _Dim:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeSelfState:
    def __init__(self, dims: dict, **kw) -> None:
        self.dimensions = {k: _Dim(v) for k, v in dims.items()}
        self.self_state_id = kw.get("sid", "self.state:tick_x:policy.v1")
        self.generated_at = kw.get("gen", datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc))
        self.overall_intensity = kw.get("intensity", 0.4)
        self.overall_condition = kw.get("condition", "steady")
        self.trajectory_condition = kw.get("trajectory", "stable")
        self.dominant_field_channels = kw.get("infra", {})


def _sample_self_state() -> _FakeSelfState:
    return _FakeSelfState(
        {
            "coherence": 0.8,
            "field_intensity": 1.0,
            "agency_readiness": 0.5,
            "execution_pressure": 0.2,
            "reasoning_pressure": 0.1,
            "resource_pressure": 1.0,
            "reliability_pressure": 0.9,
            "continuity_pressure": 0.3,
            "social_pressure": 0.1,
            "introspection_pressure": 0.0,
        },
        infra={"bus_health": 1.0},
    )


scaler = inner_state.RollingRobustScaler(maxlen=8)


def test_seed_v3_moves_reliability_to_infra_only() -> None:
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v3",
        grammar_degraded=False,
    )
    feature_names = {f.name for f in payload.features}
    infra_names = {f.name for f in payload.infra}
    assert "reliability_pressure" not in feature_names
    assert "reliability_pressure" in infra_names


def test_seed_v3_trainable_encoder_dims() -> None:
    names = inner_state.encoder_trainable_feature_names("seed-v3")
    assert names == [
        "coherence",
        "agency_readiness",
        "execution_pressure",
        "reasoning_pressure",
        "continuity_pressure",
        "social_pressure",
        "overall_intensity",
        "recall_gate_fired",
        "reasoning_present",
        "exec_step_fail_rate",
        "execution_friction",
    ]
    assert len(names) == 11


def test_seed_v3_still_emits_saturated_felt_for_audit() -> None:
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v3",
        grammar_degraded=False,
    )
    assert "field_intensity" in {f.name for f in payload.features}
