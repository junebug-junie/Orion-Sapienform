from __future__ import annotations

import importlib.util
import math
from datetime import datetime, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spark_inner_state_seed_v4",
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


def test_seed_v4_trainable_encoder_dims() -> None:
    names = inner_state.encoder_trainable_feature_names("seed-v4")
    assert names == [
        "agency_readiness",
        "execution_pressure",
        "reasoning_pressure",
        "overall_intensity",
        "recall_gate_fired",
        "reasoning_present",
        "execution_load",
        "reasoning_load",
    ]
    assert len(names) == 8


def test_seed_v4_moves_theater_trio_to_infra_only() -> None:
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
    )
    feature_names = {f.name for f in payload.features}
    infra_names = {f.name for f in payload.infra}
    for theater in ("coherence", "continuity_pressure", "social_pressure"):
        assert theater not in feature_names
        assert theater in infra_names
    # reliability_pressure still routes to infra-only under seed-v4 too.
    assert "reliability_pressure" not in feature_names
    assert "reliability_pressure" in infra_names
    # live felt dims remain trainable.
    for live in ("agency_readiness", "execution_pressure", "reasoning_pressure"):
        assert live in feature_names


def test_seed_v4_reasoning_activity_live_signals() -> None:
    reasoning_activity_projection = {
        "call_count": 10,
        "reasoning_present_rate": 0.4,
        "completion_tokens_sum": 500,
        "thinking_tokens_sum": None,
    }
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
        trajectory_projection=None,
        reasoning_activity_projection=reasoning_activity_projection,
    )
    by_name = {f.name: f for f in payload.features}

    reasoning_present = by_name["reasoning_present"]
    assert reasoning_present.raw_value == 0.4
    assert reasoning_present.source == "reasoning_activity.reasoning_present_rate"

    execution_load = by_name["execution_load"]
    assert execution_load.raw_value == round(math.log1p(500), 4)
    assert execution_load.source == "reasoning_activity.completion_tokens_sum"

    # thinking_tokens_sum is None -> truthful zero, no fake floor.
    reasoning_load = by_name["reasoning_load"]
    assert reasoning_load.raw_value == 0.0
    assert reasoning_load.source == "reasoning_activity.none"


def test_seed_v4_dark_projections_are_truthful_zero() -> None:
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
        trajectory_projection=None,
        reasoning_activity_projection=None,
    )
    by_name = {f.name: f for f in payload.features}

    for name in ("execution_load", "reasoning_present", "reasoning_load"):
        feat = by_name[name]
        assert feat.raw_value == 0.0
        assert feat.source.endswith(".none")

    recall = by_name["recall_gate_fired"]
    assert recall.raw_value == 0.0
    assert recall.source == "execution_trajectory.none"


def test_seed_v4_reasoning_load_negative_thinking_tokens_never_raises() -> None:
    """A malformed/negative thinking_tokens_sum must degrade to a truthful
    zero, not raise from math.log1p (domain error for values <= -1)."""
    reasoning_activity_projection = {
        "call_count": 3,
        "reasoning_present_rate": 0.1,
        "completion_tokens_sum": 0,
        "thinking_tokens_sum": -5,
    }
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
        trajectory_projection=None,
        reasoning_activity_projection=reasoning_activity_projection,
    )
    by_name = {f.name: f for f in payload.features}
    reasoning_load = by_name["reasoning_load"]
    assert reasoning_load.raw_value == 0.0
    assert reasoning_load.source == "reasoning_activity.none"


def test_seed_v4_reasoning_load_bool_thinking_tokens_not_treated_as_number() -> None:
    """bool is a subclass of int in Python; a JSON `true` for
    thinking_tokens_sum must not be treated as a real positive count."""
    reasoning_activity_projection = {
        "call_count": 3,
        "reasoning_present_rate": 0.1,
        "completion_tokens_sum": 0,
        "thinking_tokens_sum": True,
    }
    payload, _, _ = inner_state.build_inner_state_features(
        _sample_self_state(),
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
        trajectory_projection=None,
        reasoning_activity_projection=reasoning_activity_projection,
    )
    by_name = {f.name: f for f in payload.features}
    reasoning_load = by_name["reasoning_load"]
    assert reasoning_load.raw_value == 0.0
    assert reasoning_load.source == "reasoning_activity.none"


def test_seed_v4_execution_load_step_count_fallback() -> None:
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    trajectory_projection = {
        "runs": {
            "a": {
                "reasoning_present": True,
                "recall_observed": True,
                "step_count": 7,
                "failed_step_count": 0,
                "pressure_hints": {},
                "last_updated_at": now.isoformat(),
            },
            "b": {
                "reasoning_present": False,
                "recall_observed": False,
                "step_count": 3,
                "failed_step_count": 0,
                "pressure_hints": {},
                "last_updated_at": now.isoformat(),
            },
        }
    }
    ss = _sample_self_state()
    ss.generated_at = now
    payload, _, _ = inner_state.build_inner_state_features(
        ss,
        scaler,
        features_version="seed-v4",
        grammar_degraded=False,
        trajectory_projection=trajectory_projection,
        reasoning_activity_projection=None,  # dark -> fall back to step-count
        exec_trajectory_max_age_sec=120,
    )
    by_name = {f.name: f for f in payload.features}
    execution_load = by_name["execution_load"]
    assert execution_load.raw_value == round(math.log1p(10.0), 4)
    assert execution_load.source == "execution_trajectory.step_count_fallback"

    # recall_gate_fired should also read from the same active runs.
    recall = by_name["recall_gate_fired"]
    assert recall.raw_value == 1.0
    assert recall.source == "execution_trajectory.runs.*.recall_gate_fired"


def test_seed_v3_cognitive_features_unchanged_after_refactor() -> None:
    """Regression: the _recall_gate_fired/_active_trajectory_runs factoring
    must not change seed-v3's cognitive_features_from_trajectory output."""
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    projection = {
        "runs": {
            "a": {
                "reasoning_present": True,
                "recall_observed": True,
                "step_count": 4,
                "failed_step_count": 1,
                "pressure_hints": {"execution_friction": 0.2},
                "last_updated_at": now.isoformat(),
            }
        }
    }
    feats = inner_state.cognitive_features_from_trajectory(
        projection, now=now, max_age_sec=120
    )
    by_name = {f.name: f for f in feats}
    assert by_name["recall_gate_fired"].raw_value == 1.0
    assert by_name["reasoning_present"].raw_value == 1.0
    assert by_name["exec_step_fail_rate"].raw_value == round(1 / 4, 4)
    assert by_name["execution_friction"].raw_value == 0.2
    for feat in feats:
        assert feat.source == f"execution_trajectory.runs.*.{feat.name}"


def test_seed_v3_cognitive_features_none_case_unchanged() -> None:
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
    feats = inner_state.cognitive_features_from_trajectory(None, now=now, max_age_sec=120)
    assert len(feats) == 4
    for feat in feats:
        assert feat.raw_value == 0.0
        assert feat.source == "execution_trajectory.none"


def test_fit_phi_encoder_resolves_seed_v4_names() -> None:
    """scripts/fit_phi_encoder.py dynamically loads this same inner_state.py
    module; confirm its wrapper resolves the same 8-name seed-v4 list."""
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    fit_script = repo_root / "scripts" / "fit_phi_encoder.py"
    module_name = "fit_phi_encoder_test_v4"
    spec = importlib.util.spec_from_file_location(module_name, fit_script)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: fit_phi_encoder.py's dataclasses
    # resolve annotations via sys.modules[cls.__module__], which raises
    # AttributeError on None if the module was never registered.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    names = module.input_features_for_version("seed-v4", legacy_corpus=False)
    assert names == [
        "agency_readiness",
        "execution_pressure",
        "reasoning_pressure",
        "overall_intensity",
        "recall_gate_fired",
        "reasoning_present",
        "execution_load",
        "reasoning_load",
    ]
