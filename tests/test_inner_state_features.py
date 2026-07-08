import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "spark_inner_state",
    Path(__file__).resolve().parents[1]
    / "services" / "orion-spark-introspector" / "app" / "inner_state.py",
)
inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inner_state)


def test_robust_scale_zero_when_iqr_degenerate() -> None:
    assert inner_state.robust_scale(0.5, median=0.5, iqr=0.0) == 0.0


def test_robust_scale_standardizes() -> None:
    # value one IQR above median -> ~1.0
    assert inner_state.robust_scale(2.0, median=1.0, iqr=1.0) == 1.0
    assert inner_state.robust_scale(0.0, median=1.0, iqr=1.0) == -1.0


def test_rolling_scaler_centers_after_history() -> None:
    s = inner_state.RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s.observe("cpu", v)
    scaled = s.scale("cpu", 0.5)  # 0.5 is the median of the window
    assert abs(scaled) < 1e-6


def test_rolling_scaler_saturated_input_does_not_explode() -> None:
    s = inner_state.RollingRobustScaler(maxlen=16)
    for v in [0.0, 0.1, 0.2, 0.3, 0.4]:
        s.observe("pressure", v)
    # a single saturated 1.0 is scaled to a bounded value, not a raw 1.0 dominating
    scaled = s.scale("pressure", 1.0)
    assert scaled <= inner_state.SCALE_CLIP


class _Dim:
    def __init__(self, score: float) -> None:
        self.score = score


class _FakeSelfState:
    def __init__(self, dims: dict, **kw) -> None:
        self.dimensions = {k: _Dim(v) for k, v in dims.items()}
        self.dimension_trajectory = kw.get("traj", {})
        self.self_state_id = kw.get("sid", "self.state:tick_x:policy.v1")
        self.generated_at = kw.get("gen")
        self.overall_intensity = kw.get("intensity", 0.4)
        self.overall_condition = kw.get("condition", "steady")
        self.trajectory_condition = kw.get("trajectory", "stable")
        self.dominant_field_channels = kw.get("infra", {})


def _pinned_state():
    from datetime import datetime, timezone
    return _FakeSelfState(
        {
            "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
            "execution_pressure": 0.0, "reasoning_pressure": 0.05,
            "resource_pressure": 1.0, "reliability_pressure": 1.0,
            "continuity_pressure": 0.0, "introspection_pressure": 0.0,
            "social_pressure": 0.0, "uncertainty": 0.0, "policy_pressure": 0.0,
        },
        gen=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        infra={"contract_pressure": 1.0, "catalog_drift_pressure": 1.0, "bus_health": 1.0},
    )


def test_dead_signals_excluded_from_features() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _felt, _streak = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1",
        grammar_degraded=False, prev_felt=None, prev_headline=None,
        degenerate_streak=0, degenerate_limit=20,
    )
    names = {f.name for f in payload.features}
    assert "policy_pressure" not in names
    assert "uncertainty" not in names
    # infra signals present ONLY in the infra sub-vector
    infra_names = {f.name for f in payload.infra}
    assert "contract_pressure" in infra_names
    assert "contract_pressure" not in names


def test_features_carry_raw_scaled_source() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _f, _s = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1",
        grammar_degraded=False, prev_felt=None, prev_headline=None,
        degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.features, "expected felt features"
    for f in payload.features:
        assert f.source.startswith("self_state.")


def test_honest_headline_not_floored_on_pinned_state() -> None:
    # The live pinned-0.01 state: reliability & resource maxed, but coherence,
    # field_intensity, agency healthy. Old geometric mean -> 0.01. Honest -> >0.5.
    raw = {
        "coherence": 1.0, "field_intensity": 1.0, "agency_readiness": 0.41,
        "execution_pressure": 0.0, "reasoning_pressure": 0.05,
        "resource_pressure": 1.0, "reliability_pressure": 1.0,
        "continuity_pressure": 0.0, "social_pressure": 0.0,
    }
    h = inner_state.honest_headline(raw)
    assert h > 0.5, f"honest headline must not floor on pinned infra (got {h})"


def test_single_saturated_pressure_does_not_collapse_headline() -> None:
    healthy = {
        "coherence": 0.9, "field_intensity": 0.8, "agency_readiness": 0.8,
        "execution_pressure": 0.1, "reasoning_pressure": 0.1,
        "resource_pressure": 0.1, "reliability_pressure": 0.1,
        "continuity_pressure": 0.1, "social_pressure": 0.1,
    }
    base = inner_state.honest_headline(healthy)
    poisoned = dict(healthy, reliability_pressure=1.0)
    after = inner_state.honest_headline(poisoned)
    assert after > 0.4, "one maxed pressure must not floor the headline"
    assert after < base, "but it should still lower it somewhat"


def test_degeneracy_freeze_after_streak() -> None:
    from datetime import datetime, timezone
    scaler = inner_state.RollingRobustScaler(maxlen=64)
    st = _pinned_state()
    prev_felt = None
    prev_headline = None
    streak = 0
    payload = None
    for _ in range(21):  # limit=20 -> frozen on the 21st identical tick
        payload, prev_felt, streak = inner_state.build_inner_state_features(
            st, scaler, features_version="seed-v1", grammar_degraded=False,
            prev_felt=prev_felt, prev_headline=prev_headline,
            degenerate_streak=streak, degenerate_limit=20,
        )
        prev_headline = payload.headline
    assert payload.phi_health == "frozen"
    assert payload.phi_degenerate_streak >= 20


def test_grammar_degraded_forces_frozen() -> None:
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _f, _s = inner_state.build_inner_state_features(
        _pinned_state(), scaler, features_version="seed-v1", grammar_degraded=True,
        prev_felt=None, prev_headline=0.62, degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.phi_health == "frozen"
    assert payload.headline == 0.62


def test_corpus_sink_appends_jsonl(tmp_path) -> None:
    import importlib.util as _u
    from pathlib import Path as _P
    spec = _u.spec_from_file_location(
        "spark_inner_sink",
        _P(__file__).resolve().parents[1]
        / "services" / "orion-spark-introspector" / "app" / "inner_state_sink.py",
    )
    sink_mod = _u.module_from_spec(spec)
    spec.loader.exec_module(sink_mod)

    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1

    path = tmp_path / "corpus.jsonl"
    sink = sink_mod.InnerStateCorpusSink(str(path))
    p = InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc), headline=0.5)
    sink.append(p)
    sink.append(p)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["headline"] == 0.5


def test_corpus_sink_disabled_when_no_path() -> None:
    import importlib.util as _u
    from pathlib import Path as _P
    spec = _u.spec_from_file_location(
        "spark_inner_sink2",
        _P(__file__).resolve().parents[1]
        / "services" / "orion-spark-introspector" / "app" / "inner_state_sink.py",
    )
    sink_mod = _u.module_from_spec(spec)
    spec.loader.exec_module(sink_mod)
    sink = sink_mod.InnerStateCorpusSink("")
    assert sink.enabled is False
    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
    sink.append(InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc)))  # no-op, no raise
