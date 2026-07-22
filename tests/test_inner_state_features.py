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


# 2026-07-22 (SelfStateV1 burn): _FakeSelfState/_pinned_state removed --
# build_inner_state_features() no longer takes a self-state object, and
# honest_headline() (tested below, pre-burn) no longer exists -- see
# services/orion-spark-introspector/app/inner_state.py's module docstring
# for why no principled non-self-state headline formula replaces it.
# test_dead_signals_excluded_from_features/test_features_carry_raw_scaled_source
# (tested FELT_DIMENSIONS/self_state.-prefixed sources) and
# test_honest_headline_not_floored_on_pinned_state/
# test_single_saturated_pressure_does_not_collapse_headline (tested
# honest_headline() directly) all tested removed functionality and are
# deleted, not rewritten. test_degeneracy_freeze_after_streak and
# test_grammar_degraded_forces_frozen below are rewritten against the
# surviving degeneracy-freeze mechanism, now keyed on the 4 real cognitive
# features instead of self-state's old 10-dimension felt_tuple.


def test_degeneracy_freeze_after_streak() -> None:
    from datetime import datetime, timezone
    scaler = inner_state.RollingRobustScaler(maxlen=64)
    now = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)
    prev_felt = None
    streak = 0
    payload = None
    for _ in range(21):  # limit=20 -> frozen on the 21st identical tick
        payload, prev_felt, streak = inner_state.build_inner_state_features(
            scaler, features_version="seed-v1", grammar_degraded=False,
            trajectory_projection=None, now=now,
            prev_felt=prev_felt, degenerate_streak=streak, degenerate_limit=20,
        )
    assert payload.phi_health == "frozen"
    assert payload.phi_degenerate_streak >= 20


def test_grammar_degraded_forces_frozen() -> None:
    from datetime import datetime, timezone
    scaler = inner_state.RollingRobustScaler(maxlen=8)
    payload, _f, _s = inner_state.build_inner_state_features(
        scaler, features_version="seed-v1", grammar_degraded=True,
        trajectory_projection=None,
        now=datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc),
        prev_felt=None, degenerate_streak=0, degenerate_limit=20,
    )
    assert payload.phi_health == "frozen"


def test_corpus_sink_appends_jsonl(tmp_path) -> None:
    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
    from orion.telemetry.corpus_sink import InnerStateCorpusSink

    path = tmp_path / "corpus.jsonl"
    sink = InnerStateCorpusSink(str(path))
    p = InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc), headline=0.5)
    sink.append(p)
    sink.append(p)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["headline"] == 0.5


def test_corpus_sink_disabled_when_no_path() -> None:
    from datetime import datetime, timezone
    from orion.schemas.telemetry.inner_state import InnerStateFeaturesV1
    from orion.telemetry.corpus_sink import InnerStateCorpusSink

    sink = InnerStateCorpusSink("")
    assert sink.enabled is False
    sink.append(InnerStateFeaturesV1(generated_at=datetime(2026, 7, 7, tzinfo=timezone.utc)))  # no-op, no raise


def test_corpus_sink_init_does_not_mkdir() -> None:
    from pathlib import Path as _P
    from orion.telemetry.corpus_sink import InnerStateCorpusSink

    deep = _P("/tmp/orion_phi_sink_init_test/deep/nested/corpus.jsonl")
    sink = InnerStateCorpusSink(str(deep))
    assert sink.enabled is True
    assert not deep.parent.exists()
