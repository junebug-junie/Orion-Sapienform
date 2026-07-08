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
