"""Task 2: deviation gate fires on change, not presence."""
from __future__ import annotations

from orion.autonomy.deviation_gate import DeviationGate


def _gate() -> DeviationGate:
    return DeviationGate(alpha=0.1, z_threshold=1.5, sigma_floor=0.02,
                         impulse_k=0.25, warmup=5)


def test_cold_start_mints_nothing() -> None:
    gate = _gate()
    impulses = [gate.observe("biometrics_state", "homeostasis", 0.8, worse="down")
                for _ in range(5)]
    assert all(i == 0.0 for i in impulses)


def test_steady_input_settles_to_zero() -> None:
    """A steady stream (the scene_state flood) mints ~0 after warm-up."""
    gate = _gate()
    total = 0.0
    for _ in range(200):
        total += gate.observe("scene_state", "salience", 0.5, worse="up")
    # Tiny float jitter is fine; the flood must not accumulate real impulse.
    assert total < 1e-6, total


def test_real_drop_impulses() -> None:
    """homeostasis 0.82 steady, then a real drop to 0.55 → sized impulse
    (worse='down' means falling is bad)."""
    gate = _gate()
    for _ in range(30):
        gate.observe("biometrics_state", "homeostasis", 0.82, worse="down")
    impulse = gate.observe("biometrics_state", "homeostasis", 0.55, worse="down")
    assert impulse > 0.0, impulse


def test_wrong_direction_no_impulse() -> None:
    """A rise when only a fall is 'worse' mints nothing."""
    gate = _gate()
    for _ in range(30):
        gate.observe("biometrics_state", "homeostasis", 0.55, worse="down")
    impulse = gate.observe("biometrics_state", "homeostasis", 0.9, worse="down")
    assert impulse == 0.0, impulse


def test_sigma_floor_prevents_blowup() -> None:
    """A perfectly constant series (var→0) then a small step does not explode."""
    gate = _gate()
    for _ in range(50):
        gate.observe("mesh_health", "level", 1.0, worse="down")
    impulse = gate.observe("mesh_health", "level", 0.98, worse="down")
    # Bounded by impulse_k * excess * confidence; excess is finite because
    # sigma is floored, not zero.
    assert 0.0 <= impulse < 1.0, impulse


def test_confidence_scales_impulse() -> None:
    gate_hi = _gate()
    gate_lo = _gate()
    for _ in range(30):
        gate_hi.observe("spark_signal", "coherence", 0.8, worse="down")
        gate_lo.observe("spark_signal", "coherence", 0.8, worse="down")
    hi = gate_hi.observe("spark_signal", "coherence", 0.4, confidence=1.0, worse="down")
    lo = gate_lo.observe("spark_signal", "coherence", 0.4, confidence=0.3, worse="down")
    assert hi > lo > 0.0, (hi, lo)


def test_non_finite_degrades() -> None:
    gate = _gate()
    assert gate.observe("x", "y", float("nan"), worse="up") == 0.0
    assert gate.observe("x", "y", float("inf"), worse="up") == 0.0
    assert gate.observe("x", "y", "not a number", worse="up") == 0.0  # type: ignore[arg-type]
