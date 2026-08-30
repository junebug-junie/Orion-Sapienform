from __future__ import annotations

import pytest

from orion.grammar.atom_signals import (
    clamp01,
    uncertainty_for_cortex_exec_atom,
    uncertainty_from_abs_zscore,
    uncertainty_from_backpressure,
    uncertainty_from_catalog_drift,
    uncertainty_from_induction_volatility,
    uncertainty_from_route_arbitration,
    uncertainty_from_sample_mismatch,
    uncertainty_from_telemetry_error_rate,
)
from orion.schemas.telemetry.biometrics import (
    BiometricsInductionMetricV1,
    BiometricsInductionV1,
)


def test_clamp01() -> None:
    assert clamp01(-0.1) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.4) == pytest.approx(0.4)


def test_telemetry_error_rate() -> None:
    assert uncertainty_from_telemetry_error_rate(0.25) == pytest.approx(0.25)


def test_abs_zscore_no_baseline() -> None:
    assert uncertainty_from_abs_zscore(None) == pytest.approx(0.75)


def test_abs_zscore_three_sigma_caps_at_one() -> None:
    assert uncertainty_from_abs_zscore(9.0) == pytest.approx(1.0)


def test_sample_mismatch_ratio() -> None:
    assert uncertainty_from_sample_mismatch(2, 5) == pytest.approx(0.4)


def test_backpressure_below_threshold() -> None:
    assert uncertainty_from_backpressure(100, 400) == pytest.approx(0.0625)


def test_catalog_drift() -> None:
    assert uncertainty_from_catalog_drift(3, 10) == pytest.approx(0.3)


def test_induction_volatility_max() -> None:
    induction = BiometricsInductionV1(
        metrics={
            "cpu": BiometricsInductionMetricV1(volatility=0.2),
            "gpu": BiometricsInductionMetricV1(volatility=0.55),
        }
    )
    assert uncertainty_from_induction_volatility(induction) == pytest.approx(0.55)


def test_route_arbitration_mind_skipped() -> None:
    u = uncertainty_from_route_arbitration(
        {
            "execution_lane": "spark",
            "execution_lane_reason": "lane_routing_disabled",
            "mind_requested": False,
            "mind_skip_reason": "mind_enabled_not_true",
            "output_mode": "concise",
        }
    )
    assert u >= 0.35


def test_cortex_exec_failure_marker_high_uncertainty() -> None:
    u = uncertainty_for_cortex_exec_atom(
        atom_type="uncertainty_marker",
        semantic_role="exec_step_failed",
        confidence=0.9,
    )
    assert u == pytest.approx(0.85)
