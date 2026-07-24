"""Deterministic unit tests for measure_phase3_biometrics_drive_shadow_comparison.py.

No DB -- all pure functions, exercised directly with synthetic payload dicts,
same pattern as test_measure_ast_hot_reducer.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_phase3_biometrics_drive_shadow_comparison.py"
_spec = importlib.util.spec_from_file_location(
    "measure_phase3_biometrics_drive_shadow_comparison", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_phase3_biometrics_drive_shadow_comparison"] = mod
_spec.loader.exec_module(mod)

UTC = timezone.utc
BASE = datetime(2026, 7, 24, 0, 0, 0, tzinfo=UTC)


def _field_state_payload(pe: float | None) -> dict:
    if pe is None:
        return {"node_vectors": {}}
    return {"node_vectors": {"node:substrate.biometrics": {"prediction_error": pe}}}


def _drive_audit_payload(*, capability: float | None, continuity: float | None, dominant_drive: str | None) -> dict:
    pressures = {}
    if capability is not None:
        pressures["capability"] = capability
    if continuity is not None:
        pressures["continuity"] = continuity
    return {
        "drive_pressures": pressures,
        "dominant_drive": dominant_drive,
        "active_drives": [],
        "tension_kinds": ["tension.signal.v1"],
    }


class TestExtractBiometricsPredictionError:
    def test_extracts_present_value(self) -> None:
        assert mod.extract_biometrics_prediction_error(_field_state_payload(0.42)) == 0.42

    def test_missing_node_yields_none(self) -> None:
        assert mod.extract_biometrics_prediction_error({"node_vectors": {}}) is None

    def test_missing_node_vectors_yields_none(self) -> None:
        assert mod.extract_biometrics_prediction_error({}) is None

    def test_malformed_value_yields_none(self) -> None:
        payload = {"node_vectors": {"node:substrate.biometrics": {"prediction_error": "not-a-number"}}}
        assert mod.extract_biometrics_prediction_error(payload) is None


class TestAlignDriveAuditsToFieldState:
    def test_joins_nearest_preceding_field_state(self) -> None:
        field_state_rows = [
            (BASE, _field_state_payload(0.1)),
            (BASE + timedelta(seconds=5), _field_state_payload(0.5)),
        ]
        drive_audit_rows = [
            (BASE + timedelta(seconds=6), _drive_audit_payload(capability=0.3, continuity=0.2, dominant_drive="capability")),
        ]
        ticks = mod.align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
        assert len(ticks) == 1
        assert ticks[0].biometrics_prediction_error == 0.5
        assert ticks[0].capability_pressure == 0.3
        assert ticks[0].dominant_drive == "capability"

    def test_stale_field_state_beyond_max_join_staleness_is_none(self) -> None:
        field_state_rows = [(BASE, _field_state_payload(0.1))]
        drive_audit_rows = [
            (
                BASE + timedelta(seconds=mod.MAX_JOIN_STALENESS_SEC + 1),
                _drive_audit_payload(capability=0.3, continuity=0.2, dominant_drive="capability"),
            ),
        ]
        ticks = mod.align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
        assert len(ticks) == 1
        assert ticks[0].biometrics_prediction_error is None

    def test_no_preceding_field_state_row_yields_none(self) -> None:
        field_state_rows = [(BASE + timedelta(seconds=10), _field_state_payload(0.1))]
        drive_audit_rows = [(BASE, _drive_audit_payload(capability=0.3, continuity=None, dominant_drive="capability"))]
        ticks = mod.align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
        assert len(ticks) == 1
        assert ticks[0].biometrics_prediction_error is None

    def test_missing_pressures_yield_none_not_zero(self) -> None:
        field_state_rows = [(BASE, _field_state_payload(0.2))]
        drive_audit_rows = [(BASE + timedelta(seconds=1), _drive_audit_payload(capability=None, continuity=None, dominant_drive="relational"))]
        ticks = mod.align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
        assert ticks[0].capability_pressure is None
        assert ticks[0].continuity_pressure is None

    def test_each_drive_audit_joins_its_own_nearest_preceding_row(self) -> None:
        field_state_rows = [
            (BASE, _field_state_payload(0.1)),
            (BASE + timedelta(seconds=20), _field_state_payload(0.9)),
        ]
        drive_audit_rows = [
            (BASE + timedelta(seconds=5), _drive_audit_payload(capability=0.1, continuity=0.1, dominant_drive="capability")),
            (BASE + timedelta(seconds=25), _drive_audit_payload(capability=0.1, continuity=0.1, dominant_drive="capability")),
        ]
        ticks = mod.align_drive_audits_to_field_state(drive_audit_rows, field_state_rows)
        assert ticks[0].biometrics_prediction_error == 0.1
        assert ticks[1].biometrics_prediction_error == 0.9


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        r = mod.pearson_correlation(xs, ys)
        assert r is not None
        assert abs(r - 1.0) < 1e-9

    def test_perfect_negative_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [8.0, 6.0, 4.0, 2.0]
        r = mod.pearson_correlation(xs, ys)
        assert r is not None
        assert abs(r - (-1.0)) < 1e-9

    def test_fewer_than_two_points_yields_none(self) -> None:
        assert mod.pearson_correlation([], []) is None
        assert mod.pearson_correlation([1.0], [1.0]) is None

    def test_zero_variance_series_yields_none(self) -> None:
        assert mod.pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_mismatched_lengths_yields_none(self) -> None:
        assert mod.pearson_correlation([1.0, 2.0], [1.0]) is None


class TestComputeCorrelation:
    def test_only_uses_ticks_with_all_three_values_present(self) -> None:
        ticks = [
            mod.AlignedTick(BASE, 0.1, 0.2, 0.1, "capability", 1.0),
            mod.AlignedTick(BASE, None, 0.2, 0.1, "capability", None),
            mod.AlignedTick(BASE, 0.3, None, 0.1, "capability", 1.0),
            mod.AlignedTick(BASE, 0.5, 0.4, 0.2, "capability", 1.0),
        ]
        r, n = mod.compute_correlation(ticks)
        assert n == 2

    def test_empty_ticks_yields_none_zero(self) -> None:
        r, n = mod.compute_correlation([])
        assert r is None
        assert n == 0


class TestSplitComparison:
    def test_groups_by_biometrics_fed_drives(self) -> None:
        ticks = [
            mod.AlignedTick(BASE, 0.8, 0.5, 0.5, "capability", 1.0),
            mod.AlignedTick(BASE, 0.6, 0.5, 0.5, "continuity", 1.0),
            mod.AlignedTick(BASE, 0.1, 0.5, 0.5, "relational", 1.0),
            mod.AlignedTick(BASE, 0.2, 0.5, 0.5, "predictive", 1.0),
        ]
        fed, other = mod.split_comparison(ticks)
        assert fed.n == 2
        assert abs(fed.mean - 0.7) < 1e-9
        assert other.n == 2
        assert abs(other.mean - 0.15) < 1e-9

    def test_missing_dominant_drive_or_pe_excluded_from_both_groups(self) -> None:
        ticks = [
            mod.AlignedTick(BASE, None, 0.5, 0.5, "capability", None),
            mod.AlignedTick(BASE, 0.5, 0.5, 0.5, None, 1.0),
        ]
        fed, other = mod.split_comparison(ticks)
        assert fed.n == 0
        assert other.n == 0

    def test_empty_ticks_yields_empty_groups(self) -> None:
        fed, other = mod.split_comparison([])
        assert fed.n == 0
        assert fed.mean is None
        assert other.n == 0
        assert other.mean is None
