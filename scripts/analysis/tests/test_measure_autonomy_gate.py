"""Deterministic unit tests for the pure layer of measure_autonomy_gate.

No DB, no bus, no network. Every test builds in-memory fixtures and exercises
ONLY the pure functions (window classification, metrics, verdict rules).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Load the sibling module by path so this test file needs no package scaffolding.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_autonomy_gate.py"
_spec = importlib.util.spec_from_file_location("measure_autonomy_gate", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec so dataclasses can resolve string annotations (PEP 563).
sys.modules["measure_autonomy_gate"] = mod
_spec.loader.exec_module(mod)


UTC = timezone.utc
BASE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)


def _self_state(offset_sec: float, scores: dict[str, float], trajectory: dict[str, float], *,
                condition: str = "stable", surprise: float = 0.0) -> "mod.SelfStateRecord":
    return mod.SelfStateRecord(
        generated_at=BASE + timedelta(seconds=offset_sec),
        dimensions=dict(scores),
        dimension_trajectory=dict(trajectory),
        trajectory_condition=condition,
        overall_surprise=surprise,
        resource_pressure=scores.get("resource_pressure", 0.0),
    )


# ---------------------------------------------------------------------------
# 1. Flat silent buckets -> verdict (a) NO-GO
# ---------------------------------------------------------------------------
def test_verdict_a_flat_silent_is_no_go():
    silent_rows = [
        _self_state(i * 10, {"coherence": 0.5, "resource_pressure": 0.1}, {"coherence": 0.0, "resource_pressure": 0.0})
        for i in range(6)
    ]
    busy_rows = [
        _self_state(1000 + i * 10, {"coherence": 0.4, "resource_pressure": 0.2},
                    {"coherence": 0.05, "resource_pressure": 0.05})
        for i in range(6)
    ]
    silent = mod.compute_self_state_metrics(silent_rows)
    busy = mod.compute_self_state_metrics(busy_rows)
    assert silent.median_abs_trajectory == 0.0
    assert mod.verdict_drift(silent, busy) == "NO-GO"


# ---------------------------------------------------------------------------
# 2. Silent buckets with real movement + variance -> verdict (a) GO
# ---------------------------------------------------------------------------
def test_verdict_a_moving_silent_is_go():
    # Silent rows drift: trajectory clearly above 0.03 and scores vary a lot.
    silent_scores = [0.1, 0.5, 0.2, 0.7, 0.3, 0.6]
    silent_rows = [
        _self_state(i * 10, {"coherence": s, "resource_pressure": s / 2.0},
                    {"coherence": 0.2, "resource_pressure": 0.15}, condition="improving", surprise=0.4)
        for i, s in enumerate(silent_scores)
    ]
    # Busy rows almost flat -> tiny variance so the ratio test passes.
    busy_rows = [
        _self_state(2000 + i * 10, {"coherence": 0.50 + 0.001 * i, "resource_pressure": 0.2},
                    {"coherence": 0.0, "resource_pressure": 0.0})
        for i in range(6)
    ]
    silent = mod.compute_self_state_metrics(silent_rows)
    busy = mod.compute_self_state_metrics(busy_rows)
    assert silent.median_abs_trajectory >= mod.DRIFT_MIN_MEDIAN_ABS_TRAJECTORY
    assert silent.dim_score_variance >= mod.DRIFT_VARIANCE_RATIO * busy.dim_score_variance
    assert mod.verdict_drift(silent, busy) == "GO"


def test_verdict_a_busy_zero_variance_passes_when_silent_moves():
    silent_rows = [
        _self_state(i * 10, {"coherence": v}, {"coherence": 0.2})
        for i, v in enumerate([0.1, 0.6, 0.2, 0.7])
    ]
    silent = mod.compute_self_state_metrics(silent_rows)
    busy = mod.SelfStateMetrics()  # empty busy -> variance 0
    assert busy.dim_score_variance == 0.0
    assert mod.verdict_drift(silent, busy) == "GO"


# ---------------------------------------------------------------------------
# 3. Rare co-activation + flat pressure -> verdict (b) NO-GO
# ---------------------------------------------------------------------------
def test_verdict_b_rare_coactivation_is_no_go():
    # 20 audits with 1 active drive, 1 audit with 2 -> 1/21 < 10% co-active.
    drive = mod.drive_stats_from_histogram({1: 20, 2: 1})
    pressure_rows = [_self_state(i, {"resource_pressure": 0.05}, {}) for i in range(20)]
    pressure = mod.compute_resource_pressure_stats(pressure_rows)
    assert drive.coactivation_frac < mod.COACTIVATION_MIN_FRAC
    assert pressure.frac_gt_level < mod.RESOURCE_PRESSURE_MIN_FRAC
    assert mod.verdict_economy(drive, pressure) == "NO-GO"


# ---------------------------------------------------------------------------
# 4. Frequent co-activation + real pressure -> verdict (b) GO
# ---------------------------------------------------------------------------
def test_verdict_b_frequent_coactivation_is_go():
    # 10 audits with 2 active drives, 10 with 1 -> 50% co-active >= 10%.
    drive = mod.drive_stats_from_histogram({2: 10, 1: 10})
    # Resource pressure: 3 of 20 >= 0.3 -> 15% >= 5%.
    pressure_rows = [
        _self_state(i, {"resource_pressure": 0.5 if i < 3 else 0.1}, {})
        for i in range(20)
    ]
    pressure = mod.compute_resource_pressure_stats(pressure_rows)
    assert drive.coactivation_frac >= mod.COACTIVATION_MIN_FRAC
    assert pressure.frac_gt_level >= mod.RESOURCE_PRESSURE_MIN_FRAC
    assert mod.verdict_economy(drive, pressure) == "GO"


def test_drive_stats_from_histogram():
    stats = mod.drive_stats_from_histogram({0: 100, 2: 5, 3: 5})
    assert stats.record_count == 110
    assert stats.coactivation_frac == pytest.approx(10 / 110)
    assert stats.concurrent_active_hist == {0: 100, 2: 5, 3: 5}

    empty = mod.drive_stats_from_histogram({})
    assert empty.record_count == 0
    assert empty.coactivation_frac == 0.0
    assert empty.concurrent_active_hist == {}


# ---------------------------------------------------------------------------
# 5. Window classification: busy vs silent + boundary bucketing
# ---------------------------------------------------------------------------
def test_window_classification_busy_and_silent():
    window_start = BASE
    # A receipt inside bucket 0 -> that bucket is busy; bucket 1 stays silent.
    receipts = [window_start + timedelta(seconds=30)]
    buckets = mod.build_bucket_activity(
        receipt_timestamps=receipts,
        turn_timestamps=[],
        window_start=window_start,
    )
    assert mod.classify_bucket(buckets[0]) == "busy"
    # An unseen bucket is silent by definition.
    assert mod.bucket_class_for(window_start + timedelta(seconds=mod.WINDOW_SEC + 5), buckets, window_start) == "silent"


def test_bucket_boundary_timestamp():
    window_start = BASE
    assert mod.bucket_index(window_start, window_start) == 0
    assert mod.bucket_index(window_start + timedelta(seconds=mod.WINDOW_SEC - 1), window_start) == 0
    # Exactly on the boundary lands in the next bucket.
    assert mod.bucket_index(window_start + timedelta(seconds=mod.WINDOW_SEC), window_start) == 1
    assert mod.bucket_index(window_start + timedelta(seconds=2 * mod.WINDOW_SEC), window_start) == 2


def test_empty_bucket_is_silent():
    assert mod.classify_bucket(mod.BucketActivity()) == "silent"
    assert mod.classify_bucket(mod.BucketActivity(receipt_count=1)) == "busy"
    assert mod.classify_bucket(mod.BucketActivity(turn_count=1)) == "busy"


# ---------------------------------------------------------------------------
# 6. Empty inputs degrade gracefully (no raise, well-defined result)
# ---------------------------------------------------------------------------
def test_empty_metrics_do_not_raise():
    ss = mod.compute_self_state_metrics([])
    assert ss.row_count == 0
    assert ss.mean_abs_trajectory == 0.0
    assert ss.median_abs_trajectory == 0.0
    assert ss.dim_score_variance == 0.0
    assert ss.nonstable_frac == 0.0
    assert ss.mean_surprise == 0.0

    pressure = mod.compute_resource_pressure_stats([])
    assert pressure.row_count == 0
    assert pressure.median is None
    assert pressure.p90 is None
    assert pressure.frac_gt_level == 0.0

    drive = mod.drive_stats_from_histogram({})
    assert drive.record_count == 0
    assert drive.coactivation_frac == 0.0
    assert drive.concurrent_active_hist == {}

    assert mod.dim_score_variance([]) == 0.0
    assert mod.per_row_abs_trajectory([]) == []
    assert mod._percentile([], 0.9) is None

    # Verdicts on empty metrics are well-defined NO-GO.
    assert mod.verdict_drift(ss, ss) == "NO-GO"
    assert mod.verdict_economy(drive, pressure) == "NO-GO"


def test_percentile_and_median():
    rows = [_self_state(i, {"resource_pressure": v}, {}) for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.9])]
    stats = mod.compute_resource_pressure_stats(rows)
    assert stats.median == 0.3
    assert stats.p90 == pytest.approx(0.7, abs=1e-9)
    assert stats.frac_gt_level == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------
# 7. Durable drive-audit source (Fuseki SPARQL) — pure query + parse layer
# ---------------------------------------------------------------------------
def test_build_drive_coactivation_histogram_sparql_shape():
    q = mod.build_drive_coactivation_histogram_sparql(BASE)
    assert mod.AUTONOMY_DRIVES_GRAPH in q
    assert "orion:DriveAudit" in q
    # Assessment-level boolean shape, counted DISTINCT (bounded by drive keys).
    assert "hasDriveAssessment" in q
    assert "orion:driveActive true" in q
    assert "COUNT(DISTINCT ?activeDa)" in q
    # ts is multi-valued -> windowed via FILTER EXISTS so it cannot fan out the
    # assessment count. ?ts must NOT be bound in the aggregation group.
    assert "FILTER EXISTS" in q
    assert "orion:timestamp" in q
    assert f'FILTER(?ts >= "{BASE.isoformat()}"^^xsd:dateTime)' in q
    # Nested aggregation: inner reduces per audit, outer bins into a histogram.
    assert "GROUP BY ?audit" in q
    assert "GROUP BY ?activeCount" in q
    # Old markers (fan-out / unbound-SUM / stale predicate) must be gone.
    assert "SUM(IF" not in q
    assert "BOUND(?act)" not in q
    assert "highlightsActiveDrive" not in q
    assert "COUNT(DISTINCT ?d)" not in q


def test_parse_sparql_histogram_bindings_happy():
    bindings = [
        {"activeCount": {"value": "0"}, "audits": {"value": "444734"}},
        {"activeCount": {"value": "2"}, "audits": {"value": "20"}},
        {"activeCount": {"value": "3.0"}, "audits": {"value": "31"}},
        "not-a-dict",  # skipped, no crash
        {"activeCount": {"value": "junk"}, "audits": {"value": "5"}},  # bad count -> skipped
        {"activeCount": {"value": "1"}},  # missing audits -> skipped
    ]
    hist = mod.parse_sparql_histogram_bindings(bindings)
    assert hist == {0: 444734, 2: 20, 3: 31}
    # Feeds the pure drive-stats path.
    stats = mod.drive_stats_from_histogram(hist)
    assert stats.record_count == 444785
    assert stats.coactivation_frac == pytest.approx(51 / 444785)


def test_parse_sparql_histogram_bindings_retains_zero_bucket():
    # The query now emits an explicit bound 0 bucket for assessment-less audits
    # (BOUND(?act) guard) rather than dropping them; lock in that the parser
    # keeps a bound-0 row so those audits stay in the record_count denominator.
    bindings = [
        {"activeCount": {"value": "0"}, "audits": {"value": "5"}},
        {"activeCount": {"value": "2"}, "audits": {"value": "3"}},
    ]
    hist = mod.parse_sparql_histogram_bindings(bindings)
    assert hist == {0: 5, 2: 3}
    stats = mod.drive_stats_from_histogram(hist)
    assert stats.record_count == 8  # the 5 zero-active audits are counted
    assert stats.coactivation_frac == pytest.approx(3 / 8)


def test_parse_sparql_histogram_bindings_degrades():
    # Empty bindings -> empty dict, no raise.
    assert mod.parse_sparql_histogram_bindings([]) == {}
