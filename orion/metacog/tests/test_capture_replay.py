from __future__ import annotations

import pytest

from orion.metacog.capture_replay import (
    acceptance_failures,
    analyze,
    proxy_for,
    spearman,
    tie_limited_ceiling,
)


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2], [1, 2]) is None
    # ties get average ranks
    assert spearman([0, 0, 1, 1], [1, 2, 3, 4]) == pytest.approx(0.894427, abs=1e-5)


def test_tie_limited_ceiling_equals_rho_for_monotone_assignment_and_exceeds_misordered():
    proxies = [0.5, 1.0, 1.7, 2.2, 3.0, 4.0]
    monotone = [0, 0, 1, 1, 2, 2]
    assert spearman(monotone, proxies) == pytest.approx(tie_limited_ceiling(monotone, proxies))
    misordered = [2, 0, 1, 1, 0, 2]
    assert spearman(misordered, proxies) < tie_limited_ceiling(misordered, proxies)


def test_proxy_for_shapes():
    assert proxy_for("telemetry_anomaly", {"recon_loss": 0.03, "threshold": 0.01})[1] == pytest.approx(3.0)
    assert proxy_for("transport", {"evidence_source": "rpc_health_snapshot", "timeout_count": 2}) == (
        "rpc_health:timeout_count",
        2.0,
    )
    assert proxy_for("transport", {"evidence_source": "rpc_health_snapshot", "timeout_count": 0,
                                   "success_latency_ms_p95": 10000, "latency_p95_threshold_ms": 5000}) == (
        "rpc_health:p95_over_threshold",
        2.0,
    )
    assert proxy_for("baseline", {}) is None


def _tel(ratio, day="2026-09-20"):
    return {
        "trigger_kind": "telemetry_anomaly",
        "reason": "r",
        "upstream": {"recon_loss": 0.01 * ratio, "threshold": 0.01, "top_channels": []},
        "timestamp": f"{day}T00:00:00",
    }


def test_analyze_and_acceptance_pass_on_monotone_rows():
    rows = [_tel(1.0 + i * 0.05) for i in range(60)]
    res = analyze(rows)
    assert res["severity_by_kind"]["telemetry_anomaly"]["critical"] > 0
    assert res["density_distinct_per_day"]["2026-09-20"]["new_distinct"] > 10
    assert acceptance_failures(res, min_rows_per_day=1) == []


def test_acceptance_flags_degenerate_density_day():
    rows = [_tel(1.2) for _ in range(200)]
    fails = acceptance_failures(analyze(rows), min_rows_per_day=1)
    assert any(f.startswith("check5") for f in fails)
    # a thin day is skipped, not failed
    assert not any(f.startswith("check5") for f in acceptance_failures(analyze(rows[:5]), min_rows_per_day=100))


def test_analyze_records_old_vs_new_when_joined():
    row = _tel(3.0)
    row["old_severity"] = "nominal"
    row["old_density_score"] = 0.25
    res = analyze([row])
    assert res["old_vs_new"]["telemetry_anomaly"] == {"nominal->critical": 1}
