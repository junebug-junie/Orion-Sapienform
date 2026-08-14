"""Fleet aggregation of raw physical measurements (ROADMAP B1 consumer).

The hard part is that ABSENT-IS-NOT-ZERO has to survive aggregation. Per-node it is enforced
by omitting the key; a naive `sum()` over nodes re-introduces exactly the error it prevented,
because circe has no reachable BMC and a fleet total computed from athena+atlas is a real
number missing an entire machine.

Fixture values are the live 2026-08-14 readings.
"""
from __future__ import annotations

import pytest

from orion.telemetry.biometrics_pipeline import (
    FLEET_MAX_KEYS,
    FLEET_SUM_KEYS,
    aggregate_fleet_measurements,
)

# Live 2026-08-14. circe has no BMC -> no chassis_watts, no fan.
LIVE = {
    "athena": {"chassis_watts": 390.0, "gpu_watts_total": 44.63, "gpu_count": 1.0,
               "cpu_cores": 80.0, "temp_c_max": 65.0, "fan_pct_max": 43.0,
               "load_1m": 31.84, "load_15m": 35.41},
    "atlas": {"chassis_watts": 273.0, "gpu_watts_total": 99.04, "gpu_count": 2.0,
              "cpu_cores": 96.0, "temp_c_max": 46.0, "fan_pct_max": 52.0,
              "load_1m": 0.65, "load_15m": 0.73},
    "circe": {"gpu_watts_total": 172.96, "gpu_count": 3.0, "cpu_cores": 72.0,
              "temp_c_max": 44.0, "load_1m": 0.0, "load_15m": 0.05},
}


# ------------------------------------------------------- the invariant


def test_a_partial_total_always_names_what_it_is_missing():
    """The load-bearing property. 390 + 273 = 663 W is real AND missing a whole machine."""
    totals, missing = aggregate_fleet_measurements(LIVE)
    assert totals["chassis_watts"] == pytest.approx(663.0)
    assert missing["chassis_watts"] == ["circe"]


def test_a_complete_total_reports_nothing_missing():
    totals, missing = aggregate_fleet_measurements(LIVE)
    assert totals["gpu_count"] == 6.0  # 1 + 2 + 3, every node reported
    assert "gpu_count" not in missing


def test_a_none_measurements_node_is_missing_for_every_key_not_a_zero_contributor():
    """A producer predating the field says NOTHING about the node -- it does not report zeros."""
    per_node = {"athena": LIVE["athena"], "atlas": None}
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["chassis_watts"] == 390.0          # not 390 + 0
    assert missing["chassis_watts"] == ["atlas"]
    assert missing["cpu_cores"] == ["atlas"]


def test_a_key_nobody_measured_is_absent_at_the_fleet_level_too():
    per_node = {"circe": LIVE["circe"]}
    totals, missing = aggregate_fleet_measurements(per_node)
    assert "chassis_watts" not in totals   # not 0.0
    assert "chassis_watts" not in missing  # and not reported as a key at all
    assert "fan_pct_max" not in totals


def test_empty_input_is_empty_output():
    assert aggregate_fleet_measurements({}) == ({}, {})


# ------------------------------------------------------- composition policy


def test_extensive_quantities_are_summed():
    totals, _ = aggregate_fleet_measurements(LIVE)
    # 44.63 + 99.04 + 172.96 = 316.63
    assert totals["gpu_watts_total"] == pytest.approx(316.63)
    assert totals["cpu_cores"] == 248.0  # 80 + 96 + 72


def test_intensive_quantities_are_maxed_not_summed():
    """Summing temperatures across machines is meaningless; 65 + 46 + 44 is not 155 C."""
    totals, _ = aggregate_fleet_measurements(LIVE)
    assert totals["temp_c_max"] == 65.0    # max, not 155.0
    assert totals["fan_pct_max"] == 52.0   # max of {43, 52}; circe reports none


def test_fan_max_names_the_node_that_could_not_report_it():
    _, missing = aggregate_fleet_measurements(LIVE)
    assert missing["fan_pct_max"] == ["circe"]


def test_load_averages_are_deliberately_not_aggregated():
    """A load average is relative to its own machine's core count (80 vs 96 vs 72), so it
    neither sums nor maxes meaningfully. Omitted rather than invented."""
    totals, missing = aggregate_fleet_measurements(LIVE)
    assert "load_1m" not in totals and "load_15m" not in totals
    assert "load_1m" not in missing


def test_unknown_keys_are_skipped_rather_than_guessed():
    totals, _ = aggregate_fleet_measurements({"a": {"something_new_hz": 5.0}})
    assert totals == {}


def test_the_two_policy_tables_do_not_overlap():
    """A key in both would take whichever branch ran first -- a silent, arbitrary choice."""
    assert not (set(FLEET_SUM_KEYS) & set(FLEET_MAX_KEYS))


# ------------------------------------------------------- hostile inputs


def test_unparseable_values_count_as_missing_not_as_zero():
    per_node = {"a": {"chassis_watts": 100.0}, "b": {"chassis_watts": "n/a"}}
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["chassis_watts"] == 100.0
    assert missing["chassis_watts"] == ["b"]


def test_booleans_do_not_slip_in_as_one_watt():
    per_node = {"a": {"chassis_watts": 100.0}, "b": {"chassis_watts": True}}
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["chassis_watts"] == 100.0
    assert missing["chassis_watts"] == ["b"]


def test_non_dict_node_payloads_are_missing_not_crashes():
    totals, missing = aggregate_fleet_measurements({"a": {"cpu_cores": 8.0}, "b": "nope"})
    assert totals["cpu_cores"] == 8.0
    assert missing["cpu_cores"] == ["b"]


def test_missing_node_lists_are_sorted_for_stable_output():
    per_node = {"z": None, "a": None, "m": {"cpu_cores": 4.0}}
    _, missing = aggregate_fleet_measurements(per_node)
    assert missing["cpu_cores"] == ["a", "z"]


def test_zero_is_a_real_contribution_not_an_absence():
    per_node = {"a": {"chassis_watts": 0.0}, "b": {"chassis_watts": 100.0}}
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["chassis_watts"] == 100.0
    assert "chassis_watts" not in missing  # a is present, it just drew nothing


# ------------------------------------------------------- schema wiring


def test_cluster_schema_carries_totals_and_their_coverage():
    from orion.schemas.telemetry.biometrics import BiometricsClusterV1

    totals, missing = aggregate_fleet_measurements(LIVE)
    c = BiometricsClusterV1(sources=sorted(LIVE), measurements=totals, measurements_missing=missing)
    rt = BiometricsClusterV1.model_validate_json(c.model_dump_json())
    assert rt.measurements["chassis_watts"] == pytest.approx(663.0)
    assert rt.measurements_missing["chassis_watts"] == ["circe"]


def test_cluster_schema_defaults_to_none_for_producers_predating_this():
    from orion.schemas.telemetry.biometrics import BiometricsClusterV1

    c = BiometricsClusterV1.model_validate({"sources": ["athena"]})
    assert c.measurements is None and c.measurements_missing is None


def test_fleet_measurements_are_not_clamped_like_the_pressures():
    """The regression that would make this worthless: watts pushed through min(1.0, ...)."""
    from orion.schemas.telemetry.biometrics import BiometricsClusterV1

    totals, _ = aggregate_fleet_measurements(LIVE)
    c = BiometricsClusterV1(measurements=totals, pressures={"cpu": 0.5})
    assert c.measurements["chassis_watts"] > 1.0
    assert all(0.0 <= v <= 1.0 for v in c.pressures.values())
