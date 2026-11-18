"""Tests for raw physical measurements on BiometricsSummaryV1 (ROADMAP B1).

The load-bearing property is ABSENT-IS-NOT-ZERO. Everything else here is ordinary parsing;
that one invariant is the difference between a fleet total that is honest about what it
could not see and one that silently understates by a whole machine.

Fixture values are the real ones observed on 2026-08-13: atlas reports chassis watts and fan
percent over iLO, circe reports neither because its BMC is unreachable.
"""
from __future__ import annotations

import pytest

from orion.telemetry.biometrics_pipeline import (
    BiometricsPipeline,
    PipelineConfig,
    extract_measurements,
)

# atlas: iLO reachable -> chassis watts and fan percent present.
ATLAS_SAMPLE = {
    "timestamp": "2026-08-13T17:54:29.965811+00:00",
    "node": "atlas",
    "ilo": {"ilo_power_watts": 412.0, "ilo_fan_pct": {"Fan 1": 61.0, "Fan 2": 47.0}},
    "power": {"gpu_power_watts": [66.2, 92.9]},
    "cpu": {"cores": 96, "util": 0.0052, "loadavg": {"1m": 0.53, "15m": 0.53}},
    # Present in the real sample, deliberately NOT promoted to measurements (see below).
    "disk": {"read_bytes_per_sec": 1_000_000.0, "write_bytes_per_sec": 500_000.0},
    "network": {"rx_bytes_per_sec": 2_000.0, "tx_bytes_per_sec": 3_000.0},
    "temps": {"max_c": 56.0},
}

# circe: no reachable BMC -> no chassis watts, no fan. GPU watts still come from nvidia-smi.
CIRCE_SAMPLE = {
    "timestamp": "2026-08-13T08:21:45.419144+00:00",
    "node": "circe",
    "power": {"gpu_power_watts": [41.7, 78.3, 40.7]},
    "cpu": {"cores": 72, "util": 0.0021, "loadavg": {"1m": 0.0, "15m": 0.16}},
}


# ------------------------------------------------------ the invariant


def test_unmeasured_quantities_are_absent_not_zero():
    """circe has no BMC. A 0.0 here understates a fleet total by an entire machine."""
    m = extract_measurements(CIRCE_SAMPLE)
    assert "chassis_watts" not in m
    assert "fan_pct_max" not in m
    assert "temp_c_max" not in m
    assert m.get("chassis_watts") is None  # the access pattern callers must use


def test_fleet_total_can_tell_measured_from_unmeasured():
    """The whole point of B1: sum what exists, and know what is missing."""
    per_node = {
        "atlas": extract_measurements(ATLAS_SAMPLE),
        "circe": extract_measurements(CIRCE_SAMPLE),
    }
    measured = {n: m["chassis_watts"] for n, m in per_node.items() if "chassis_watts" in m}
    unmeasured = [n for n, m in per_node.items() if "chassis_watts" not in m]
    assert measured == {"atlas": 412.0}
    assert unmeasured == ["circe"]
    assert sum(measured.values()) == 412.0  # honest partial total, not 412.0 + a fake 0.0


def test_empty_sample_yields_an_empty_dict_not_a_dict_of_zeros():
    assert extract_measurements({}) == {}


def test_nonsense_container_types_are_ignored_not_coerced():
    assert extract_measurements({"ilo": "not a dict", "cpu": None, "power": []}) == {}


# ------------------------------------------------------ values


def test_atlas_measurements_are_the_raw_numbers():
    m = extract_measurements(ATLAS_SAMPLE)
    assert m["chassis_watts"] == 412.0
    assert m["fan_pct_max"] == 61.0          # max of {61, 47}
    assert m["gpu_watts_total"] == pytest.approx(159.1)   # 66.2 + 92.9, SUM not mean
    assert m["temp_c_max"] == 56.0
    assert m["cpu_cores"] == 96.0
    assert m["load_1m"] == 0.53
    assert m["load_15m"] == 0.53


def test_gpu_watts_is_a_sum_not_the_mean_that_power_pressure_uses():
    """`_power_pressure` averages the same list, which makes a 3-GPU box look like a 1-GPU box.

    Fine for a self-relative band, wrong for a fleet total. circe's three cards draw
    41.7+78.3+40.7 = 160.7 W in total and 53.57 W on average; only the first is a cost.
    """
    m = extract_measurements(CIRCE_SAMPLE)
    assert m["gpu_watts_total"] == pytest.approx(160.7)
    assert m["gpu_watts_total"] != pytest.approx(160.7 / 3)


def test_gpu_watts_falls_back_to_per_gpu_entries():
    """nvidia-smi shape, with space-padded strings as they actually arrive."""
    sample = {"gpu": {"gpus": [{"power_draw_watts": " 58.08"}, {"power_draw_watts": " 55.24"}]}}
    assert extract_measurements(sample)["gpu_watts_total"] == pytest.approx(113.32)


def test_flat_power_list_wins_over_the_per_gpu_fallback():
    sample = {
        "power": {"gpu_power_watts": [10.0, 20.0]},
        "gpu": {"gpus": [{"power_draw_watts": 999.0}]},
    }
    assert extract_measurements(sample)["gpu_watts_total"] == 30.0


def test_review_disk_and_net_rates_are_not_promoted_to_physical_quantities():
    """Review findings 1 and 2: both are wrong at node scale, by ~1000x and ~1.97x.

    network reads a namespaced /proc/net/dev from inside a bridged container, so it measures
    its own veth (athena: host 5,093,078 B/s vs container ~4,000). disk sums /proc/diskstats
    rows for whole disks AND their partitions (athena: 26,147,226 vs 13,288,243 B/s).

    Survivable as self-relative 0-1 bands, which is all the pressures ever claimed. Not
    survivable as numbers someone might sum. Fixing the collector moves the shipped
    pressures, so it is parked rather than silently emitted here.
    """
    m = extract_measurements(ATLAS_SAMPLE)
    assert "disk_bytes_per_sec" not in m
    assert "net_bytes_per_sec" not in m
    # ...while the pressures that DO consume them are untouched.
    summary, _ = BiometricsPipeline(PipelineConfig()).update(ATLAS_SAMPLE)
    assert summary.pressures["disk"] > 0.0
    assert summary.pressures["net"] > 0.0


# ------------------------------------------------------ hostile inputs


@pytest.mark.parametrize("bad", [None, "", "  ", "abc", [], {}, float("nan"), float("inf")])
def test_unparseable_chassis_watts_is_absent(bad):
    assert "chassis_watts" not in extract_measurements({"ilo": {"ilo_power_watts": bad}})


def test_booleans_are_rejected_rather_than_read_as_one_watt():
    """float(True) is 1.0 -- a perfectly plausible-looking reading."""
    assert "chassis_watts" not in extract_measurements({"ilo": {"ilo_power_watts": True}})


def test_padded_numeric_strings_parse():
    assert extract_measurements({"ilo": {"ilo_power_watts": " 412.5 "}})["chassis_watts"] == 412.5


def test_review_a_partial_gpu_list_is_absent_not_a_plausible_partial_total():
    """Review finding 4: absent-is-not-zero broken INSIDE a key.

    On a 3-GPU box where one card reports `[N/A]` (nvidia-smi really emits that, and the
    collector drops it), summing the survivors gives 120.0 -- byte-identical to a genuine
    2-GPU total, understating while looking complete. All-or-nothing instead.
    """
    assert "gpu_watts_total" not in extract_measurements(
        {"power": {"gpu_power_watts": [10.0, "bad", None, 20.0]}}
    )
    assert "gpu_watts_total" not in extract_measurements(
        {"power": {"gpu_power_watts": ["bad", None]}}
    )


def test_gpu_count_accompanies_the_total_so_a_partial_is_detectable():
    m = extract_measurements(CIRCE_SAMPLE)
    assert m["gpu_count"] == 3.0  # checkable against the real box
    assert m["gpu_watts_total"] == pytest.approx(160.7)


def test_review_partial_gpu_list_falls_back_to_per_gpu_entries():
    """A partial flat list must not shadow a complete per-GPU list."""
    sample = {
        "power": {"gpu_power_watts": [10.0, "bad"]},
        "gpu": {"gpus": [{"power_draw_watts": 1.0}, {"power_draw_watts": 2.0}]},
    }
    m = extract_measurements(sample)
    assert m["gpu_watts_total"] == 3.0
    assert m["gpu_count"] == 2.0


def test_fan_dict_with_no_usable_values_is_absent():
    assert "fan_pct_max" not in extract_measurements({"ilo": {"ilo_fan_pct": {"Fan 1": "n/a"}}})


def test_review_fan_as_a_list_does_not_crash():
    """Review finding 10: `(x or {}).values()` raised AttributeError on a list."""
    assert extract_measurements({"ilo": {"ilo_fan_pct": [61.0, 47.0]}}) == {}


def test_review_negative_watts_are_a_sensor_fault_not_a_reading():
    assert "chassis_watts" not in extract_measurements({"ilo": {"ilo_power_watts": -1.0}})


def test_missing_max_c_is_absent_the_real_collector_shape_when_sensors_are_gone():
    assert "temp_c_max" not in extract_measurements({"temps": {"max_c": None}})


def test_non_dict_loadavg_is_ignored():
    assert extract_measurements({"cpu": {"loadavg": [1.0, 2.0]}}) == {}


def test_gpus_list_with_non_dict_entries_is_skipped_not_crashed():
    assert "gpu_watts_total" not in extract_measurements({"gpu": {"gpus": ["nope", 5]}})


def test_zero_is_preserved_when_it_is_a_real_reading():
    """Absent-is-not-zero must not become zero-is-not-real: 0 W is a legitimate value."""
    m = extract_measurements({"ilo": {"ilo_power_watts": 0.0}, "cpu": {"loadavg": {"1m": 0.0}}})
    assert m["chassis_watts"] == 0.0
    assert m["load_1m"] == 0.0


# ------------------------------------------------------ wiring


def test_pipeline_populates_measurements_on_the_summary():
    summary, _ = BiometricsPipeline(PipelineConfig()).update(ATLAS_SAMPLE)
    assert summary.measurements["chassis_watts"] == 412.0
    assert summary.measurements["gpu_watts_total"] == pytest.approx(159.1)


def test_measurements_are_not_normalised_unlike_every_other_field():
    """The regression that matters: if someone clamps this to 0-1, watts become useless."""
    summary, _ = BiometricsPipeline(PipelineConfig()).update(ATLAS_SAMPLE)
    assert summary.measurements["chassis_watts"] > 1.0
    assert all(0.0 <= v <= 1.0 for v in summary.pressures.values())


def test_review_none_and_empty_dict_mean_different_things():
    """Review finding 3: a producer that predates the field is not a node with nothing to say.

    atlas is emitting {} right now mid-rollout while its iLO is demonstrably live (its `fan`
    pressure is non-zero in the same rows). Defaulting to {} would let a fleet total count
    ~400 W of real chassis draw as "measured zero".
    """
    from orion.schemas.telemetry.biometrics import BiometricsSummaryV1

    old = BiometricsSummaryV1.model_validate({"node": "atlas", "pressures": {"cpu": 0.5}})
    assert old.measurements is None  # says NOTHING about the node

    current = BiometricsSummaryV1.model_validate({"node": "circe", "measurements": {}})
    assert current.measurements == {}  # a current producer measured nothing


def test_summary_round_trips_measurements_through_json():
    from orion.schemas.telemetry.biometrics import BiometricsSummaryV1

    s = BiometricsSummaryV1(node="atlas", measurements={"chassis_watts": 412.0})
    assert BiometricsSummaryV1.model_validate_json(s.model_dump_json()).measurements == {
        "chassis_watts": 412.0
    }


def test_chassis_watts_contains_gpu_watts_and_the_contract_says_so():
    """Review finding 5: chassis is measured at the PSU and already includes GPU draw.

    A naive SUM(chassis_watts) + SUM(gpu_watts_total) over-reports by ~11% on athena live.
    The containment must be stated where a consumer will see it.
    """
    from orion.schemas.telemetry import biometrics as bio_schema
    import inspect as _inspect

    schema_src = _inspect.getsource(bio_schema)
    assert "ALREADY INCLUDES" in schema_src or "already includes" in schema_src
    assert "Never sum the two" in schema_src or "never sum the two" in schema_src

    from orion.telemetry.biometrics_pipeline import extract_measurements as _em

    assert "CONTAINMENT" in (_em.__doc__ or "")


# NB: the SQL column shape is asserted against the real mapper in
# services/orion-sql-writer/tests/test_biometrics_summary_sql_shape.py. An earlier version of
# this file grepped the model's source text for "measurements = Column(JSONB", which still
# matches when the line is commented out -- it would have passed with the column removed.


def test_gpu_temperature_is_per_card_plus_hottest():
    # Real circe reading 2026-09-29 (nvidia-smi temperature.gpu, space-padded strings).
    sample = {"gpu": {"gpus": [
        {"gpu_index": "0", "temperature_gpu_c": " 39"},
        {"gpu_index": "1", "temperature_gpu_c": " 41"},
        {"gpu_index": "3", "temperature_gpu_c": " 71"},
    ]}}
    out = extract_measurements(sample)
    assert out["gpu0_temp_c"] == 39.0
    assert out["gpu1_temp_c"] == 41.0
    assert out["gpu3_temp_c"] == 71.0
    assert out["gpu_temp_c_max"] == 71.0
    assert "gpu2_temp_c" not in out


def test_gpu_temperature_absent_when_the_csv_predates_the_column():
    # A host still running the old gpu_host_stats.sh: no temperature column at all.
    out = extract_measurements({"gpu": {"gpus": [{"gpu_index": "0", "power_draw_watts": " 38.7"}]}})
    assert "gpu_temp_c_max" not in out
    assert not any(k.startswith("gpu0_temp") for k in out)


@pytest.mark.parametrize("bad", ["", " [N/A]", None, True, "-5"])
def test_unreadable_gpu_temperature_is_absent_not_zero(bad):
    out = extract_measurements({"gpu": {"gpus": [{"gpu_index": "0", "temperature_gpu_c": bad}]}})
    assert "gpu_temp_c_max" not in out and "gpu0_temp_c" not in out


def test_gpu_temperature_without_a_usable_index_still_counts_in_the_max():
    out = extract_measurements({"gpu": {"gpus": [{"gpu_index": " x", "temperature_gpu_c": "80"},
                                                 {"gpu_index": "1", "temperature_gpu_c": "60"}]}})
    assert out["gpu_temp_c_max"] == 80.0
    assert out["gpu1_temp_c"] == 60.0


# Real BMC sensor names, captured live 2026-10-09 (subset). Board heat must pick the
# chipset/VR parts only -- never the CPU, DIMM, GPU, PSU, BMC, or ambient readings, which
# are often hotter (athena's BMC chip sits at 73 C, a GPU at 77 C).
ATHENA_ILO_THERMAL = {
    "01-Inlet Ambient": 29.0,
    "02-CPU 1": 40.0,
    "04-P1 DIMM 1-6": 38.0,
    "16-VR P1": 43.0,
    "17-VR P2": 44.0,
    "18-VR P1 Mem 1": 36.0,
    "22-Chipset": 45.0,
    "23-BMC": 73.0,
    "42.1-GPU 7-GPU ASIC": 77.0,
    "57-P/S 2": 59.0,
    "97-CPU 2 PkgTmp": 61.0,
}
CIRCE_BMC_THERMAL = {
    "CPU0_TEMP": 53.0,
    "CPU1_DTS": 60.0,
    "DIMMG0_TEMP": 46.0,
    "GPU2_PROC": 72.0,
    "INLET_AIR_TEMP": 32.0,
    "PCH_TEMP": 41.0,
    "PSU2_HOTSPOT": 51.0,
    "VR_DIMMG0_TEMP": 50.0,
    "VR_VCCIN_P0_TEMP": 49.0,
}


@pytest.mark.parametrize(
    "thermal, expected",
    [(ATHENA_ILO_THERMAL, 45.0), (CIRCE_BMC_THERMAL, 50.0)],
    ids=["hpe-ilo", "ami-bmc"],
)
def test_board_temp_is_hottest_chipset_or_vr_sensor(thermal, expected):
    m = extract_measurements({"ilo": {"ilo_thermal_c": thermal}})
    assert m["board_temp_c_max"] == expected


def test_board_temp_absent_without_board_sensors():
    # No BMC at all, and a BMC that reports only non-board sensors: absent, never 0.0.
    assert "board_temp_c_max" not in extract_measurements(CIRCE_SAMPLE)
    only_cpu = {"ilo": {"ilo_thermal_c": {"02-CPU 1": 40.0, "23-BMC": 73.0}}}
    assert "board_temp_c_max" not in extract_measurements(only_cpu)


def test_board_temp_ignores_zero_and_non_numeric_readings():
    sample = {"ilo": {"ilo_thermal_c": {"22-Chipset": 0.0, "16-VR P1": "n/a", "17-VR P2": 44.0}}}
    assert extract_measurements(sample)["board_temp_c_max"] == 44.0
