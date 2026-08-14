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
    assert m["disk_bytes_per_sec"] == 1_500_000.0
    assert m["net_bytes_per_sec"] == 5_000.0
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


def test_one_sided_disk_and_net_rates_still_report():
    m = extract_measurements({"disk": {"read_bytes_per_sec": 42.0}})
    assert m["disk_bytes_per_sec"] == 42.0
    assert "net_bytes_per_sec" not in m


# ------------------------------------------------------ hostile inputs


@pytest.mark.parametrize("bad", [None, "", "  ", "abc", [], {}, float("nan"), float("inf")])
def test_unparseable_chassis_watts_is_absent(bad):
    assert "chassis_watts" not in extract_measurements({"ilo": {"ilo_power_watts": bad}})


def test_booleans_are_rejected_rather_than_read_as_one_watt():
    """float(True) is 1.0 -- a perfectly plausible-looking reading."""
    assert "chassis_watts" not in extract_measurements({"ilo": {"ilo_power_watts": True}})


def test_padded_numeric_strings_parse():
    assert extract_measurements({"ilo": {"ilo_power_watts": " 412.5 "}})["chassis_watts"] == 412.5


def test_a_partially_unusable_gpu_list_sums_what_it_can():
    m = extract_measurements({"power": {"gpu_power_watts": [10.0, "bad", None, 20.0]}})
    assert m["gpu_watts_total"] == 30.0


def test_an_entirely_unusable_gpu_list_is_absent_not_zero():
    assert "gpu_watts_total" not in extract_measurements(
        {"power": {"gpu_power_watts": ["bad", None]}}
    )


def test_fan_dict_with_no_usable_values_is_absent():
    assert "fan_pct_max" not in extract_measurements({"ilo": {"ilo_fan_pct": {"Fan 1": "n/a"}}})


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


def test_summary_defaults_to_empty_measurements_for_old_payloads():
    """Backward compatibility: rows written before B1 validate unchanged."""
    from orion.schemas.telemetry.biometrics import BiometricsSummaryV1

    s = BiometricsSummaryV1.model_validate({"node": "atlas", "pressures": {"cpu": 0.5}})
    assert s.measurements == {}


def test_summary_round_trips_measurements_through_json():
    from orion.schemas.telemetry.biometrics import BiometricsSummaryV1

    s = BiometricsSummaryV1(node="atlas", measurements={"chassis_watts": 412.0})
    assert BiometricsSummaryV1.model_validate_json(s.model_dump_json()).measurements == {
        "chassis_watts": 412.0
    }


def test_sql_model_has_the_column_the_writer_will_look_for():
    """_write_row filters payload keys against the mapper's columns; a missing column drops
    the field silently rather than erroring."""
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "services", "orion-sql-writer", "app", "models", "biometrics_summary.py",
    )
    src = open(path, encoding="utf-8").read()
    assert "measurements = Column(JSONB" in src
