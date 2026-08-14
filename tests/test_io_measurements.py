"""net_pressure's denominator is measured, and raw I/O reaches `measurements` (ROADMAP B3).

`NET_BW_MBPS=125` was one constant for three heterogeneous hosts. It is right for athena's
single 1 GbE port by coincidence and was never verified on atlas or circe. The kernel already
knows the answer per node, so the constant is now a fallback and the live denominator is the
summed link speed of the node's up physical NICs.

The unit trap this guards: sysfs reports Mb/s (bits), the pipeline's scale is bytes/sec. A
missing factor of 8 is invisible inside a 0-1 band -- the same class of silent error that let
net_pressure sit at 3e-05 on every node without anyone noticing.
"""
from __future__ import annotations

import pytest

from datetime import datetime, timezone

from orion.telemetry.biometrics_pipeline import (
    FLEET_MAX_KEYS,
    FLEET_SUM_KEYS,
    aggregate_fleet_measurements,
    extract_measurements,
)


# ------------------------------------------------------------- the unit conversion

def test_one_gigabit_link_is_one_hundred_twenty_five_megabytes_per_second():
    """Hand-computed: 1000 Mb/s * 1e6 / 8 = 125,000,000 B/s. Saturating that link must read
    1.0, and the historical NET_BW_MBPS=125 must agree with it -- that agreement is why the
    old constant was invisible-wrong on athena and visibly wrong nowhere until atlas and
    circe were checked."""
    from orion.telemetry.biometrics_pipeline import NormalizationConfig, normalize_rate

    scale_from_link = 1000.0 * 1_000_000 / 8.0
    scale_from_const = NormalizationConfig().net_bw_mbps * 1_000_000
    assert scale_from_link == scale_from_const == 125_000_000
    assert normalize_rate(125_000_000, scale=scale_from_link) == pytest.approx(1.0)
    assert normalize_rate(12_500_000, scale=scale_from_link) == pytest.approx(0.1)


def test_a_ten_gigabit_node_is_not_read_as_eighty_times_saturated():
    """atlas/circe link speed is unknown to me and may be 10 GbE. Under the old constant a
    10 Gb node moving 1 GB/s would read net_pressure 1.0 (saturated) when it is at 0.8."""
    from orion.telemetry.biometrics_pipeline import normalize_rate

    ten_gig = 10_000.0 * 1_000_000 / 8.0
    assert normalize_rate(1_000_000_000, scale=ten_gig) == pytest.approx(0.8)
    assert normalize_rate(1_000_000_000, scale=125 * 1_000_000) == pytest.approx(1.0)


# ------------------------------------------------------------- measurements extraction

ATHENA_SAMPLE = {
    "disk": {"read_bytes_per_sec": 413001.4, "write_bytes_per_sec": 9492764.6},
    "network": {
        "rx_bytes_per_sec": 52736.0,
        "tx_bytes_per_sec": 106568.0,
        "error_rate": 0.0,
        "scope": "host",
        "interfaces": ["eno1"],
        "link_mbps": 1000.0,
    },
}


def test_disk_and_net_rates_now_reach_measurements():
    m = extract_measurements(ATHENA_SAMPLE)
    assert m["disk_bytes_per_sec"] == pytest.approx(9_905_766.0)   # 413001.4 + 9492764.6
    assert m["net_bytes_per_sec"] == pytest.approx(159_304.0)      # 52736 + 106568
    assert m["net_link_mbps"] == 1000.0


def test_container_scoped_network_is_withheld_entirely():
    """The load-bearing rule. A veth reading is a DIFFERENT quantity, not a scaled one, so it
    is absent -- which the fleet aggregate then reports as a missing node."""
    sample = {**ATHENA_SAMPLE, "network": {**ATHENA_SAMPLE["network"], "scope": "container"}}
    m = extract_measurements(sample)
    assert "net_bytes_per_sec" not in m
    assert "net_link_mbps" not in m
    assert "disk_bytes_per_sec" in m  # /proc/diskstats is not namespaced; disk is unaffected


def test_absent_scope_is_treated_as_not_host():
    """A producer predating this field says nothing about namespace, so it cannot be trusted
    as a node reading."""
    sample = {**ATHENA_SAMPLE, "network": {"rx_bytes_per_sec": 1.0, "tx_bytes_per_sec": 2.0}}
    assert "net_bytes_per_sec" not in extract_measurements(sample)


def test_a_partial_disk_reading_is_absent_not_half_counted():
    """_sum_of is all-or-nothing: a missing write counter must not publish read-only bytes as
    the node's total I/O."""
    sample = {"disk": {"read_bytes_per_sec": 100.0}}
    assert "disk_bytes_per_sec" not in extract_measurements(sample)


def test_no_disk_or_network_blob_is_absent_not_zero():
    m = extract_measurements({"cpu": {"cores": 80.0}})
    assert "disk_bytes_per_sec" not in m and "net_bytes_per_sec" not in m


def test_zero_traffic_is_a_real_reading():
    sample = {
        "disk": {"read_bytes_per_sec": 0.0, "write_bytes_per_sec": 0.0},
        "network": {"rx_bytes_per_sec": 0.0, "tx_bytes_per_sec": 0.0, "scope": "host"},
    }
    m = extract_measurements(sample)
    assert m["disk_bytes_per_sec"] == 0.0
    assert m["net_bytes_per_sec"] == 0.0


# ------------------------------------------------------------- fleet composition

def test_io_rates_and_link_capacity_sum_across_the_fleet():
    per_node = {
        "athena": {"disk_bytes_per_sec": 9_905_766.0, "net_bytes_per_sec": 159_304.0,
                   "net_link_mbps": 1000.0},
        "atlas": {"disk_bytes_per_sec": 2_550.0, "net_bytes_per_sec": 21_000.0,
                  "net_link_mbps": 10_000.0},
    }
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["disk_bytes_per_sec"] == pytest.approx(9_908_316.0)
    assert totals["net_bytes_per_sec"] == pytest.approx(180_304.0)
    assert totals["net_link_mbps"] == 11_000.0
    assert missing == {}


def test_a_node_still_on_container_scope_is_named_missing_not_summed_as_zero():
    """Mid-rollout: athena redeployed, atlas not yet. The fleet net total is real and missing
    a machine, and says so -- the same invariant as circe's absent chassis_watts."""
    per_node = {
        "athena": {"net_bytes_per_sec": 159_304.0, "disk_bytes_per_sec": 9_905_766.0},
        "atlas": {"disk_bytes_per_sec": 2_550.0},
    }
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["net_bytes_per_sec"] == pytest.approx(159_304.0)
    assert missing["net_bytes_per_sec"] == ["atlas"]
    assert "disk_bytes_per_sec" not in missing


def test_the_new_keys_are_extensive_not_intensive():
    for key in ("disk_bytes_per_sec", "net_bytes_per_sec", "net_link_mbps"):
        assert key in FLEET_SUM_KEYS
        assert key not in FLEET_MAX_KEYS


def test_the_two_policy_tables_still_do_not_overlap():
    assert not (set(FLEET_SUM_KEYS) & set(FLEET_MAX_KEYS))


# ------------------------------------------------------------- end-to-end through the reducer


def _pipe(net_bw_mbps=125.0):
    from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig

    return BiometricsPipeline(PipelineConfig(net_bw_mbps=net_bw_mbps))


def _net_sample(**net):
    """_summarize requires a timestamp; nothing here depends on its value."""
    return {"timestamp": datetime.now(timezone.utc), "network": net}


def test_measured_link_speed_beats_the_configured_constant():
    """1 GB/s on a 10 Gb link is 0.8, not the 1.0 the 125 MB/s constant would report."""
    s = _net_sample(rx_bytes_per_sec=1_000_000_000.0, tx_bytes_per_sec=0.0,
                    scope="host", link_mbps=10_000.0)
    assert _pipe()._summarize(s).pressures["net"] == pytest.approx(0.8)


def test_the_constant_is_used_when_no_link_speed_was_readable():
    s = _net_sample(rx_bytes_per_sec=62_500_000.0, tx_bytes_per_sec=0.0, scope="container")
    assert _pipe()._summarize(s).pressures["net"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, True, "1000", None])
def test_a_nonsense_link_speed_falls_back_rather_than_dividing_by_it(bad):
    """0 would be a ZeroDivisionError; -1 (a driver that does not know the speed) would
    invert the pressure; True would be 1 Mb/s and read every node as saturated."""
    s = _net_sample(rx_bytes_per_sec=62_500_000.0, tx_bytes_per_sec=0.0,
                    scope="host", link_mbps=bad)
    assert _pipe()._summarize(s).pressures["net"] == pytest.approx(0.5)


def test_the_degenerate_reading_this_replaces():
    """Regression pin on the live numbers. athena's net_pressure was 3.2e-05 because the
    numerator was the container's veth. With the host numerator the same node reads four
    orders of magnitude higher -- and, unlike before, can actually reach 1.0."""
    veth = _net_sample(rx_bytes_per_sec=687.6, tx_bytes_per_sec=3323.2)
    host = _net_sample(rx_bytes_per_sec=52_736.0, tx_bytes_per_sec=106_568.0,
                       scope="host", link_mbps=1000.0)
    saturated = _net_sample(rx_bytes_per_sec=125_000_000.0, tx_bytes_per_sec=0.0,
                            scope="host", link_mbps=1000.0)

    assert _pipe()._summarize(veth).pressures["net"] < 1e-4
    assert _pipe()._summarize(host).pressures["net"] == pytest.approx(159_304 / 125_000_000)
    assert _pipe()._summarize(saturated).pressures["net"] == pytest.approx(1.0)


# ------------------------------------------------------------- CPU package power (B4)

def test_cpu_package_watts_reaches_measurements():
    m = extract_measurements({"power": {"cpu_package_watts": 210.35, "gpu_power_watts": [44.63]}})
    assert m["cpu_watts_total"] == pytest.approx(210.35)


def test_cpu_watts_is_absent_when_the_node_has_no_rapl():
    m = extract_measurements({"power": {"gpu_power_watts": [44.63]}})
    assert "cpu_watts_total" not in m   # not 0.0


def test_cpu_watts_sums_across_the_fleet_and_names_what_is_missing():
    per_node = {
        "athena": {"cpu_watts_total": 210.35, "chassis_watts": 407.0},
        "atlas": {"cpu_watts_total": 95.0, "chassis_watts": 273.0},
        "circe": {"gpu_watts_total": 172.96},          # no BMC, and no RAPL reading yet
    }
    totals, missing = aggregate_fleet_measurements(per_node)
    assert totals["cpu_watts_total"] == pytest.approx(305.35)
    assert missing["cpu_watts_total"] == ["circe"]


def test_the_containment_rule_is_why_these_must_not_be_added_together():
    """Documentation as a test. athena is a 407 W machine; chassis + cpu + gpu reports 661 W,
    a 62% over-count. Anything summing every watt-keyed measurement is wrong by construction."""
    athena = {"chassis_watts": 407.0, "cpu_watts_total": 210.35, "gpu_watts_total": 44.63}
    naive_total = sum(v for k, v in athena.items() if k.endswith("_watts") or k.endswith("watts_total"))
    assert naive_total == pytest.approx(661.98)
    assert athena["chassis_watts"] == 407.0
    # And the decomposition must not exceed the whole.
    assert athena["cpu_watts_total"] + athena["gpu_watts_total"] < athena["chassis_watts"]
