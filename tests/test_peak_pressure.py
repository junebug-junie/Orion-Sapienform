"""The binding constraint, as a signal Orion can read (ROADMAP B2, additive).

`strain` cannot answer "what is about to stop this machine". It is the MEAN of 7 of the 11
pressures, so a saturated channel is averaged away by its calm neighbours, and the 4 it omits
(`gpu_mem`, `swap`, `disk_capacity`, `fan`) are the top channel on two of three nodes.

`strain` is NOT fixed here. It is written straight into the field lattice's `cpu_pressure`
channel (`state_deltas.py:125`, mode="replace") and feeds the substrate prediction-error diff,
so changing its formula moves Orion's substrate. This adds a second reduction of the same
pressures alongside it and leaves it untouched — which is why these tests assert, repeatedly,
that `strain` still reads exactly what it always did.

Fixture values are the live 2026-08-14/15 readings.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.telemetry.biometrics import BiometricsClusterV1, BiometricsSummaryV1
from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig


def _pipe() -> BiometricsPipeline:
    return BiometricsPipeline(PipelineConfig())


# Live athena, 2026-08-14: `power` fully saturated.
ATHENA_SATURATED = {
    "cpu": 0.486, "gpu_util": 0.06, "gpu_mem": 0.334, "mem": 0.082, "swap": 0.003,
    "disk": 0.130, "net": 0.0000, "thermal": 0.600, "power": 1.000,
    "disk_capacity": 0.771, "fan": 0.410,
}


# ------------------------------------------------------------- the point of the signal

def test_a_saturated_channel_is_visible_where_strain_hides_it():
    """THE REASON THIS EXISTS. Hand-computed from the live athena vector:
        strain = mean(cpu, gpu_util, mem, disk, net, thermal, power)
               = (0.486+0.06+0.082+0.130+0.0+0.600+1.000)/7 = 2.358/7 = 0.3369
        peak   = max(all eleven) = power = 1.000
    A pegged resource reads 0.34 through the mean and 1.00 through the max."""
    peak, channel = _pipe()._peak_pressure(ATHENA_SATURATED)
    assert peak == 1.0
    assert channel == "power"

    strain_inputs = [ATHENA_SATURATED[k] for k in
                     ("cpu", "gpu_util", "mem", "disk", "net", "thermal", "power")]
    assert sum(strain_inputs) / 7 == pytest.approx(0.3369, abs=1e-4)


def test_the_four_pressures_strain_omits_can_win():
    """athena's actual peak on 2026-08-15 was `disk_capacity` 0.772 -- a channel `strain` does
    not include at all, so no change to strain's inputs could have surfaced it."""
    peak, channel = _pipe()._peak_pressure({"cpu": 0.10, "disk_capacity": 0.772, "power": 0.30})
    assert channel == "disk_capacity"
    assert peak == pytest.approx(0.772)


@pytest.mark.parametrize("channel", ["gpu_mem", "swap", "disk_capacity", "fan"])
def test_each_strain_excluded_channel_is_eligible_to_be_the_peak(channel):
    peak, found = _pipe()._peak_pressure({"cpu": 0.1, channel: 0.9})
    assert found == channel and peak == pytest.approx(0.9)


def test_it_reports_the_channel_below_the_constraint_threshold():
    """`constraint` says nothing under 0.7. atlas peaked at `power` 0.683 and circe at
    `gpu_mem` 0.540 -- both real binding constraints, both invisible as `constraint=NONE`."""
    peak, channel = _pipe()._peak_pressure({"power": 0.683, "cpu": 0.004})
    assert channel == "power" and peak == pytest.approx(0.683)


def test_it_reports_channels_the_constraints_map_omits():
    """CONSTRAINTS has no entry for swap/disk_capacity/fan, so a peak in one of them reports
    NONE however high it goes. Live: athena disk_capacity 0.772, over threshold, NONE."""
    from orion.telemetry.biometrics_pipeline import CONSTRAINTS

    for channel in ("swap", "disk_capacity", "fan"):
        assert channel not in CONSTRAINTS
        _, found = _pipe()._peak_pressure({channel: 0.95, "cpu": 0.1})
        assert found == channel


# ------------------------------------------------------------- absence and edges

def test_no_pressures_is_absent_not_zero():
    """Nothing measured is not 'nothing under pressure'. Same rule as `measurements`."""
    assert _pipe()._peak_pressure({}) == (None, None)


def test_all_calm_is_a_real_zero():
    """The metric must be able to report a genuine rest state, not just vary."""
    peak, channel = _pipe()._peak_pressure({"cpu": 0.0, "power": 0.0})
    assert peak == 0.0
    assert channel in {"cpu", "power"}


def test_a_single_channel_is_its_own_peak():
    assert _pipe()._peak_pressure({"cpu": 0.42}) == (0.42, "cpu")


def test_the_value_is_clamped_into_the_band():
    peak, _ = _pipe()._peak_pressure({"cpu": 1.4})
    assert peak == 1.0


# ------------------------------------------------------------- strain is untouched

def _sample(pressures_source):
    """A sample that drives the real pipeline, not a hand-built pressures dict."""
    return {
        "timestamp": datetime.now(timezone.utc),
        "node": "athena",
        "cpu": {"cores": 80, "loadavg": {"1m": 30.0, "5m": 30.0, "15m": 30.0}},
        "memory": {"mem_total_kb": 1000, "mem_available_kb": 1000 - int(1000 * pressures_source)},
        "power": {"gpu_power_watts": []},
        "temps": {},
    }


def test_strain_and_homeostasis_still_come_from_the_mean_of_seven():
    """Regression guard on the whole point of doing this additively. If a later patch quietly
    redefines strain, this fails and the field lattice does not silently move."""
    summary = _pipe()._summarize(_sample(0.7))
    pressures = summary.pressures
    expected = sum(pressures[k] for k in
                   ("cpu", "gpu_util", "mem", "disk", "net", "thermal", "power")) / 7
    assert summary.composites["strain"] == pytest.approx(expected)
    assert summary.composites["homeostasis"] == pytest.approx(1.0 - expected)


def test_the_new_fields_ride_alongside_without_displacing_anything():
    summary = _pipe()._summarize(_sample(0.7))
    assert summary.peak_pressure == pytest.approx(max(summary.pressures.values()))
    assert summary.peak_pressure_channel in summary.pressures
    # everything that was there before is still there
    assert set(summary.composites) == {"strain", "homeostasis", "stability"}
    assert summary.constraint is not None


def test_constraint_is_unchanged_by_this_patch():
    """`constraint` has its own consumers and keeps its 0.7 threshold and its partial map."""
    p = _pipe()
    assert p._constraint_from_pressures({"disk_capacity": 0.99}) == "NONE"  # unmapped, as before
    assert p._constraint_from_pressures({"power": 0.99}) == "POWER"
    assert p._constraint_from_pressures({"power": 0.5}) == "NONE"
    # ...while the new signal reports all three honestly.
    assert p._peak_pressure({"disk_capacity": 0.99})[1] == "disk_capacity"
    assert p._peak_pressure({"power": 0.5}) == (0.5, "power")


# ------------------------------------------------------------- schema carriage

def test_summary_schema_round_trips_the_pair():
    s = BiometricsSummaryV1(peak_pressure=0.772, peak_pressure_channel="disk_capacity")
    rt = BiometricsSummaryV1.model_validate_json(s.model_dump_json())
    assert rt.peak_pressure == pytest.approx(0.772)
    assert rt.peak_pressure_channel == "disk_capacity"


def test_cluster_schema_carries_the_node_too():
    """Without the node a fleet peak of 0.772 says something is nearly full and gives no way
    to act on it."""
    c = BiometricsClusterV1(peak_pressure=0.772, peak_pressure_channel="disk_capacity",
                            peak_pressure_node="athena")
    rt = BiometricsClusterV1.model_validate_json(c.model_dump_json())
    assert (rt.peak_pressure, rt.peak_pressure_channel, rt.peak_pressure_node) == (
        pytest.approx(0.772), "disk_capacity", "athena")


def test_producers_predating_this_report_none_not_zero():
    s = BiometricsSummaryV1.model_validate({"node": "athena"})
    assert s.peak_pressure is None and s.peak_pressure_channel is None
    c = BiometricsClusterV1.model_validate({"sources": ["athena"]})
    assert c.peak_pressure is None and c.peak_pressure_node is None


def test_the_fleet_peak_is_a_max_across_nodes_not_a_mean():
    """The reduction that matters: one saturated machine among three calm ones must read 1.0,
    not 0.33. Mirrors publish_cluster's loop."""
    per_node = {"athena": (1.0, "power"), "atlas": (0.2, "cpu"), "circe": (0.1, "gpu_mem")}
    best_node, (best_val, best_ch) = max(per_node.items(), key=lambda kv: kv[1][0])
    assert (best_val, best_ch, best_node) == (1.0, "power", "athena")
    assert best_val != pytest.approx(sum(v for v, _ in per_node.values()) / 3)


def test_a_node_without_the_field_does_not_win_or_break_the_fleet_peak():
    """Mid-rollout: one node upgraded, two not. The peak must come from what reported."""
    summaries = {
        "athena": BiometricsSummaryV1(peak_pressure=0.4, peak_pressure_channel="cpu"),
        "atlas": BiometricsSummaryV1(),  # predates the field
    }
    best = None
    for node, s in summaries.items():
        if s.peak_pressure is None:
            continue
        if best is None or s.peak_pressure > best[0]:
            best = (s.peak_pressure, s.peak_pressure_channel, node)
    assert best == (0.4, "cpu", "athena")
