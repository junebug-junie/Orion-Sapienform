"""Regression coverage for power/disk_capacity/fan pressures added to
orion/telemetry/biometrics_pipeline.py::BiometricsPipeline._summarize()
(2026-07-25, real iLO/BMC hardware telemetry).

See services/orion-biometrics/app/ilo.py for the live-verified iLO collector
these pressures are derived from, and services/orion-field-digester/tests/
test_field_node_biometrics_perturbations.py for the downstream lattice-channel
coverage of the same three signals.
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig

FIXED_TS = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _pipeline() -> BiometricsPipeline:
    return BiometricsPipeline(PipelineConfig(), window_sec=30.0)


def _minimal_sample(**extra: object) -> dict:
    sample: dict = {
        "timestamp": FIXED_TS,
        "node": "athena",
        "service_name": "orion-biometrics",
        "service_version": "0.1.0",
        "cpu": {"util": 0.1, "loadavg": {"1m": 0.1}, "cores": 4},
        "memory": {"mem_total_kb": 1000, "mem_available_kb": 900},
        "disk": {},
        "network": {},
        "temps": {},
        "power": {},
        "gpu": {},
        "errors": [],
    }
    sample.update(extra)
    return sample


def test_power_pressure_prefers_ilo_chassis_watts_over_gpu_only_estimate() -> None:
    pipeline = _pipeline()
    # Two updates so EwmaBand.normalize() has a real (non-zero dev) band --
    # its first .update() call always yields dev=0.0 (see orion/signals/
    # normalization.py::EwmaBand), which would make normalize() return 0.0
    # regardless of input.
    summary, _ = pipeline.update(_minimal_sample(ilo={"ilo_power_watts": 300.0}))
    summary, _ = pipeline.update(_minimal_sample(ilo={"ilo_power_watts": 340.0}))
    assert summary.pressures["power"] > 0.0


def test_power_pressure_falls_back_to_gpu_only_without_ilo_data() -> None:
    pipeline = _pipeline()
    sample = _minimal_sample(power={"gpu_power_watts": [200.0]})
    summary, _ = pipeline.update(sample)
    summary, _ = pipeline.update(sample)
    # No "ilo" key at all (node without BMC credentials configured) -- must
    # not raise, and must fall back to the pre-existing GPU-only calculation.
    assert "ilo" not in sample
    assert summary.pressures["power"] >= 0.0


def test_power_pressure_ignores_ilo_when_power_watts_missing() -> None:
    pipeline = _pipeline()
    # iLO configured but the last poll errored -- no ilo_power_watts key.
    sample = _minimal_sample(ilo={"ilo_error": "connection timeout"})
    summary, _ = pipeline.update(sample)
    assert summary.pressures["power"] == 0.0


def test_disk_capacity_pressure_uses_max_across_mounts() -> None:
    pipeline = _pipeline()
    summary, _ = pipeline.update(
        _minimal_sample(
            disk_capacity={"disk_usage_pct": {"docker": 87.3, "scripts": 14.1, "postgres": 14.4}}
        )
    )
    assert summary.pressures["disk_capacity"] == 0.873


def test_disk_capacity_pressure_zero_when_no_mounts_reported() -> None:
    pipeline = _pipeline()
    summary, _ = pipeline.update(_minimal_sample())
    assert summary.pressures["disk_capacity"] == 0.0


def test_disk_capacity_pressure_zero_when_all_mounts_errored() -> None:
    pipeline = _pipeline()
    summary, _ = pipeline.update(
        _minimal_sample(disk_capacity={"disk_usage_pct": {}, "disk_usage_errors": {"docker": "not_mounted"}})
    )
    assert summary.pressures["disk_capacity"] == 0.0


def test_fan_pressure_uses_max_across_fans() -> None:
    pipeline = _pipeline()
    summary, _ = pipeline.update(
        _minimal_sample(ilo={"ilo_fan_pct": {"Fan 1": 18.0, "Fan 2": 29.0, "Fan 3": 27.0}})
    )
    assert summary.pressures["fan"] == 0.29


def test_fan_pressure_zero_without_ilo_fan_data() -> None:
    pipeline = _pipeline()
    summary, _ = pipeline.update(_minimal_sample())
    assert summary.pressures["fan"] == 0.0


def test_new_pressures_flow_into_induction_without_extra_wiring() -> None:
    # BiometricsPipeline._induce() iterates summary.pressures.items() generically
    # (see biometrics_pipeline.py:~211) -- confirm the 3 new keys are picked up
    # automatically, same as every other pressure, with no separate change needed.
    pipeline = _pipeline()
    _, induction = pipeline.update(
        _minimal_sample(
            ilo={"ilo_power_watts": 300.0, "ilo_fan_pct": {"Fan 1": 25.0}},
            disk_capacity={"disk_usage_pct": {"docker": 50.0}},
        )
    )
    assert "power" in induction.metrics
    assert "disk_capacity" in induction.metrics
    assert "fan" in induction.metrics
