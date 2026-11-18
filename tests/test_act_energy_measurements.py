"""Per-card watts/VRAM and per-container CPU watts in biometrics measurements (act energy cost).

Fixture values: circe's live nvidia-smi row 2026-10-29 20:12 (gpu1 at 172.15 W, 24888/32768 MiB)
and athena's fitted CPU line (99 W idle + ~4.5 W per busy core).
"""
from __future__ import annotations

import pytest

from orion.telemetry.biometrics_pipeline import (
    CONTAINER_CPU_WATTS_PREFIX,
    FLEET_SUM_KEYS,
    container_cpu_measurements,
    extract_measurements,
)

CIRCE_GPUS = {"gpus": [
    {"gpu_index": "0", "power_draw_watts": " 38.46", "memory_used_mb": " 27962", "memory_total_mb": " 32768",
     "temperature_gpu_c": " 46"},
    {"gpu_index": "1", "power_draw_watts": " 172.15", "memory_used_mb": " 24888", "memory_total_mb": " 32768",
     "temperature_gpu_c": " 72"},
]}


def test_per_card_watts_and_vram_come_from_the_same_row():
    m = extract_measurements({"gpu": CIRCE_GPUS})
    assert m["gpu0_watts"] == pytest.approx(38.46) and m["gpu1_watts"] == pytest.approx(172.15)
    assert m["gpu1_mem_used_mb"] == 24888.0 and m["gpu1_mem_total_mb"] == 32768.0
    assert m["gpu1_temp_c"] == 72.0  # unchanged
    # Per-card keys are already inside gpu_watts_total: never summed across the fleet.
    assert not any(k.startswith("gpu0_") or k.startswith("gpu1_") for k in FLEET_SUM_KEYS)


def test_per_card_keys_are_independent_and_absent_not_zero():
    gpus = {"gpus": [{"gpu_index": "2", "power_draw_watts": "[N/A]", "memory_used_mb": " 1158",
                      "memory_total_mb": "", "temperature_gpu_c": "bad"}]}
    m = extract_measurements({"gpu": gpus})
    assert "gpu2_watts" not in m and "gpu2_mem_total_mb" not in m and "gpu2_temp_c" not in m
    assert m["gpu2_mem_used_mb"] == 1158.0


def test_a_card_without_an_index_gets_no_per_card_key():
    m = extract_measurements({"gpu": {"gpus": [{"power_draw_watts": "50", "temperature_gpu_c": "40"}]}})
    assert not any(k.startswith("gpu") and k[3:4].isdigit() for k in m)


def _sample(busy, cores, idle=99.0):
    return {"containers": {"host_busy_cores": busy, "cpu_idle_watts": idle, "cpu_cores": cores}}


def test_container_split_sums_to_the_watts_above_idle():
    # 30 busy cores, 390 W package: 291 W above idle split by core share.
    cores = {"orion-athena-vector-host": 6.0, "zelda-dojo-live-3": 1.0, "tiny": 0.01}
    m = container_cpu_measurements(_sample(30.0, cores), 390.0)
    above = 390.0 - 99.0
    assert m[CONTAINER_CPU_WATTS_PREFIX + "orion-athena-vector-host"] == pytest.approx(above * 6.0 / 30.0)
    assert CONTAINER_CPU_WATTS_PREFIX + "tiny" not in m  # under the name floor
    assert m["container_cpu_watts_small"] == pytest.approx(above * 0.01 / 30.0)
    total = sum(v for k, v in m.items() if k.startswith(CONTAINER_CPU_WATTS_PREFIX))
    total += m["container_cpu_watts_small"] + m["cpu_watts_host_other"]
    assert total == pytest.approx(above)
    assert m["cpu_busy_cores"] == 30.0 and m["cpu_idle_watts"] == 99.0


def test_containers_reading_more_than_the_host_are_normalised_not_over_charged():
    m = container_cpu_measurements(_sample(4.0, {"a": 3.0, "b": 3.0}), 199.0)
    assert m[CONTAINER_CPU_WATTS_PREFIX + "a"] + m[CONTAINER_CPU_WATTS_PREFIX + "b"] == pytest.approx(100.0)
    assert m["cpu_watts_host_other"] == 0.0


def test_below_the_fitted_floor_there_is_no_split_not_a_split_of_zero():
    """Review 2026-10-29: circe reads down to ~46 W against its 100 W intercept (~10% of samples).
    Publishing 0 W for busy containers there would be a fake zero."""
    m = container_cpu_measurements(_sample(2.0, {"a": 1.0}), 80.0)
    assert not any(k.startswith(CONTAINER_CPU_WATTS_PREFIX) for k in m)
    assert "container_cpu_watts_small" not in m and "cpu_watts_host_other" not in m
    assert m["cpu_busy_cores"] == 2.0 and m["cpu_idle_watts"] == 99.0


@pytest.mark.parametrize("sample,cpu_w,keys", [
    ({}, 300.0, set()),                                             # no container read at all
    (_sample(None, {"a": 1.0}), 300.0, set()),                     # first tick: no busy cores
    (_sample(10.0, {"a": 1.0}), None, {"cpu_busy_cores"}),         # no RAPL: no watt split
    (_sample(10.0, {"a": 1.0}, idle=None), 300.0, {"cpu_busy_cores"}),  # node has no idle floor
    (_sample(10.0, None), 300.0, {"cpu_busy_cores"}),              # no container delta yet
])
def test_split_is_absent_when_an_input_is_missing(sample, cpu_w, keys):
    assert set(container_cpu_measurements(sample, cpu_w)) == keys


def test_extract_measurements_carries_the_split():
    sample = {"power": {"cpu_package_watts": 300.0}, **_sample(10.0, {"a": 5.0})}
    m = extract_measurements(sample)
    assert m["cpu_watts_total"] == 300.0
    assert m[CONTAINER_CPU_WATTS_PREFIX + "a"] == pytest.approx(201.0 * 0.5)
