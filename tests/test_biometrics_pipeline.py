from datetime import datetime, timezone

from orion.telemetry.biometrics_pipeline import (
    BiometricsPipeline,
    PipelineConfig,
    clamp01,
    filter_temps,
)


def _sample(cpu_util: float) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc),
        "node": "athena",
        "service_name": "orion-biometrics",
        "service_version": "0.1.0",
        "cpu": {
            "util": cpu_util,
            "loadavg": {"1m": cpu_util, "5m": cpu_util, "15m": cpu_util},
            "cores": 4,
        },
        "memory": {
            "mem_total_kb": 1000,
            "mem_available_kb": 500,
            "swap_total_kb": 1000,
            "swap_free_kb": 500,
        },
        "disk": {
            "read_bytes_per_sec": 0.0,
            "write_bytes_per_sec": 0.0,
        },
        "network": {
            "rx_bytes_per_sec": 0.0,
            "tx_bytes_per_sec": 0.0,
            "error_rate": 0.0,
        },
        "temps": {
            "max_c": 60.0,
        },
        "power": {
            "gpu_power_watts": [50.0],
        },
        "gpu": {
            "gpus": [
                {
                    "utilization_gpu": 10,
                    "memory_used_mb": 100,
                    "memory_total_mb": 1000,
                    "power_draw_watts": 50,
                }
            ]
        },
        "errors": [],
    }


def test_clamp01_bounds() -> None:
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(2.0) == 1.0


def test_filter_temps_bogus_values() -> None:
    temps = filter_temps([-273.15, 65261.85, 35.0, 90.0])
    assert temps == [35.0, 90.0]


def test_stability_drops_on_spike() -> None:
    pipeline = BiometricsPipeline(PipelineConfig())
    for _ in range(3):
        summary, _ = pipeline.update(_sample(0.2))
    stable = summary.composites["stability"]

    summary_spike, _ = pipeline.update(_sample(1.0))
    spiky = summary_spike.composites["stability"]
    assert spiky < stable
