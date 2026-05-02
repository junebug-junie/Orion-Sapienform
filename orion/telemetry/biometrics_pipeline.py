from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from orion.schemas.telemetry.biometrics import BiometricsSummaryV1, BiometricsInductionV1, BiometricsInductionMetricV1
from orion.signals.normalization import clamp01, EwmaBand, InductionTracker


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def filter_temps(values: Iterable[float]) -> List[float]:
    cleaned: List[float] = []
    for v in values:
        if v in (-273.15, 65261.85):
            continue
        if v < -50 or v > 200:
            continue
        cleaned.append(v)
    return cleaned


@dataclass
class NormalizationConfig:
    thermal_min_c: float = 50.0
    thermal_max_c: float = 85.0
    disk_bw_mbps: float = 200.0
    net_bw_mbps: float = 125.0


def normalize_thermal(temp_c: Optional[float], cfg: NormalizationConfig) -> float:
    if temp_c is None:
        return 0.0
    span = cfg.thermal_max_c - cfg.thermal_min_c
    if span <= 0:
        return 0.0
    return clamp01((temp_c - cfg.thermal_min_c) / span)


def normalize_rate(rate: Optional[float], *, scale: float) -> float:
    if rate is None or scale <= 0:
        return 0.0
    return clamp01(rate / scale)


CONSTRAINTS = {
    "thermal": "THERMAL",
    "power": "POWER",
    "gpu_mem": "GPU_MEM",
    "gpu_util": "GPU_UTIL",
    "mem": "MEM",
    "disk": "DISK",
    "net": "NET",
    "containers": "CONTAINERS",
}


@dataclass
class PipelineConfig:
    thermal_min_c: float = 50.0
    thermal_max_c: float = 85.0
    disk_bw_mbps: float = 200.0
    net_bw_mbps: float = 125.0
    power_band_alpha: float = 0.1


class BiometricsPipeline:
    def __init__(self, cfg: PipelineConfig, *, window_sec: float = 30.0) -> None:
        self.cfg = cfg
        self.window_sec = window_sec
        self.power_band = EwmaBand(alpha=cfg.power_band_alpha)
        self.tracker = InductionTracker()

    def update(self, sample: Dict[str, object]) -> Tuple[BiometricsSummaryV1, BiometricsInductionV1]:
        summary = self._summarize(sample)
        induction = self._induce(summary)
        stability = self._stability_from_induction(induction)
        summary.composites["stability"] = stability
        return summary, induction

    def _summarize(self, sample: Dict[str, object]) -> BiometricsSummaryV1:
        cpu = sample.get("cpu") or {}
        memory = sample.get("memory") or {}
        disk = sample.get("disk") or {}
        network = sample.get("network") or {}
        temps = sample.get("temps") or {}
        power = sample.get("power") or {}
        errors = sample.get("errors") or []

        cpu_util = float(cpu.get("util") or 0.0)
        loadavg = cpu.get("loadavg") or {}
        cores = max(int(cpu.get("cores") or 1), 1)
        load1 = float(loadavg.get("1m") or 0.0)
        cpu_pressure = clamp01(max(cpu_util, load1 / cores))

        gpu_util, gpu_mem = self._gpu_pressures(sample.get("gpu") or {})

        mem_total = memory.get("mem_total_kb") or 0
        mem_avail = memory.get("mem_available_kb") or 0
        mem_pressure = clamp01((mem_total - mem_avail) / mem_total) if mem_total else 0.0

        swap_total = memory.get("swap_total_kb") or 0
        swap_free = memory.get("swap_free_kb") or 0
        swap_pressure = clamp01((swap_total - swap_free) / swap_total) if swap_total else 0.0

        disk_rate = (disk.get("read_bytes_per_sec") or 0.0) + (disk.get("write_bytes_per_sec") or 0.0)
        disk_pressure = normalize_rate(disk_rate, scale=self.cfg.disk_bw_mbps * 1_000_000)

        net_rate = (network.get("rx_bytes_per_sec") or 0.0) + (network.get("tx_bytes_per_sec") or 0.0)
        net_pressure = normalize_rate(net_rate, scale=self.cfg.net_bw_mbps * 1_000_000)
        net_error_rate = float(network.get("error_rate") or 0.0)
        net_pressure = clamp01(max(net_pressure, net_error_rate))

        thermal_pressure = normalize_thermal(
            temps.get("max_c"),
            NormalizationConfig(
                thermal_min_c=self.cfg.thermal_min_c,
                thermal_max_c=self.cfg.thermal_max_c,
                disk_bw_mbps=self.cfg.disk_bw_mbps,
                net_bw_mbps=self.cfg.net_bw_mbps,
            ),
        )

        power_pressure = self._power_pressure(power)

        pressures = {
            "cpu": cpu_pressure,
            "gpu_util": gpu_util,
            "gpu_mem": gpu_mem,
            "mem": mem_pressure,
            "swap": swap_pressure,
            "disk": disk_pressure,
            "net": net_pressure,
            "thermal": thermal_pressure,
            "power": power_pressure,
        }

        headroom = {k: clamp01(1.0 - v) for k, v in pressures.items()}

        strain_inputs = [cpu_pressure, gpu_util, mem_pressure, disk_pressure, net_pressure, thermal_pressure, power_pressure]
        strain = sum(strain_inputs) / max(len(strain_inputs), 1)
        composites = {
            "strain": clamp01(strain),
            "homeostasis": clamp01(1.0 - strain),
            "stability": 0.0,
        }

        constraint = self._constraint_from_pressures(pressures)
        telemetry_error_rate = clamp01(len(errors) / 5) if isinstance(errors, list) else 0.0

        return BiometricsSummaryV1(
            timestamp=sample.get("timestamp"),
            node=sample.get("node"),
            service_name=sample.get("service_name"),
            service_version=sample.get("service_version"),
            pressures=pressures,
            headroom=headroom,
            composites=composites,
            constraint=constraint,
            telemetry_error_rate=telemetry_error_rate,
        )

    def _gpu_pressures(self, gpu: Dict[str, object]) -> Tuple[float, float]:
        gpus = gpu.get("gpus") if isinstance(gpu, dict) else None
        util_vals = []
        mem_vals = []
        if isinstance(gpus, list):
            for entry in gpus:
                if not isinstance(entry, dict):
                    continue
                util = entry.get("utilization_gpu")
                mem_used = entry.get("memory_used_mb")
                mem_total = entry.get("memory_total_mb")
                try:
                    if util is not None:
                        util_vals.append(float(util) / 100.0)
                except (TypeError, ValueError):
                    pass
                try:
                    if mem_used is not None and mem_total:
                        mem_vals.append(float(mem_used) / float(mem_total))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        gpu_util = clamp01(sum(util_vals) / len(util_vals)) if util_vals else 0.0
        gpu_mem = clamp01(sum(mem_vals) / len(mem_vals)) if mem_vals else 0.0
        return gpu_util, gpu_mem

    def _power_pressure(self, power: Dict[str, object]) -> float:
        gpu_power = power.get("gpu_power_watts") if isinstance(power, dict) else None
        if not isinstance(gpu_power, list) or not gpu_power:
            return 0.0
        try:
            avg_power = sum(float(v) for v in gpu_power) / len(gpu_power)
        except (TypeError, ValueError):
            return 0.0
        self.power_band.update(avg_power)
        return self.power_band.normalize(avg_power)

    def _constraint_from_pressures(self, pressures: Dict[str, float]) -> str:
        if not pressures:
            return "NONE"
        key, value = max(pressures.items(), key=lambda item: item[1])
        if value < 0.7:
            return "NONE"
        return CONSTRAINTS.get(key, "NONE")

    def _induce(self, summary: BiometricsSummaryV1) -> BiometricsInductionV1:
        metrics: Dict[str, BiometricsInductionMetricV1] = {}
        for name, value in summary.pressures.items():
            state = self.tracker.update(name, value)
            metrics[name] = BiometricsInductionMetricV1(
                level=state.level,
                trend=state.trend,
                volatility=clamp01(state.volatility),
                spike_rate=clamp01(state.spike_rate),
            )
        for name, value in summary.composites.items():
            state = self.tracker.update(name, value)
            metrics[name] = BiometricsInductionMetricV1(
                level=state.level,
                trend=state.trend,
                volatility=clamp01(state.volatility),
                spike_rate=clamp01(state.spike_rate),
            )
        return BiometricsInductionV1(
            timestamp=summary.timestamp,
            node=summary.node,
            service_name=summary.service_name,
            service_version=summary.service_version,
            window_sec=self.window_sec,
            metrics=metrics,
        )

    def _stability_from_induction(self, induction: BiometricsInductionV1) -> float:
        metric = induction.metrics.get("strain")
        if not metric:
            return 0.5
        return clamp01(1.0 - (0.5 * metric.volatility + 0.5 * metric.spike_rate))
