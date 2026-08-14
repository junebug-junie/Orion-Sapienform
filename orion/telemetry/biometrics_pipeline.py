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


def _as_float(value: object) -> Optional[float]:
    """Strict: None for anything that is not a real, finite number.

    Booleans are rejected too -- `float(True)` is 1.0, which would enter a watt field as a
    perfectly plausible reading. nvidia-smi values arrive as space-padded strings
    (`" 58.08"`), so strings are parsed rather than refused.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def _sum_of(values: object) -> Optional[Tuple[float, int]]:
    """(sum, count) only if EVERY element parses. None if any element does not.

    All-or-nothing, deliberately. A partial sum is the absent-is-not-zero failure moved
    inside a key instead of between keys: on a 3-GPU box where one card reports `[N/A]`
    (nvidia-smi really does emit that), summing the survivors yields a number that is
    byte-identical to a genuine 2-GPU total and understates while looking complete. The
    caller pairs the sum with its count so a consumer can check it against the hardware.
    """
    if not isinstance(values, list) or not values:
        return None
    parsed = [_as_float(v) for v in values]
    if any(f is None for f in parsed):
        return None
    return sum(parsed), len(parsed)  # type: ignore[arg-type]


def extract_measurements(sample: Dict[str, object]) -> Dict[str, float]:
    """Raw physical quantities from a biometrics sample, in their own units.

    ROADMAP B1. Everything else on BiometricsSummaryV1 is normalised to 0-1, which cannot be
    summed across hosts or compared over time. These are the numbers Juniper actually pays.

    THE INVARIANT: a key is absent when the quantity is not measured. Never 0.0, never a
    placeholder. circe has no reachable BMC, so writing `chassis_watts: 0.0` for it would
    make a fleet total silently understate by a whole machine while looking complete. Callers
    must use `.get(key)` and handle None, not `.get(key, 0.0)`.

    CONTAINMENT, and a consumer must not sum across it: `chassis_watts` is the whole-node
    draw measured at the PSU and ALREADY INCLUDES `gpu_watts_total`. On athena, live, that
    is 410 W total of which 45 W is GPU -- so `SUM(chassis_watts) + SUM(gpu_watts_total)`
    over-reports by ~11%. For a fleet total use `chassis_watts` alone where present;
    `gpu_watts_total` is for the nodes that have no BMC, and for attributing what share of a
    node's draw is inference.

    WHAT IS DELIBERATELY NOT HERE: disk and network byte rates. Both are collected
    (`services/orion-biometrics/app/metrics.py`) and both are wrong as *node-scale physical
    quantities*, which is what this dict claims to hold:

      - network reads a network-namespaced /proc/net/dev from inside a bridged container, so
        it sees its own veth. Measured on athena: host 5,093,078 B/s vs container ~4,000 B/s,
        a factor of ~1000.
      - disk sums /proc/diskstats rows for whole disks AND their partitions, double-counting.
        Measured on athena over the same window: 26,147,226 vs 13,288,243 B/s, a factor of 1.97.

    Both are survivable as self-relative 0-1 bands (which is all `disk_pressure`/`net_pressure`
    ever claimed) and neither is survivable as a number someone might sum. Fixing them means
    changing the collector, which moves the shipped pressures too -- a live behaviour change
    that belongs in its own patch. Parked, not silently emitted.
    """
    out: Dict[str, float] = {}

    def put(key: str, value: Optional[float]) -> None:
        if value is not None:
            out[key] = value

    ilo = sample.get("ilo") if isinstance(sample.get("ilo"), dict) else {}
    power = sample.get("power") if isinstance(sample.get("power"), dict) else {}
    gpu = sample.get("gpu") if isinstance(sample.get("gpu"), dict) else {}
    cpu = sample.get("cpu") if isinstance(sample.get("cpu"), dict) else {}
    temps = sample.get("temps") if isinstance(sample.get("temps"), dict) else {}

    # Chassis draw from the BMC. The dominant cost term, and the only one that composes
    # across hosts. Present on nodes with iLO configured; absent elsewhere.
    watts = _as_float(ilo.get("ilo_power_watts"))
    if watts is not None and watts >= 0.0:  # a negative draw is a sensor fault, not a reading
        out["chassis_watts"] = watts

    # SUM, deliberately. `_power_pressure` takes the *mean* of the same list, which makes a
    # 3-GPU box normalise like a 1-GPU box -- fine for a self-relative band, wrong for a
    # fleet total. The two numbers are different quantities and both are kept.
    totals = _sum_of(power.get("gpu_power_watts"))
    if totals is None:
        # Fall back to the per-GPU entries when the flat list is absent or partial.
        gpus = gpu.get("gpus")
        if isinstance(gpus, list):
            totals = _sum_of([e.get("power_draw_watts") for e in gpus if isinstance(e, dict)])
    if totals is not None and totals[0] >= 0.0:
        out["gpu_watts_total"] = totals[0]
        out["gpu_count"] = float(totals[1])  # so a consumer can check the total against the box

    fan_map = ilo.get("ilo_fan_pct")
    if isinstance(fan_map, dict):
        fan_values = [f for f in (_as_float(v) for v in fan_map.values()) if f is not None]
        if fan_values:
            out["fan_pct_max"] = max(fan_values)

    put("temp_c_max", _as_float(temps.get("max_c")))
    put("cpu_cores", _as_float(cpu.get("cores")))
    loadavg = cpu.get("loadavg") if isinstance(cpu.get("loadavg"), dict) else {}
    put("load_1m", _as_float(loadavg.get("1m")))
    put("load_15m", _as_float(loadavg.get("15m")))
    return out


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

        # `power` (via `_power_pressure`) was GPU-power-only (nvidia-smi power_draw_watts)
        # -- 0.0 on any node without an NVIDIA GPU, and even with one, blind to actual
        # whole-system draw (PSU, CPU, disks, fans). `ilo` (present only when this node
        # has iLO/BMC credentials configured, see app/ilo.py) carries a real chassis-level
        # PowerConsumedWatts reading -- strictly more complete, so prefer it over the
        # GPU-only estimate when available rather than trying to combine the two (they'd
        # otherwise double-count GPU draw, which is already a component of chassis watts).
        # Same adaptive EwmaBand as before, not a new fixed-max config -- per-node draw
        # baselines vary too much by hardware for one global max to make sense.
        ilo = sample.get("ilo") if isinstance(sample.get("ilo"), dict) else {}
        chassis_watts = ilo.get("ilo_power_watts")
        if isinstance(chassis_watts, (int, float)):
            self.power_band.update(float(chassis_watts))
            power_pressure = self.power_band.normalize(float(chassis_watts))
        else:
            power_pressure = self._power_pressure(power)

        # `disk_capacity` (how full a filesystem is, from
        # services/orion-biometrics/app/metrics.py::collect_disk_capacity) is
        # deliberately distinct from `disk` above (I/O throughput). Present only when
        # DISK_CAPACITY_MOUNTS is configured; a node with no mounts configured omits it.
        # Direct 0-100 -> 0-1 normalization, no adaptive band needed -- "percent full" is
        # already a bounded, meaningful scale on its own.
        disk_capacity_data = sample.get("disk_capacity") if isinstance(sample.get("disk_capacity"), dict) else {}
        disk_capacity_values = list((disk_capacity_data.get("disk_usage_pct") or {}).values())
        disk_capacity_pressure = clamp01(max(disk_capacity_values) / 100.0) if disk_capacity_values else 0.0

        # `fan` (RedFish fan speed, percent) -- present only when iLO is configured and
        # reports fan units as "Percent" (see app/ilo.py's ReadingUnits guard). Elevated
        # fan speed is the BMC's own real-time thermal-stress response, not an independent
        # hazard signal on its own -- theory anchor: sustained high fan percent indicates
        # the chassis is actively compensating for thermal load, a legitimate physical
        # strain signal even though it is a *response to* (not sole cause of) that load,
        # similar to elevated heart rate under exertion.
        fan_values = list((ilo.get("ilo_fan_pct") or {}).values())
        fan_pressure = clamp01(max(fan_values) / 100.0) if fan_values else 0.0

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
            "disk_capacity": disk_capacity_pressure,
            "fan": fan_pressure,
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
            measurements=extract_measurements(sample),
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
