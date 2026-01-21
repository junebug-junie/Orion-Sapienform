# services/orion-biometrics/app/metrics.py
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.settings import settings
from app.utils import collect_gpu_stats
from orion.telemetry.biometrics_pipeline import filter_temps

logger = logging.getLogger("orion-biometrics")


@dataclass
class _DiskStats:
    read_bytes: int
    write_bytes: int


@dataclass
class _NetStats:
    rx_bytes: int
    tx_bytes: int
    rx_packets: int
    tx_packets: int
    rx_errors: int
    tx_errors: int
    rx_drops: int
    tx_drops: int


@dataclass
class _CpuTimes:
    total: int
    idle: int


class BiometricsCollector:
    def __init__(self) -> None:
        self._prev_cpu: Optional[_CpuTimes] = None
        self._prev_disk: Optional[_DiskStats] = None
        self._prev_net: Optional[_NetStats] = None
        self._prev_ts: Optional[float] = None

    def collect(self) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc)
        now_ts = time.time()
        dt = None
        if self._prev_ts is not None:
            dt = max(now_ts - self._prev_ts, 1e-3)
        self._prev_ts = now_ts

        errors: List[str] = []

        gpu_data = self._collect_gpu(errors)
        cpu_data, cpu_util = self._collect_cpu(errors)
        memory_data = self._collect_memory(errors)
        disk_data = self._collect_disk(errors, dt)
        net_data = self._collect_network(errors, dt)
        temps_data = self._collect_temps(errors)
        power_data = self._collect_power(gpu_data, temps_data)

        return {
            "timestamp": timestamp,
            "node": settings.NODE_NAME,
            "service_name": settings.SERVICE_NAME,
            "service_version": settings.SERVICE_VERSION,
            "gpu": gpu_data,
            "cpu": {
                **cpu_data,
                "util": cpu_util,
            },
            "memory": memory_data,
            "disk": disk_data,
            "network": net_data,
            "temps": temps_data,
            "power": power_data,
            "errors": errors,
        }

    def _collect_gpu(self, errors: List[str]) -> Dict[str, Any]:
        try:
            gpu_data = collect_gpu_stats()
            if gpu_data is None:
                return {"status": "no_data", "gpus": []}
            return gpu_data
        except Exception as exc:
            errors.append(f"gpu:{exc}")
            return {"status": "failed", "error": str(exc), "gpus": []}

    def _collect_cpu(self, errors: List[str]) -> Tuple[Dict[str, Any], float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as handle:
                line = handle.readline()
            parts = line.split()
            if parts[0] != "cpu":
                raise ValueError("cpu line missing")
            values = [int(v) for v in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
            current = _CpuTimes(total=total, idle=idle)
            util = 0.0
            if self._prev_cpu:
                total_delta = current.total - self._prev_cpu.total
                idle_delta = current.idle - self._prev_cpu.idle
                if total_delta > 0:
                    util = max(0.0, min(1.0, 1.0 - (idle_delta / total_delta)))
            self._prev_cpu = current
            load1, load5, load15 = self._read_loadavg()
            return {
                "loadavg": {"1m": load1, "5m": load5, "15m": load15},
                "cores": os.cpu_count() or 1,
            }, util
        except Exception as exc:
            errors.append(f"cpu:{exc}")
            return {"error": str(exc)}, 0.0

    def _read_loadavg(self) -> Tuple[float, float, float]:
        with open("/proc/loadavg", "r", encoding="utf-8") as handle:
            parts = handle.read().split()
        return float(parts[0]), float(parts[1]), float(parts[2])

    def _collect_memory(self, errors: List[str]) -> Dict[str, Any]:
        try:
            info = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    info[key.strip()] = int(value.strip().split()[0])
            return {
                "mem_total_kb": info.get("MemTotal"),
                "mem_available_kb": info.get("MemAvailable"),
                "swap_total_kb": info.get("SwapTotal"),
                "swap_free_kb": info.get("SwapFree"),
            }
        except Exception as exc:
            errors.append(f"memory:{exc}")
            return {"error": str(exc)}

    def _collect_disk(self, errors: List[str], dt: Optional[float]) -> Dict[str, Any]:
        try:
            stats = self._read_diskstats()
            read_bytes = stats.read_bytes
            write_bytes = stats.write_bytes
            read_rate = 0.0
            write_rate = 0.0
            if self._prev_disk and dt:
                read_rate = max(0.0, (read_bytes - self._prev_disk.read_bytes) / dt)
                write_rate = max(0.0, (write_bytes - self._prev_disk.write_bytes) / dt)
            self._prev_disk = stats
            return {
                "read_bytes_per_sec": read_rate,
                "write_bytes_per_sec": write_rate,
            }
        except Exception as exc:
            errors.append(f"disk:{exc}")
            return {"error": str(exc)}

    def _read_diskstats(self) -> _DiskStats:
        read_sectors = 0
        write_sectors = 0
        with open("/proc/diskstats", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 14:
                    continue
                name = parts[2]
                if not self._is_disk_device(name):
                    continue
                read_sectors += int(parts[5])
                write_sectors += int(parts[9])
        return _DiskStats(read_bytes=read_sectors * 512, write_bytes=write_sectors * 512)

    def _is_disk_device(self, name: str) -> bool:
        if name.startswith("loop") or name.startswith("ram"):
            return False
        if name.startswith("sd") or name.startswith("nvme") or name.startswith("vd") or name.startswith("mmc"):
            return True
        return False

    def _collect_network(self, errors: List[str], dt: Optional[float]) -> Dict[str, Any]:
        try:
            stats = self._read_netdev()
            rx_rate = 0.0
            tx_rate = 0.0
            err_rate = 0.0
            if self._prev_net and dt:
                rx_rate = max(0.0, (stats.rx_bytes - self._prev_net.rx_bytes) / dt)
                tx_rate = max(0.0, (stats.tx_bytes - self._prev_net.tx_bytes) / dt)
                err_delta = (stats.rx_errors - self._prev_net.rx_errors) + (stats.tx_errors - self._prev_net.tx_errors)
                drop_delta = (stats.rx_drops - self._prev_net.rx_drops) + (stats.tx_drops - self._prev_net.tx_drops)
                pkt_delta = (stats.rx_packets - self._prev_net.rx_packets) + (stats.tx_packets - self._prev_net.tx_packets)
                if pkt_delta > 0:
                    err_rate = max(0.0, (err_delta + drop_delta) / pkt_delta)
            self._prev_net = stats
            return {
                "rx_bytes_per_sec": rx_rate,
                "tx_bytes_per_sec": tx_rate,
                "error_rate": err_rate,
            }
        except Exception as exc:
            errors.append(f"network:{exc}")
            return {"error": str(exc)}

    def _read_netdev(self) -> _NetStats:
        rx_bytes = tx_bytes = rx_packets = tx_packets = 0
        rx_errors = tx_errors = rx_drops = tx_drops = 0
        with open("/proc/net/dev", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                iface, rest = line.split(":", 1)
                iface = iface.strip()
                if iface == "lo":
                    continue
                parts = rest.split()
                rx_bytes += int(parts[0])
                rx_packets += int(parts[1])
                rx_errors += int(parts[2])
                rx_drops += int(parts[3])
                tx_bytes += int(parts[8])
                tx_packets += int(parts[9])
                tx_errors += int(parts[10])
                tx_drops += int(parts[11])
        return _NetStats(
            rx_bytes=rx_bytes,
            tx_bytes=tx_bytes,
            rx_packets=rx_packets,
            tx_packets=tx_packets,
            rx_errors=rx_errors,
            tx_errors=tx_errors,
            rx_drops=rx_drops,
            tx_drops=tx_drops,
        )

    def _collect_temps(self, errors: List[str]) -> Dict[str, Any]:
        temps: List[float] = []
        try:
            sensors = subprocess.run(
                ["sensors", "-j"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if sensors.stdout:
                data = json.loads(sensors.stdout)
                for chip in data.values():
                    if not isinstance(chip, dict):
                        continue
                    for feature in chip.values():
                        if not isinstance(feature, dict):
                            continue
                        for key, value in feature.items():
                            if "temp" in key and key.endswith("_input") and isinstance(value, (int, float)):
                                temps.append(float(value))
        except subprocess.TimeoutExpired:
            errors.append("temps:timeout")
        except Exception as exc:
            errors.append(f"temps:{exc}")
        cleaned = filter_temps(temps)
        return {
            "celsius": cleaned,
            "max_c": max(cleaned) if cleaned else None,
        }

    def _collect_power(self, gpu_data: Dict[str, Any], temps_data: Dict[str, Any]) -> Dict[str, Any]:
        gpu_power = []
        gpus = gpu_data.get("gpus") if isinstance(gpu_data, dict) else None
        if isinstance(gpus, list):
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                power = gpu.get("power_draw_watts")
                try:
                    if power is not None:
                        gpu_power.append(float(power))
                except (TypeError, ValueError):
                    continue
        return {
            "gpu_power_watts": gpu_power,
            "temps_max_c": temps_data.get("max_c"),
        }


_COLLECTOR = BiometricsCollector()


def collect_biometrics() -> Dict[str, Any]:
    return _COLLECTOR.collect()
