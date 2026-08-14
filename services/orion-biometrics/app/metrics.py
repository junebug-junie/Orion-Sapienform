# services/orion-biometrics/app/metrics.py
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.settings import settings
from app.utils import collect_gpu_stats
from orion.telemetry.biometrics_pipeline import filter_temps

logger = logging.getLogger("orion-biometrics")

# Block-device name grammar for /proc/diskstats. Whole devices vs their partitions:
#   sda / sda1        nvme0n1 / nvme0n1p1        vda / vda1        mmcblk0 / mmcblk0p1
# NVMe and mmc separate the partition with a literal 'p'; SCSI/SATA/virtio just append digits.
_WHOLE_DISK_RE = re.compile(r"^(sd[a-z]+|nvme\d+n\d+|vd[a-z]+|mmcblk\d+)")
_PARTITION_RE = re.compile(r"^(sd[a-z]+\d+|nvme\d+n\d+p\d+|vd[a-z]+\d+|mmcblk\d+p\d+)$")


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
        self._prev_net_scope: Optional[str] = None
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
        """Whole block devices only -- never their partitions.

        `/proc/diskstats` has a row for `sda` AND a row for `sda1`, and the partition's
        counters are also included in the whole disk's. Accepting both double-counts every
        byte on any partitioned device. Measured on athena 2026-08-14 over the same 10 s
        window: 19,378,995 B/s counting partitions vs 9,905,766 B/s without -- a factor of
        1.956, matching the 1.97 measured independently on 2026-08-13.

        athena has ten devices of which six are partitioned (nvme0n1p1, sdd1-3, sde1, sdf1,
        sdg1-2), so this is not an edge case here, it is the normal reading.
        """
        if name.startswith("loop") or name.startswith("ram"):
            return False
        if not _WHOLE_DISK_RE.match(name):
            return False
        return not _PARTITION_RE.match(name)

    # ---------------------------------------------------------------- host namespace

    def _host_path(self, kind: str, rel: str) -> Optional[str]:
        """Resolve a path under the read-only host /proc or /sys bind mount, if mounted.

        Returns None when the mount is absent, which is the correct answer for a collector
        running outside a container or in a deployment that predates these mounts -- the
        callers fall back to the container's own view and SAY SO in the payload rather than
        silently reporting a container-scoped number as a node-scoped one.
        """
        root = settings.HOST_PROC_PATH if kind == "proc" else settings.HOST_SYS_PATH
        if not root:
            return None
        candidate = os.path.join(root, rel)
        return candidate if os.path.exists(candidate) else None

    def _physical_interfaces(self) -> Optional[List[str]]:
        """Up physical NICs, from the host sysfs mount. None when it is unavailable.

        Two filters, both load-bearing:

        - `device` symlink present -> a real NIC behind a real driver. docker0, br-*, veth*,
          tailscale0 and lo have no `device` and are excluded. This matters for the DENOMINATOR
          because the docker bridges report a fabricated `speed` of 10000 Mb/s on athena, which
          would inflate a link-capacity sum by 40 Gb/s of hardware that does not exist.
        - `operstate == up` -> athena has eno1..eno6 and only eno1 is plugged in. Counting the
          five dark ports would claim 6 Gb/s of capacity against one 1 Gb link.

        It matters for the NUMERATOR too: a packet leaving a container traverses veth -> bridge
        -> eno1 and is counted once on each, so summing every interface multiply-counts
        container traffic. Physical-only is the definition that answers "how close is this
        node's uplink to saturation", which is the only question `net_pressure` ever claimed to
        answer.
        """
        base = self._host_path("sys", "class/net")
        if not base:
            return None
        found: List[str] = []
        try:
            for name in sorted(os.listdir(base)):
                iface = os.path.join(base, name)
                if not os.path.exists(os.path.join(iface, "device")):
                    continue
                try:
                    with open(os.path.join(iface, "operstate"), "r", encoding="utf-8") as fh:
                        if fh.read().strip() != "up":
                            continue
                except OSError:
                    continue
                found.append(name)
        except OSError:
            return None
        return found

    def _link_speed_mbps(self, interfaces: List[str]) -> Optional[float]:
        """Summed link speed of the given NICs, in Mb/s, read from the kernel.

        This replaces the `NET_BW_MBPS` guess with a measurement. The constant was 125 MB/s
        (1 Gb/s) fleet-wide across three heterogeneous hosts -- correct for athena by
        coincidence, unverified for atlas and circe. A number the kernel already knows should
        not be a hand-maintained env key on three machines.

        Returns None (not 0.0) when nothing could be read, so the caller falls back to the
        configured constant instead of dividing by zero or treating "unknown" as "no capacity".
        """
        base = self._host_path("sys", "class/net")
        if not base:
            return None
        total = 0.0
        for name in interfaces:
            try:
                with open(os.path.join(base, name, "speed"), "r", encoding="utf-8") as fh:
                    value = float(fh.read().strip())
            except (OSError, ValueError):
                continue
            # -1 means "unknown to the driver" (tailscale0 reports this); not a capacity.
            if value > 0:
                total += value
        return total if total > 0 else None

    def _collect_network(self, errors: List[str], dt: Optional[float]) -> Dict[str, Any]:
        try:
            interfaces = self._physical_interfaces()
            host_netdev = self._host_path("proc", "1/net/dev")
            # PID 1 on the host is the host init, so its /proc/<pid>/net/dev is the HOST network
            # namespace -- the one where the real NICs live. The container's own /proc/net/dev is
            # namespaced to its veth pair and sees only this service's own traffic. Measured on
            # athena 2026-08-14: host 159,304 B/s vs container 1,357 B/s. On 2026-08-13 the same
            # comparison gave ~1000x. The ratio is unstable BECAUSE it is not a scale error --
            # the two numbers are different quantities, and no calibration constant could have
            # rescued the container-scoped one.
            if host_netdev and interfaces:
                stats = self._read_netdev(path=host_netdev, only=interfaces)
                scope = "host"
            else:
                stats = self._read_netdev()
                scope = "container"
            rx_rate = 0.0
            tx_rate = 0.0
            err_rate = 0.0
            # Counters from different namespaces are not comparable. If the scope flips between
            # ticks (the host mount appears on redeploy, or disappears), the previous sample
            # belongs to a different set of interfaces and the delta would be meaningless --
            # usually a huge positive spike. Drop the baseline and report 0 for one tick.
            if self._prev_net and dt and self._prev_net_scope == scope:
                rx_rate = max(0.0, (stats.rx_bytes - self._prev_net.rx_bytes) / dt)
                tx_rate = max(0.0, (stats.tx_bytes - self._prev_net.tx_bytes) / dt)
                err_delta = (stats.rx_errors - self._prev_net.rx_errors) + (stats.tx_errors - self._prev_net.tx_errors)
                drop_delta = (stats.rx_drops - self._prev_net.rx_drops) + (stats.tx_drops - self._prev_net.tx_drops)
                pkt_delta = (stats.rx_packets - self._prev_net.rx_packets) + (stats.tx_packets - self._prev_net.tx_packets)
                if pkt_delta > 0:
                    err_rate = max(0.0, (err_delta + drop_delta) / pkt_delta)
            self._prev_net = stats
            self._prev_net_scope = scope
            out: Dict[str, Any] = {
                "rx_bytes_per_sec": rx_rate,
                "tx_bytes_per_sec": tx_rate,
                "error_rate": err_rate,
                # Which namespace produced the bytes above. A consumer that sums these across
                # nodes needs to know a "container" reading is this service's own veth traffic
                # and NOT the node's, so it can exclude the node rather than understate the sum.
                "scope": scope,
            }
            if interfaces:
                out["interfaces"] = list(interfaces)
                link_mbps = self._link_speed_mbps(interfaces)
                if link_mbps is not None:
                    # The measured denominator for net_pressure, replacing NET_BW_MBPS.
                    out["link_mbps"] = link_mbps
            return out
        except Exception as exc:
            errors.append(f"network:{exc}")
            return {"error": str(exc)}

    def _read_netdev(self, path: str = "/proc/net/dev", only: Optional[List[str]] = None) -> _NetStats:
        allowed = set(only) if only is not None else None
        rx_bytes = tx_bytes = rx_packets = tx_packets = 0
        rx_errors = tx_errors = rx_drops = tx_drops = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                iface, rest = line.split(":", 1)
                iface = iface.strip()
                if iface == "lo":
                    continue
                if allowed is not None and iface not in allowed:
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


def collect_disk_capacity(mounts: Dict[str, str]) -> Dict[str, Any]:
    """Real host disk-capacity (percent-used) telemetry, keyed by mount name.

    This is deliberately separate from `BiometricsCollector._collect_disk()`, which
    reads `/proc/diskstats` for I/O *throughput* (read/write bytes-per-sec) and feeds
    `BiometricsPipeline`'s `disk` pressure -- a different metric with an existing
    consumer chain (pipeline -> summary.pressures.disk -> grammar_emit's
    disk_pressure_signal -> field-digester's disk_pressure lattice channel). Disk
    *capacity* (how full a filesystem is) has no existing collector anywhere in this
    repo (confirmed by design-doc grep: zero hits for `disk_usage`/
    `docker system prune` under `scripts/`) and is not part of that pipeline -- it
    rides the bus-native `SystemHealthV1` heartbeat's free-form `details` dict
    instead. See `services/orion-biometrics/README.md`'s "Disk capacity telemetry"
    section for the documented `details.disk_usage_pct.<mount_name>` key convention.

    `mounts` maps a short mount name (e.g. "docker") to the in-container path it is
    bind-mounted at (e.g. "/host_mnt/docker", read-only -- see docker-compose.yml).
    A mount that doesn't exist inside this container (not yet bind-mounted on this
    node, or a host path that genuinely doesn't exist on atlas/circe) is skipped, not
    fatal -- one missing mount must never take down the whole heartbeat.
    """
    percent_used: Dict[str, float] = {}
    errors: Dict[str, str] = {}
    for name, path in mounts.items():
        try:
            if not path or not os.path.isdir(path):
                errors[name] = "not_mounted"
                continue
            if not os.path.ismount(path):
                # If a bind-mount source doesn't exist on the host, Docker silently
                # auto-creates an empty directory at the target instead of failing --
                # os.path.isdir() alone would report that empty dir as present and
                # shutil.disk_usage() would then report the *container's own* overlay
                # filesystem capacity, a plausible-looking but meaningless number, not
                # a "not mounted" error. os.path.ismount() catches that case.
                errors[name] = "not_mounted"
                continue
            usage = shutil.disk_usage(path)
            if usage.total <= 0:
                errors[name] = "zero_total_bytes"
                continue
            percent_used[name] = round((usage.used / usage.total) * 100.0, 2)
        except Exception as exc:
            errors[name] = str(exc)

    result: Dict[str, Any] = {"disk_usage_pct": percent_used}
    if errors:
        result["disk_usage_errors"] = errors
    return result
