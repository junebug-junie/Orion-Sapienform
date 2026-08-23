from __future__ import annotations

"""Nano ESP32 cabinet sensor node: raw measurements and baseline-relative pressures.

ROADMAP: physical senses. `orion-biometrics` already measures the host
machines (CPU/GPU/thermal/power/...); this module extends the same
raw->measurements->pressures shape to the physical cabinet environment
(temperature, humidity, air quality, magnetic field, vibration, proximity,
UV) via a Nano ESP32 reachable over host USB serial.
See `orion/schemas/telemetry/cabinet_sensor_frame.py` for the wire contract
and `services/orion-biometrics/README.md` ("Cabinet sensor node") for the
full pipeline.

All v1 cabinet pressures are baseline-relative only: EWMA band, delta,
volatility -> anomaly in [0, 1]. No absolute comfort or AQI thresholds.
Pressure keys: cabinet_climate_activity, cabinet_particulate_activity,
cabinet_em_activity, cabinet_uv_activity, cabinet_vibration_activity,
cabinet_proximity_activity.

HAND-VERIFIED REST POINT (per CLAUDE.md 0A step 4): for a raw magnitude
that is EXACTLY constant tick to tick, `EwmaBand.update()` computes
`delta = value - mean == 0` every call, so `dev` converges to exactly `0.0`;
`EwmaBand.normalize()` then returns exactly `0.0`; and
`InductionTracker.volatility` also converges to exactly `0.0`. Proven in
`tests/test_cabinet_sensors.py::test_activity_signal_rests_at_zero_for_
constant_input`.
"""

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, Optional

from orion.signals.normalization import EwmaBand, InductionTracker, clamp01

GRAVITY_MPS2 = 9.80665

CLIMATE_MEASUREMENT_KEYS = (
    "cabinet_temp_c",
    "cabinet_humidity_pct",
    "cabinet_pressure_hpa",
    "cabinet_gas_resistance_ohm",
)
PARTICULATE_MEASUREMENT_KEYS = (
    "cabinet_pm1_ug_m3",
    "cabinet_pm25_ug_m3",
    "cabinet_pm10_ug_m3",
)


def _as_float(value: Any) -> Optional[float]:
    """Same strict parse as biometrics_pipeline.extract_measurements: None
    for anything that is not a real, finite, non-bool number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


@dataclass
class CabinetPressureConfig:
    climate_band_alpha: float = 0.1
    particulate_band_alpha: float = 0.1
    em_band_alpha: float = 0.1
    uv_band_alpha: float = 0.1
    vibration_band_alpha: float = 0.1
    proximity_band_alpha: float = 0.1


@dataclass
class _ActivityChannel:
    band: EwmaBand
    tracker: InductionTracker = field(default_factory=InductionTracker)

    def activity(self, name: str, raw: float) -> float:
        self.band.update(raw)
        level = self.band.normalize(raw)
        return clamp01(self.tracker.update(name, level).volatility)


def extract_cabinet_measurements(sensors: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Raw cabinet physical quantities, in native units, keyed with the unit
    in the name -- same absent-is-not-zero invariant as
    biometrics_pipeline.extract_measurements. `sensors` is the
    BiometricsSampleV1.sensors dict; returns {} (not None) when there is
    nothing to report, same "producer ran, measured nothing it could vouch
    for" meaning documented on that field.
    """
    out: Dict[str, float] = {}
    if not isinstance(sensors, dict) or sensors.get("stale"):
        return out
    frame = sensors.get("frame")
    if not isinstance(frame, dict):
        return out

    def put(key: str, value: Optional[float]) -> None:
        if value is not None:
            out[key] = value

    env = frame.get("environment") if isinstance(frame.get("environment"), dict) else {}
    put("cabinet_temp_c", _as_float(env.get("temp_c")))
    put("cabinet_humidity_pct", _as_float(env.get("humidity_pct")))
    put("cabinet_pressure_hpa", _as_float(env.get("pressure_hpa")))
    put("cabinet_gas_resistance_ohm", _as_float(env.get("gas_resistance_ohm")))

    uv = frame.get("uv") if isinstance(frame.get("uv"), dict) else {}
    put("cabinet_uv_raw", _as_float(uv.get("raw")))
    put("cabinet_als_raw", _as_float(uv.get("als_raw")))

    mag = frame.get("magnetic") if isinstance(frame.get("magnetic"), dict) else {}
    put("cabinet_magnetic_ut", _as_float(mag.get("magnitude_ut")))

    part = frame.get("particulate") if isinstance(frame.get("particulate"), dict) else {}
    put("cabinet_pm1_ug_m3", _as_float(part.get("pm1_ug_m3")))
    put("cabinet_pm25_ug_m3", _as_float(part.get("pm25_ug_m3")))
    put("cabinet_pm10_ug_m3", _as_float(part.get("pm10_ug_m3")))

    # LiDAR: only trust distance_mm when status == 0 (VL53L1X RangeStatus,
    # 0 == valid). Any other status means the reading, if present at all,
    # is a fault code or an out-of-range/no-target result, not a distance.
    lidar = frame.get("lidar") if isinstance(frame.get("lidar"), dict) else {}
    if lidar.get("status") == 0:
        put("cabinet_lidar_mm", _as_float(lidar.get("distance_mm")))

    imu = frame.get("imu") if isinstance(frame.get("imu"), dict) else {}
    ax, ay, az = _as_float(imu.get("accel_x")), _as_float(imu.get("accel_y")), _as_float(imu.get("accel_z"))
    if ax is not None and ay is not None and az is not None:
        accel_g = sqrt(ax * ax + ay * ay + az * az) / GRAVITY_MPS2
        out["cabinet_vibration_g"] = abs(accel_g - 1.0)

    return out


class CabinetSensorTracker:
    """Per-node persistent EWMA state for baseline-relative cabinet activity signals."""

    def __init__(self, cfg: CabinetPressureConfig) -> None:
        self.cfg = cfg
        self.climate_channels: Dict[str, _ActivityChannel] = {}
        self.particulate_channels: Dict[str, _ActivityChannel] = {}
        self.em_channel = _ActivityChannel(EwmaBand(alpha=cfg.em_band_alpha))
        self.uv_channel = _ActivityChannel(EwmaBand(alpha=cfg.uv_band_alpha))
        self.vibration_channel = _ActivityChannel(EwmaBand(alpha=cfg.vibration_band_alpha))
        self.proximity_channel = _ActivityChannel(EwmaBand(alpha=cfg.proximity_band_alpha))

    def _climate_channel(self, key: str) -> _ActivityChannel:
        ch = self.climate_channels.get(key)
        if ch is None:
            ch = _ActivityChannel(EwmaBand(alpha=self.cfg.climate_band_alpha))
            self.climate_channels[key] = ch
        return ch

    def _particulate_channel(self, key: str) -> _ActivityChannel:
        ch = self.particulate_channels.get(key)
        if ch is None:
            ch = _ActivityChannel(EwmaBand(alpha=self.cfg.particulate_band_alpha))
            self.particulate_channels[key] = ch
        return ch

    def _domain_activity(self, keys: tuple[str, ...], measurements: Dict[str, float], getter) -> Optional[float]:
        activities = []
        for key in keys:
            if key in measurements:
                activities.append(getter(key, measurements[key]))
        if not activities:
            return None
        return max(activities)


def compute_cabinet_pressures(
    measurements: Dict[str, float],
    tracker: CabinetSensorTracker,
) -> Dict[str, float]:
    """Baseline-relative 0-1 cabinet activity signals, computed ONLY for
    measurements that are actually present. Returns {} when `measurements`
    is empty -- no fabricated 0.0 for unmeasured channels.
    """
    out: Dict[str, float] = {}

    climate = tracker._domain_activity(
        CLIMATE_MEASUREMENT_KEYS,
        measurements,
        lambda k, v: tracker._climate_channel(k).activity(k, v),
    )
    if climate is not None:
        out["cabinet_climate_activity"] = climate

    particulate = tracker._domain_activity(
        PARTICULATE_MEASUREMENT_KEYS,
        measurements,
        lambda k, v: tracker._particulate_channel(k).activity(k, v),
    )
    if particulate is not None:
        out["cabinet_particulate_activity"] = particulate

    if "cabinet_magnetic_ut" in measurements:
        out["cabinet_em_activity"] = tracker.em_channel.activity(
            "em", measurements["cabinet_magnetic_ut"]
        )
    if "cabinet_uv_raw" in measurements:
        out["cabinet_uv_activity"] = tracker.uv_channel.activity(
            "uv", measurements["cabinet_uv_raw"]
        )
    if "cabinet_vibration_g" in measurements:
        out["cabinet_vibration_activity"] = tracker.vibration_channel.activity(
            "vibration", measurements["cabinet_vibration_g"]
        )
    if "cabinet_lidar_mm" in measurements:
        out["cabinet_proximity_activity"] = tracker.proximity_channel.activity(
            "proximity", measurements["cabinet_lidar_mm"]
        )

    return out
