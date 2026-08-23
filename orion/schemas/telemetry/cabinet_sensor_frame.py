from __future__ import annotations

"""Host-local transport contract: Nano ESP32 cabinet sensor frame.

This is NOT a bus payload and is deliberately not registered in
`orion/schemas/registry.py`. It validates the NDJSON the Arduino Nano ESP32
firmware writes over USB serial, and the atomic snapshot
(`scripts/sensor_serial_reader.py`) writes to `/run/orion-sensors/latest.json`
on the Athena host. `services/orion-biometrics` reads that file (bind-mounted
read-only) and folds a validated frame into `BiometricsSampleV1.sensors` --
see that model's docstring for the raw-vs-normalized boundary this schema
sits on the raw side of.

Firmware contract (v1), one NDJSON line per ~1s tick:

    {
      "schema": "orion.sensor_frame.v1",
      "seq": 4812,
      "uptime_ms": 992813,
      "environment": {"temp_c": 24.61, "humidity_pct": 31.8,
                       "pressure_hpa": 857.2, "gas_resistance_ohm": 138421},
      "uv": {"raw": 17, "als_raw": 1292},
      "magnetic": {"x_ut": 31.2, "y_ut": -8.4, "z_ut": 42.1, "magnitude_ut": 53.0},
      "particulate": {"pm1_ug_m3": 2, "pm25_ug_m3": 4, "pm10_ug_m3": 5},
      "lidar": {"distance_mm": 438, "status": 0},
      "imu": {"accel_x": 0.01, "accel_y": -0.02, "accel_z": 9.79,
              "yaw_deg": 12.4, "pitch_deg": 0.8, "roll_deg": -1.1}
    }

Every sub-payload is OPTIONAL -- a sensor that failed init on the Nano side
(BME680 not found, LiDAR I2C timeout, etc.) omits its key rather than sending
zeros. That is the same absent-is-not-zero invariant
`orion.telemetry.biometrics_pipeline.extract_measurements` already enforces
for host hardware telemetry, extended one hop further out to the firmware
itself. `orion/telemetry/cabinet_sensors.py` (the raw->measurements/pressures
step) preserves it: a missing sub-payload here means an absent key there, not
a 0.0.

Units are raw and native (uT, mm, ug/m3, hPa, ...) -- this model does not
normalize or judge. See `orion/telemetry/cabinet_sensors.py` for that step.
Audio is not part of the Nano frame (MAX9814 removed from this design).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

FRAME_SCHEMA_V1 = "orion.sensor_frame.v1"


class CabinetEnvironmentV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    gas_resistance_ohm: Optional[float] = None


class CabinetUvV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw: Optional[float] = None
    als_raw: Optional[float] = None


class CabinetMagneticV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x_ut: Optional[float] = None
    y_ut: Optional[float] = None
    z_ut: Optional[float] = None
    magnitude_ut: Optional[float] = None


class CabinetParticulateV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pm1_ug_m3: Optional[float] = None
    pm25_ug_m3: Optional[float] = None
    pm10_ug_m3: Optional[float] = None


class CabinetLidarV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    distance_mm: Optional[float] = None
    # VL53L1X RangeStatus: 0 == valid. Any other value means distance_mm, if
    # present at all, should not be trusted -- cabinet_sensors.py checks this.
    status: Optional[int] = None


class CabinetImuV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None


class CabinetSensorFrameV1(BaseModel):
    """One validated NDJSON frame from the Nano ESP32 firmware."""

    model_config = ConfigDict(extra="ignore")

    schema_: str = Field(alias="schema")
    seq: int
    uptime_ms: Optional[int] = None

    environment: Optional[CabinetEnvironmentV1] = None
    uv: Optional[CabinetUvV1] = None
    magnetic: Optional[CabinetMagneticV1] = None
    particulate: Optional[CabinetParticulateV1] = None
    lidar: Optional[CabinetLidarV1] = None
    imu: Optional[CabinetImuV1] = None
