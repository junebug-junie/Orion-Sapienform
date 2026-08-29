#!/usr/bin/env python3
"""Host serial reader for the Athena cabinet Nano ESP32 sensor node.

Reads line-oriented NDJSON from the stable by-id USB serial path, validates
each line against ``orion.sensor_frame.v1``, and atomically writes
``/run/orion-sensors/latest.json``. Malformed lines never overwrite the last
good frame. No normalization, EWMA, pressures, or cognition — biometrics owns
that downstream.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import ValidationError

from orion.schemas.telemetry.cabinet_sensor_frame import (
    FRAME_SCHEMA_V1,
    CabinetSensorFrameV1,
)

Status = Literal["ok", "stale", "error", "missing"]

DEFAULT_OUTPUT_PATH = Path("/run/orion-sensors/latest.json")
BOOT_OUTPUT_PATH = Path("/run/orion-sensors/boot.json")
BOOT_SCHEMA_V1 = "orion.sensor_boot.v1"
DEVICE_GLOB = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_*"
DEFAULT_STALE_AFTER_SEC = 10.0
DEFAULT_RECONNECT_SEC = 2.0
DEFAULT_BAUD = 115200


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def output_path() -> Path:
    raw = os.environ.get("ORION_CABINET_SENSORS_PATH", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_OUTPUT_PATH


def stale_after_sec() -> float:
    raw = os.environ.get("ORION_CABINET_SENSOR_STALE_AFTER_SEC", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_STALE_AFTER_SEC


def discover_device(glob_pattern: str = DEVICE_GLOB) -> Optional[str]:
    """Return the first sorted by-id path matching the Nano ESP32."""
    matches = sorted(glob.glob(glob_pattern))
    if not matches:
        return None
    return matches[0]


def device_glob_pattern() -> str:
    raw = os.environ.get("ORION_CABINET_DEVICE_GLOB", "").strip()
    return raw or DEVICE_GLOB


def boot_output_path() -> Path:
    raw = os.environ.get("ORION_CABINET_BOOT_PATH", "").strip()
    if raw:
        return Path(raw)
    return BOOT_OUTPUT_PATH


def parse_boot_line(line: str) -> Optional[dict[str, Any]]:
    """Parse a boot diagnostic NDJSON line. Returns None if not boot schema."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") != BOOT_SCHEMA_V1:
        return None
    return raw


def ingest_boot_line(line: str, *, boot_path: Optional[Path] = None) -> bool:
    boot = parse_boot_line(line)
    if boot is None:
        return False
    atomic_write_json(boot_path or boot_output_path(), boot)
    return True


def parse_frame_line(line: str) -> Optional[dict[str, Any]]:
    """Parse and validate one NDJSON line. Returns None for malformed input."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema") != FRAME_SCHEMA_V1:
        return None
    try:
        model = CabinetSensorFrameV1.model_validate(raw)
    except ValidationError:
        return None
    return model.model_dump(by_alias=True)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via temp file + rename in the destination dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class SnapshotState:
    """In-memory reader state with last-good preservation."""

    def __init__(self, stale_after_sec: float = DEFAULT_STALE_AFTER_SEC) -> None:
        self.stale_after_sec = stale_after_sec
        self.last_good_frame: Optional[dict[str, Any]] = None
        self.last_good_received_at: Optional[str] = None
        self.device: Optional[str] = None
        self.error: Optional[str] = None

    def ingest_good_frame(
        self,
        frame: dict[str, Any],
        *,
        device: str,
        received_at: str,
    ) -> None:
        self.last_good_frame = frame
        self.last_good_received_at = received_at
        self.device = device
        self.error = None

    def ingest_bad_line(self) -> None:
        """Malformed line — preserve last good snapshot fields."""

    def set_error(self, message: str, *, device: Optional[str] = None) -> None:
        self.error = message
        if device is not None:
            self.device = device

    def set_missing_device(self) -> None:
        self.error = "device not found"

    def process_line(self, line: str, *, device: str, received_at: Optional[str] = None) -> bool:
        """Validate one serial line. Returns True when the frame was accepted."""
        frame = parse_frame_line(line)
        if frame is None:
            self.ingest_bad_line()
            return False
        self.ingest_good_frame(
            frame,
            device=device,
            received_at=received_at or utc_now_iso(),
        )
        return True

    def compute_status(self, now: Optional[datetime] = None) -> Status:
        now = now or datetime.now(timezone.utc)
        if self.last_good_frame is None and self.device is None:
            return "missing"
        if self.last_good_received_at:
            received = datetime.fromisoformat(
                self.last_good_received_at.replace("Z", "+00:00")
            )
            age = (now - received).total_seconds()
            if age > self.stale_after_sec:
                return "stale"
        if self.error and self.last_good_frame is None:
            return "error"
        if self.error:
            return "error"
        if self.last_good_frame is None:
            return "missing"
        return "ok"

    def to_snapshot(self, now: Optional[datetime] = None) -> dict[str, Any]:
        status = self.compute_status(now)
        snap: dict[str, Any] = {
            "status": status,
            "received_at": self.last_good_received_at,
            "device": self.device,
            "frame": self.last_good_frame,
        }
        if self.error and status in ("error", "missing"):
            snap["error"] = self.error
        return snap


def write_snapshot(path: Path, state: SnapshotState, now: Optional[datetime] = None) -> None:
    atomic_write_json(path, state.to_snapshot(now))


def run_loop(
    *,
    output: Path,
    state: SnapshotState,
    reconnect_sec: float = DEFAULT_RECONNECT_SEC,
    baud: int = DEFAULT_BAUD,
    device_glob_pattern: str = DEVICE_GLOB,
    sleep_fn=time.sleep,
) -> None:
    import serial  # pyserial — installed by setup_athena_cabinet_sensors.sh

    while True:
        device = discover_device(device_glob_pattern)
        if device is None:
            state.set_missing_device()
            write_snapshot(output, state)
            sleep_fn(reconnect_sec)
            continue

        try:
            ser = serial.Serial(device, baudrate=baud, timeout=1.0)
        except serial.SerialException as exc:
            state.set_error(str(exc), device=device)
            write_snapshot(output, state)
            sleep_fn(reconnect_sec)
            continue

        state.device = device
        state.error = None
        write_snapshot(output, state)

        try:
            while True:
                try:
                    raw = ser.readline()
                except serial.SerialException as exc:
                    state.set_error(str(exc), device=device)
                    write_snapshot(output, state)
                    break

                if not raw:
                    write_snapshot(output, state)
                    continue

                text = raw.decode("utf-8", errors="replace")
                if ingest_boot_line(text):
                    continue
                state.process_line(text, device=device)
                write_snapshot(output, state)
        finally:
            ser.close()


def main(argv: Optional[list[str]] = None) -> int:
    _ = argv
    path = output_path()
    state = SnapshotState(stale_after_sec=stale_after_sec())
    try:
        run_loop(output=path, state=state, device_glob_pattern=device_glob_pattern())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
