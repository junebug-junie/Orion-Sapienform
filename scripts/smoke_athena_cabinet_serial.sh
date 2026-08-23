#!/usr/bin/env bash
# Read N valid cabinet sensor NDJSON frames from the by-id serial device.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
VENV_PYTHON="$REPO_ROOT/venv/bin/python"
NEEDED="${1:-3}"
TIMEOUT_SEC="${SMOKE_SERIAL_TIMEOUT_SEC:-30}"

DEVICE="$("$SCRIPT_DIR/discover_athena_cabinet_serial.sh")"

export PYTHONPATH="$REPO_ROOT"
export ORION_CABINET_SERIAL_DEVICE="$DEVICE"
export ORION_CABINET_SERIAL_NEEDED="$NEEDED"
export ORION_CABINET_SERIAL_TIMEOUT_SEC="$TIMEOUT_SEC"

exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import os
import sys
import time

from scripts.orion_cabinet_sensor_reader import parse_frame_line

try:
    import serial
except ImportError:
    print("error: pyserial not installed; run scripts/setup_athena_cabinet_sensors.sh", file=sys.stderr)
    sys.exit(1)

device = os.environ["ORION_CABINET_SERIAL_DEVICE"]
needed = int(os.environ["ORION_CABINET_SERIAL_NEEDED"])
timeout_sec = float(os.environ["ORION_CABINET_SERIAL_TIMEOUT_SEC"])

deadline = time.monotonic() + timeout_sec
valid = 0
malformed = 0

with serial.Serial(device, baudrate=115200, timeout=1.0) as ser:
    while valid < needed and time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        if parse_frame_line(text) is None:
            malformed += 1
            continue
        valid += 1

if valid < needed:
    print(
        f"error: only {valid}/{needed} valid frames in {timeout_sec}s "
        f"({malformed} malformed lines)",
        file=sys.stderr,
    )
    sys.exit(1)

if valid == 0 and malformed > 0:
    print("error: only malformed serial lines observed", file=sys.stderr)
    sys.exit(1)

print(f"ok: {valid} valid cabinet sensor frames from {device}")
PY
