#!/usr/bin/env bash
# Validate cabinet sensor NDJSON frames (direct serial or via live reader snapshot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"
NEEDED="${1:-3}"
TIMEOUT_SEC="${SMOKE_SERIAL_TIMEOUT_SEC:-30}"
SNAP="${ORION_CABINET_SENSORS_PATH:-/run/orion-sensors/latest.json}"

export PYTHONPATH="$RUNTIME_ROOT"
export ORION_CABINET_SERIAL_NEEDED="$NEEDED"
export ORION_CABINET_SERIAL_TIMEOUT_SEC="$TIMEOUT_SEC"
export ORION_CABINET_SENSORS_PATH="$SNAP"

# After DFU upload the by-id node can lag a few seconds.
export ORION_CABINET_DISCOVER_WAIT_SEC="${ORION_CABINET_DISCOVER_WAIT_SEC:-30}"

if systemctl is-active --quiet orion-cabinet-sensors.service 2>/dev/null; then
    echo "note: orion-cabinet-sensors.service owns serial; smoking via ${SNAP}" >&2
    exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from scripts.orion_cabinet_sensor_reader import parse_frame_line

snap = Path(os.environ["ORION_CABINET_SENSORS_PATH"])
needed = int(os.environ["ORION_CABINET_SERIAL_NEEDED"])
timeout_sec = float(os.environ["ORION_CABINET_SERIAL_TIMEOUT_SEC"])

if not snap.is_file():
    print(f"error: snapshot missing at {snap}", file=sys.stderr)
    sys.exit(1)

deadline = time.monotonic() + timeout_sec
seen_seq: set[int] = set()
last_device = ""

while len(seen_seq) < needed and time.monotonic() < deadline:
    try:
        payload = json.loads(snap.read_text())
    except (OSError, json.JSONDecodeError):
        time.sleep(0.25)
        continue

    status = payload.get("status")
    frame = payload.get("frame")
    if status != "ok" or not isinstance(frame, dict):
        time.sleep(0.25)
        continue

    line = json.dumps(frame, separators=(",", ":"))
    if parse_frame_line(line) is None:
        time.sleep(0.25)
        continue

    seq = frame.get("seq")
    if isinstance(seq, int):
        seen_seq.add(seq)
    last_device = str(payload.get("device") or snap)
    time.sleep(0.25)

if len(seen_seq) < needed:
    print(
        f"error: only {len(seen_seq)}/{needed} distinct valid seq values in {timeout_sec}s "
        f"(last device={last_device or 'unknown'})",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"ok: {len(seen_seq)} valid cabinet sensor frames via {snap} (device={last_device})")
PY
fi

DEVICE="$("$SCRIPT_DIR/discover_athena_cabinet_serial.sh")"
export ORION_CABINET_SERIAL_DEVICE="$DEVICE"

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
