#!/usr/bin/env bash
# Human-readable cabinet sensor health from boot + latest snapshots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"

BOOT="${ORION_CABINET_BOOT_PATH:-/run/orion-sensors/boot.json}"
LATEST="${ORION_CABINET_SENSORS_PATH:-/run/orion-sensors/latest.json}"

export BOOT LATEST
export PYTHONPATH="$RUNTIME_ROOT"

exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

boot_path = Path(__import__("os").environ["BOOT"])
latest_path = Path(__import__("os").environ["LATEST"])

DETAIL_HINTS = {
    "not_on_bus": "no I2C ACK at boot — check wiring, power, or sensor not populated",
    "begin_failed": "device ACKs on I2C but driver init failed — address conflict or bad part",
    "probe_nack": "PMSA003I not responding at 0x12 — wrong variant (UART PMS5003?) or SET pin",
    "uart_no_sync": "BNO085 UART-RVC: no 0xAA packets — check D6/D7, PS0/PS1 jumper for RVC mode",
}

SENSOR_TO_FRAME = {
    "bme680": "environment",
    "ltr390": "uv",
    "lis3mdl": "magnetic",
    "pmsa003i": "particulate",
    "vl53l1x": "lidar",
    "bno085": "imu",
}


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


print(f"== boot snapshot: {boot_path} ==")
boot = load(boot_path)
if boot is None:
    print("MISSING — flash firmware with boot diagnostics and restart reader, or reboot Nano")
else:
    i2c = boot.get("i2c") or {}
    addrs = i2c.get("addresses") or []
    print(f"I2C scan ({i2c.get('sda_pin','?')}/{i2c.get('scl_pin','?')}): {', '.join(addrs) or 'none'}")
    sensors = boot.get("sensors") or {}
    for name, meta in sensors.items():
        if not isinstance(meta, dict):
            continue
        ok = meta.get("ok")
        detail = meta.get("detail")
        addr = meta.get("addr")
        line = f"  {name}: {'OK' if ok else 'FAIL'}"
        if addr:
            line += f" addr={addr}"
        if detail:
            line += f" ({detail})"
            hint = DETAIL_HINTS.get(str(detail))
            if hint:
                line += f"\n      → {hint}"
        print(line)

print(f"\n== live frame: {latest_path} ==")
latest = load(latest_path)
if latest is None:
    print("MISSING")
    sys.exit(1)

print(f"status={latest.get('status')} device={latest.get('device')}")
frame = latest.get("frame") or {}
for sensor, block in SENSOR_TO_FRAME.items():
    present = block in frame and frame.get(block) is not None
    print(f"  {block}: {'present' if present else 'absent'}")

if boot is None:
    sys.exit(1)
PY
