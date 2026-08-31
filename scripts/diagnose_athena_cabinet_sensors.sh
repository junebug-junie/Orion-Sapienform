#!/usr/bin/env bash
# Human-readable cabinet sensor health from boot + latest snapshots (one or two Nanos).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"

BOOT="${ORION_CABINET_BOOT_PATH:-/run/orion-sensors/boot.json}"
LATEST="${ORION_CABINET_SENSORS_PATH:-/run/orion-sensors/latest.json}"
BOOT_B="${ORION_CABINET_BOOT_B_PATH:-/run/orion-sensors/b/boot.json}"
LATEST_B="${ORION_CABINET_SENSORS_B_PATH:-/run/orion-sensors/b/latest.json}"

export BOOT LATEST BOOT_B LATEST_B
export PYTHONPATH="$RUNTIME_ROOT"

exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

boot_path = Path(__import__("os").environ["BOOT"])
latest_path = Path(__import__("os").environ["LATEST"])
boot_b_path = Path(__import__("os").environ["BOOT_B"])
latest_b_path = Path(__import__("os").environ["LATEST_B"])

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
    "mmc5603": "magnetic",
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


def print_boot(label: str, path: Path) -> None:
    print(f"== boot snapshot ({label}): {path} ==")
    boot = load(path)
    if boot is None:
        print("MISSING — flash firmware with boot diagnostics and restart reader, or reboot Nano")
        return
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


def print_live(label: str, path: Path) -> None:
    print(f"\n== live frame ({label}): {path} ==")
    latest = load(path)
    if latest is None:
        print("MISSING")
        return
    print(f"status={latest.get('status')} device={latest.get('device')}")
    frame = latest.get("frame") or {}
    for block in sorted(set(SENSOR_TO_FRAME.values())):
        present = block in frame and frame.get(block) is not None
        print(f"  {block}: {'present' if present else 'absent'}")


print_boot("nano-a", boot_path)
print_live("nano-a", latest_path)

if latest_b_path.is_file() or boot_b_path.is_file():
    print()
    print_boot("nano-b", boot_b_path)
    print_live("nano-b", latest_b_path)

    from orion.telemetry.cabinet_snapshot_merge import load_merged_cabinet_sensors

    merged = load_merged_cabinet_sensors(
        latest_path,
        secondary_path=latest_b_path,
        stale_after_sec=10.0,
    )
    print("\n== merged (biometrics view) ==")
    if merged is None:
        print("MISSING")
        sys.exit(1)
    print(f"stale={merged.get('stale')} received_at={merged.get('received_at')}")
    frame = merged.get("frame") or {}
    for block in sorted(set(SENSOR_TO_FRAME.values())):
        present = block in frame and frame.get(block) is not None
        print(f"  {block}: {'present' if present else 'absent'}")

if load(boot_path) is None and not latest_b_path.is_file():
    sys.exit(1)
PY
