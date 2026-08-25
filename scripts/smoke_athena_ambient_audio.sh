#!/usr/bin/env bash
# Validate ambient audio snapshots (direct arecord or via live reader snapshot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/orion_runtime_root.sh
source "$SCRIPT_DIR/lib/orion_runtime_root.sh"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
RUNTIME_ROOT="$(orion_resolve_runtime_root "$REPO_ROOT")"
VENV_PYTHON="$(orion_resolve_runtime_python "$REPO_ROOT")"
NEEDED="${1:-3}"
TIMEOUT_SEC="${SMOKE_AMBIENT_AUDIO_TIMEOUT_SEC:-30}"
SNAP="${ORION_AMBIENT_AUDIO_PATH:-/run/orion-audio/latest.json}"

export PYTHONPATH="$RUNTIME_ROOT"
export ORION_AMBIENT_AUDIO_NEEDED="$NEEDED"
export ORION_AMBIENT_AUDIO_TIMEOUT_SEC="$TIMEOUT_SEC"
export ORION_AMBIENT_AUDIO_PATH="$SNAP"

if systemctl is-active --quiet orion-ambient-audio.service 2>/dev/null; then
    echo "note: orion-ambient-audio.service owns capture; smoking via ${SNAP}" >&2
    exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from orion.schemas.telemetry.ambient_audio import AmbientAudioSnapshotV1

snap = Path(os.environ["ORION_AMBIENT_AUDIO_PATH"])
needed = int(os.environ["ORION_AMBIENT_AUDIO_NEEDED"])
timeout_sec = float(os.environ["ORION_AMBIENT_AUDIO_TIMEOUT_SEC"])

if not snap.is_file():
    print(f"error: snapshot missing at {snap}", file=sys.stderr)
    sys.exit(1)

deadline = time.monotonic() + timeout_sec
ok_count = 0
last_device = ""
last_rms = 0.0

while ok_count < needed and time.monotonic() < deadline:
    try:
        payload = json.loads(snap.read_text())
        model = AmbientAudioSnapshotV1.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError):
        time.sleep(0.25)
        continue

    if model.status != "ok":
        time.sleep(0.25)
        continue

    ok_count += 1
    last_device = model.device
    last_rms = model.rms
    time.sleep(0.25)

if ok_count < needed:
    print(
        f"error: only {ok_count}/{needed} valid ok snapshots in {timeout_sec}s "
        f"(last device={last_device or 'unknown'})",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"ok: {ok_count} valid ambient audio snapshots via {snap} "
    f"(device={last_device}, last_rms={last_rms:.1f})"
)
PY
fi

DEVICE="$("$SCRIPT_DIR/discover_athena_ambient_audio.sh")"
export ORION_AMBIENT_AUDIO_DEVICE="$DEVICE"

exec "$VENV_PYTHON" - <<'PY'
from __future__ import annotations

import os
import sys
import time

from scripts.orion_ambient_audio_reader import capture_pcm_via_arecord, compute_levels_from_pcm

device = os.environ["ORION_AMBIENT_AUDIO_DEVICE"]
needed = int(os.environ["ORION_AMBIENT_AUDIO_NEEDED"])
timeout_sec = float(os.environ["ORION_AMBIENT_AUDIO_TIMEOUT_SEC"])

deadline = time.monotonic() + timeout_sec
valid = 0
errors = 0
last_rms = 0.0

while valid < needed and time.monotonic() < deadline:
    try:
        pcm = capture_pcm_via_arecord(
            device=device,
            sample_rate=16000,
            channels=1,
            duration_sec=0.5,
        )
        rms, peak = compute_levels_from_pcm(pcm)
        if peak < 0:
            errors += 1
            continue
        valid += 1
        last_rms = rms
    except (OSError, RuntimeError):
        errors += 1
        time.sleep(0.25)
        continue
    time.sleep(0.25)

if valid < needed:
    print(
        f"error: only {valid}/{needed} successful captures in {timeout_sec}s "
        f"({errors} errors, device={device})",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"ok: {valid} ambient audio captures from {device} (last_rms={last_rms:.1f})")
PY
