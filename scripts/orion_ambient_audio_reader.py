#!/usr/bin/env python3
"""Host ALSA reader for Athena cabinet ambient audio levels.

Captures short PCM windows via ``arecord``, computes RMS/peak over int16
samples, and atomically writes ``/run/orion-audio/latest.json``. Failed
captures never overwrite the last good levels on disk. No normalization,
EWMA, pressures, or cognition — biometrics owns that downstream.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from orion.schemas.telemetry.ambient_audio import AMBIENT_AUDIO_SCHEMA_V1

Status = Literal["ok", "stale", "error", "missing"]

DEFAULT_OUTPUT_PATH = Path("/run/orion-audio/latest.json")
DEFAULT_DEVICE = "plughw:CARD=CMTECK,DEV=0"
DEFAULT_WINDOW_SEC = 0.5
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_STALE_AFTER_SEC = 10.0
DEFAULT_LOOP_SEC = 1.0


def datetime_to_iso(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def utc_now_iso() -> str:
    return datetime_to_iso(datetime.now(timezone.utc))


def output_path() -> Path:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_PATH", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_OUTPUT_PATH


def audio_device() -> str:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_DEVICE", "").strip()
    if raw:
        return raw
    return DEFAULT_DEVICE


def window_sec() -> float:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_WINDOW_SEC", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_WINDOW_SEC


def sample_rate() -> int:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_SAMPLE_RATE", "").strip()
    if raw:
        return int(raw)
    return DEFAULT_SAMPLE_RATE


def capture_channels() -> int:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_CHANNELS", "").strip()
    if raw:
        return int(raw)
    return DEFAULT_CHANNELS


def stale_after_sec() -> float:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_STALE_AFTER_SEC", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_STALE_AFTER_SEC


def loop_interval_sec() -> float:
    raw = os.environ.get("ORION_AMBIENT_AUDIO_LOOP_SEC", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_LOOP_SEC


def compute_levels_from_pcm(pcm: bytes) -> tuple[float, int]:
    """Compute RMS and peak from little-endian int16 PCM bytes."""
    peak = 0
    sum_sq = 0.0
    count = 0
    usable = len(pcm) - (len(pcm) % 2)
    for i in range(0, usable, 2):
        sample = struct.unpack_from("<h", pcm, i)[0]
        abs_sample = abs(sample)
        if abs_sample > peak:
            peak = abs_sample
        sum_sq += sample * sample
        count += 1
    if count == 0:
        return 0.0, 0
    return (sum_sq / count) ** 0.5, peak


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


def build_snapshot(
    *,
    status: Status,
    received_at: str,
    device: Optional[str],
    window_sec: float,
    sample_rate: int,
    channels: int,
    rms: float,
    peak: int,
    error: Optional[str] = None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "schema": AMBIENT_AUDIO_SCHEMA_V1,
        "status": status,
        "received_at": received_at,
        "device": device or "",
        "window_sec": window_sec,
        "sample_rate": sample_rate,
        "channels": channels,
        "rms": rms,
        "peak": peak,
    }
    if error and status in ("error", "missing"):
        snap["error"] = error
    return snap


class SnapshotState:
    """In-memory reader state with last-good level preservation."""

    def __init__(
        self,
        *,
        stale_after_sec: float = DEFAULT_STALE_AFTER_SEC,
        window_sec: float = DEFAULT_WINDOW_SEC,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        self.stale_after_sec = stale_after_sec
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.channels = channels
        self.last_good_rms: Optional[float] = None
        self.last_good_peak: Optional[int] = None
        self.last_good_received_at: Optional[str] = None
        self.device: Optional[str] = None
        self.error: Optional[str] = None

    def ingest_good_capture(
        self,
        *,
        rms: float,
        peak: int,
        device: str,
        received_at: str,
        window_sec: float,
        sample_rate: int,
        channels: int,
    ) -> None:
        self.last_good_rms = rms
        self.last_good_peak = peak
        self.last_good_received_at = received_at
        self.device = device
        self.window_sec = window_sec
        self.sample_rate = sample_rate
        self.channels = channels
        self.error = None

    def ingest_failed_capture(self, message: str, *, device: Optional[str] = None) -> None:
        self.error = message
        if device is not None:
            self.device = device

    def compute_status(self, now: Optional[datetime] = None) -> Status:
        now = now or datetime.now(timezone.utc)
        if self.error:
            return "error"
        if self.last_good_rms is None and self.device is None:
            return "missing"
        if self.last_good_received_at:
            received = datetime.fromisoformat(
                self.last_good_received_at.replace("Z", "+00:00")
            )
            age = (now - received).total_seconds()
            if age > self.stale_after_sec:
                return "stale"
        if self.last_good_rms is None:
            return "missing"
        return "ok"

    def to_snapshot(self, now: Optional[datetime] = None) -> dict[str, Any]:
        effective_now = now or datetime.now(timezone.utc)
        status = self.compute_status(effective_now)
        rms = self.last_good_rms if self.last_good_rms is not None else 0.0
        peak = self.last_good_peak if self.last_good_peak is not None else 0
        return build_snapshot(
            status=status,
            received_at=self.last_good_received_at or datetime_to_iso(effective_now),
            device=self.device,
            window_sec=self.window_sec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            rms=rms,
            peak=peak,
            error=self.error if status in ("error", "missing") else None,
        )


def write_snapshot(path: Path, state: SnapshotState, now: Optional[datetime] = None) -> None:
    atomic_write_json(path, state.to_snapshot(now))


def write_snapshot_if_changed(
    path: Path, state: SnapshotState, now: Optional[datetime] = None
) -> None:
    """Write snapshot; never clobber disk with empty/invalid levels on failure."""
    snap = state.to_snapshot(now)
    if state.error and state.last_good_rms is None:
        if path.is_file():
            return
    write_snapshot(path, state, now)


def capture_pcm_via_arecord(
    *,
    device: str,
    sample_rate: int,
    channels: int,
    duration_sec: float,
    arecord_bin: str = "arecord",
) -> bytes:
    """Capture raw S16_LE PCM from ALSA via arecord subprocess."""
    samples = int(sample_rate * duration_sec)
    if samples <= 0:
        raise ValueError("capture duration must produce at least one sample")
    proc = subprocess.run(
        [
            arecord_bin,
            "-D",
            device,
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            str(channels),
            "-t",
            "raw",
            "--samples",
            str(samples),
            "-q",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"arecord exited {proc.returncode}")
    if not proc.stdout:
        raise RuntimeError("arecord returned no PCM data")
    return proc.stdout


def run_loop(
    *,
    output: Path,
    state: SnapshotState,
    device: str,
    window: float,
    rate: int,
    channels: int,
    loop_sec: float = DEFAULT_LOOP_SEC,
    sleep_fn=time.sleep,
    capture_fn=capture_pcm_via_arecord,
) -> None:
    while True:
        received_at = utc_now_iso()
        try:
            pcm = capture_fn(
                device=device,
                sample_rate=rate,
                channels=channels,
                duration_sec=window,
            )
            rms, peak = compute_levels_from_pcm(pcm)
            state.ingest_good_capture(
                rms=rms,
                peak=peak,
                device=device,
                received_at=received_at,
                window_sec=window,
                sample_rate=rate,
                channels=channels,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            state.ingest_failed_capture(str(exc), device=device)

        write_snapshot_if_changed(output, state)
        sleep_fn(loop_sec)


def main(argv: Optional[list[str]] = None) -> int:
    _ = argv
    path = output_path()
    state = SnapshotState(
        stale_after_sec=stale_after_sec(),
        window_sec=window_sec(),
        sample_rate=sample_rate(),
        channels=capture_channels(),
    )
    try:
        run_loop(
            output=path,
            state=state,
            device=audio_device(),
            window=window_sec(),
            rate=sample_rate(),
            channels=capture_channels(),
            loop_sec=loop_interval_sec(),
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
