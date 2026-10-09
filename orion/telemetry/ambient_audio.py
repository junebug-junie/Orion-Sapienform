from __future__ import annotations

"""Athena host cabinet ambient audio: raw measurements and baseline-relative pressure.

ROADMAP: physical senses. `orion-biometrics` already measures the host
machines (CPU/GPU/thermal/power/...) and the Nano ESP32 cabinet environment;
this module extends the same raw->measurements->pressures shape to continuous
USB mic acoustic levels on the Athena host.

See `orion/schemas/telemetry/ambient_audio.py` for the wire contract and
`docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md`
for the full pipeline.

v1 pressure is baseline-relative only: EWMA band, delta, volatility ->
anomaly in [0, 1] on RMS only. Peak is a measurement for debug/smokes;
not a second field channel in v1. Pressure key: cabinet_ambient_audio_activity.

HAND-VERIFIED REST POINT (per CLAUDE.md 0A step 4): for a raw RMS magnitude
that is EXACTLY constant tick to tick, `EwmaBand.update()` computes
`delta = value - mean == 0` every call, so `dev` converges to exactly `0.0`;
`EwmaBand.normalize()` then returns exactly `0.0`; and
`InductionTracker.volatility` also converges to exactly `0.0`. Proven in
`tests/test_ambient_audio.py::test_activity_signal_rests_at_zero_for_constant_rms`.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)
from orion.signals.normalization import EwmaBand, InductionTracker, clamp01

logger = logging.getLogger(__name__)

_STALE_STATUSES = frozenset({"stale", "error", "missing"})
_VALID_STATUSES = _STALE_STATUSES | {"ok"}

# The host reader captures signed 16-bit PCM (S16_LE), so RMS and peak are in
# [0, 32768]. dBFS is level relative to that full scale: 0 dBFS is clipping,
# every -6 dB is half the amplitude. It is NOT calibrated sound pressure (dB
# SPL / dBA) -- the mic's gain sets the offset, and no calibration exists.
PCM16_FULL_SCALE = 32768.0
# Floor for a digitally silent window (rms == 0): 20*log10(1/32768), the
# smallest non-zero 16-bit level, so dBFS stays finite.
DBFS_FLOOR = round(20.0 * math.log10(1.0 / PCM16_FULL_SCALE), 1)


def pcm16_to_dbfs(level: float) -> float:
    """RMS or peak in 16-bit PCM units -> dBFS (<= 0, floored at DBFS_FLOOR)."""
    if level <= 1.0:
        return DBFS_FLOOR
    return min(0.0, 20.0 * math.log10(level / PCM16_FULL_SCALE))


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
class AmbientAudioPressureConfig:
    rms_band_alpha: float = 0.1


@dataclass
class _ActivityChannel:
    band: EwmaBand
    tracker: InductionTracker = field(default_factory=InductionTracker)

    def activity(self, name: str, raw: float) -> float:
        self.band.update(raw)
        level = self.band.normalize(raw)
        return clamp01(self.tracker.update(name, level).volatility)


def extract_ambient_audio_measurements(ambient_audio: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Raw cabinet ambient audio levels, in native PCM units, keyed with the
    unit in the name -- same absent-is-not-zero invariant as
    biometrics_pipeline.extract_measurements. `ambient_audio` is the
    BiometricsSampleV1.ambient_audio dict; returns {} (not None) when there
    is nothing to report.
    """
    out: Dict[str, float] = {}
    if not isinstance(ambient_audio, dict) or ambient_audio.get("stale"):
        return out

    def put(key: str, value: Optional[float]) -> None:
        if value is not None:
            out[key] = value

    put("cabinet_ambient_rms", _as_float(ambient_audio.get("rms")))
    put("cabinet_ambient_peak", _as_float(ambient_audio.get("peak")))

    return out


def _parse_received_at(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_ambient_audio_snapshot(
    path: str | Path,
    *,
    stale_after_sec: float,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Read the host reader's `/run/orion-audio/latest.json`.

    Returns the `BiometricsSampleV1.ambient_audio` dict shape (rms, peak,
    received_at, stale, device, window_sec), or ``None`` when the file is
    missing, unreadable, or not a supported snapshot. A readable snapshot
    that is too old or has a non-ok status comes back with ``stale=True``
    -- callers must not treat its levels as current.

    Shared by orion-biometrics (sample ingest) and the situation brief.
    """
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        return None

    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = AmbientAudioSnapshotV1.model_validate(data)
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValidationError,
    ) as exc:
        logger.warning("Ambient audio snapshot unreadable at %s: %s", snapshot_path, exc)
        return None

    if snapshot.schema_ != AMBIENT_AUDIO_SCHEMA_V1 or snapshot.status.lower() not in _VALID_STATUSES:
        logger.warning(
            "Ambient audio snapshot at %s has unsupported schema/status",
            snapshot_path,
        )
        return None

    stale = snapshot.status.lower() in _STALE_STATUSES
    if not stale:
        received_dt = _parse_received_at(snapshot.received_at)
        if received_dt is None:
            stale = True
        else:
            now_dt = now or datetime.now(timezone.utc)
            age_sec = (now_dt.astimezone(timezone.utc) - received_dt).total_seconds()
            stale = age_sec > stale_after_sec

    return {
        "rms": snapshot.rms,
        "peak": snapshot.peak,
        "received_at": snapshot.received_at,
        "stale": stale,
        "device": snapshot.device,
        "window_sec": snapshot.window_sec,
    }


class AmbientAudioTracker:
    """Per-node persistent EWMA state for baseline-relative ambient audio activity."""

    def __init__(self, cfg: AmbientAudioPressureConfig) -> None:
        self.cfg = cfg
        self.rms_channel = _ActivityChannel(EwmaBand(alpha=cfg.rms_band_alpha))


def compute_ambient_audio_pressures(
    measurements: Dict[str, float],
    tracker: AmbientAudioTracker,
) -> Dict[str, float]:
    """Baseline-relative 0-1 ambient audio activity from RMS only. Returns {}
    when `measurements` lacks `cabinet_ambient_rms` -- no fabricated 0.0."""
    out: Dict[str, float] = {}
    if "cabinet_ambient_rms" in measurements:
        out["cabinet_ambient_audio_activity"] = tracker.rms_channel.activity(
            "ambient_rms", measurements["cabinet_ambient_rms"]
        )
    return out
