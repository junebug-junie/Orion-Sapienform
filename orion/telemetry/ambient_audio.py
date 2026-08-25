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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from orion.signals.normalization import EwmaBand, InductionTracker, clamp01


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
