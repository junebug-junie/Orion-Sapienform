"""Tests for cabinet ambient audio spike detection and bus payload shape."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _name in list(sys.modules):
    if _name == "app" or _name.startswith("app."):
        del sys.modules[_name]
sys.path.insert(0, str(_SERVICE_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from app.ambient_spike_detector import AmbientSpikeDetector, AmbientSpikeDetectorConfig  # noqa: E402
from orion.schemas.telemetry.biometrics import BiometricsSummaryV1  # noqa: E402
from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1  # noqa: E402

TS0 = datetime(2026, 8, 28, 4, 0, 0, tzinfo=timezone.utc)


def _summary(
    *,
    activity: float | None,
    rms: float | None = 5000.0,
    peak: float | None = 12000.0,
    ts: datetime = TS0,
) -> BiometricsSummaryV1:
    pressures = {}
    if activity is not None:
        pressures["cabinet_ambient_audio_activity"] = activity
    measurements = {}
    if rms is not None:
        measurements["cabinet_ambient_rms"] = rms
    if peak is not None:
        measurements["cabinet_ambient_peak"] = peak
    return BiometricsSummaryV1(
        timestamp=ts,
        node="athena",
        pressures=pressures,
        measurements=measurements or None,
    )


def test_no_activity_pressure_never_fires() -> None:
    det = AmbientSpikeDetector(AmbientSpikeDetectorConfig(activity_threshold=0.30, consecutive_ticks=2))
    assert det.observe(node="athena", timestamp=TS0, summary=_summary(activity=None), source_service="orion-biometrics") is None


def test_requires_consecutive_ticks_above_threshold() -> None:
    cfg = AmbientSpikeDetectorConfig(activity_threshold=0.30, consecutive_ticks=2, cooldown_sec=0.0)
    det = AmbientSpikeDetector(cfg)

    assert det.observe(node="athena", timestamp=TS0, summary=_summary(activity=0.35), source_service="orion-biometrics") is None

    spike = det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=30),
        summary=_summary(activity=0.40, ts=TS0 + timedelta(seconds=30)),
        source_service="orion-biometrics",
        source_node="athena",
    )

    assert isinstance(spike, CabinetAmbientSpikeV1)
    assert spike.activity == pytest.approx(0.40)
    assert spike.rms == pytest.approx(5000.0)
    assert spike.peak == pytest.approx(12000.0)
    assert spike.consecutive_ticks == 2
    assert spike.source_service == "orion-biometrics"


def test_drop_below_threshold_resets_consecutive() -> None:
    cfg = AmbientSpikeDetectorConfig(activity_threshold=0.30, consecutive_ticks=2, cooldown_sec=0.0)
    det = AmbientSpikeDetector(cfg)

    det.observe(node="athena", timestamp=TS0, summary=_summary(activity=0.35), source_service="orion-biometrics")
    det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=30),
        summary=_summary(activity=0.10, ts=TS0 + timedelta(seconds=30)),
        source_service="orion-biometrics",
    )
    assert det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=60),
        summary=_summary(activity=0.35, ts=TS0 + timedelta(seconds=60)),
        source_service="orion-biometrics",
    ) is None


def test_cooldown_suppresses_repeat_until_elapsed() -> None:
    cfg = AmbientSpikeDetectorConfig(activity_threshold=0.30, consecutive_ticks=2, cooldown_sec=300.0)
    det = AmbientSpikeDetector(cfg)

    det.observe(node="athena", timestamp=TS0, summary=_summary(activity=0.35), source_service="orion-biometrics")
    det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=30),
        summary=_summary(activity=0.35, ts=TS0 + timedelta(seconds=30)),
        source_service="orion-biometrics",
    )

    assert det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=60),
        summary=_summary(activity=0.35, ts=TS0 + timedelta(seconds=60)),
        source_service="orion-biometrics",
    ) is None

    det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=90),
        summary=_summary(activity=0.35, ts=TS0 + timedelta(seconds=90)),
        source_service="orion-biometrics",
    )
    after_cooldown = det.observe(
        node="athena",
        timestamp=TS0 + timedelta(seconds=330),
        summary=_summary(activity=0.35, ts=TS0 + timedelta(seconds=330)),
        source_service="orion-biometrics",
    )
    assert after_cooldown is not None
