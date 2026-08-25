"""Tests for host ambient audio snapshot ingest and biometrics pipeline wiring."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _name in list(sys.modules):
    if _name == "app" or _name.startswith("app."):
        del sys.modules[_name]
sys.path.insert(0, str(_SERVICE_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from app.ambient_audio_snapshot import load_ambient_audio_snapshot  # noqa: E402
from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig  # noqa: E402


NOW = datetime(2026, 8, 25, 5, 0, 5, tzinfo=timezone.utc)


def _write_snapshot(path: Path, **overrides) -> None:
    payload = {
        "schema": "orion.ambient_audio.v1",
        "status": "ok",
        "received_at": "2026-08-25T05:00:00.000Z",
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
        "sample_rate": 16000,
        "channels": 1,
        "rms": 412.3,
        "peak": 1820,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_or_invalid_snapshot_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_ambient_audio_snapshot(missing, stale_after_sec=5.0, now=NOW) is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert load_ambient_audio_snapshot(invalid, stale_after_sec=5.0, now=NOW) is None


def test_fresh_snapshot_preserves_levels_without_zero_fill(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path)

    audio = load_ambient_audio_snapshot(path, stale_after_sec=5.0, now=NOW)

    assert audio == {
        "rms": pytest.approx(412.3),
        "peak": 1820,
        "received_at": "2026-08-25T05:00:00.000Z",
        "stale": False,
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "orion.ambient_audio.v2"},
        {"status": "unknown"},
    ],
)
def test_unsupported_contract_returns_none(
    tmp_path: Path,
    overrides: dict,
) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, **overrides)

    assert load_ambient_audio_snapshot(path, stale_after_sec=5.0, now=NOW) is None


@pytest.mark.parametrize("status", ["stale", "error", "missing"])
def test_reader_status_marks_snapshot_stale(
    tmp_path: Path,
    status: str,
) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status=status)

    audio = load_ambient_audio_snapshot(path, stale_after_sec=5.0, now=NOW)

    assert audio is not None
    assert audio["stale"] is True


def test_snapshot_age_over_threshold_marks_stale(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, received_at="2026-08-25T04:59:59.000Z")

    audio = load_ambient_audio_snapshot(path, stale_after_sec=5.0, now=NOW)

    assert audio is not None
    assert audio["stale"] is True


def test_pipeline_adds_audio_after_host_peak_pressure() -> None:
    pipeline = BiometricsPipeline(PipelineConfig())
    sample = {
        "timestamp": NOW,
        "node": "athena",
        "cpu": {"util": 0.0, "cores": 1, "loadavg": {"1m": 0.0}},
        "ambient_audio": {
            "rms": 412.3,
            "peak": 1820,
            "received_at": "2026-08-25T05:00:00.000Z",
            "stale": False,
        },
    }

    for rms in (412.3, 412.3, 412.3, 9000.0):
        sample["ambient_audio"]["rms"] = rms
        summary, _ = pipeline.update(sample)

    assert summary.measurements["cabinet_ambient_rms"] == pytest.approx(9000.0)
    assert summary.measurements["cabinet_ambient_peak"] == pytest.approx(1820.0)
    assert summary.pressures["cabinet_ambient_audio_activity"] > 0.0
    assert summary.peak_pressure_channel != "cabinet_ambient_audio_activity"
