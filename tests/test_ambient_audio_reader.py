from __future__ import annotations

import json
import os
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)
from scripts.orion_ambient_audio_reader import (
    SnapshotState,
    atomic_write_json,
    build_snapshot,
    capture_pcm_via_arecord,
    compute_levels_from_pcm,
    write_snapshot_if_changed,
)


def _pcm_from_samples(samples: list[int]) -> bytes:
    return b"".join(struct.pack("<h", s) for s in samples)


def test_compute_levels_from_pcm_silent():
    pcm = _pcm_from_samples([0] * 100)
    rms, peak = compute_levels_from_pcm(pcm)
    assert rms == 0.0
    assert peak == 0


def test_compute_levels_from_pcm_tone():
    pcm = _pcm_from_samples([0] * 50 + [12000] * 50)
    rms, peak = compute_levels_from_pcm(pcm)
    assert peak == 12000
    assert rms > 100.0


def test_compute_levels_from_pcm_empty():
    rms, peak = compute_levels_from_pcm(b"")
    assert rms == 0.0
    assert peak == 0


def test_compute_levels_from_pcm_odd_length_ignored():
    pcm = _pcm_from_samples([100, 200, 300]) + b"\x00"
    rms, peak = compute_levels_from_pcm(pcm)
    assert peak == 300
    assert rms > 0


def test_build_snapshot_validates_schema():
    snap = build_snapshot(
        status="ok",
        received_at="2026-08-25T05:00:00.123Z",
        device="plughw:CARD=CMTECK,DEV=0",
        window_sec=0.5,
        sample_rate=16000,
        channels=1,
        rms=412.3,
        peak=1820,
    )
    model = AmbientAudioSnapshotV1.model_validate(snap)
    assert model.schema_ == AMBIENT_AUDIO_SCHEMA_V1
    assert model.rms == 412.3
    assert model.peak == 1820


def test_failed_capture_preserves_last_good_levels():
    state = SnapshotState(stale_after_sec=60.0)
    device = "plughw:CARD=CMTECK,DEV=0"
    ts = "2026-08-25T05:00:00.123Z"

    state.ingest_good_capture(
        rms=412.3,
        peak=1820,
        device=device,
        received_at=ts,
        window_sec=0.5,
        sample_rate=16000,
        channels=1,
    )
    good_rms = state.last_good_rms
    good_peak = state.last_good_peak

    state.ingest_failed_capture("arecord: device busy")
    snap = state.to_snapshot(datetime.fromisoformat("2026-08-25T05:00:01+00:00"))
    assert snap["status"] == "error"
    assert snap["rms"] == good_rms
    assert snap["peak"] == good_peak
    assert snap["received_at"] == ts
    assert snap["error"] == "arecord: device busy"


def test_failed_capture_without_last_good_is_error():
    state = SnapshotState()
    state.ingest_failed_capture("device not found")
    now = datetime.fromisoformat("2026-08-25T05:00:01+00:00")
    snap = state.to_snapshot(now)
    assert snap["status"] == "error"
    assert snap["received_at"] == "2026-08-25T05:00:01.000Z"
    assert snap["rms"] == 0.0
    assert snap["peak"] == 0
    assert snap["error"] == "device not found"


def test_arecord_capture_requests_raw_pcm_and_exact_sample_count(monkeypatch):
    seen: dict[str, object] = {}

    def fake_run(argv, *, capture_output, check):
        seen["argv"] = argv
        assert capture_output is True
        assert check is False
        return subprocess.CompletedProcess(argv, 0, stdout=b"\x00\x00", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    pcm = capture_pcm_via_arecord(
        device="plughw:CARD=CMTECK,DEV=0",
        sample_rate=16000,
        channels=1,
        duration_sec=0.5,
    )

    assert pcm == b"\x00\x00"
    assert seen["argv"] == [
        "arecord",
        "-D",
        "plughw:CARD=CMTECK,DEV=0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-t",
        "raw",
        "--samples",
        "8000",
        "-q",
        "-",
    ]


def test_stale_status_when_last_good_aged_out():
    state = SnapshotState(stale_after_sec=5.0)
    device = "plughw:CARD=CMTECK,DEV=0"
    ts = "2026-08-25T05:00:00.123Z"
    state.ingest_good_capture(
        rms=100.0,
        peak=500,
        device=device,
        received_at=ts,
        window_sec=0.5,
        sample_rate=16000,
        channels=1,
    )
    now = datetime.fromisoformat("2026-08-25T05:00:10+00:00")
    snap = state.to_snapshot(now)
    assert snap["status"] == "stale"
    assert snap["rms"] == 100.0


def test_atomic_write_json_uses_replace(tmp_path: Path):
    dest = tmp_path / "latest.json"
    atomic_write_json(dest, {"schema": AMBIENT_AUDIO_SCHEMA_V1, "rms": 1.0, "peak": 2})
    data = json.loads(dest.read_text())
    assert data["rms"] == 1.0

    atomic_write_json(dest, {"schema": AMBIENT_AUDIO_SCHEMA_V1, "rms": 3.0, "peak": 4})
    data = json.loads(dest.read_text())
    assert data["rms"] == 3.0
    assert list(tmp_path.glob(".tmp-*")) == []


def test_atomic_write_leaves_destination_on_mid_write_failure(tmp_path: Path, monkeypatch):
    dest = tmp_path / "latest.json"
    atomic_write_json(dest, {"schema": AMBIENT_AUDIO_SCHEMA_V1, "rms": 1.0, "peak": 2})

    original_replace = os.replace

    def boom_replace(src, dst):
        if str(dst) == str(dest):
            raise OSError("simulated crash before replace")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom_replace)

    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_json(dest, {"schema": AMBIENT_AUDIO_SCHEMA_V1, "rms": 99.0, "peak": 99})

    data = json.loads(dest.read_text())
    assert data["rms"] == 1.0


def test_write_snapshot_does_not_clobber_on_invalid_payload(tmp_path: Path):
    dest = tmp_path / "latest.json"
    state = SnapshotState()
    state.ingest_good_capture(
        rms=200.0,
        peak=1000,
        device="plughw:CARD=CMTECK,DEV=0",
        received_at="2026-08-25T05:00:00.000Z",
        window_sec=0.5,
        sample_rate=16000,
        channels=1,
    )
    write_snapshot_if_changed(dest, state)
    before = json.loads(dest.read_text())

    state.ingest_failed_capture("capture timeout")
    write_snapshot_if_changed(dest, state)
    after = json.loads(dest.read_text())
    assert after["rms"] == before["rms"]
    assert after["peak"] == before["peak"]
    assert after["status"] == "error"
