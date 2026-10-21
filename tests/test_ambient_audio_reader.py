from __future__ import annotations

import json
import os
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_LIVE_SLOT_HEADER as LIVE_SLOT_HEADER,
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)
from scripts.orion_ambient_audio_reader import (
    LiveRing,
    SnapshotState,
    atomic_write_json,
    build_snapshot,
    compute_levels_from_pcm,
    run_loop,
    stream_pcm_via_arecord,
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


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 1):
        import io

        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None
        self._exit = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = self._exit

    def wait(self):
        return self.returncode


def test_arecord_stream_is_one_continuous_process_cut_into_windows():
    seen: dict[str, object] = {}
    # 2.5 windows of 0.5s @ 16 kHz mono: two full chunks then a short tail.
    pcm = _pcm_from_samples(list(range(8000)) * 2 + [7] * 4000)

    def fake_popen(argv, *, stdout, stderr):
        seen["argv"] = argv
        proc = _FakeProc(pcm, stderr=b"arecord: overrun")
        seen["proc"] = proc
        return proc

    chunks = []
    with pytest.raises(RuntimeError, match="overrun"):
        for chunk in stream_pcm_via_arecord(
            device="plughw:CARD=CMTECK,DEV=0",
            sample_rate=16000,
            channels=1,
            duration_sec=0.5,
            popen=fake_popen,
        ):
            chunks.append(chunk)

    assert [len(c) for c in chunks] == [16000, 16000]
    assert seen["proc"].killed is True
    # No --samples: one stream, not a fresh device open per window.
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
        "-q",
        "-",
    ]


def test_live_ring_slots_carry_seq_and_rate_and_wrap(tmp_path: Path):
    ring = LiveRing(tmp_path / "live", slots=4)
    for i in range(6):
        ring.write(_pcm_from_samples([i] * 4), sample_rate=16000, channels=1)

    files = sorted(p.name for p in (tmp_path / "live").iterdir())
    assert files == ["slot-00.pcm", "slot-01.pcm", "slot-02.pcm", "slot-03.pcm"]
    raw = ring.slot_path(6).read_bytes()
    seq, rate, channels, _ = LIVE_SLOT_HEADER.unpack_from(raw)
    assert (seq, rate, channels) == (6, 16000, 1)
    assert raw[LIVE_SLOT_HEADER.size:] == _pcm_from_samples([5] * 4)
    assert oct(ring.slot_path(6).stat().st_mode & 0o777) == "0o644"


def test_run_loop_writes_levels_and_live_chunks_then_retries_after_failure(tmp_path: Path):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_stream(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _pcm_from_samples([1000] * 8)
            yield _pcm_from_samples([2000] * 8)
            raise RuntimeError("arecord died")
        yield _pcm_from_samples([3000] * 8)

    out = tmp_path / "latest.json"
    ring = LiveRing(tmp_path / "live")
    run_loop(
        output=out,
        state=SnapshotState(stale_after_sec=60.0),
        device="dev",
        window=0.5,
        rate=16000,
        channels=1,
        loop_sec=1.5,
        live=ring,
        sleep_fn=sleeps.append,
        stream_fn=fake_stream,
        max_windows=4,
    )
    assert calls["n"] == 2
    assert sleeps == [1.5]
    assert ring.seq == 3
    snap = json.loads(out.read_text())
    assert snap["status"] == "ok"
    assert snap["peak"] == 3000


def test_run_loop_without_live_ring_writes_no_audio(tmp_path: Path):
    def fake_stream(**_kwargs):
        yield _pcm_from_samples([500] * 8)

    run_loop(
        output=tmp_path / "latest.json",
        state=SnapshotState(),
        device="dev",
        window=0.5,
        rate=16000,
        channels=1,
        live=None,
        stream_fn=fake_stream,
        max_windows=1,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest.json"]


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
    assert (dest.stat().st_mode & 0o777) == 0o644

    atomic_write_json(dest, {"schema": AMBIENT_AUDIO_SCHEMA_V1, "rms": 3.0, "peak": 4})
    data = json.loads(dest.read_text())
    assert data["rms"] == 3.0
    assert (dest.stat().st_mode & 0o777) == 0o644
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


def test_live_ring_default_layout_is_the_shared_contract_hub_reads(tmp_path: Path):
    from orion.schemas.telemetry.ambient_audio import ambient_audio_live_slot_name

    ring = LiveRing(tmp_path)
    for seq in (1, 16, 17, 40):
        assert ring.slot_path(seq).name == ambient_audio_live_slot_name(seq)
