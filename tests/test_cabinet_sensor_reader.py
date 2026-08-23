from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.orion_cabinet_sensor_reader import (
    BOOT_SCHEMA_V1,
    SnapshotState,
    atomic_write_json,
    ingest_boot_line,
    parse_boot_line,
    parse_frame_line,
    write_snapshot,
)


def _valid_line(**overrides) -> str:
    payload = {
        "schema": "orion.sensor_frame.v1",
        "seq": 1,
        "uptime_ms": 1000,
        "environment": {"temp_c": 24.6},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_frame_line_accepts_valid_ndjson():
    frame = parse_frame_line(_valid_line(seq=42))
    assert frame is not None
    assert frame["schema"] == "orion.sensor_frame.v1"
    assert frame["seq"] == 42


@pytest.mark.parametrize(
    "line",
    [
        "",
        "{not json",
        json.dumps({"schema": "wrong.schema", "seq": 1}),
        json.dumps({"schema": "orion.sensor_frame.v1"}),  # missing seq
        json.dumps(["not", "a", "dict"]),
    ],
)
def test_parse_frame_line_rejects_malformed(line: str):
    assert parse_frame_line(line) is None


def test_malformed_line_preserves_last_good_frame():
    state = SnapshotState(stale_after_sec=60.0)
    device = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_TEST-if01"
    ts = "2026-08-23T06:00:00.123Z"

    assert state.process_line(_valid_line(seq=1), device=device, received_at=ts)
    good_frame = state.last_good_frame
    assert good_frame is not None

    assert not state.process_line("{broken", device=device, received_at=ts)
    assert state.last_good_frame == good_frame
    assert state.last_good_received_at == ts
    now = datetime.fromisoformat("2026-08-23T06:00:01+00:00")
    assert state.compute_status(now) == "ok"


def test_snapshot_shape_when_ok():
    state = SnapshotState(stale_after_sec=60.0)
    device = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_TEST-if01"
    ts = "2026-08-23T06:00:00.123Z"
    state.process_line(_valid_line(seq=7), device=device, received_at=ts)

    snap = state.to_snapshot(datetime.fromisoformat("2026-08-23T06:00:01+00:00"))
    assert snap["status"] == "ok"
    assert snap["received_at"] == ts
    assert snap["device"] == device
    assert snap["frame"]["seq"] == 7


def test_stale_status_when_last_good_aged_out():
    state = SnapshotState(stale_after_sec=5.0)
    device = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_TEST-if01"
    ts = "2026-08-23T06:00:00.123Z"
    state.process_line(_valid_line(seq=1), device=device, received_at=ts)

    now = datetime.fromisoformat("2026-08-23T06:00:10+00:00")
    snap = state.to_snapshot(now)
    assert snap["status"] == "stale"
    assert snap["frame"]["seq"] == 1


def test_missing_status_when_never_connected():
    state = SnapshotState()
    state.set_missing_device()
    snap = state.to_snapshot()
    assert snap["status"] == "missing"
    assert snap["frame"] is None
    assert snap["error"] == "device not found"


def test_atomic_write_json_uses_replace(tmp_path: Path):
    dest = tmp_path / "latest.json"
    atomic_write_json(dest, {"status": "ok", "seq": 1})
    data = json.loads(dest.read_text())
    assert data["status"] == "ok"

    atomic_write_json(dest, {"status": "ok", "seq": 2})
    data = json.loads(dest.read_text())
    assert data["seq"] == 2
    leftovers = list(tmp_path.glob(".tmp-*"))
    assert leftovers == []


def test_atomic_write_leaves_destination_on_mid_write_failure(tmp_path: Path, monkeypatch):
    dest = tmp_path / "latest.json"
    atomic_write_json(dest, {"status": "ok", "seq": 1})

    original_replace = os.replace

    def boom_replace(src, dst):
        if str(dst) == str(dest):
            raise OSError("simulated crash before replace")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", boom_replace)

    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_json(dest, {"status": "ok", "seq": 99})

    data = json.loads(dest.read_text())
    assert data["seq"] == 1


def test_write_snapshot_round_trip(tmp_path: Path):
    path = tmp_path / "snap.json"
    state = SnapshotState(stale_after_sec=60.0)
    device = "/dev/serial/by-id/usb-Arduino_Nano_ESP32_TEST-if01"
    state.process_line(
        _valid_line(seq=3),
        device=device,
        received_at="2026-08-23T06:00:00.000Z",
    )
    write_snapshot(path, state, datetime.fromisoformat("2026-08-23T06:00:01+00:00"))
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "ok"
    assert loaded["frame"]["seq"] == 3


def test_parse_boot_line_accepts_boot_schema():
    line = json.dumps(
        {
            "schema": BOOT_SCHEMA_V1,
            "uptime_ms": 4000,
            "i2c": {"addresses": ["0x29", "0x76"]},
            "sensors": {"bme680": {"ok": True}},
        }
    )
    boot = parse_boot_line(line)
    assert boot is not None
    assert boot["schema"] == BOOT_SCHEMA_V1


def test_ingest_boot_line_writes_boot_snapshot(tmp_path: Path):
    boot_path = tmp_path / "boot.json"
    line = json.dumps({"schema": BOOT_SCHEMA_V1, "uptime_ms": 1, "sensors": {}})
    assert ingest_boot_line(line, boot_path=boot_path)
    data = json.loads(boot_path.read_text())
    assert data["schema"] == BOOT_SCHEMA_V1
