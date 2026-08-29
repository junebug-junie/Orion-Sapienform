"""Tests for dual-Nano cabinet snapshot merge."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.telemetry.cabinet_snapshot_merge import (
    load_merged_cabinet_sensors,
    merge_cabinet_sensors_payload,
    load_cabinet_snapshot_file,
    CabinetSnapshotSource,
    merge_cabinet_frame_channels,
)

NOW = datetime(2026, 8, 29, 12, 0, 5, tzinfo=timezone.utc)
FRESH = "2026-08-29T12:00:00.000Z"
OLDER = "2026-08-29T11:59:50.000Z"


def _write(path: Path, frame: dict, *, received_at: str = FRESH, status: str = "ok") -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "received_at": received_at,
                "device": f"/dev/serial/by-id/test-{path.name}",
                "frame": frame,
            }
        ),
        encoding="utf-8",
    )


def _frame(**channels):
    base = {"schema": "orion.sensor_frame.v1", "seq": 1, "uptime_ms": 1000}
    base.update(channels)
    return base


def test_single_primary_unchanged_behavior(tmp_path: Path) -> None:
    primary = tmp_path / "latest.json"
    _write(primary, _frame(environment={"temp_c": 24.0}))

    merged = load_merged_cabinet_sensors(primary, stale_after_sec=10.0, now=NOW)

    assert merged is not None
    assert merged["stale"] is False
    assert merged["frame"]["environment"]["temp_c"] == 24.0
    assert merged["sources"]["a"]["stale"] is False


def test_merge_splits_channels_across_nanos(tmp_path: Path) -> None:
    primary = tmp_path / "a.json"
    secondary = tmp_path / "b.json"
    _write(
        primary,
        _frame(
            environment={"temp_c": 28.0, "humidity_pct": 20.0},
            uv={"raw": 0.0, "als_raw": 50.0},
            lidar={"distance_mm": 120.0, "status": 0},
        ),
    )
    _write(
        secondary,
        _frame(
            magnetic={"magnitude_ut": 40.0, "x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0},
            imu={"accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.8, "yaw_deg": 1.0},
        ),
        received_at=FRESH,
    )

    merged = load_merged_cabinet_sensors(
        primary,
        secondary_path=secondary,
        stale_after_sec=10.0,
        now=NOW,
    )

    assert merged is not None
    assert merged["stale"] is False
    frame = merged["frame"]
    assert frame["environment"]["temp_c"] == 28.0
    assert frame["uv"]["als_raw"] == 50.0
    assert frame["lidar"]["distance_mm"] == 120.0
    assert frame["magnetic"]["magnitude_ut"] == 40.0
    assert frame["imu"]["yaw_deg"] == 1.0


def test_stale_secondary_does_not_contribute_channels(tmp_path: Path) -> None:
    primary = tmp_path / "a.json"
    secondary = tmp_path / "b.json"
    _write(primary, _frame(lidar={"distance_mm": 10.0, "status": 0}))
    _write(
        secondary,
        _frame(magnetic={"magnitude_ut": 99.0}),
        status="stale",
        received_at=OLDER,
    )

    merged = load_merged_cabinet_sensors(
        primary,
        secondary_path=secondary,
        stale_after_sec=10.0,
        now=NOW,
    )

    assert merged is not None
    assert merged["stale"] is False
    assert "magnetic" not in merged["frame"]
    assert merged["frame"]["lidar"]["distance_mm"] == 10.0


def test_all_stale_merged_payload_is_stale(tmp_path: Path) -> None:
    primary = tmp_path / "a.json"
    secondary = tmp_path / "b.json"
    _write(primary, _frame(environment={"temp_c": 1.0}), status="stale")
    _write(secondary, _frame(magnetic={"magnitude_ut": 2.0}), status="stale")

    merged = load_merged_cabinet_sensors(
        primary,
        secondary_path=secondary,
        stale_after_sec=10.0,
        now=NOW,
    )

    assert merged is None


def test_newer_source_wins_channel_conflict() -> None:
    older = CabinetSnapshotSource(
        source_id="a",
        frame=_frame(magnetic={"magnitude_ut": 10.0}),
        received_at=OLDER,
        stale=False,
        status="ok",
        device="a",
        raw=None,
    )
    newer = CabinetSnapshotSource(
        source_id="b",
        frame=_frame(magnetic={"magnitude_ut": 20.0}),
        received_at=FRESH,
        stale=False,
        status="ok",
        device="b",
        raw=None,
    )
    merged = merge_cabinet_frame_channels([older, newer])
    assert merged["magnetic"]["magnitude_ut"] == 20.0


def test_missing_secondary_file_primary_only(tmp_path: Path) -> None:
    primary = tmp_path / "a.json"
    _write(primary, _frame(uv={"raw": 1.0, "als_raw": 2.0}))
    missing = tmp_path / "missing.json"

    merged = load_merged_cabinet_sensors(
        primary,
        secondary_path=missing,
        stale_after_sec=10.0,
        now=NOW,
    )

    assert merged is not None
    assert "uv" in merged["frame"]
    assert len(merged["sources"]) == 1
