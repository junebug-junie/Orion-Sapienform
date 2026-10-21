"""Tests for Hub cabinet ambient latest and history APIs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for path in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import cabinet_ambient_routes  # noqa: E402


NOW = datetime(2026, 8, 26, 3, 0, 5, tzinfo=timezone.utc)


def _snapshot(**overrides) -> dict:
    payload = {
        "schema": "orion.ambient_audio.v1",
        "status": "ok",
        "received_at": "2026-08-26T03:00:00Z",
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
        "sample_rate": 16000,
        "channels": 1,
        "rms": 5055.4,
        "peak": 19725,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    path = tmp_path / "latest.json"
    monkeypatch.setattr(cabinet_ambient_routes.settings, "AMBIENT_AUDIO_PATH", str(path))
    monkeypatch.setattr(cabinet_ambient_routes.settings, "AMBIENT_AUDIO_STALE_AFTER_SEC", 5.0)
    monkeypatch.setattr(cabinet_ambient_routes.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena")
    monkeypatch.setattr(cabinet_ambient_routes.settings, "CABINET_AMBIENT_HISTORY_MAX_POINTS", 800)
    monkeypatch.setattr(cabinet_ambient_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "athena"
        assert hours == 24
        return []

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", no_rows)
    app = FastAPI()
    app.include_router(cabinet_ambient_routes.router)
    return TestClient(app), path


def test_parse_window_accepts_only_supported_windows():
    assert cabinet_ambient_routes.parse_window("24h") == 24
    assert cabinet_ambient_routes.parse_window("3d") == 72
    assert cabinet_ambient_routes.parse_window("7d") == 168
    with pytest.raises(ValueError):
        cabinet_ambient_routes.parse_window("1h")


def test_rows_to_points_omits_missing_optional_values_instead_of_zero_filling():
    rows = [
        {
            "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
            "rms": "7457.8",
            "peak": None,
            "activity": None,
        }
    ]
    assert cabinet_ambient_routes.rows_to_points(rows) == [
        {"t": "2026-08-26T02:54:09Z", "rms": 7457.8}
    ]


def test_rows_to_points_preserves_peak_as_integer():
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 7457.8,
                "peak": "19148",
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["peak"] == 19148
    assert isinstance(points[0]["peak"], int)


def test_rows_to_points_accepts_float_peak_from_biometrics_json():
    """Biometrics stores cabinet_ambient_peak as float; Postgres ->> yields '16213.0'."""
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 4132.5,
                "peak": 16213.0,
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["peak"] == 16213


def test_rows_to_points_accepts_sql_writer_varchar_timestamp():
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": "2026-08-26 02:41:42.067071+00",
                "rms": 3806.35,
                "peak": 16213.0,
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["t"] == "2026-08-26T02:41:42.067071Z"


def test_parse_db_timestamp_pads_variable_fraction_for_python310():
    parsed = cabinet_ambient_routes._parse_db_timestamp("2026-08-26 02:46:53.36052+00")
    assert parsed.isoformat() == "2026-08-26T02:46:53.360520+00:00"


def test_downsample_points_averages_each_bucket_and_respects_cap():
    points = [
        {"t": f"2026-08-26T00:00:0{i}Z", "rms": float(i), "activity": i / 10}
        for i in range(6)
    ]
    sampled = cabinet_ambient_routes.downsample_points(points, 3)
    assert sampled == [
        {"t": "2026-08-26T00:00:00Z", "rms": 0.5, "activity": 0.05},
        {"t": "2026-08-26T00:00:02Z", "rms": 2.5, "activity": 0.25},
        {"t": "2026-08-26T00:00:04Z", "rms": 4.5, "activity": 0.45},
    ]


def test_biometrics_summary_cutoff_matches_sql_writer_varchar_form():
    """sql-writer stores TEXT timestamps as `YYYY-MM-DD HH:MM:SS.ffffff+00`.

    ISO-Z cutoffs (`...T...Z`) sort AFTER same-day stored rows because
    space < 'T', so a 24h window falsely returns zero. Confirmed live
    2026-09-13 against orion_biometrics_summary.
    """
    cutoff = cabinet_ambient_routes.biometrics_summary_cutoff(NOW)
    assert cutoff == "2026-08-26 03:00:05.000000+00"
    # Same calendar day, later sample in sql-writer form must clear the cutoff.
    assert "2026-08-26 07:02:11.914495+00" >= cutoff
    # The old ISO-Z form must NOT win against that same-day sample.
    assert not ("2026-08-26 07:02:11.914495+00" >= "2026-08-26T03:00:05.000000Z")


def test_history_query_uses_index_compatible_timestamp_range(monkeypatch):
    captured = {}

    class FakeConnection:
        async def fetch(self, sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return []

        async def close(self):
            captured["closed"] = True

    class FakeAsyncpg:
        @staticmethod
        async def connect(*, dsn):
            captured["dsn"] = dsn
            return FakeConnection()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setitem(sys.modules, "asyncpg", FakeAsyncpg)
    monkeypatch.setattr(cabinet_ambient_routes, "_now_utc", lambda: NOW)

    asyncio.run(cabinet_ambient_routes.query_history_rows(node="athena", hours=72))

    assert captured["args"] == ("athena", "2026-08-23 03:00:05.000000+00")
    assert "timestamp >= $2" in captured["sql"]
    assert "ORDER BY timestamp ASC" in captured["sql"]
    assert "timestamp::timestamptz" not in captured["sql"]
    assert "cabinet_ambient_peak')::double precision" in captured["sql"]
    assert "cabinet_ambient_peak')::bigint" not in captured["sql"]
    assert captured["closed"] is True


def test_latest_missing_returns_ok_false(client):
    tc, path = client
    assert not path.exists()
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body == {"ok": False, "age_sec": None, "snapshot": None}


def test_latest_fresh_valid_snapshot_returns_age(client):
    tc, path = client
    payload = _snapshot()
    path.write_text(json.dumps(payload), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body["ok"] is True
    assert body["age_sec"] == pytest.approx(5.0)
    assert body["snapshot"] == payload


def test_latest_stale_keeps_last_snapshot(client):
    tc, path = client
    payload = _snapshot(received_at="2026-08-26T02:59:50Z")
    path.write_text(json.dumps(payload), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body["ok"] is False
    assert body["age_sec"] == pytest.approx(15.0)
    assert body["snapshot"] == payload


def test_latest_invalid_schema_is_unreadable(client):
    tc, path = client
    path.write_text(json.dumps(_snapshot(schema="wrong.v1")), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body == {"ok": False, "age_sec": None, "snapshot": None}


def test_history_defaults_to_24h_and_returns_empty_points(client):
    tc, _path = client
    body = tc.get("/api/cabinet/ambient/history").json()
    assert body == {
        "ok": True,
        "node": "athena",
        "window": "24h",
        "grain_sec": 30,
        "points": [],
        "stats": {
            "n_raw": 0,
            "n": 0,
            "rms_min": None,
            "rms_max": None,
            "activity_max": None,
        },
    }


def test_history_uses_query_rows_and_reports_stats(client, monkeypatch):
    tc, _path = client

    async def rows(*, node: str, hours: int):
        assert (node, hours) == ("athena", 72)
        return [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 1200.0,
                "peak": 19000,
                "activity": 0.2,
            },
            {
                "t": datetime(2026, 8, 26, 2, 54, 39, tzinfo=timezone.utc),
                "rms": 9000.0,
                "peak": 20000,
                "activity": 0.85,
            },
        ]

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", rows)
    body = tc.get("/api/cabinet/ambient/history?window=3d").json()
    assert body["ok"] is True
    assert body["window"] == "3d"
    assert len(body["points"]) == 2
    assert body["stats"] == {
        "n_raw": 2,
        "n": 2,
        "rms_min": 1200.0,
        "rms_max": 9000.0,
        "activity_max": 0.85,
    }


def test_history_invalid_window_returns_400(client):
    tc, _path = client
    response = tc.get("/api/cabinet/ambient/history?window=1h")
    assert response.status_code == 400


def test_history_db_failure_returns_ok_false(client, monkeypatch):
    tc, _path = client

    async def failed(*, node: str, hours: int):
        raise OSError("db unavailable")

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", failed)
    body = tc.get("/api/cabinet/ambient/history").json()
    assert body["ok"] is False
    assert body["points"] == []
    assert body["error"] == "ambient_history_unavailable"


def test_rows_to_spikes_maps_db_rows():
    rows = [
        {
            "spike_id": "abc-123",
            "t": datetime(2026, 8, 29, 6, 20, tzinfo=timezone.utc),
            "activity": 0.41,
            "rms": 7340.0,
            "peak": 16213.0,
            "activity_threshold": 0.30,
        }
    ]
    assert cabinet_ambient_routes.rows_to_spikes(rows) == [
        {
            "t": "2026-08-29T06:20:00Z",
            "spike_id": "abc-123",
            "activity": 0.41,
            "rms": 7340.0,
            "peak": 16213.0,
            "activity_threshold": 0.30,
        }
    ]


def test_spikes_uses_query_rows(client, monkeypatch):
    tc, _path = client

    async def spikes(*, node: str, hours: int):
        assert (node, hours) == ("athena", 24)
        return [
            {
                "spike_id": "spike-1",
                "t": datetime(2026, 8, 29, 6, 20, tzinfo=timezone.utc),
                "activity": 0.41,
                "rms": 7340.0,
                "peak": 16213.0,
                "activity_threshold": 0.30,
                "consecutive_ticks": 2,
            }
        ]

    monkeypatch.setattr(cabinet_ambient_routes, "_spike_query", spikes)
    body = tc.get("/api/cabinet/ambient/spikes?window=24h").json()
    assert body["ok"] is True
    assert body["n"] == 1
    assert body["spikes"][0]["spike_id"] == "spike-1"


def test_spikes_db_failure_returns_ok_false(client, monkeypatch):
    tc, _path = client

    async def failed(*, node: str, hours: int):
        raise OSError("db unavailable")

    monkeypatch.setattr(cabinet_ambient_routes, "_spike_query", failed)
    body = tc.get("/api/cabinet/ambient/spikes").json()
    assert body["ok"] is False
    assert body["spikes"] == []
    assert body["error"] == "ambient_spikes_unavailable"


def test_router_registered_on_api_routes():
    from scripts import api_routes

    paths = {getattr(route, "path", None) for route in api_routes.router.routes}
    assert "/api/cabinet/ambient/latest" in paths
    assert "/api/cabinet/ambient/history" in paths
    assert "/api/cabinet/ambient/spikes" in paths


# --- Live listen ----------------------------------------------------------

from orion.schemas.telemetry.ambient_audio import (  # noqa: E402
    AMBIENT_AUDIO_LIVE_SLOT_HEADER,
    ambient_audio_live_slot_name,
)


class LiveRing:
    """Writes slots exactly as the host reader's LiveRing does (shared contract);
    the reader side is covered by tests/test_ambient_audio_reader.py."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.seq = 0

    def write(self, pcm: bytes, *, sample_rate: int, channels: int) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seq += 1
        header = AMBIENT_AUDIO_LIVE_SLOT_HEADER.pack(self.seq, sample_rate, channels, 0)
        (self.directory / ambient_audio_live_slot_name(self.seq)).write_bytes(header + pcm)


def _live_app(monkeypatch, live_dir: Path) -> TestClient:
    monkeypatch.setattr(cabinet_ambient_routes.settings, "AMBIENT_AUDIO_LIVE_DIR", str(live_dir))
    app = FastAPI()
    app.include_router(cabinet_ambient_routes.router)
    return TestClient(app)


def test_live_reads_the_ring_the_host_reader_writes(tmp_path: Path) -> None:
    ring = LiveRing(tmp_path)
    for i in range(3):
        ring.write(bytes([i, 0]) * 4, sample_rate=16000, channels=1)
    slots = cabinet_ambient_routes.read_live_slots(tmp_path)
    assert sorted(slots) == [1, 2, 3]
    assert slots[3] == (16000, 1, bytes([2, 0]) * 4)


def test_next_live_chunks_starts_at_newest_then_follows_in_order() -> None:
    slots = {5: (16000, 1, b"e"), 6: (16000, 1, b"f")}
    chunks, nxt = cabinet_ambient_routes.next_live_chunks(slots, None)
    assert (chunks, nxt) == ([b"f"], 7)
    slots = {6: (16000, 1, b"f"), 7: (16000, 1, b"g"), 8: (16000, 1, b"h")}
    assert cabinet_ambient_routes.next_live_chunks(slots, nxt) == ([b"g", b"h"], 9)
    assert cabinet_ambient_routes.next_live_chunks(slots, 9) == ([], 9)


def test_next_live_chunks_resyncs_after_reader_restart_and_lag() -> None:
    # Reader restarted: seq reset to 1..2 while we wait for 40.
    restarted = {1: (16000, 1, b"a"), 2: (16000, 1, b"b")}
    assert cabinet_ambient_routes.next_live_chunks(restarted, 40) == ([b"b"], 3)
    # Listener fell a full ring behind: jump to newest, no backlog burst.
    ahead = {30: (16000, 1, b"x"), 31: (16000, 1, b"y")}
    assert cabinet_ambient_routes.next_live_chunks(ahead, 3) == ([b"y"], 32)


def test_live_endpoint_503_when_reader_has_written_nothing(monkeypatch, tmp_path: Path) -> None:
    client = _live_app(monkeypatch, tmp_path / "missing")
    resp = client.get("/api/cabinet/ambient/live")
    assert resp.status_code == 503
    assert "no chunks" in resp.json()["detail"]


def test_live_endpoint_streams_pcm_with_format_headers_and_ends_on_stall(
    monkeypatch, tmp_path: Path
) -> None:
    ring = LiveRing(tmp_path)
    ring.write(b"\x01\x00" * 4, sample_rate=16000, channels=1)
    ring.write(b"\x02\x00" * 4, sample_rate=16000, channels=1)
    monkeypatch.setattr(cabinet_ambient_routes, "_LIVE_STALL_SEC", 0.05)
    monkeypatch.setattr(cabinet_ambient_routes, "_LIVE_POLL_SEC", 0.01)
    client = _live_app(monkeypatch, tmp_path)
    resp = client.get("/api/cabinet/ambient/live")
    assert resp.status_code == 200
    assert resp.headers["x-orion-sample-rate"] == "16000"
    assert resp.headers["x-orion-channels"] == "1"
    assert resp.headers["x-orion-sample-format"] == "s16le"
    # Newest chunk only, then the stream ends because the reader went quiet.
    assert resp.content == b"\x02\x00" * 4


def test_next_live_chunks_skips_a_seq_the_reader_failed_to_write() -> None:
    slots = {5: (16000, 1, b"e"), 7: (16000, 1, b"g"), 8: (16000, 1, b"h")}
    assert cabinet_ambient_routes.next_live_chunks(slots, 6) == ([b"g", b"h"], 9)


def test_live_endpoint_503_when_reader_stopped_but_slots_remain(monkeypatch, tmp_path: Path) -> None:
    ring = LiveRing(tmp_path)
    ring.write(b"\x01\x00" * 4, sample_rate=16000, channels=1)
    old = datetime.now(timezone.utc).timestamp() - 60
    for path in tmp_path.glob("slot-*.pcm"):
        os.utime(path, (old, old))
    client = _live_app(monkeypatch, tmp_path)
    resp = client.get("/api/cabinet/ambient/live")
    assert resp.status_code == 503
    assert "stopped writing" in resp.json()["detail"]
