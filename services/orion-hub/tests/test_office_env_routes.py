"""Tests for Hub office environment latest and history APIs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
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

from scripts import office_env_routes  # noqa: E402


NOW = datetime(2026, 10, 27, 20, 0, 30, tzinfo=timezone.utc)
SAMPLE_TS = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    row = {
        "ts": SAMPLE_TS,
        "office_temp_c": 21.4,
        "office_humidity_pct": 33.0,
        "sensor_online": True,
        "stale": False,
        "sample_age_sec": 2.0,
        "payload_json": None,
    }
    row.update(overrides)
    return row


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(office_env_routes.settings, "OFFICE_ENV_NODE", "pan-1")
    monkeypatch.setattr(office_env_routes.settings, "OFFICE_ENV_STALE_AFTER_SEC", 90.0)
    monkeypatch.setattr(office_env_routes.settings, "CABINET_AMBIENT_HISTORY_MAX_POINTS", 800)
    monkeypatch.setattr(office_env_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "pan-1"
        assert hours == 24
        return []

    async def no_latest(*, node: str):
        assert node == "pan-1"
        return None

    monkeypatch.setattr(office_env_routes, "_history_query", no_rows)
    monkeypatch.setattr(office_env_routes, "_latest_query", no_latest)
    app = FastAPI()
    app.include_router(office_env_routes.router)
    return TestClient(app)


def test_router_prefix():
    assert office_env_routes.router.prefix == "/api/office/env"


def test_row_to_sample_omits_absent_readings():
    sample = office_env_routes.row_to_sample(
        _row(office_temp_c=None, office_humidity_pct=None, sensor_online=False, stale=True)
    )
    assert sample == {"ts": "2026-10-27T20:00:00Z", "sensor_online": False}


def test_rows_to_points_never_zero_fills():
    points = office_env_routes.rows_to_points(
        [
            {"t": SAMPLE_TS, "office_temp_c": 21.0, "office_humidity_pct": None},
            {"t": SAMPLE_TS, "office_temp_c": None, "office_humidity_pct": None},
        ]
    )
    assert points == [{"t": "2026-10-27T20:00:00Z", "office_temp_c": 21.0}]


def test_downsample_averages_buckets_and_skips_absent_fields():
    pts = [
        {"t": (SAMPLE_TS + timedelta(seconds=30 * i)).isoformat().replace("+00:00", "Z"), "office_temp_c": float(i)}
        for i in range(10)
    ]
    out = office_env_routes.downsample_points(pts, 2)
    assert [p["office_temp_c"] for p in out] == [2.0, 7.0]
    assert all("office_humidity_pct" not in p for p in out)


def test_latest_no_rows(client):
    assert client.get("/api/office/env/latest").json() == {"ok": False, "age_sec": None, "sample": None}


def test_latest_fresh_row_is_ok(client, monkeypatch):
    async def latest(*, node: str):
        return _row()

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is True
    assert body["age_sec"] == pytest.approx(30.0)
    assert body["sample"]["office_temp_c"] == 21.4
    assert body["sensor_stale"] is False


def test_latest_sensor_stale_row_is_not_ok_and_has_no_readings(client, monkeypatch):
    async def latest(*, node: str):
        return _row(office_temp_c=None, office_humidity_pct=None, sensor_online=False, stale=True)

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["sensor_stale"] is True
    assert "office_temp_c" not in body["sample"]


def test_latest_old_row_is_not_ok_even_if_sensor_was_fine(client, monkeypatch):
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 10, 27, 19, 0, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["sensor_stale"] is False


def test_history_rejects_unknown_window(client):
    assert client.get("/api/office/env/history?window=1h").status_code == 400


def test_history_returns_points_and_stats(client, monkeypatch):
    async def rows(*, node: str, hours: int):
        return [
            {"t": SAMPLE_TS, "office_temp_c": 20.0, "office_humidity_pct": 30.0},
            {"t": SAMPLE_TS + timedelta(seconds=30), "office_temp_c": 22.0, "office_humidity_pct": None},
        ]

    monkeypatch.setattr(office_env_routes, "_history_query", rows)
    body = client.get("/api/office/env/history?window=24h").json()
    assert body["ok"] is True
    assert len(body["points"]) == 2
    assert body["stats"]["temp_min"] == 20.0 and body["stats"]["temp_max"] == 22.0
    assert body["stats"]["humidity_min"] == 30.0 and body["stats"]["humidity_max"] == 30.0


def test_history_db_failure_is_ok_false_not_empty_success(client, monkeypatch):
    async def boom(*, node: str, hours: int):
        raise RuntimeError("db down")

    monkeypatch.setattr(office_env_routes, "_history_query", boom)
    body = client.get("/api/office/env/history?window=24h").json()
    assert body["ok"] is False
    assert body["error"] == "office_env_history_unavailable"
    assert body["points"] == []


def test_latest_publisher_dead_hides_the_old_reading(client, monkeypatch):
    """Publisher down 2h: the API must not hand the UI a temperature it could show as live."""
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["publisher_stale"] is True
    assert "office_temp_c" not in body["sample"]
    assert "office_humidity_pct" not in body["sample"]


def test_latest_db_failure_is_flagged_as_an_error(client, monkeypatch):
    async def boom(*, node: str):
        raise RuntimeError("db down")

    monkeypatch.setattr(office_env_routes, "_latest_query", boom)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is False
    assert body["error"] == "office_env_latest_unavailable"


def test_downsample_never_averages_across_an_outage():
    from datetime import timedelta as td

    start = datetime(2026, 10, 27, 0, 0, 0, tzinfo=timezone.utc)
    iso = lambda s: (start + td(seconds=s)).isoformat().replace("+00:00", "Z")
    before = [{"t": iso(30 * i), "office_temp_c": 20.0} for i in range(100)]
    after = [{"t": iso(30 * i + 3 * 3600), "office_temp_c": 30.0} for i in range(100)]
    out = office_env_routes.downsample_points(before + after, 20)
    assert not any(p["office_temp_c"] not in (20.0, 30.0) for p in out)  # no 25.0 blend
    assert [p.get("gap") for p in out].count(True) == 1
    first_after = next(p for p in out if p.get("gap"))
    assert first_after["office_temp_c"] == 30.0


def test_latest_carries_air_readings(client, monkeypatch):
    async def latest(*, node: str):
        return _row(office_eco2_ppm=612, office_tvoc_ppb=31, air_warming=False)

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    assert sample["office_eco2_ppm"] == 612.0
    assert sample["office_tvoc_ppb"] == 31.0
    assert sample["air_warming"] is False


def test_latest_warming_row_has_flag_and_no_air_numbers(client, monkeypatch):
    async def latest(*, node: str):
        return _row(office_eco2_ppm=None, office_tvoc_ppb=None, air_warming=True)

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    body = client.get("/api/office/env/latest").json()
    assert body["ok"] is True
    assert body["sample"]["air_warming"] is True
    assert "office_eco2_ppm" not in body["sample"]


def test_latest_publisher_dead_hides_old_air_reading(client, monkeypatch):
    async def latest(*, node: str):
        return _row(
            ts=datetime(2026, 10, 27, 19, 0, 0, tzinfo=timezone.utc),
            office_eco2_ppm=612, office_tvoc_ppb=31, air_warming=False,
        )

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    assert "office_eco2_ppm" not in sample and "office_tvoc_ppb" not in sample
    assert "air_warming" not in sample


def test_history_includes_air_points_and_stats(client, monkeypatch):
    async def rows(*, node: str, hours: int):
        return [
            {"t": SAMPLE_TS, "office_temp_c": None, "office_humidity_pct": None,
             "office_eco2_ppm": 500, "office_tvoc_ppb": 10},
            {"t": SAMPLE_TS + timedelta(seconds=30), "office_temp_c": 21.0, "office_humidity_pct": None,
             "office_eco2_ppm": 700, "office_tvoc_ppb": None},
        ]

    monkeypatch.setattr(office_env_routes, "_history_query", rows)
    body = client.get("/api/office/env/history?window=24h").json()
    assert len(body["points"]) == 2
    assert body["stats"]["eco2_min"] == 500.0 and body["stats"]["eco2_max"] == 700.0
    assert body["stats"]["tvoc_min"] == 10.0 and body["stats"]["tvoc_max"] == 10.0


def test_downsample_marks_a_per_field_gap_rows_do_not_show():
    """CCS811 warm-up: temp keeps rows flowing, eCO2 has a hole. The chart must break eCO2."""
    pts = []
    for i in range(40):
        p = {"t": (SAMPLE_TS + timedelta(seconds=30 * i)).isoformat().replace("+00:00", "Z"), "office_temp_c": 21.0}
        if not 10 <= i < 30:
            p["office_eco2_ppm"] = 600.0
        pts.append(p)
    out = office_env_routes.downsample_points(pts, 8)
    marked = [p for p in out if "office_eco2_ppm" in (p.get("gap_fields") or [])]
    assert len(marked) == 1
    assert all("office_temp_c" not in (p.get("gap_fields") or []) for p in out)


# --- PMSA003I + TSL2591 (read from payload_json for deploy-order safety) ---


def _payload(measurements: dict, state: dict | None = None) -> str:
    # asyncpg returns the json column as text.
    return json.dumps({"measurements": measurements, "state": state or {}})


def test_latest_reads_pm_and_lux_from_payload_json(client, monkeypatch):
    async def latest(*, node: str):
        return _row(payload_json=_payload(
            {"office_temp_c": 21.4, "office_pm1_ug_m3": 2, "office_pm25_ug_m3": 5,
             "office_pm10_ug_m3": 7, "office_lux": 312.5, "office_sound_dba": 41.5},
            {"light_saturated": False},
        ))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    assert sample["office_pm1_ug_m3"] == 2.0
    assert sample["office_pm25_ug_m3"] == 5.0
    assert sample["office_pm10_ug_m3"] == 7.0
    assert sample["office_lux"] == 312.5
    assert sample["office_sound_dba"] == 41.5
    assert sample["light_saturated"] is False


def test_latest_saturated_light_has_flag_and_no_lux(client, monkeypatch):
    async def latest(*, node: str):
        return _row(payload_json=_payload({"office_temp_c": 21.4}, {"light_saturated": True}))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    assert sample["light_saturated"] is True
    assert "office_lux" not in sample  # too bright to measure, never 0


def test_latest_pre_sensor_row_has_no_pm_or_light_keys(client, monkeypatch):
    """A row written before the PMSA003I/TSL2591 existed: absent, not zero."""
    async def latest(*, node: str):
        return _row(payload_json=_payload({"office_temp_c": 21.4}))

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    for key in ("office_pm1_ug_m3", "office_pm25_ug_m3", "office_pm10_ug_m3", "office_lux", "light_saturated"):
        assert key not in sample
    assert sample["office_temp_c"] == 21.4


def test_latest_publisher_dead_hides_pm_lux_and_saturation(client, monkeypatch):
    async def latest(*, node: str):
        return _row(
            ts=datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc),
            payload_json=_payload({"office_pm25_ug_m3": 5, "office_lux": 10.0}, {"light_saturated": True}),
        )

    monkeypatch.setattr(office_env_routes, "_latest_query", latest)
    sample = client.get("/api/office/env/latest").json()["sample"]
    assert "office_pm25_ug_m3" not in sample and "office_lux" not in sample
    assert "light_saturated" not in sample


def test_history_includes_pm_and_lux_points_and_stats(client, monkeypatch):
    async def rows(*, node: str, hours: int):
        return [
            {"t": SAMPLE_TS, "office_temp_c": 21.0, "office_pm25_ug_m3": 4.0, "office_pm10_ug_m3": 6.0, "office_lux": None},
            {"t": SAMPLE_TS + timedelta(seconds=30), "office_temp_c": 21.0, "office_pm25_ug_m3": 8.0, "office_lux": 120.0},
        ]

    monkeypatch.setattr(office_env_routes, "_history_query", rows)
    body = client.get("/api/office/env/history?window=24h").json()
    assert body["points"][0]["office_pm25_ug_m3"] == 4.0
    assert "office_lux" not in body["points"][0]
    stats = body["stats"]
    assert stats["pm25_min"] == 4.0 and stats["pm25_max"] == 8.0
    assert stats["pm10_min"] == 6.0 and stats["pm10_max"] == 6.0
    assert stats["lux_min"] == 120.0 and stats["pm1_min"] is None
    # The pre-existing stats keys are still there (response shape is backward compatible).
    for key in ("temp_min", "humidity_max", "eco2_min", "tvoc_max", "n_raw", "n"):
        assert key in stats


_NEW_COLUMNS = ("office_pm1_ug_m3", "office_pm25_ug_m3", "office_pm10_ug_m3", "office_lux", "office_sound_dba", "light_saturated")


def _bare_column_refs(sql: str) -> list[str]:
    """New-column names used as bare SQL identifiers, i.e. not inside payload_json->>'x'
    and not as an ``AS x`` alias or a reference to that alias in the outer query."""
    import re

    inner = sql.split("FROM office_env_sample")[0]
    inner = inner[inner.rfind("SELECT"):]  # the subquery's select list, which reads the table
    inner = re.sub(r"\(payload_json->'measurements'->>'[a-z0-9_]+'\)::double precision AS [a-z0-9_]+", "", inner)
    return [c for c in _NEW_COLUMNS if re.search(rf"\b{c}\b", inner)]


def test_queries_never_name_the_new_columns_directly():
    """Deploy-order safety: Hub may run before sql-writer has added these columns. A query
    that names one fails with UndefinedColumn and blanks the whole office strip."""
    assert _bare_column_refs(office_env_routes.HISTORY_SQL) == []
    assert _bare_column_refs(office_env_routes.LATEST_SQL) == []
    for col in _NEW_COLUMNS[:-1]:
        assert f"payload_json->'measurements'->>'{col}'" in office_env_routes.HISTORY_SQL


def test_history_sql_reads_only_pre_existing_columns():
    """Every column the inner SELECT reads from the table is one that existed before the
    PM/light sensors (or payload_json), so the query works on an un-migrated table."""
    import re

    inner = office_env_routes.HISTORY_SQL.split("FROM office_env_sample")[0]
    inner = inner[inner.rfind("SELECT") + len("SELECT"):]
    inner = re.sub(r"\(payload_json->'measurements'->>'[a-z0-9_]+'\)::double precision AS [a-z0-9_]+", "payload_json", inner)
    read = {tok.strip().split(" AS ")[0].strip() for tok in inner.split(",")}
    old_columns = {"ts", "office_temp_c", "office_humidity_pct", "office_eco2_ppm", "office_tvoc_ppb", "payload_json"}
    assert read <= old_columns, read - old_columns
