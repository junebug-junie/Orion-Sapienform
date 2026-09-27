"""Tests for Hub energy (house electricity) read APIs."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
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

from scripts import energy_routes  # noqa: E402

AS_OF = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(energy_routes.router)
    return TestClient(app)


def test_latest_shapes_and_keeps_unknown_null(client, monkeypatch) -> None:
    stakes = {"as_of": AS_OF, "pressure": "unknown", "pressure_reason": "no_forecast_total",
              "cycle_to_date_total_usd": 17.2, "forecast_total_usd": None, "cycle_start": date(2026, 9, 1)}
    importer = {"as_of": AS_OF, "state": "healthy", "reason": "usage_fresh", "usage_lag_hours": 26.0}
    reconcile = [{"reconcile_kind": "actual", "billing_period_start": date(2026, 8, 12),
                  "orion_total_usd": 99.0, "utility_total_usd": 97.13, "delta_usd": 1.87,
                  "reconcile_gap": None, "bucket_deltas": '{"customer_charge": 0.16}'}]

    async def fake():
        return stakes, importer, reconcile

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    body = client.get("/api/energy/latest").json()
    assert body["ok"] is True
    assert body["stakes"]["forecast_total_usd"] is None
    assert body["stakes"]["as_of"] == "2026-09-27T18:00:00Z"
    assert body["stakes"]["cycle_start"] == "2026-09-01"
    assert body["stakes"]["pressure_reason"] == "no_forecast_total"
    assert body["importer"]["state"] == "healthy"
    assert body["reconcile"]["actual"]["bucket_deltas"] == {"customer_charge": 0.16}
    assert body["reconcile"]["actual"]["reconcile_gap"] is None
    assert "forecast" not in body["reconcile"]


def test_latest_accepts_already_decoded_bucket_deltas(client, monkeypatch) -> None:
    reconcile = [{"reconcile_kind": "forecast", "billing_period_start": date(2026, 9, 11),
                  "orion_total_usd": None, "reconcile_gap": "usage_incomplete", "bucket_deltas": {}}]

    async def fake():
        return {"as_of": AS_OF, "pressure": "normal"}, None, reconcile

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    body = client.get("/api/energy/latest").json()
    assert body["importer"] is None
    assert body["reconcile"]["forecast"]["bucket_deltas"] == {}
    assert body["reconcile"]["forecast"]["orion_total_usd"] is None


def test_latest_without_rows_is_not_ok(client, monkeypatch) -> None:
    async def fake():
        return None, None, []

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    assert client.get("/api/energy/latest").json() == {
        "ok": False, "stale": None, "as_of": None, "covered_through": None,
        "stakes": None, "importer": None, "reconcile": {},
    }


def _latest_with(monkeypatch, stakes):
    async def fake():
        return stakes, None, []

    monkeypatch.setattr(energy_routes, "_latest_query", fake)
    monkeypatch.setattr(energy_routes, "_now", lambda: AS_OF)


def test_latest_fresh_snapshot_is_not_stale_and_surfaces_coverage(client, monkeypatch) -> None:
    covered = datetime(2026, 9, 26, 6, 0, tzinfo=timezone.utc)
    _latest_with(monkeypatch, {"as_of": AS_OF - timedelta(seconds=60), "covered_through": covered, "pressure": "normal"})
    body = client.get("/api/energy/latest").json()
    assert body["stale"] is False
    assert body["as_of"] == "2026-09-27T17:59:00Z"
    assert body["covered_through"] == "2026-09-26T06:00:00Z"


def test_latest_old_snapshot_is_flagged_stale(client, monkeypatch) -> None:
    max_age = float(energy_routes.settings.ORION_ENERGY_STAKES_MAX_AGE_SEC)
    _latest_with(monkeypatch, {"as_of": AS_OF - timedelta(seconds=max_age + 1), "covered_through": None,
                               "pressure": "over_forecast", "orion_projected_total_usd": 88.4})
    body = client.get("/api/energy/latest").json()
    assert body["stale"] is True
    assert body["covered_through"] is None
    # The numbers still ship (for debugging); the flag is what stops them rendering as current.
    assert body["stakes"]["orion_projected_total_usd"] == 88.4


def test_latest_unreadable_as_of_is_stale_not_fresh(client, monkeypatch) -> None:
    _latest_with(monkeypatch, {"as_of": None, "pressure": "normal"})
    assert client.get("/api/energy/latest").json()["stale"] is True


def test_reconcile_newest_utility_version_wins_a_computed_at_tie() -> None:
    sql = " ".join(energy_routes._RECONCILE_SQL.split())
    assert "ORDER BY reconcile_kind, billing_period_start DESC, computed_at DESC, utility_as_of DESC" in sql


def test_latest_db_failure_is_reported(client, monkeypatch) -> None:
    async def boom():
        raise RuntimeError("no db")

    monkeypatch.setattr(energy_routes, "_latest_query", boom)
    body = client.get("/api/energy/latest").json()
    assert body["ok"] is False and body["error"] == "energy_unavailable"
    assert body["stakes"] is None and body["reconcile"] == {}
    assert body["stale"] is None


def test_daily_usage(client, monkeypatch) -> None:
    async def fake(days, tz):
        assert (days, tz) == (7, energy_routes.settings.HUB_ENERGY_TIMEZONE)
        return [{"day": date(2026, 9, 26), "kwh": 24.5, "hours": 24.0}]

    monkeypatch.setattr(energy_routes, "_daily_query", fake)
    body = client.get("/api/energy/usage/daily?days=7").json()
    assert body == {"ok": True, "days": 7, "points": [{"day": "2026-09-26", "kwh": 24.5, "hours": 24}]}


def test_daily_usage_defaults_to_14_days_and_rejects_out_of_range(client, monkeypatch) -> None:
    seen = []

    async def fake(days, tz):
        seen.append(days)
        return []

    monkeypatch.setattr(energy_routes, "_daily_query", fake)
    assert client.get("/api/energy/usage/daily").json() == {"ok": True, "days": 14, "points": []}
    assert seen == [14]
    assert client.get("/api/energy/usage/daily?days=0").status_code == 422
    assert client.get("/api/energy/usage/daily?days=91").status_code == 422


def test_daily_usage_db_failure_is_reported(client, monkeypatch) -> None:
    async def boom(days, tz):
        raise RuntimeError("no db")

    monkeypatch.setattr(energy_routes, "_daily_query", boom)
    body = client.get("/api/energy/usage/daily?days=3").json()
    assert body == {"ok": False, "error": "energy_unavailable", "days": 3, "points": []}


def test_timezone_setting_defaults_to_denver() -> None:
    assert energy_routes.settings.HUB_ENERGY_TIMEZONE == os.getenv("HUB_ENERGY_TIMEZONE", "America/Denver")


def test_daily_sql_buckets_by_local_day_and_counts_covered_hours() -> None:
    sql = energy_routes._DAILY_SQL
    assert "AT TIME ZONE $2" in sql
    assert "interval_end - interval_start" in sql
    assert "FROM energy_usage_interval" in sql


def test_router_is_mounted_in_api_routes() -> None:
    # Source check: importing api_routes pulls the full Hub dependency set,
    # which the minimal reading CI job does not install.
    source = (HUB_ROOT / "scripts" / "api_routes.py").read_text()
    assert "from .energy_routes import router as energy_router" in source
    assert "router.include_router(energy_router)" in source
