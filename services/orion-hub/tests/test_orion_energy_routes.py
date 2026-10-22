"""Hub Orion energy API (/api/energy/orion): Orion's own kWh against the house."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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

for _key in list(sys.modules):
    if _key in ("scripts", "app") or _key.startswith(("scripts.", "app.")):
        del sys.modules[_key]
for _path in (str(REPO_ROOT), str(HUB_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HUB_ROOT))

from scripts import energy_routes  # noqa: E402

DENVER = ZoneInfo("America/Denver")


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


def orion(hour, source, kwh, covered=3600.0):
    return {"hour_start": hour, "source": source, "kwh": kwh, "covered_sec": covered}


def agg(orion_rows=(), house_rows=(), price_rows=(), grain="day", periods=2, now=utc(2026, 10, 22, 18)):
    return energy_routes.aggregate_orion_energy(
        orion_rows=orion_rows, house_rows=house_rows, price_rows=price_rows,
        grain=grain, periods=periods, tz=DENVER, now=now)


def test_share_uses_only_hours_where_both_exist() -> None:
    h1, h2 = utc(2026, 10, 22, 14), utc(2026, 10, 22, 15)
    out = agg(
        orion_rows=[orion(h1, "athena", 0.4), orion(h1, "cooling", 0.6), orion(h2, "athena", 0.5)],
        house_rows=[{"hour_start": h1, "kwh": 4.0}],  # RMP has not delivered h2 yet
        periods=1,
    )
    b = out["buckets"][0]
    assert b["orion_kwh"] == pytest.approx(1.5)
    assert b["house_kwh"] == pytest.approx(4.0)
    assert b["overlap_hours"] == 1
    assert b["orion_share"] == pytest.approx(0.25)  # 1.0 of 4.0, not 1.5 of 4.0
    assert out["sources"] == ["athena", "cooling"]


def test_unknown_is_null_not_zero() -> None:
    out = agg(periods=2)
    for b in out["buckets"]:
        assert b["orion_kwh"] is None and b["house_kwh"] is None
        assert b["orion_share"] is None and b["orion_usd"] is None and b["house_usd"] is None


def test_day_buckets_follow_local_midnight() -> None:
    # 05:00 UTC Oct 22 is 23:00 MDT Oct 21: yesterday's bucket.
    out = agg(orion_rows=[orion(utc(2026, 10, 22, 5), "circe", 0.9), orion(utc(2026, 10, 22, 6), "circe", 0.8)])
    yesterday, today = out["buckets"]
    assert yesterday["label"] == "Wed Oct 21" and yesterday["orion_kwh"] == pytest.approx(0.9)
    assert today["orion_kwh"] == pytest.approx(0.8)
    assert today["complete"] is False and yesterday["complete"] is True


def test_dst_fall_back_day_has_25_hours() -> None:
    out = agg(grain="day", periods=1, now=utc(2026, 11, 2, 12))  # Nov 1 2026: DST ends
    day = agg(grain="day", periods=2, now=utc(2026, 11, 2, 12))["buckets"][0]
    assert day["label"] == "Sun Nov 1" and day["hours_elapsed"] == 25.0
    assert out["buckets"][0]["label"] == "Mon Nov 2"


@pytest.mark.parametrize("grain,labels", [
    ("week", ["Week of Oct 12", "Week of Oct 19"]),
    ("month", ["Sep 2026", "Oct 2026"]),
    ("quarter", ["Q3 2026", "Q4 2026"]),
    ("year", ["2025", "2026"]),
])
def test_calendar_grains(grain, labels) -> None:
    assert [b["label"] for b in agg(grain=grain)["buckets"]] == labels


def test_quarter_rolls_over_year() -> None:
    out = agg(grain="quarter", periods=3, now=utc(2026, 2, 1, 12))
    assert [b["label"] for b in out["buckets"]] == ["Q3 2025", "Q4 2025", "Q1 2026"]


def test_orion_dollars_at_marginal_rate_and_partial_pricing_is_visible() -> None:
    h1, h2 = utc(2026, 10, 22, 14), utc(2026, 10, 22, 15)
    out = agg(
        orion_rows=[orion(h1, "athena", 1.0), orion(h2, "athena", 2.0)],
        house_rows=[{"hour_start": h1, "kwh": 3.0}],
        price_rows=[{"hour_start": h1, "usd": 0.30, "marginal_usd_per_kwh": 0.16}],
        periods=1,
    )
    b = out["buckets"][0]
    assert b["orion_usd"] == pytest.approx(0.16)
    assert b["orion_priced_kwh"] == pytest.approx(1.0)  # h2 has no price: shown, not guessed
    assert b["house_usd"] == pytest.approx(0.30)


def test_hour_grain_labels_and_window() -> None:
    out = agg(grain="hour", periods=3, now=utc(2026, 10, 22, 18, 30))
    assert [b["label"] for b in out["buckets"]] == ["Oct 22 10am", "Oct 22 11am", "Oct 22 12pm"]
    assert out["window_start"] == "2026-10-22T16:00:00Z" and out["window_end"] == "2026-10-22T19:00:00Z"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(energy_routes.router)
    return TestClient(app)


def test_route_caps_periods_and_rejects_unknown_grain(client, monkeypatch) -> None:
    seen = {}

    async def fake_query(start, end):
        seen["span"] = end - start
        return [], [], []

    monkeypatch.setattr(energy_routes, "_orion_query", fake_query)
    body = client.get("/api/energy/orion", params={"grain": "year", "periods": 500}).json()
    assert body["ok"] is True and body["periods"] == 10 and len(body["buckets"]) == 10
    assert seen["span"] > timedelta(days=9 * 365)
    assert client.get("/api/energy/orion", params={"grain": "fortnight"}).status_code == 422


def test_route_reports_unavailable_without_inventing_numbers(client, monkeypatch) -> None:
    async def broken(start, end):
        raise RuntimeError("db down")

    monkeypatch.setattr(energy_routes, "_orion_query", broken)
    body = client.get("/api/energy/orion").json()
    assert body["ok"] is False and body["buckets"] == []
