"""Walkway history route (walkway rollups, 2026-10-18).

The route reads vision_rollup newest first, validates grain/stream, never sends
individual ids, and hides story content while the spot-check gate is closed. A
missing table is a 503 named after rollups, not the ask card.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from scripts import walkway_rollup_routes as wrr  # noqa: E402

T = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)


def _row(grain="day", gate=False):
    rollup = {
        "total": {"passings": 185, "distinct": 93, "sightings": 400},
        "kinds": {"person": {"passings": 105, "distinct": 50, "sightings": 200}},
        "named": {"Maeve": {"times": ["15:31"], "sightings": 2, "individual_ids": ["secret-ind"]}},
        "individuals": {"secret-ind": {"kind": "person"}},
        "regulars": {"count": 3, "by_kind": {"person": 3}, "ids": ["secret-reg"]},
        "camera": {"up_minutes": 1212, "expected_minutes": 1440, "status": "ok", "down_spans": ["13:00–14:00"]},
        "captions": [{"at": "14:23", "text": "A dog on a leash.", "detector": ["dog"]}],
        "stories": ({"gate_passed": True, "checked": 60, "count": 1, "sentences": [{"at": "1", "text": "A person walked."}]}
                    if gate else {"gate_passed": False, "checked": 8, "sentences": [{"text": "leak"}]}),
    }
    return {"stream_id": "walkway", "grain": grain, "period_start": T, "period_end": T, "label": "2026-10-17",
            "complete": True, "line": "Walkway Sat 17 Oct 2026: 185 passings.", "rollup": json.dumps(rollup),
            "computed_at": T}


class _Conn:
    def __init__(self, rows):
        self.rows, self.args = rows, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        self.args = args
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self.conn


def _client(conn):
    app = FastAPI()
    app.include_router(wrr.router)
    app.state.memory_pg_pool = _Pool(conn)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def test_routes_are_mounted_on_the_hub_app() -> None:
    main = (HUB_ROOT / "scripts" / "main.py").read_text()
    assert "from scripts.walkway_rollup_routes import router as walkway_rollup_router" in main
    assert "app.include_router(walkway_rollup_router)" in main
    assert {r.path for r in wrr.router.routes} == {"/api/vision/rollups"}


def test_lists_newest_first_without_ids_and_with_the_gate_respected() -> None:
    async def go():
        conn = _Conn([_row()])
        async with _client(conn) as c:
            r = await c.get("/api/vision/rollups?grain=day")
            assert r.status_code == 200
            body = r.json()
            assert conn.args == ("walkway", "day", 14)
            row = body["rows"][0]
            assert row["line"].endswith("185 passings.") and row["total"]["passings"] == 185
            assert "secret" not in json.dumps(body)
            assert row["stories"] == {"gate_passed": False, "checked": 8}
            assert row["captions"][0]["text"] == "A dog on a leash."
            conn2 = _Conn([_row(gate=True)])
        async with _client(conn2) as c:
            row = (await c.get("/api/vision/rollups?grain=hour&limit=500")).json()["rows"][0]
            assert conn2.args == ("walkway", "hour", wrr.MAX_LIMIT["hour"])
            assert row["stories"]["sentences"][0]["text"] == "A person walked."

    asyncio.run(go())


def test_bad_grain_or_stream_is_400_and_missing_table_is_503() -> None:
    async def go():
        async with _client(_Conn([])) as c:
            assert (await c.get("/api/vision/rollups?grain=year")).status_code == 400
            assert (await c.get("/api/vision/rollups?stream_id=../x")).status_code == 400
        try:
            import asyncpg

            err = asyncpg.exceptions.UndefinedTableError('relation "vision_rollup" does not exist')
        except ImportError:  # pragma: no cover
            pytest.skip("asyncpg missing")
        async with _client(_Conn(err)) as c:
            r = await c.get("/api/vision/rollups?grain=week")
            assert r.status_code == 503 and r.json()["detail"] == "rollup_schema_missing"

    asyncio.run(go())


def test_template_mounts_the_panel_under_the_live_picture_and_loads_the_script() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text()
    card = html.index('id="walkwayHistoryCard"')
    assert html.index('id="visionDockedContainer"') < card < html.index('id="visionFloatingContainer"')
    assert html.index('id="walkwayHistoryTabs"') < html.index('id="walkwayHistoryList"')
    assert '<script src="/static/js/walkway-history.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in html
    assert (HUB_ROOT / "static" / "js" / "walkway-history.js").is_file()
