"""Walkway stories spot-check routes (motion stories slice 1, 2026-10-16).

Pure gate maths first, then the REAL routes against a throwaway Postgres with the
real migration (set ORION_MEMORY_EPISODE_TEST_DATABASE_URL, the admin DSN of a
throwaway server; each test creates and drops its own database). Synthetic rows only.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import motion_story_routes as msr  # noqa: E402

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
MIGRATION = REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_vision_motion_story_v1.sql"
needs_pg = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")


# --- pure ----------------------------------------------------------------------------------------------


def test_gate_needs_fifty_checked_eighty_percent_right_and_no_wrong_name() -> None:
    assert msr.tally_from_counts({"right": 49}, 0, 1)["gate"]["passed"] is False
    t = msr.tally_from_counts({"right": 40, "wrong_motion": 10}, 3, 1)
    assert (t["checked"], t["pct_right"], t["gate"]["passed"]) == (50, 0.8, True)
    assert msr.tally_from_counts({"right": 49, "wrong_name": 1}, 1, 1)["gate"]["passed"] is False
    assert msr.tally_from_counts({"right": 39, "wrong_other": 11}, 0, 1)["gate"]["passed"] is False
    empty = msr.tally_from_counts({}, 0, None)
    assert empty["checked"] == 0 and empty["pct_right"] is None and empty["gate"]["passed"] is False


def test_routes_are_mounted_on_the_hub_app() -> None:
    main = (HUB_ROOT / "scripts" / "main.py").read_text()
    assert "from scripts.motion_story_routes import router as motion_story_router" in main
    assert "app.include_router(motion_story_router)" in main
    paths = {r.path for r in msr.router.routes}
    assert paths == {"/api/vision/motion-stories", "/api/vision/motion-stories/{story_id}/verdict"}


def test_template_mounts_the_card_under_the_live_picture_and_loads_the_script() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text()
    card = html.index('id="walkwayStoriesCard"')
    assert html.index('id="visionDockedContainer"') < card < html.index('id="visionFloatingContainer"')
    assert html.index('id="walkwayStoriesTally"') < html.index('id="walkwayStoriesList"')
    assert '<script src="/static/js/walkway-stories.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in html
    assert (HUB_ROOT / "static" / "js" / "walkway-stories.js").is_file()


# --- real Postgres -------------------------------------------------------------------------------------


def _statements(sql: str) -> list[str]:
    body = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in body.split(";") if s.strip()]


def _run(body):
    async def _go():
        import asyncpg
        import httpx
        from fastapi import FastAPI

        name = f"motionstory_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
        try:
            for st in _statements(MIGRATION.read_text()):   # CONCURRENTLY: one statement at a time
                await pool.execute(st)
            app = FastAPI()
            app.include_router(msr.router)
            app.state.memory_pg_pool = pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub") as client:
                await body(pool, client)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await admin.close()

    asyncio.run(_go())


async def _story(pool, i: int, *, published=True, clean=True, name=None, version=1, minute=0) -> str:
    sid = f"{i:032x}"
    await pool.execute(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "speed_class, zone_path, published, drop_reason, clean, sentence, crop_ids, evidence, name_label, "
        "rules_version) VALUES ($1, 'walkway', 'person', now() - make_interval(mins => $2), now(), 3, 'walking', "
        "ARRAY['walkway'], $3, $4, $5, $6, ARRAY['c1'], $7::jsonb, $8, $9)",
        sid, 60 - minute, published, None if published else "glimpse", clean, f"A person walked {i}.",
        json.dumps([{"crop_id": "c1", "box": [1, 2, 3, 4], "clipped": False, "thumb": None}]), name, version)
    return sid


@needs_pg
def test_list_shows_published_clean_stories_newest_first_with_tally() -> None:
    async def body(pool, client):
        a = await _story(pool, 1, minute=1)
        b = await _story(pool, 2, minute=2, name="Alex")
        await _story(pool, 3, published=False)
        await _story(pool, 4, clean=False)
        r = await client.get("/api/vision/motion-stories")
        assert r.status_code == 200, r.text
        data = r.json()
        assert [s["story_id"] for s in data["stories"]] == [b, a]
        assert data["stories"][0]["name_label"] == "Alex"
        assert data["stories"][0]["evidence"][0]["box"] == [1, 2, 3, 4]
        assert data["stories"][0]["zone_path"] == ["walkway"] and data["stories"][0]["verdict"] is None
        assert data["tally"]["checked"] == 0 and data["tally"]["rules_version"] == 1
        r = await client.get("/api/vision/motion-stories", params={"clean_only": "false"})
        assert len(r.json()["stories"]) == 3

    _run(body)


@needs_pg
def test_verdicts_upsert_count_against_the_gate_and_unchecked_filter_works() -> None:
    async def body(pool, client):
        a = await _story(pool, 1)
        b = await _story(pool, 2, name="Alex")
        r = await client.post(f"/api/vision/motion-stories/{a}/verdict", json={"verdict": "right"})
        assert r.status_code == 200, r.text
        assert r.json()["tally"]["checked"] == 1 and r.json()["tally"]["right"] == 1
        r = await client.post(f"/api/vision/motion-stories/{b}/verdict", json={"verdict": "wrong_name"})
        t = r.json()["tally"]
        assert (t["checked"], t["wrong_name"], t["named_checked"], t["pct_right"]) == (2, 1, 1, 0.5)
        # A second click replaces the first.
        r = await client.post(f"/api/vision/motion-stories/{b}/verdict", json={"verdict": "right"})
        t = r.json()["tally"]
        assert (t["checked"], t["right"], t["wrong_name"]) == (2, 2, 0)
        assert await pool.fetchval("SELECT count(*) FROM vision_motion_story_verdict") == 2
        c = await _story(pool, 3)
        r = await client.get("/api/vision/motion-stories", params={"unchecked_only": "true"})
        assert [s["story_id"] for s in r.json()["stories"]] == [c]
        r = await client.get("/api/vision/motion-stories")
        assert {s["story_id"]: s["verdict"] for s in r.json()["stories"]}[a] == "right"

    _run(body)


@needs_pg
def test_tally_counts_only_the_newest_rules_version() -> None:
    async def body(pool, client):
        old = await _story(pool, 1, version=1)
        await client.post(f"/api/vision/motion-stories/{old}/verdict", json={"verdict": "right"})
        new = await _story(pool, 2, version=2)
        r = await client.get("/api/vision/motion-stories")
        assert r.json()["tally"]["rules_version"] == 2 and r.json()["tally"]["checked"] == 0
        r = await client.get("/api/vision/motion-stories")
        assert [x["story_id"] for x in r.json()["stories"]] == [new]   # old-rules stories are not offered
        r = await client.post(f"/api/vision/motion-stories/{new}/verdict", json={"verdict": "wrong_motion"})
        assert r.json()["tally"]["checked"] == 1 and r.json()["tally"]["wrong_motion"] == 1

    _run(body)


@needs_pg
def test_bad_ids_unknown_or_unpublished_stories_and_bad_verdicts_are_refused() -> None:
    async def body(pool, client):
        hidden = await _story(pool, 1, published=False)
        assert (await client.post("/api/vision/motion-stories/nothex/verdict",
                                  json={"verdict": "right"})).status_code == 400
        assert (await client.post(f"/api/vision/motion-stories/{'f' * 32}/verdict",
                                  json={"verdict": "right"})).status_code == 404
        assert (await client.post(f"/api/vision/motion-stories/{hidden}/verdict",
                                  json={"verdict": "right"})).status_code == 404
        ok = await _story(pool, 2)
        assert (await client.post(f"/api/vision/motion-stories/{ok}/verdict",
                                  json={"verdict": "maybe"})).status_code == 422
        assert await pool.fetchval("SELECT count(*) FROM vision_motion_story_verdict") == 0

    _run(body)


@needs_pg
def test_missing_migration_is_a_503_not_a_crash() -> None:
    async def body(pool, client):
        await pool.execute("DROP TABLE vision_motion_story_verdict")
        r = await client.get("/api/vision/motion-stories")
        assert r.status_code == 503 and r.json()["detail"] == "motion_story_schema_missing"

    _run(body)
