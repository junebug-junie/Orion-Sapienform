"""People Orion knows: Hub routes and panel (2026-10-22, scripts/people_routes.py, static/js/people.js).

Pure checks first, then the REAL routes against a throwaway Postgres with the real
migrations (set ORION_MEMORY_EPISODE_TEST_DATABASE_URL, the admin DSN of a throwaway
server; each test creates and drops its own database). Synthetic rows only.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import people_routes as pr  # noqa: E402

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL = REPO_ROOT / "services" / "orion-sql-db"
MIGRATIONS = [SQL / f"manual_migration_{f}.sql" for f in (
    "walkway_camera_v1", "walkway_privacy_v2", "vision_confirm_asks_v1", "vision_ask_names_v1", "person_record_v1")]
needs_pg = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")
JS = HUB_ROOT / "static" / "js" / "people.js"
THUMB = "ab" * 32


def test_routes_are_mounted_on_the_hub_app() -> None:
    main = (HUB_ROOT / "scripts" / "main.py").read_text()
    assert "from scripts.people_routes import router as people_router" in main
    assert "app.include_router(people_router)" in main
    assert {r.path for r in pr.router.routes} == {
        "/api/people", "/api/people/{person_id}/facts", "/api/people/{person_id}/facts/{fact_id}/remove",
        "/api/people/{person_id}/status"}


def test_template_mounts_the_panel_under_orion_is_asking_and_loads_the_script() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text()
    assert html.index('id="visionAsksCard"') < html.index('id="peopleCard"') < html.index('id="ekgDrawer"')
    assert html.index('id="peopleStatus"') < html.index('id="peopleList"')
    assert '<script src="/static/js/people.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in html


def _node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = f"const api = require({json.dumps(str(JS))}); process.stdout.write(JSON.stringify({expr}));"
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_panel_helpers() -> None:
    got = _node(f"""[
      api.thumbUrl("thumb:{THUMB}"), api.thumbUrl("thumb:../etc"), api.thumbUrl(null),
      api.seenLine({{days: 0}}), api.seenLine({{days: 1}}), api.seenLine({{days: 4}}),
      api.statusLine([]), api.statusLine([{{status: "active"}}, {{status: "not_a_person"}}]),
      api.request("add-fact", "p/1", {{category: "age", value: "  ten "}}),
      api.request("remove-fact", "p1", {{fact_id: "f 1"}}),
      api.request("status", "p1", {{status: "not_a_person"}})
    ]""")
    assert got == [
        f"/api/vision/crop-thumbs/{THUMB}", "", "",
        "Not seen on camera", "Seen on camera on 1 day", "Seen on camera on 4 days",
        "Orion does not know anyone by name yet.", "Orion knows 1 person by name.",
        {"url": "/api/people/p%2F1/facts", "body": {"category": "age", "value": "ten"}},
        {"url": "/api/people/p1/facts/f%201/remove", "body": {}},
        {"url": "/api/people/p1/status", "body": {"status": "not_a_person"}},
    ]


# --- real Postgres -------------------------------------------------------------------------------------

def _run(body, published=None):
    async def _go():
        import asyncpg
        import httpx
        from fastapi import FastAPI

        name = f"people_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
        try:
            for f in MIGRATIONS:
                await pool.execute(f.read_text())
            app = FastAPI()
            app.include_router(pr.router)
            app.state.memory_pg_pool = pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub") as client:
                await body(pool, client)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await admin.close()

    async def _pub(event):
        (published if published is not None else []).append(event)
        return True

    orig = pr.ask_routes._publish_answered
    pr.ask_routes._publish_answered = _pub
    try:
        asyncio.run(_go())
    finally:
        pr.ask_routes._publish_answered = orig


async def _world(pool) -> None:
    await pool.execute("INSERT INTO person (person_id, display_name, role) VALUES "
                       "('p-sam', 'Sam', 'other'), ('p-ups', 'Parcel van', 'other'), ('p-typo', 'Smа', 'other'), "
                       "('r-you', 'Alex', 'interlocutor'), ('r-self', 'Orion', 'self')")
    for iid, days in (("sam-1", 4), ("sam-2", 2), ("you-1", 1)):
        await pool.execute(
            "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
            "last_seen_at, sighting_count, distinct_days, label) VALUES ($1, 'walkway', 'person', '{1,0}', 1, "
            "now(), now(), $2, $2, 'x')", iid, days)
    await pool.execute(
        "INSERT INTO vision_individual_sighting (sighting_id, individual_id, stream_id, started_at, ended_at, "
        "dwell_sec, observation_count, thumb_ref) VALUES ('s1', 'sam-1', 'walkway', now(), now(), 0, 1, $1)",
        f"thumb:{THUMB}")
    await pool.execute("INSERT INTO person_link (link_id, person_id, source_kind, source_ref, decided_by) "
                       "VALUES ('l5', 'p-ups', 'vision_individual', 'ups-1', 'camera_name')")
    for lid, pid, kind, ref in (("l1", "p-sam", "vision_individual", "sam-1"), ("l2", "p-sam", "vision_individual", "sam-2"),
                                ("l3", "p-sam", "vision_belonging", "bike"), ("l4", "r-you", "vision_individual", "you-1")):
        await pool.execute("INSERT INTO person_link (link_id, person_id, source_kind, source_ref, decided_by) "
                           "VALUES ($1, $2, $3, $4, 'camera_name')", lid, pid, kind, ref)
    await pool.execute("INSERT INTO person_fact (fact_id, person_id, category, value_text, voice) VALUES "
                       "('f1', 'p-sam', 'relationship', 'my neighbor', 'ask_answer')")
    await pool.execute(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, status, created_at, expires_at) VALUES "
        "('ask-ups', 'Who is Parcel van to you?', 'person_fact', 'p-ups|relationship', 'open', now(), "
        "now() + interval '1 day')")


@needs_pg
def test_list_shows_people_most_seen_first_without_orion() -> None:
    async def body(pool, client):
        await _world(pool)
        r = (await client.get("/api/people")).json()
        assert [p["name"] for p in r["people"]] == ["Sam", "Alex", "Parcel van"]
        sam = r["people"][0]
        assert (sam["clusters"], sam["belongings"], sam["days"], sam["image_ref"]) == (2, 1, 6, f"thumb:{THUMB}")
        assert [(f["label"], f["value"]) for f in sam["facts"]] == [("Relationship", "my neighbor")]
        assert r["people"][2]["open_question"] == {"ask_id": "ask-ups", "question": "Who is Parcel van to you?"}
        assert r["categories"][0] == {"key": "relationship", "label": "Relationship"}
    _run(body)


@needs_pg
def test_add_and_remove_a_fact() -> None:
    async def body(pool, client):
        await _world(pool)
        r = await client.post("/api/people/p-sam/facts", json={"category": "age", "value": "  about   ten "})
        assert r.status_code == 200
        fid = r.json()["fact_id"]
        assert await pool.fetchrow("SELECT category, value_text, voice FROM person_fact WHERE fact_id=$1", fid) == (
            "age", "about ten", "hub_edit")
        assert (await client.post("/api/people/p-sam/facts", json={"category": "shoe", "value": "x"})).status_code == 422
        assert (await client.post("/api/people/nobody/facts", json={"category": "age", "value": "x"})).status_code == 404
        assert (await client.post("/api/people/r-self/facts", json={"category": "age", "value": "x"})).status_code == 404
        assert (await client.post(f"/api/people/p-sam/facts/{fid}/remove")).status_code == 200
        assert (await client.post(f"/api/people/p-sam/facts/{fid}/remove")).status_code == 404
        assert (await client.post("/api/people/r-you/facts/f1/remove")).status_code == 404   # not their fact
        facts = (await client.get("/api/people")).json()["people"][0]["facts"]
        assert [f["fact_id"] for f in facts] == ["f1"]
    _run(body)


@needs_pg
def test_not_a_person_closes_its_question_and_sinks_to_the_bottom() -> None:
    published: list = []

    async def body(pool, client):
        await _world(pool)
        r = (await client.post("/api/people/p-ups/status", json={"status": "not_a_person"})).json()
        assert r["closed_asks"] == ["ask-ups"]
        assert await pool.fetchval("SELECT status FROM orion_ask WHERE ask_id='ask-ups'") == "dismissed"
        people = (await client.get("/api/people")).json()["people"]
        assert people[-1]["name"] == "Parcel van" and people[-1]["status"] == "not_a_person"
        assert people[-1]["open_question"] is None
        assert (await client.post("/api/people/p-ups/status", json={"status": "active"})).status_code == 200
        assert (await client.post("/api/people/p-ups/status", json={"status": "gone"})).status_code == 422
    _run(body, published)
    assert [(e.ask_id, e.status, e.source_kind) for e in published] == [("ask-ups", "dismissed", "person_fact")]
