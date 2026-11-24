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
MIGRATION_V2 = REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_vision_motion_story_strip_v2.sql"
# The person route writes orion_ask rows and reads vision_individual (2026-10-18).
WALKWAY_MIGRATIONS = [REPO_ROOT / "services" / "orion-sql-db" / f"manual_migration_{f}.sql" for f in (
    "walkway_camera_v1", "walkway_privacy_v2", "vision_confirm_asks_v1", "vision_ask_names_v1")]
# vision_crop_observation.individual_id: the late individual links (2026-10-20).
SPLIT_MIGRATION = REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_vision_split_v1.sql"
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
    assert paths == {"/api/vision/motion-stories", "/api/vision/motion-stories/{story_id}/verdict",
                     "/api/vision/motion-stories/{story_id}/strip/{name}",
                     "/api/vision/motion-stories/{story_id}/people/{individual_id}",
                     "/api/vision/motion-stories/{story_id}/persons/{person_key}/not-same",
                     "/api/vision/known-names"}


def test_cant_tell_never_counts_toward_the_gate() -> None:
    t = msr.tally_from_counts({"right": 2, "cant_tell": 5}, 0, 1)
    assert (t["checked"], t["right"], t["cant_tell"], t["pct_right"]) == (2, 2, 5, 1.0)


# --- strip images: path whitelist, no traversal, no links ------------------------------------------


def _strip_client(tmp_path, monkeypatch):
    import httpx
    from fastapi import FastAPI

    monkeypatch.setattr(msr, "_strip_dir", lambda: tmp_path)
    app = FastAPI()
    app.include_router(msr.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub")


def test_strip_route_serves_only_whitelisted_files_inside_the_story_dir(tmp_path, monkeypatch) -> None:
    sid = "a" * 32
    (tmp_path / sid).mkdir()
    (tmp_path / sid / "frame.jpg").write_bytes(b"\xff\xd8frame")
    (tmp_path / sid / "crop_0.jpg").write_bytes(b"\xff\xd8crop")
    (tmp_path / sid / "notes.txt").write_text("secret")
    (tmp_path / "secret.jpg").write_bytes(b"outside")
    outside = tmp_path.parent / f"outside_{uuid.uuid4().hex}"
    outside.mkdir()
    (outside / "frame.jpg").write_bytes(b"linked")
    (tmp_path / ("b" * 32)).symlink_to(outside, target_is_directory=True)
    (tmp_path / ("c" * 32)).mkdir()
    (tmp_path / ("c" * 32) / "frame.jpg").symlink_to(tmp_path / "secret.jpg")

    async def go():
        async with _strip_client(tmp_path, monkeypatch) as c:
            r = await c.get(f"/api/vision/motion-stories/{sid}/strip/frame.jpg")
            assert r.status_code == 200 and r.content == b"\xff\xd8frame"
            assert r.headers["content-type"] == "image/jpeg" and r.headers["x-content-type-options"] == "nosniff"
            assert (await c.get(f"/api/vision/motion-stories/{sid}/strip/crop_0.jpg")).status_code == 200
            assert (await c.get(f"/api/vision/motion-stories/{sid}/strip/crop_1.jpg")).status_code == 404
            for bad in ("notes.txt", "crop_4.jpg", "..%2Fsecret.jpg", "frame.jpg.bak", "FRAME.jpg"):
                assert (await c.get(f"/api/vision/motion-stories/{sid}/strip/{bad}")).status_code in (400, 404), bad
            assert (await c.get("/api/vision/motion-stories/..%2F..%2Fetc/strip/frame.jpg")).status_code in (400, 404)
            assert (await c.get("/api/vision/motion-stories/AAAA/strip/frame.jpg")).status_code == 400
            # A story dir that is a link out of the root, or a frame that is a link: never followed.
            assert (await c.get(f"/api/vision/motion-stories/{'b' * 32}/strip/frame.jpg")).status_code == 404
            assert (await c.get(f"/api/vision/motion-stories/{'c' * 32}/strip/frame.jpg")).status_code == 404

    asyncio.run(go())


def test_template_mounts_the_card_under_the_live_picture_and_loads_the_script() -> None:
    html = (HUB_ROOT / "templates" / "index.html").read_text()
    card = html.index('id="walkwayStoriesCard"')
    assert html.index('id="visionDockedContainer"') < card < html.index('id="visionFloatingContainer"')
    assert html.index('id="walkwayStoriesTally"') < html.index('id="walkwayStoriesList"')
    assert '<script src="/static/js/walkway-stories.js?v={{HUB_UI_ASSET_VERSION}}" defer></script>' in html
    assert (HUB_ROOT / "static" / "js" / "walkway-stories.js").is_file()


# --- real Postgres -------------------------------------------------------------------------------------


def _people(story: dict) -> list[dict]:
    """The card's persons as one entry per person, with its first camera group's id (Patch B
    dropped ``people``; these tests' stories have one camera group per person unless linked)."""
    return [dict(p, individual_id=p["individual_ids"][0],
                 picture_crop_ids=[c for g in p["groups"] for c in g["picture_crop_ids"]],
                 thumb=next((c.get("thumb") for c in p["pictures"] if c.get("thumb")), None))
            for p in story["persons"]]


def _statements(sql: str) -> list[str]:
    body = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in body.split(";") if s.strip()]


def _run(body, published=None):
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
            for f in WALKWAY_MIGRATIONS:
                await pool.execute(f.read_text())
            for st in (_statements(MIGRATION.read_text()) + _statements(MIGRATION_V2.read_text())
                       + _statements(SPLIT_MIGRATION.read_text())):
                await pool.execute(st)   # CONCURRENTLY: one statement at a time
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

    if published is not None:
        # The module msr itself calls (conftest re-imports `scripts` per test, so never re-import here).
        ask_routes = msr.ask_routes

        async def _pub(event):
            published.append(event)
            return True

        orig = ask_routes._publish_answered
        ask_routes._publish_answered = _pub
        try:
            asyncio.run(_go())
        finally:
            ask_routes._publish_answered = orig
    else:
        asyncio.run(_go())


async def _story(pool, i: int, *, published=True, clean=True, name=None, version=1, minute=0,
                 strip="frame", dark=False) -> str:
    sid = f"{i:032x}"
    rec = None
    if strip == "frame":
        rec = {"version": 1, "frame": {"file": "frame.jpg", "t": "2026-10-17T00:47:09+00:00", "crop_id": "c1"},
               "crops": [{"crop_id": "c1", "t": "2026-10-17T00:47:09+00:00", "file": "crop_0.jpg"}],
               "dark": dark, "scene_contrast": 0.02 if dark else 0.25}
    elif strip == "none_saved":
        rec = {"version": 1, "frame": None, "crops": [{"crop_id": "c1", "missing": "frame_gone"}], "dark": None}
    await pool.execute(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "speed_class, zone_path, published, drop_reason, clean, sentence, crop_ids, evidence, name_label, "
        "rules_version) VALUES ($1, 'walkway', 'person', now() - make_interval(mins => $2), now(), 3, 'walking', "
        "ARRAY['walkway'], $3, $4, $5, $6, ARRAY['c1'], $7::jsonb, $8, $9)",
        sid, 60 - minute, published, None if published else "glimpse", clean, f"A person walked {i}.",
        json.dumps([{"crop_id": "c1", "box": [1, 2, 3, 4], "clipped": False, "thumb": None}]), name, version)
    if rec is not None:
        await pool.execute(
            "UPDATE vision_motion_story SET evidence_strip = $2::jsonb, dark = $3 WHERE story_id = $1",
            sid, json.dumps(rec), rec["dark"])
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


@needs_pg
def test_checkable_filter_is_on_by_default_and_hides_dark_and_pictureless_stories() -> None:
    async def body(pool, client):
        ok = await _story(pool, 1, minute=1)
        dark = await _story(pool, 2, minute=2, dark=True)
        nopic = await _story(pool, 3, minute=3, strip=None)
        gone = await _story(pool, 4, minute=4, strip="none_saved")
        pruned = await _story(pool, 5, minute=5)
        await pool.execute("UPDATE vision_motion_story SET evidence_strip = evidence_strip || "
                           "'{\"pruned_at\": \"x\"}'::jsonb WHERE story_id = $1", pruned)
        r = await client.get("/api/vision/motion-stories")
        assert r.status_code == 200, r.text
        assert [s["story_id"] for s in r.json()["stories"]] == [ok]
        assert r.json()["stories"][0]["evidence_strip"]["frame"]["file"] == "frame.jpg"
        assert r.json()["tally"]["not_checkable"] == 4
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        got = {s["story_id"]: s for s in r.json()["stories"]}
        assert set(got) == {ok, dark, nopic, gone, pruned}
        assert got[dark]["dark"] is True and got[nopic]["evidence_strip"] is None

    _run(body)


@needs_pg
def test_cant_tell_is_recorded_leaves_the_queue_and_never_counts() -> None:
    async def body(pool, client):
        a = await _story(pool, 1, strip=None)
        b = await _story(pool, 2)
        r = await client.post(f"/api/vision/motion-stories/{a}/verdict", json={"verdict": "cant_tell"})
        assert r.status_code == 200, r.text
        t = r.json()["tally"]
        assert (t["checked"], t["right"], t["cant_tell"], t["pct_right"]) == (0, 0, 1, None)
        row = await pool.fetchrow("SELECT verdict, cant_tell FROM vision_motion_story_verdict WHERE story_id = $1", a)
        assert row["verdict"] is None and row["cant_tell"] is True
        r = await client.post(f"/api/vision/motion-stories/{b}/verdict", json={"verdict": "right"})
        t = r.json()["tally"]
        assert (t["checked"], t["right"], t["cant_tell"], t["pct_right"]) == (1, 1, 1, 1.0)
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false", "unchecked_only": "true"})
        assert r.json()["stories"] == []
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert {s["story_id"]: s["verdict"] for s in r.json()["stories"]} == {a: "cant_tell", b: "right"}
        # Changing her mind: a real verdict replaces "can't tell" and counts.
        r = await client.post(f"/api/vision/motion-stories/{a}/verdict", json={"verdict": "wrong_motion"})
        t = r.json()["tally"]
        assert (t["checked"], t["cant_tell"], t["wrong_motion"]) == (2, 0, 1)
        assert await pool.fetchval("SELECT cant_tell FROM vision_motion_story_verdict WHERE story_id = $1", a) is False

    _run(body)


# --- who the story is about (2026-10-18) ---------------------------------------------------------------

THUMB = "e" * 64


def test_people_in_lists_each_individual_once_with_its_crops_and_first_thumb() -> None:
    ev = [{"crop_id": "c1", "individual_id": "ind-a", "thumb": None},
          {"crop_id": "c2", "individual_id": "ind-a", "thumb": THUMB},
          {"crop_id": "c3", "individual_id": None, "thumb": THUMB},
          {"crop_id": "c4", "individual_id": "ind-b", "thumb": "not-a-hash"}]
    got = msr.people_in(ev, "walker", "person")
    assert [(p["individual_id"], p["crop_ids"], p["thumb"], p["short_id"]) for p in got] == [
        ("ind-a", ["c1", "c2"], THUMB, "ind-a"), ("ind-b", ["c4"], None, "ind-b")]


def test_fill_late_links_fills_only_null_ids_and_keeps_order() -> None:
    ev = [{"crop_id": "c1", "individual_id": None}, {"crop_id": "c2", "individual_id": "ind-a"},
          {"crop_id": "c3", "individual_id": None}, {"individual_id": None}]
    assert msr.unlinked_crop_ids(ev) == ["c1", "c3"]
    got = msr.fill_late_links(ev, {"c1": "ind-j", "c2": "ind-z"})
    assert [e.get("individual_id") for e in got] == ["ind-j", "ind-a", None, None]  # c2 kept, c3 unknown
    assert ev[0]["individual_id"] is None                                           # input not mutated


async def _crop(pool, crop_id: str, individual_id) -> None:
    await pool.execute(
        "INSERT INTO vision_crop_observation (crop_id, observation_id, stream_id, observed_at, label, score, "
        "box_xyxy, individual_id) VALUES ($1, 'obs', 'walkway', now(), 'person', 0.9, ARRAY[1,2,3,4]::real[], $2)",
        crop_id, individual_id)


@needs_pg
def test_a_story_frozen_before_its_crops_were_linked_still_shows_and_names_the_person() -> None:
    # Live 2026-10-20: story 3ca2d1c1... had individual_id null on all 3 evidence
    # crops while vision_crop_observation linked them to Juniper's individual.
    async def body(pool, client):
        await _individual(pool, "ind-juniper", "Juniper")
        sid = await _walk(pool, 1, [None, None, None])
        for k in range(3):
            await _crop(pool, f"s1c{k}", "ind-juniper")
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert r.status_code == 200, r.text
        story = next(s for s in r.json()["stories"] if s["story_id"] == sid)
        assert [(p["individual_id"], p["label"], p["crop_ids"]) for p in _people(story)] == [
            ("ind-juniper", "Juniper", ["s1c0", "s1c1", "s1c2"])]
        assert story["display_sentence"].startswith("Juniper walked") and story["name_source"] == "rule"
        # The naming button the card now shows works on the late link too.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-juniper",
                              json={"action": "name", "name": "June"})
        assert r.status_code == 200, r.text

    _run(body, [])


def _ev(*ids, clipped=False):
    return [{"crop_id": f"c{i}", "individual_id": iid, "clipped": clipped, "thumb": None} for i, iid in enumerate(ids)]


def test_story_display_rule_uses_todays_names_and_juniper_wins_on_her_story() -> None:
    story = {"kind": "person", "name_label": None, "sentence": "A person walked left to right, over 5 s (3 detections).",
             "evidence": _ev("a", "a", "a")}
    story["people"] = msr.people_in(story["evidence"], "walker", "person")
    assert msr.story_display(story, {}, [])[1] == story["sentence"]
    name, sentence, why = msr.story_display(story, {"a": "Maya"}, [])
    assert (name, why) == ("Maya", "rule") and sentence.startswith("Maya walked left to right")
    # The rule still holds: an unnamed second id withholds the automatic name.
    other = dict(story, evidence=_ev("a", "a", "b"))
    other["people"] = msr.people_in(other["evidence"], "walker", "person")
    assert msr.story_display(other, {"a": "Maya"}, [])[0] is None
    # ...but Juniper's own answer on this story overrides it, for this story.
    theo = {"individual_id": "a", "action": "name", "name": "Theo", "ask_id": "x"}
    name, sentence, why = msr.story_display(other, {"a": "Maya"}, [theo])
    assert (name, why) == ("Theo", "juniper") and sentence.startswith("Theo walked")
    # "Someone else" with no name: the story says "A person" again, and a later "not sure" keeps that.
    named = dict(story, name_label="Maya", sentence="Maya walked left to right, over 5 s (3 detections).")
    se = {"individual_id": "a", "action": "someone_else", "name": None, "ask_id": "y"}
    unsure = {"individual_id": "a", "action": "unsure", "name": None, "ask_id": "z"}
    assert msr.story_display(named, {"a": "Maya"}, [se])[1].startswith("A person walked")
    assert msr.story_display(named, {"a": "Maya"}, [se, unsure])[1].startswith("A person walked")
    # "Not sure" alone changes nothing; an answer about a companion never names the walker.
    assert msr.story_display(named, {"a": "Maya"}, [unsure]) == ("Maya", named["sentence"], "rule")
    assert msr.story_display(named, {"a": "Maya"}, [dict(theo, individual_id="dog")])[2] == "rule"


def test_wrong_name_only_for_a_correction_of_the_walkers_written_name() -> None:
    assert msr.wrong_name_correction("Maya", "walker", "name", "Theo") is True
    assert msr.wrong_name_correction("Maya", "walker", "name", "maya") is False      # a confirmation
    assert msr.wrong_name_correction("Maya", "walker", "someone_else", "") is True
    assert msr.wrong_name_correction("Maya", "walker", "unsure", "") is False
    assert msr.wrong_name_correction("Maya", "companion", "name", "Theo") is False
    assert msr.wrong_name_correction(None, "walker", "name", "Theo") is False
    # Re-saving the name the individual already carries (renamed elsewhere) is not a verdict on the rule.
    assert msr.wrong_name_correction("Maya", "walker", "name", "mia", "Mia") is False


async def _individual(pool, iid: str, label=None, kind="person") -> None:
    await pool.execute(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, label_copied) "
        "VALUES ($1, 'walkway', $2, ARRAY[1.0, 0.0]::real[], 3, now(), now(), 1, 1, $3, false)", iid, kind, label)


async def _walk(pool, i: int, ids, *, name=None, together=None, kind="person", published=True,
                strip=None, thumbs=(0,), boxes=None) -> str:
    """``strip``: indexes of the evidence crops saved in the story strip (crop_0..);
    with any, the strip also has a frame. ``thumbs``: indexes with a vision-host thumb.
    ``boxes``: evidence index -> box (default a 2 x 2 box)."""
    sid = f"{i:032x}"
    ev = [{"crop_id": f"s{i}c{k}", "box": (boxes or {}).get(k, [1, 2, 3, 4]), "clipped": False, "individual_id": iid,
           "thumb": THUMB if k in thumbs else None} for k, iid in enumerate(ids)]
    subj = name or ("A dog" if kind == "dog" else "A person")
    await pool.execute(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "speed_class, zone_path, published, drop_reason, clean, sentence, crop_ids, evidence, name_label, "
        "individual_id, together_story_ids, rules_version) VALUES ($1, 'walkway', $2, now() - interval '1 minute', "
        "now(), $3, 'walking', ARRAY['walkway'], $4, $5, true, $6, $7, $8::jsonb, $9, $10, $11, 1)",
        sid, kind, len(ids), published, None if published else "companion",
        f"{subj} walked right to left on the walkway, over 5 s ({len(ids)} detections).",
        [e["crop_id"] for e in ev], json.dumps(ev), name, ids[0] if name else None, together)
    if strip:
        rec = {"version": 1, "dark": False,
               "frame": {"file": "frame.jpg", "t": "2026-11-22T01:28:33+00:00", "crop_id": f"s{i}c{strip[0]}"},
               "crops": [{"crop_id": f"s{i}c{k}", "t": f"2026-11-22T01:28:3{n}+00:00", "file": f"crop_{n}.jpg"}
                         for n, k in enumerate(strip)]}
        await pool.execute("UPDATE vision_motion_story SET evidence_strip = $2::jsonb, dark = false "
                           "WHERE story_id = $1", sid, json.dumps(rec))
    return sid


@needs_pg
def test_list_shows_who_each_story_is_about_with_todays_names() -> None:
    async def body(pool, client):
        await _individual(pool, "ind-maya", "Maya")
        await _individual(pool, "ind-anon")
        await _individual(pool, "ind-dog", kind="dog")
        dog = await _walk(pool, 9, ["ind-dog"] * 3, kind="dog", published=False)
        a = await _walk(pool, 1, ["ind-maya"] * 3, together=[dog])
        b = await _walk(pool, 2, ["ind-anon", "ind-anon", None])
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert r.status_code == 200, r.text
        got = {s["story_id"]: s for s in r.json()["stories"]}
        assert set(got) == {a, b}                       # the unpublished companion story is not listed
        # Named after the story was written: the rule names it now; what was written is kept.
        assert got[a]["display_sentence"].startswith("Maya walked right to left")
        assert got[a]["sentence"].startswith("A person walked") and got[a]["name_source"] == "rule"
        assert [(p["individual_id"], p["role"], p["label"], p["kind"]) for p in _people(got[a])] == [
            ("ind-maya", "walker", "Maya", "person"), ("ind-dog", "companion", None, "dog")]
        assert _people(got[a])[1]["story_id"] == dog and _people(got[a])[0]["thumb"] == THUMB
        anon = _people(got[b])[0]
        assert (anon["label"], anon["short_id"], anon["known"], anon["story_action"]) == (None, "ind-anon", True, None)
        assert got[b]["display_sentence"] == got[b]["sentence"]

    _run(body)


@needs_pg
def test_naming_a_person_is_a_who_is_this_answer_and_renames_the_story() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-anon")
        await _individual(pool, "ind-other")
        sid = await _walk(pool, 1, ["ind-anon", "ind-anon", "ind-other"])
        # A who-is-this card the opener already had open for this person stays for sql-writer to close.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-anon",
                              json={"action": "name", "name": "  Theo  "})
        assert r.status_code == 200, r.text
        out = r.json()
        row = await pool.fetchrow("SELECT * FROM orion_ask WHERE ask_id = $1", out["ask_id"])
        assert (row["source_kind"], row["source_ref"], row["status"], row["answer"]) == (
            "vision_individual", "ind-anon", "answered", "Theo")
        assert row["applied_at"] is None and row["answered_at"] is not None   # sql-writer applies it
        assert json.loads(row["answer_detail"]) == {"story_id": sid, "action": "name", "via": "motion_story",
                                                    "name": "Theo"}
        # Marked as Juniper's own card (attention patch 4a): it opens no vision.ask act, counts in no ask stat.
        assert json.loads(row["evidence_refs"]) == [f"motion_story:{sid}", "individual:ind-anon",
                                                    "rule:juniper_story_naming"]
        from orion.grammar.camera_ask_acts import is_juniper_card, is_orion_camera_ask
        assert is_juniper_card(row["evidence_refs"], None)
        assert not is_orion_camera_ask(row["source_kind"], row["evidence_refs"], None)
        assert row["image_ref"] == f"thumb:{THUMB}"
        # The substrate entity event, the same one a Hub "who is this?" answer publishes.
        assert [(e.ask_id, e.source_kind, e.source_ref, e.answer, e.status) for e in published] == [
            (out["ask_id"], "vision_individual", "ind-anon", "Theo", "answered")]
        # Her answer on this story wins even though another unnamed id is in it.
        assert out["story"]["display_sentence"].startswith("Theo walked right to left")
        assert out["story"]["name_source"] == "juniper"
        person = next(p for p in _people(out["story"]) if p["individual_id"] == "ind-anon")
        assert person["story_action"]["action"] == "name" and person["story_action"]["name"] == "Theo"
        assert out["wrong_name_recorded"] is False     # nothing was named before: not a correction
        assert await pool.fetchval("SELECT count(*) FROM vision_motion_story_verdict") == 0
        # A second name on the same story: the last one shows.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-anon", json={"action": "name", "name": "Theodore"})
        assert r.json()["story"]["display_sentence"].startswith("Theodore walked")
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == 2

    _run(body, published)


@needs_pg
def test_correcting_the_written_name_records_wrong_name_and_someone_else_records_pictures() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-maya", "Maya")
        # Crops 0 and 2 are in the story's strip; crop 1 is never shown, so it gets no verdict.
        sid = await _walk(pool, 1, ["ind-maya"] * 3, name="Maya", strip=[0, 2])
        await client.post(f"/api/vision/motion-stories/{sid}/verdict", json={"verdict": "right"})
        # Same name (any case): a confirmation, the "right" verdict stays.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya", json={"action": "name", "name": "maya"})
        assert r.status_code == 200 and r.json()["wrong_name_recorded"] is False
        assert await pool.fetchval("SELECT verdict FROM vision_motion_story_verdict WHERE story_id=$1", sid) == "right"
        # "Someone else, called Cora": picture verdicts for every picture of that individual the card showed.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya",
                              json={"action": "someone_else", "name": "Cora"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["wrong_name_recorded"] is True and out["published"] is False
        assert out["tally"]["wrong_name"] == 1 and out["story"]["verdict"] == "wrong_name"
        row = await pool.fetchrow("SELECT * FROM orion_ask WHERE ask_id = $1", out["ask_id"])
        assert (row["source_kind"], row["source_ref"], row["status"], row["answer"]) == (
            "vision_story_person", "ind-maya", "answered", None)
        detail = json.loads(row["answer_detail"])
        assert [(p["crop_id"], p["verdict"], p["name"], p["index"]) for p in detail["pictures"]] == [
            ("s1c0", "different", "Cora", 0), ("s1c2", "different", "Cora", 1)]
        assert out["story"]["display_sentence"].startswith("Cora walked")
        assert len(published) == 1                     # only the name was an entity event
        assert "why" not in detail                     # no reason typed: none stored
        # "Someone else" with no name, and how she could tell: the reason rides on the same ask row
        # (answer_detail.why) and comes back on the card.
        why = "  Back turned, but pushing a stroller.\n\n  Maya has no young kids;   not her build. "
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya",
                              json={"action": "someone_else", "why": why})
        assert r.json()["story"]["display_sentence"].startswith("A person walked")
        cleaned = "Back turned, but pushing a stroller.\nMaya has no young kids; not her build."
        row = await pool.fetchrow("SELECT answer_detail FROM orion_ask WHERE ask_id = $1", r.json()["ask_id"])
        assert json.loads(row["answer_detail"])["why"] == cleaned
        person = next(p for p in _people(r.json()["story"]) if p["individual_id"] == "ind-maya")
        assert person["story_action"]["why"] == cleaned
        # Too long a reason is refused, not cut silently.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya",
                              json={"action": "unsure", "why": "x" * 1001})
        assert r.status_code == 422
        # "Not sure": recorded, the latest direct answer still holds for the story.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya",
                              json={"action": "unsure", "name": "ignored"})
        assert r.status_code == 200
        d = json.loads(await pool.fetchval("SELECT answer_detail FROM orion_ask WHERE ask_id=$1", r.json()["ask_id"]))
        assert {p["verdict"] for p in d["pictures"]} == {"unsure"} and "name" not in d
        assert _people(r.json()["story"])[0]["story_action"]["action"] == "unsure"
        assert r.json()["story"]["display_sentence"].startswith("A person walked")   # "someone else" still holds
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert r.json()["stories"][0]["display_sentence"].startswith("A person walked")

    _run(body, published)


@needs_pg
def test_a_name_correction_never_overwrites_a_motion_verdict_and_a_failed_readback_still_publishes() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-maya", "Maya")
        sid = await _walk(pool, 1, ["ind-maya"] * 3, name="Maya")
        await client.post(f"/api/vision/motion-stories/{sid}/verdict", json={"verdict": "wrong_motion"})
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya", json={"action": "name", "name": "Theo"})
        assert r.status_code == 200 and r.json()["wrong_name_recorded"] is False
        assert await pool.fetchval("SELECT verdict FROM vision_motion_story_verdict WHERE story_id=$1", sid) == "wrong_motion"
        # The name is committed and published even if reading the story back fails.
        orig = msr._enrich

        async def boom(conn, stories):
            raise OSError("read-back down")

        msr._enrich = boom
        try:
            r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya", json={"action": "name", "name": "Ash"})
        finally:
            msr._enrich = orig
        assert r.status_code == 200 and r.json()["story"] is None
        assert [e.answer for e in published] == ["Theo", "Ash"]

    _run(body, published)


@needs_pg
def test_story_person_refusals_change_nothing() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-maya", "Maya")
        await _individual(pool, "ind-elsewhere")
        await _individual(pool, "ind-nopic")
        await _individual(pool, "ind-anon")
        sid = await _walk(pool, 1, ["ind-maya", "ind-maya", "ind-ghost", "ind-nopic", "ind-anon"], strip=[0, 2, 4])
        hidden = await _walk(pool, 2, ["ind-maya"] * 3, published=False)
        url = f"/api/vision/motion-stories/{sid}/people"
        cases = [
            (f"/api/vision/motion-stories/nothex/people/ind-maya", {"action": "name", "name": "X"}, 400, "bad_story_id"),
            (f"{url}/bad%20id", {"action": "name", "name": "X"}, 400, "bad_individual_id"),
            (f"/api/vision/motion-stories/{hidden}/people/ind-maya", {"action": "name", "name": "X"}, 404, "story_not_found"),
            (f"{url}/ind-elsewhere", {"action": "name", "name": "X"}, 422, "individual_not_in_story"),
            (f"{url}/ind-ghost", {"action": "name", "name": "X"}, 404, "individual_not_found"),
            (f"{url}/ind-maya", {"action": "name", "name": "   "}, 422, "name_empty"),
            (f"{url}/ind-maya", {"action": "name", "name": "x" * 121}, 422, "name_too_long"),
            (f"{url}/ind-maya", {"action": "someone_else", "name": "MAYA"}, 422, "someone_else_takes_another_name"),
            (f"{url}/ind-maya", {"action": "rename", "name": "X"}, 422, None),
            # No picture of them on the card: never asked, so never a blind answer.
            (f"{url}/ind-nopic", {"action": "name", "name": "X"}, 422, "no_picture_of_person"),
            (f"{url}/ind-nopic", {"action": "someone_else", "name": ""}, 422, "no_picture_of_person"),
            (f"{url}/ind-nopic", {"action": "unsure", "name": ""}, 422, "no_picture_of_person"),
            # "Yes, that's <name>" needs a name to say yes to.
            (f"{url}/ind-anon", {"action": "confirm", "name": ""}, 422, "nothing_to_confirm"),
            (f"{url}/ind-maya", {"action": "confirm", "name": "Mia"}, 409, "label_changed"),
        ]
        for path, payload, code, detail in cases:
            r = await client.post(path, json=payload)
            assert r.status_code == code, (path, payload, r.text)
            if detail:
                assert r.json()["detail"] == detail, (path, payload)
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == 0
        assert await pool.fetchval("SELECT count(*) FROM vision_motion_story_verdict") == 0
        assert published == []

    _run(body, published)


@needs_pg
def test_a_companion_from_a_story_walked_together_can_be_named_there() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-walker")
        await _individual(pool, "ind-friend")
        friend = await _walk(pool, 2, ["ind-friend"] * 3)
        sid = await _walk(pool, 1, ["ind-walker"] * 3, together=[friend])
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-friend", json={"action": "name", "name": "Ash"})
        assert r.status_code == 200, r.text
        story = r.json()["story"]
        assert [(p["individual_id"], p["role"]) for p in _people(story)] == [
            ("ind-walker", "walker"), ("ind-friend", "companion")]
        # A companion's name is not the walker's: the sentence keeps "A person".
        assert story["display_sentence"].startswith("A person walked") and story["name_source"] == "rule"
        assert published[0].source_ref == "ind-friend"

    _run(body, published)


# --- street continuity line (2026-10-20) ---------------------------------------------------------------

HANDOFF_MIGRATION = REPO_ROOT / "services" / "orion-sql-db" / "manual_migration_vision_handoff_v1.sql"


def test_handoff_info_says_which_camera_and_how_long() -> None:
    row = {"source_story_id": "a", "target_story_id": "b", "source_stream": "doorbell", "target_stream": "walkway",
           "transit_sec": 38.4, "status": "met"}
    assert msr.handoff_info("b", row) == {"kind": "from", "camera": "door camera", "seconds": 38}
    assert msr.handoff_info("a", row) == {"kind": "next", "camera": "walkway camera", "seconds": 38}
    assert msr.handoff_info("z", row) is None
    assert msr.handoff_info("b", {**row, "status": "paired", "transit_sec": -2})["kind"] == "same"
    assert msr.handoff_info("b", {**row, "status": "missed"}) is None


@needs_pg
def test_list_shows_the_handoff_line_and_is_silent_without_the_table() -> None:
    async def body(pool, client):
        a = await _story(pool, 1, minute=1)
        r = await client.get("/api/vision/motion-stories")
        assert r.status_code == 200 and r.json()["stories"][0]["handoff"] is None   # no table yet: fail-open
        for st in _statements(HANDOFF_MIGRATION.read_text()):
            await pool.execute(st)
        await pool.execute(
            "INSERT INTO vision_handoff (handoff_id, portal_id, portal_kind, source_stream, target_stream, "
            "source_story_id, target_story_id, status, transit_sec, scored_at) VALUES ('h1', 'door_to_walkway', "
            "'gap', 'doorbell', 'walkway', 'door-story', $1, 'met', 38.0, now())", a)
        r = await client.get("/api/vision/motion-stories")
        assert r.json()["stories"][0]["handoff"] == {"kind": "from", "camera": "door camera", "seconds": 38}

    _run(body)


def test_the_juniper_card_marker_is_the_shared_one() -> None:
    """Hub stamps the marker; sql-writer, the act ledger and the replay filter on it (patch 4a)."""
    from orion.grammar.camera_ask_acts import JUNIPER_STORY_CARD_REF

    assert msr.JUNIPER_STORY_CARD_REF == JUNIPER_STORY_CARD_REF


# --- each person's own pictures (2026-11-24) -----------------------------------------------------------
# Juniper, on story c486f5d8... (doorbell, 2026-11-22 01:28:30Z): six "who is this?" rows (Juniper,
# Sasha, Sasha, unnamed 38e112de, two companions) and one unlabelled picture. "how the f would i know
# which is which". Each row now carries the pictures of its own individual and its box numbers.


def test_box_numbers_follow_the_order_sql_writer_drew_them() -> None:
    ev = [{"crop_id": "a", "box": [1, 2, 3, 4]}, {"crop_id": "b", "box": None}, {"box": [1, 2, 3, 4]},
          {"crop_id": "c", "box": [1, 2, 3, 4]}, {"crop_id": "d", "box": [1, 2, 3, 4]}]
    # Only entries with a crop_id and a 4-number box are drawn: 1 = first (green), last = red.
    assert msr.box_numbers(ev) == {"a": {"n": 1, "colour": "green"}, "c": {"n": 2, "colour": "blue"},
                                   "d": {"n": 3, "colour": "red"}}
    assert msr.box_numbers([{"crop_id": "x", "box": [1, 2, 3, 4]}]) == {"x": {"n": 1, "colour": "green"}}


def test_attach_pictures_gives_each_person_only_their_own_crops() -> None:
    ev = [{"crop_id": "c0", "individual_id": "jun", "thumb": None, "box": [1, 2, 3, 4]},
          {"crop_id": "c1", "individual_id": "sasha", "thumb": THUMB, "box": [1, 2, 3, 4]},
          {"crop_id": "c2", "individual_id": "sasha", "thumb": None, "box": [1, 2, 3, 4]},
          {"crop_id": "c3", "individual_id": "thumbonly", "thumb": THUMB, "box": [1, 2, 3, 4]},
          {"crop_id": "c4", "individual_id": "nopic", "thumb": None, "box": [1, 2, 3, 4]}]
    strip = {"frame": {"file": "frame.jpg"}, "crops": [
        {"crop_id": "c0", "file": "crop_0.jpg", "t": "t0"}, {"crop_id": "c1", "file": "crop_1.jpg", "t": "t1"},
        {"crop_id": "c4", "missing": "frame_gone"}, {"crop_id": "c3", "file": "../../etc/passwd"}]}
    boxes = msr.box_numbers(ev)
    people = {p["individual_id"]: msr.attach_pictures(p, "s" * 32, strip, boxes)
              for p in msr.people_in(ev, "walker", "person")}
    assert people["jun"]["pictures"] == [{"story_id": "s" * 32, "file": "crop_0.jpg", "t": "t0", "crop_id": "c0"}]
    assert [b["n"] for b in people["sasha"]["boxes"]] == [2, 3]
    assert people["sasha"]["picture_crop_ids"] == ["c1"]          # c2 was never shown
    # No strip crop, but vision-host's thumbnail of c3: that is a picture of them.
    assert people["thumbonly"]["pictures"] == [] and people["thumbonly"]["picture_crop_ids"] == ["c3"]
    assert people["thumbonly"]["pictured"] is True
    # A crop whose frame was gone and no thumb: not pictured, so not asked.
    assert people["nopic"]["pictured"] is False and people["nopic"]["picture_crop_ids"] == []
    assert [b["colour"] for b in people["nopic"]["boxes"]] == ["red"]
    # A pruned strip shows no crops; not boxed when there is no frame to box them on.
    p = msr.attach_pictures(msr.people_in(ev[:1], "walker", "person")[0], "s" * 32, {**strip, "pruned_at": "x"}, None)
    assert (p["pictures"], p["boxes"], p["pictured"]) == ([], [], False)


@needs_pg
def test_list_gives_every_person_their_own_pictures_boxes_and_companions_their_own_story() -> None:
    async def body(pool, client):
        for iid, label in (("ind-jun", "Juniper"), ("ind-sasha1", "Sasha"), ("ind-sasha2", "Sasha"), ("ind-anon", None),
                           ("ind-c1", None), ("ind-c2", None)):
            await _individual(pool, iid, label)
        # The companions' story (not listed: it is a companion story) has its own strip.
        partner = await _walk(pool, 2, ["ind-c1", "ind-c2", "ind-c2"], published=False, strip=[0, 1], thumbs=())
        sid = await _walk(pool, 1, ["ind-jun", "ind-sasha1", "ind-sasha1", "ind-sasha2", "ind-anon"],
                          together=[partner], strip=[0, 1, 3, 4], thumbs=())
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert r.status_code == 200, r.text
        story = next(s for s in r.json()["stories"] if s["story_id"] == sid)
        got = {p["individual_id"]: p for p in _people(story)}
        assert list(got) == ["ind-jun", "ind-sasha1", "ind-sasha2", "ind-anon", "ind-c1", "ind-c2"]
        assert [(c["story_id"], c["file"]) for c in got["ind-jun"]["pictures"]] == [(sid, "crop_0.jpg")]
        assert [(c["story_id"], c["file"]) for c in got["ind-sasha1"]["pictures"]] == [(sid, "crop_1.jpg")]
        assert [b["n"] for b in got["ind-sasha1"]["boxes"]] == [2, 3]
        assert [(c["file"], b["n"]) for c, b in zip(got["ind-sasha2"]["pictures"], got["ind-sasha2"]["boxes"])] == [
            ("crop_2.jpg", 4)]
        assert got["ind-anon"]["boxes"] == [{"n": 5, "colour": "red"}]
        # Companions: their own crops, from the story they walked in; not boxed on this frame.
        assert [(c["story_id"], c["file"]) for c in got["ind-c1"]["pictures"]] == [(partner, "crop_0.jpg")]
        assert got["ind-c1"]["boxes"] == [] and got["ind-c1"]["role"] == "companion"
        assert got["ind-c2"]["picture_crop_ids"] == ["s2c1"]
        assert all(p["pictured"] for p in got.values())

    _run(body)


@needs_pg
def test_yes_thats_them_is_her_answer_with_same_pictures_for_the_pictures_shown() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-sasha", "Sasha")
        await _individual(pool, "ind-anon")
        sid = await _walk(pool, 1, ["ind-sasha", "ind-sasha", "ind-anon"], name="Sasha", strip=[0, 2], thumbs=())
        await client.post(f"/api/vision/motion-stories/{sid}/people/ind-sasha", json={"action": "someone_else"})
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-sasha",
                              json={"action": "confirm", "name": "sasha"})
        assert r.status_code == 200, r.text
        out = r.json()
        row = await pool.fetchrow("SELECT * FROM orion_ask WHERE ask_id = $1", out["ask_id"])
        # Her own "who is this?" answer: sql-writer makes the name a typed label (a copied one becomes hers).
        assert (row["source_kind"], row["source_ref"], row["status"], row["answer"]) == (
            "vision_individual", "ind-sasha", "answered", "Sasha")
        assert "rule:juniper_story_naming" in json.loads(row["evidence_refs"])
        detail = json.loads(row["answer_detail"])
        assert (detail["action"], detail["name"], detail["story_id"]) == ("confirm", "Sasha", sid)
        # One "same" picture per picture the card showed (s1c1 has no picture: no answer about it).
        assert [(p["crop_id"], p["verdict"], p["name"]) for p in detail["pictures"]] == [("s1c0", "same", "Sasha")]
        assert row["question"] == "Is this Sasha? Yes (said on a walkway story)"
        # Not a name correction; the substrate entity event like any typed name.
        assert out["wrong_name_recorded"] is False
        assert [(e.source_kind, e.answer) for e in published] == [("vision_individual", "Sasha")]
        # Her latest answer on this story wins: the yes undoes the earlier "someone else".
        assert out["story"]["display_sentence"].startswith("Sasha walked") and out["story"]["name_source"] == "juniper"
        person = next(p for p in _people(out["story"]) if p["individual_id"] == "ind-sasha")
        assert person["story_action"]["action"] == "confirm"
        # Renamed elsewhere after the card rendered: her yes to "Sasha" is refused, nothing written.
        await pool.execute("UPDATE vision_individual SET label = 'Alice' WHERE individual_id = 'ind-sasha'")
        before = await pool.fetchval("SELECT count(*) FROM orion_ask")
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-sasha",
                              json={"action": "confirm", "name": "Sasha"})
        assert (r.status_code, r.json()["detail"]) == (409, "label_changed")
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-sasha", json={"action": "confirm"})
        assert r.status_code == 409
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == before

    _run(body, published)


@needs_pg
def test_a_companion_in_two_stories_is_answered_on_the_pictures_the_list_showed() -> None:
    async def body(pool, client):
        await _individual(pool, "ind-walker")
        await _individual(pool, "ind-friend")
        # The friend walked in two together-stories. The list shows the first in together_story_ids
        # (story ...0c, whose id sorts AFTER ...0b): the answer must cover that story's pictures.
        late = await _walk(pool, 12, ["ind-friend"], published=False, strip=[0], thumbs=())
        early = await _walk(pool, 11, ["ind-friend"], published=False, strip=[0], thumbs=())
        sid = await _walk(pool, 1, ["ind-walker"], together=[late, early], strip=[0], thumbs=())
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        friend = next(p for s in r.json()["stories"] if s["story_id"] == sid
                      for p in _people(s) if p["individual_id"] == "ind-friend")
        assert friend["pictures"][0]["story_id"] == late and friend["picture_crop_ids"] == ["s12c0"]
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-friend", json={"action": "unsure"})
        assert r.status_code == 200, r.text
        d = json.loads(await pool.fetchval("SELECT answer_detail FROM orion_ask WHERE ask_id=$1", r.json()["ask_id"]))
        assert [p["crop_id"] for p in d["pictures"]] == ["s12c0"]

    _run(body)


def test_sql_writer_records_a_story_yes_as_its_same_picture_verdict() -> None:
    """The yes reuses the existing "same" picture verdict: sql-writer's picture_rows keeps it as written."""
    pytest.importorskip("numpy")
    import importlib.util

    path = REPO_ROOT / "services" / "orion-sql-writer" / "app" / "vision_individuals.py"
    spec = importlib.util.spec_from_file_location("_sqlwriter_vision_individuals_for_hub_test", path)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod          # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    assert mod.SOURCE_KIND_STORY_PERSON == msr.STORY_PERSON_KIND
    detail = {"pictures": [{"image_ref": "crop:s1c0", "index": 0, "verdict": "same", "crop_id": "s1c0",
                            "name": "Sasha"}]}
    rows = mod.picture_rows("ask-1", "ind-sasha", detail, None)
    assert [(r["v"], r["name"], r["crop_id"]) for r in rows] == [("same", "Sasha", "s1c0")]


def test_box_numbers_match_what_sql_writer_actually_draws(tmp_path, monkeypatch) -> None:
    """Parity with the real writer: run sql-writer's assemble_strip on one evidence list, record the
    numbers it draws at each box, and compare with the Hub's box_numbers for the same list."""
    pytest.importorskip("PIL")
    import importlib.util
    import sys
    from datetime import datetime, timezone

    from PIL import Image, ImageDraw

    path = REPO_ROOT / "services" / "orion-sql-writer" / "app" / "vision_story_strip.py"
    spec = importlib.util.spec_from_file_location("_sqlwriter_vision_story_strip_for_hub_test", path)
    vss = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = vss
    try:
        spec.loader.exec_module(vss)
    finally:
        sys.modules.pop(spec.name, None)
    t = datetime(2026, 11, 22, 1, 28, 33, tzinfo=timezone.utc)
    ev = [{"crop_id": "a", "box": [10, 10, 50, 90], "t": t.isoformat()},
          {"crop_id": None, "box": [1, 1, 2, 2], "t": t.isoformat()},        # no crop: never drawn
          {"crop_id": "b", "box": None, "t": t.isoformat()},                 # no box: not numbered
          {"crop_id": "c", "box": [60, 10, 100, 90], "t": t.isoformat()},
          {"crop_id": "d", "box": [110, 10, 150, 90], "t": t.isoformat()}]
    cfg = vss.StripConfig(strip_dir=str(tmp_path), frame_w=200.0, frame_h=100.0)
    pending = tmp_path / vss.PENDING
    pending.mkdir()
    Image.new("RGB", (200, 100)).save(pending / f"{vss.frame_key('walkway', t)}.jpg")
    drawn: list = []
    real = ImageDraw.Draw

    class Recorder:
        def __init__(self, im):
            self._d = real(im)

        def rectangle(self, xy, outline=None, width=1):
            drawn.append({"box": [round(v) for v in xy], "colour": outline})

        def text(self, xy, s, fill=None):
            drawn[-1]["n"] = int(s)

    monkeypatch.setattr(ImageDraw, "Draw", Recorder)
    vss.assemble_strip("f" * 32, "walkway", ev, cfg=cfg)
    names = {vss.FIRST: "green", vss.MID: "blue", vss.LAST: "red"}
    by_box = {tuple(d["box"]): {"n": d["n"], "colour": names[d["colour"]]} for d in drawn}
    want = {e["crop_id"]: by_box[tuple(e["box"])] for e in ev if e["crop_id"] and e["box"]}
    assert msr.box_numbers(ev) == want


def test_clean_why_trims_spaces_and_blank_lines_but_keeps_line_breaks() -> None:
    assert msr.clean_why("  a  b \r\n\n\t c ") == "a b\nc"
    assert msr.clean_why("") == "" and msr.clean_why("   \n ") == ""
    assert msr.clean_why("no\x00 kids\x07") == "no kids"   # NUL would break the jsonb insert


# --- one person per linked group (who-card redesign Patch A, 2026-11-24) ------------------------------


async def _same(pool, a: str, b: str, verdict: str = "same", ask_id: str = "ask-link") -> None:
    lo, hi = sorted((a, b))
    await pool.execute(
        "INSERT INTO vision_individual_pair_verdict (individual_a, individual_b, verdict, ask_id, decided_at) "
        "VALUES ($1, $2, $3, $4, now())", lo, hi, verdict, ask_id)


@needs_pg
def test_list_shows_linked_camera_groups_as_one_person_with_each_groups_pictures() -> None:
    async def body(pool, client):
        for iid, label in (("ind-sasha1", "Sasha"), ("ind-sasha2", "Sasha"), ("ind-x", "Sasha"), ("ind-anon", None),
                           ("ind-ann", "Ann"), ("ind-bea", "Bea")):
            await _individual(pool, iid, label)
        # sasha1 and sasha2 are joined through ind-x, a group that is not in this story.
        await _same(pool, "ind-sasha1", "ind-x")
        await _same(pool, "ind-x", "ind-sasha2")
        # A link that joins two typed names is a conflict: shown as two people, never one.
        await _same(pool, "ind-ann", "ind-bea")
        sid = await _walk(pool, 1, ["ind-sasha1", "ind-sasha1", "ind-sasha2", "ind-anon", "ind-ann", "ind-bea"],
                          strip=[0, 2, 3, 4], thumbs=(), boxes={2: [0, 0, 100, 200]})
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        assert r.status_code == 200, r.text
        story = next(s for s in r.json()["stories"] if s["story_id"] == sid)
        assert "people" not in story                                       # dropped in Patch B: persons only
        persons = story["persons"]
        assert [p["individual_ids"] for p in persons] == [["ind-sasha1", "ind-sasha2"], ["ind-anon"], ["ind-ann"],
                                                          ["ind-bea"]]
        sasha = persons[0]
        assert (sasha["person_key"], sasha["label"], sasha["role"]) == ("ind-sasha1", "Sasha", "walker")
        # Every group's pictures; the largest box first.
        assert [c["crop_id"] for c in sasha["pictures"]] == ["s1c2", "s1c0"]
        assert [b["n"] for b in sasha["boxes"]] == [1, 2, 3]
        assert [(g["individual_id"], g["picture_crop_ids"]) for g in sasha["groups"]] == [
            ("ind-sasha1", ["s1c0"]), ("ind-sasha2", ["s1c2"])]
        assert sasha["links"] == []                     # joined only through ind-x: no direct link here
        assert persons[1]["links"] == []

    _run(body)


@needs_pg
def test_one_answer_on_a_linked_person_writes_one_row_per_camera_group_on_its_own_pictures() -> None:
    """Orion's question 2: one card answer writes at most one ask per camera group (max 4), each on
    that group's SHOWN pictures; "can't see them" writes 'unsure' per group."""
    published: list = []

    async def body(pool, client):
        ids = [f"ind-g{i}" for i in range(5)]
        for iid in ids:
            await _individual(pool, iid, "Sasha")
        for a, b in zip(ids, ids[1:]):
            await _same(pool, a, b)
        await _individual(pool, "ind-anon")
        # Five camera groups of one person; the strip saved four crops (crop_0..3): g4 is never shown.
        sid = await _walk(pool, 1, ids + ["ind-anon"], name="Sasha", strip=[0, 1, 2, 3], thumbs=())
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-g2",
                              json={"action": "someone_else", "name": "Cora", "group": True})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["person_key"] == "ind-g0" and len(out["ask_ids"]) == 4 and out["skipped_individual_ids"] == []
        rows = await pool.fetch("SELECT source_kind, source_ref, answer_detail FROM orion_ask ORDER BY source_ref")
        assert [r["source_ref"] for r in rows] == ids[:4]
        for k, row in enumerate(rows):
            d = json.loads(row["answer_detail"])
            assert row["source_kind"] == "vision_story_person"
            assert [(p["crop_id"], p["verdict"], p["name"]) for p in d["pictures"]] == [(f"s1c{k}", "different", "Cora")]
            assert (d["person_key"], d["group_size"], d["story_id"]) == ("ind-g0", 4, sid)
        # One story verdict, not one per group.
        assert out["wrong_name_recorded"] is True
        assert await pool.fetchval("SELECT count(*) FROM vision_motion_story_verdict") == 1
        # "Can't see them": 'unsure' for each group's shown pictures, again one row per group.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-g0", json={"action": "unsure", "group": True})
        assert r.status_code == 200 and len(r.json()["ask_ids"]) == 4
        verdicts = [json.loads(v) for v in await pool.fetchval(
            "SELECT array_agg(answer_detail::text) FROM orion_ask WHERE answer_detail->>'action' = 'unsure'")]
        assert sorted(p["verdict"] for d in verdicts for p in d["pictures"]) == ["unsure"] * 4
        # The unnamed person beside them is still its own person.
        assert [p["individual_ids"] for p in r.json()["story"]["persons"]][-1] == ["ind-anon"]
        assert published == []                           # no typed name: no entity event

    _run(body, published)


@needs_pg
def test_answering_a_merged_person_again_writes_no_second_answer_per_group() -> None:
    """Orion's question 4: after a mid-story link joins two groups, answering the merged person
    again writes no second vision_individual answer per (story, group)."""
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-a", "Sasha")
        await _individual(pool, "ind-b", "Sasha")
        sid = await _walk(pool, 1, ["ind-a", "ind-b"], strip=[0, 1], thumbs=())
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a", json={"action": "confirm", "name": "Sasha"})
        assert r.status_code == 200 and len(r.json()["ask_ids"]) == 1
        await _same(pool, "ind-a", "ind-b")             # linked mid-story
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-b",
                              json={"action": "confirm", "name": "Sasha", "group": True})
        assert r.status_code == 200, r.text
        assert r.json()["skipped_individual_ids"] == ["ind-a"] and len(r.json()["ask_ids"]) == 1
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a",
                              json={"action": "confirm", "name": "sasha", "group": True})
        assert r.json()["ask_ids"] == [] and r.json()["skipped_individual_ids"] == ["ind-a", "ind-b"]
        counts = await pool.fetch("SELECT source_ref, count(*) AS n FROM orion_ask WHERE source_kind = "
                                  "'vision_individual' GROUP BY 1 ORDER BY 1")
        assert [(c["source_ref"], c["n"]) for c in counts] == [("ind-a", 1), ("ind-b", 1)]
        # A different answer is never skipped.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a", json={"action": "unsure", "group": True})
        assert len(r.json()["ask_ids"]) == 2

    _run(body, published)


@needs_pg
def test_yes_on_a_linked_person_refuses_two_names_and_names_an_unnamed_group() -> None:
    async def body(pool, client):
        await _individual(pool, "ind-a", "Sasha")
        await _individual(pool, "ind-b")
        await _same(pool, "ind-a", "ind-b")
        sid = await _walk(pool, 1, ["ind-a", "ind-b"], strip=[0, 1], thumbs=())
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-b",
                              json={"action": "confirm", "name": "Sasha", "group": True})
        assert r.status_code == 200 and len(r.json()["ask_ids"]) == 2
        assert {row["answer"] for row in await pool.fetch("SELECT answer FROM orion_ask")} == {"Sasha"}
        await pool.execute("UPDATE vision_individual SET label = 'Sam' WHERE individual_id = 'ind-b'")
        before = await pool.fetchval("SELECT count(*) FROM orion_ask")
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a",
                              json={"action": "unsure", "group": True})
        assert r.status_code == 200                    # two names: shown as two persons, ind-a alone
        assert len(r.json()["ask_ids"]) == 1
        await pool.execute("UPDATE vision_individual SET label = 'Sasha' WHERE individual_id = 'ind-b'")
        await pool.execute("UPDATE vision_individual SET label = 'Other' WHERE individual_id = 'ind-a'")
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-b",
                              json={"action": "someone_else", "name": "sasha", "group": True})
        assert (r.status_code, r.json()["detail"]) == (422, "someone_else_takes_another_name")
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == before + 1

    _run(body, [])


@needs_pg
def test_not_the_same_person_writes_her_pair_answer_for_one_direct_link() -> None:
    """Orion's question 1: a wrong 'same' link hides once linked groups show as one person. The grouped
    row offers "not the same person?", which writes the existing "same person?" answer, no."""
    async def body(pool, client):
        for iid in ("ind-a", "ind-b", "ind-c", "ind-d"):
            await _individual(pool, iid, "Sasha")
        await _same(pool, "ind-a", "ind-b", ask_id="ask-typed")
        sid = await _walk(pool, 1, ["ind-a", "ind-b"], strip=[0, 1], thumbs=())
        r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false"})
        person = r.json()["stories"][0]["persons"][0]
        assert person["links"] == [{"a": "ind-a", "b": "ind-b"}]
        url = f"/api/vision/motion-stories/{sid}/persons/{person['person_key']}/not-same"
        r = await client.post(url, json={"a": "ind-b", "b": "ind-a"})
        assert r.status_code == 200, r.text
        row = await pool.fetchrow("SELECT * FROM orion_ask WHERE ask_id = $1", r.json()["ask_id"])
        assert (row["source_kind"], row["source_ref"], row["answer"], row["status"]) == (
            "vision_same_individual", "ind-a|ind-b", "different", "answered")
        from orion.grammar.camera_ask_acts import is_orion_camera_ask
        assert not is_orion_camera_ask(row["source_kind"], row["evidence_refs"], row["answer_detail"])
        assert json.loads(row["answer_detail"])["action"] == "not_same"
        # Not a direct link of this person: refused, nothing written.
        r = await client.post(url, json={"a": "ind-a", "b": "ind-c"})
        assert (r.status_code, r.json()["detail"]) == (422, "not_a_direct_link_of_this_person")
        # Still joined through another path (a-c-b): the pair answer could not split them. Said so plainly.
        await _same(pool, "ind-a", "ind-c")
        await _same(pool, "ind-c", "ind-b")
        before = await pool.fetchval("SELECT count(*) FROM orion_ask")
        r = await client.post(url, json={"a": "ind-a", "b": "ind-b"})
        assert (r.status_code, r.json()["detail"]) == (409, "still_linked")
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == before

    _run(body)


# --- the one-at-a-time card (who-card redesign Patch B, 2026-11-24) -----------------------------------


@needs_pg
def test_known_names_are_typed_names_most_seen_first_one_per_name() -> None:
    async def body(pool, client):
        await _individual(pool, "ind-a", "Sasha")
        await _individual(pool, "ind-b", "sasha")
        await _individual(pool, "ind-c", "Theo")
        await _individual(pool, "ind-d", "Copied")
        await _individual(pool, "ind-dog", "Rex", kind="dog")
        await pool.execute("UPDATE vision_individual SET sighting_count = 5 WHERE individual_id = 'ind-c'")
        await pool.execute("UPDATE vision_individual SET label_copied = true WHERE individual_id = 'ind-d'")
        r = await client.get("/api/vision/known-names")
        assert r.status_code == 200, r.text
        assert r.json()["names"] == [{"name": "Theo", "seen": 5}, {"name": "Sasha", "seen": 2}]
        assert [n["name"] for n in (await client.get("/api/vision/known-names", params={"kind": "dog"})).json()["names"]] == ["Rex"]
        assert (await client.get("/api/vision/known-names", params={"kind": "x;drop"})).status_code == 422

    _run(body)


@needs_pg
def test_not_sure_stories_go_back_in_the_unchecked_queue_once() -> None:
    """Juniper's answer 5: a story whose person she could only answer "not sure" on is unchecked again,
    once: the first answer from the one-at-a-time card closes it (its old verdict stays)."""
    async def body(pool, client):
        await _individual(pool, "ind-a")
        await _individual(pool, "ind-b")
        unsure = await _walk(pool, 1, ["ind-a", "ind-b"], strip=[0, 1], thumbs=())
        named = await _walk(pool, 2, ["ind-a"], strip=[0], thumbs=())
        for sid in (unsure, named):
            await client.post(f"/api/vision/motion-stories/{sid}/verdict", json={"verdict": "right"})
        await client.post(f"/api/vision/motion-stories/{unsure}/people/ind-a", json={"action": "unsure"})
        await client.post(f"/api/vision/motion-stories/{named}/people/ind-a", json={"action": "unsure"})
        # A later real answer on the same group: no longer a "not sure" story.
        await client.post(f"/api/vision/motion-stories/{named}/people/ind-a", json={"action": "name", "name": "Ann"})

        async def queue():
            r = await client.get("/api/vision/motion-stories", params={"checkable_only": "false", "unchecked_only": "true"})
            return {s["story_id"]: s["reopened"] for s in r.json()["stories"]}

        assert await queue() == {unsure: True}
        # The old list card (a cached page) does not close it; the one-at-a-time card does.
        await client.post(f"/api/vision/motion-stories/{unsure}/people/ind-b", json={"action": "unsure"})
        assert await queue() == {unsure: True}
        r = await client.post(f"/api/vision/motion-stories/{unsure}/people/ind-b",
                              json={"action": "unsure", "card": "one_at_a_time", "dont_know": True, "group": True})
        assert r.status_code == 200, r.text
        d = json.loads(await pool.fetchval("SELECT answer_detail FROM orion_ask WHERE ask_id = $1", r.json()["ask_id"]))
        assert (d["card"], d["dont_know"], d["action"]) == ("one_at_a_time", True, "unsure")
        assert await queue() == {}
        assert await pool.fetchval("SELECT verdict FROM vision_motion_story_verdict WHERE story_id = $1", unsure) == "right"

    _run(body)


@needs_pg
def test_tally_counts_cant_see_answers_apart() -> None:
    async def body(pool, client):
        await _individual(pool, "ind-a")
        sid = await _walk(pool, 1, ["ind-a"], strip=[0], thumbs=())
        await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a", json={"action": "unsure"})
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-a", json={"action": "someone_else"})
        assert (r.json()["tally"]["person_answers"], r.json()["tally"]["cant_see"]) == (2, 1)

    _run(body)
