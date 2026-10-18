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
                     "/api/vision/motion-stories/{story_id}/people/{individual_id}"}


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
            for st in _statements(MIGRATION.read_text()) + _statements(MIGRATION_V2.read_text()):
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


def _ev(*ids, clipped=False):
    return [{"crop_id": f"c{i}", "individual_id": iid, "clipped": clipped, "thumb": None} for i, iid in enumerate(ids)]


def test_story_display_rule_uses_todays_names_and_juniper_wins_on_her_story() -> None:
    story = {"kind": "person", "name_label": None, "sentence": "A person walked left to right, over 5 s (3 detections).",
             "evidence": _ev("a", "a", "a")}
    story["people"] = msr.people_in(story["evidence"], "walker", "person")
    assert msr.story_display(story, {}, {})[1] == story["sentence"]
    name, sentence, why = msr.story_display(story, {"a": "Maya"}, {})
    assert (name, why) == ("Maya", "rule") and sentence.startswith("Maya walked left to right")
    # The rule still holds: an unnamed second id withholds the automatic name.
    other = dict(story, evidence=_ev("a", "a", "b"))
    other["people"] = msr.people_in(other["evidence"], "walker", "person")
    assert msr.story_display(other, {"a": "Maya"}, {})[0] is None
    # ...but Juniper's own answer on this story overrides it, for this story.
    acts = {"a": {"action": "name", "name": "Theo", "answered_at": "2026-10-18T01:00:00+00:00", "ask_id": "x"}}
    name, sentence, why = msr.story_display(other, {"a": "Maya"}, acts)
    assert (name, why) == ("Theo", "juniper") and sentence.startswith("Theo walked")
    # "Someone else" with no name: the story says "A person" again; "not sure" changes nothing.
    named = dict(story, name_label="Maya", sentence="Maya walked left to right, over 5 s (3 detections).")
    acts = {"a": {"action": "someone_else", "name": None, "answered_at": "t", "ask_id": "y"}}
    assert msr.story_display(named, {"a": "Maya"}, acts)[1].startswith("A person walked")
    acts = {"a": {"action": "unsure", "name": None, "answered_at": "t", "ask_id": "z"}}
    assert msr.story_display(named, {"a": "Maya"}, acts) == ("Maya", named["sentence"], "rule")


def test_wrong_name_only_for_a_correction_of_the_walkers_written_name() -> None:
    assert msr.wrong_name_correction("Maya", "walker", "name", "Theo") is True
    assert msr.wrong_name_correction("Maya", "walker", "name", "maya") is False      # a confirmation
    assert msr.wrong_name_correction("Maya", "walker", "someone_else", "") is True
    assert msr.wrong_name_correction("Maya", "walker", "unsure", "") is False
    assert msr.wrong_name_correction("Maya", "companion", "name", "Theo") is False
    assert msr.wrong_name_correction(None, "walker", "name", "Theo") is False


async def _individual(pool, iid: str, label=None, kind="person") -> None:
    await pool.execute(
        "INSERT INTO vision_individual (individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, "
        "last_seen_at, sighting_count, distinct_days, label, label_copied) "
        "VALUES ($1, 'walkway', $2, ARRAY[1.0, 0.0]::real[], 3, now(), now(), 1, 1, $3, false)", iid, kind, label)


async def _walk(pool, i: int, ids, *, name=None, together=None, kind="person", published=True) -> str:
    sid = f"{i:032x}"
    ev = [{"crop_id": f"s{i}c{k}", "box": [1, 2, 3, 4], "clipped": False, "individual_id": iid,
           "thumb": THUMB if k == 0 else None} for k, iid in enumerate(ids)]
    subj = name or ("A dog" if kind == "dog" else "A person")
    await pool.execute(
        "INSERT INTO vision_motion_story (story_id, stream_id, kind, started_at, ended_at, detection_count, "
        "speed_class, zone_path, published, drop_reason, clean, sentence, crop_ids, evidence, name_label, "
        "individual_id, together_story_ids, rules_version) VALUES ($1, 'walkway', $2, now() - interval '1 minute', "
        "now(), $3, 'walking', ARRAY['walkway'], $4, $5, true, $6, $7, $8::jsonb, $9, $10, $11, 1)",
        sid, kind, len(ids), published, None if published else "companion",
        f"{subj} walked right to left on the walkway, over 5 s ({len(ids)} detections).",
        [e["crop_id"] for e in ev], json.dumps(ev), name, ids[0] if name else None, together)
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
        assert [(p["individual_id"], p["role"], p["label"], p["kind"]) for p in got[a]["people"]] == [
            ("ind-maya", "walker", "Maya", "person"), ("ind-dog", "companion", None, "dog")]
        assert got[a]["people"][1]["story_id"] == dog and got[a]["people"][0]["thumb"] == THUMB
        anon = got[b]["people"][0]
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
        assert json.loads(row["evidence_refs"]) == [f"motion_story:{sid}", "individual:ind-anon"]
        assert row["image_ref"] == f"thumb:{THUMB}"
        # The substrate entity event, the same one a Hub "who is this?" answer publishes.
        assert [(e.ask_id, e.source_kind, e.source_ref, e.answer, e.status) for e in published] == [
            (out["ask_id"], "vision_individual", "ind-anon", "Theo", "answered")]
        # Her answer on this story wins even though another unnamed id is in it.
        assert out["story"]["display_sentence"].startswith("Theo walked right to left")
        assert out["story"]["name_source"] == "juniper"
        person = next(p for p in out["story"]["people"] if p["individual_id"] == "ind-anon")
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
        sid = await _walk(pool, 1, ["ind-maya"] * 3, name="Maya")
        await client.post(f"/api/vision/motion-stories/{sid}/verdict", json={"verdict": "right"})
        # Same name (any case): a confirmation, the "right" verdict stays.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya", json={"action": "name", "name": "maya"})
        assert r.status_code == 200 and r.json()["wrong_name_recorded"] is False
        assert await pool.fetchval("SELECT verdict FROM vision_motion_story_verdict WHERE story_id=$1", sid) == "right"
        # "Someone else, called Cora": picture verdicts for every crop of that individual in the story.
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
            ("s1c0", "different", "Cora", 0), ("s1c1", "different", "Cora", 1), ("s1c2", "different", "Cora", 2)]
        assert out["story"]["display_sentence"].startswith("Cora walked")
        assert len(published) == 1                     # only the name was an entity event
        # "Someone else" with no name: the story says "A person".
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya", json={"action": "someone_else"})
        assert r.json()["story"]["display_sentence"].startswith("A person walked")
        # "Not sure": recorded, the latest direct answer still holds for the story.
        r = await client.post(f"/api/vision/motion-stories/{sid}/people/ind-maya",
                              json={"action": "unsure", "name": "ignored"})
        assert r.status_code == 200
        d = json.loads(await pool.fetchval("SELECT answer_detail FROM orion_ask WHERE ask_id=$1", r.json()["ask_id"]))
        assert {p["verdict"] for p in d["pictures"]} == {"unsure"} and "name" not in d
        assert r.json()["story"]["people"][0]["story_action"]["action"] == "unsure"

    _run(body, published)


@needs_pg
def test_story_person_refusals_change_nothing() -> None:
    published: list = []

    async def body(pool, client):
        await _individual(pool, "ind-maya", "Maya")
        await _individual(pool, "ind-elsewhere")
        sid = await _walk(pool, 1, ["ind-maya", "ind-maya", "ind-ghost"])
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
        assert [(p["individual_id"], p["role"]) for p in story["people"]] == [
            ("ind-walker", "walker"), ("ind-friend", "companion")]
        # A companion's name is not the walker's: the sentence keeps "A person".
        assert story["display_sentence"].startswith("A person walked") and story["name_source"] == "rule"
        assert published[0].source_ref == "ind-friend"

    _run(body, published)
