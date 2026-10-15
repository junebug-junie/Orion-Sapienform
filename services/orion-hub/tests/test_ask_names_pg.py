"""Names typed on visual cards, through the REAL Hub answer route and real Postgres (2026-10-15).

A "same person?" card with a name stores the names on the card, closes the named people's open
"who is this?" cards in the same transaction, and a refused name rolls everything back. A
multi-picture "who is this?" card stores one verdict per picture. sql-writer's side (applying
these rows) is tested against Postgres in services/orion-sql-writer/tests/test_vision_confirm_asks.py.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server). Synthetic rows only.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import ask_routes  # noqa: E402

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL_DIR = REPO_ROOT / "services" / "orion-sql-db"
MIGRATIONS = (
    "manual_migration_walkway_camera_v1.sql",
    "manual_migration_walkway_privacy_v2.sql",
    "manual_migration_vision_confirm_asks_v1.sql",
    "manual_migration_vision_ask_names_v1.sql",
)
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")
FACES = ["thumb:" + c * 64 for c in "abc"]


def _run(body, monkeypatch):
    published: list = []

    async def _pub(event):
        published.append(event)
        return True

    monkeypatch.setattr(ask_routes, "_publish_answered", _pub)

    async def _go():
        import asyncpg
        import httpx
        from fastapi import FastAPI

        name = f"asknames_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
        try:
            for f in MIGRATIONS:
                await pool.execute((SQL_DIR / f).read_text())
            app = FastAPI()
            app.include_router(ask_routes.router)
            app.state.memory_pg_pool = pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub") as client:
                await body(pool, client, published)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await admin.close()

    asyncio.run(_go())


async def _ask(pool, ask_id: str, kind: str, ref: str, refs=None) -> None:
    refs = refs or FACES[:2]
    await pool.execute(
        "INSERT INTO orion_ask (ask_id, question, source_kind, source_ref, image_ref, image_refs) "
        "VALUES ($1, 'q', $2, $3, $4, $5::jsonb)", ask_id, kind, ref, refs[0], json.dumps(refs))


async def _card(pool, ask_id: str):
    row = await pool.fetchrow("SELECT status, answer, answer_detail::text AS d FROM orion_ask WHERE ask_id=$1", ask_id)
    return row["status"], row["answer"], (json.loads(row["d"]) if row["d"] else None)


def test_named_same_person_answer_and_its_who_is_this_close_together(monkeypatch):
    async def body(pool, client, published):
        await _ask(pool, "pair", "vision_same_individual", "aaa|bbb")
        await _ask(pool, "who-bbb", "vision_individual", "bbb", FACES[1:2])
        await _ask(pool, "who-zzz", "vision_individual", "zzz", FACES[2:3])
        r = await client.post("/api/asks/pair/answer", json={"answer": "Same person", "names": {"bbb": "Maya"}})
        assert r.status_code == 422  # free text is still refused on a pair card
        r = await client.post("/api/asks/pair/answer", json={"answer": "yes", "names": {"bbb": " Maya "}})
        assert r.status_code == 200, r.text
        assert r.json()["closed_asks"] == ["who-bbb"]
        assert await _card(pool, "pair") == ("answered", "yes", {"names": {"aaa": "Maya", "bbb": "Maya"}})
        assert await _card(pool, "who-bbb") == ("answered", "Maya", {"resolved_by": "pair"})
        assert (await _card(pool, "who-zzz"))[0] == "open"
        assert [e.ask_id for e in published] == ["pair", "who-bbb"]
        assert published[0].names == {"aaa": "Maya", "bbb": "Maya"}

    _run(body, monkeypatch)


def test_a_refused_name_rolls_back_the_whole_answer(monkeypatch):
    async def body(pool, client, published):
        await _ask(pool, "pair", "vision_same_individual", "aaa|bbb")
        await _ask(pool, "who-aaa", "vision_individual", "aaa", FACES[:1])
        r = await client.post("/api/asks/pair/answer", json={"answer": "yes", "names": {"aaa": "Theo", "bbb": "Maya"}})
        assert r.status_code == 422 and r.json()["detail"] == "same_person_takes_one_name"
        assert await _card(pool, "pair") == ("open", None, None)
        assert (await _card(pool, "who-aaa"))[0] == "open"
        assert published == []
        # Dismiss (Not sure) still works and stores nothing extra.
        assert (await client.post("/api/asks/pair/dismiss")).status_code == 200
        assert await _card(pool, "pair") == ("dismissed", None, None)

    _run(body, monkeypatch)


def test_multi_picture_who_is_this_stores_each_picture(monkeypatch):
    async def body(pool, client, published):
        await _ask(pool, "who", "vision_individual", "ind-1", FACES)
        r = await client.post("/api/asks/who/answer", json={"answer": "Theo", "pictures": [
            {"image_ref": FACES[1], "verdict": "unsure"},
            {"image_ref": FACES[2], "verdict": "different", "name": "Other kid"}]})
        assert r.status_code == 200, r.text
        status, answer, detail = await _card(pool, "who")
        assert (status, answer) == ("answered", "Theo")
        assert [(p["index"], p["verdict"], p["name"]) for p in detail["pictures"]] == [
            (0, "same", "Theo"), (1, "unsure", None), (2, "different", "Other kid")]

    _run(body, monkeypatch)
