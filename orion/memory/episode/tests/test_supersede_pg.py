"""Supersede writes against a disposable Postgres (issue #3587). Synthetic rows only.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server); CI does, and fails on
a skip. Proves, with the real migration and the real reader SQL:

- a distilled correction supersedes its target at write time: all three marks (status, pointer,
  one ``superseded`` event) in the persist transaction;
- retrieval never serves the superseded row (``orion.memory.lived_memory.fetch_live_records``, the
  recall gate, both the full read and the id re-check), and the core-page fold input drops it;
- re-applying a handled pair is a no-op with no second event (idempotent pass);
- undo restores the row and pointer, and an undone pair is never re-applied;
- Juniper-confirmed rows and Orion's views are never superseded.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.memory.episode import supersede as s
from orion.memory.episode.store import persist_episode
from orion.memory.episode.tests.test_validate import TURNS, _mem
from orion.memory.episode.validate import validate_distillation
from orion.schemas.memory_episode import EpisodeDistillationV1

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL_DIR = Path(__file__).resolve().parents[4] / "services" / "orion-sql-db"
MIGRATION = SQL_DIR / "manual_migration_episode_memory_v1.sql"
TEMPORAL = SQL_DIR / "manual_migration_episode_memory_temporal_v1.sql"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


async def _with_db(fn):
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    name = f"supersede_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
        await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=2, open=False,
                               kwargs={"autocommit": True, "row_factory": dict_row})
    await pool.open()
    try:
        async with pool.connection() as conn:
            await conn.execute(MIGRATION.read_text())
            await conn.execute(TEMPORAL.read_text())
        await fn(pool, dsn)
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


async def _old_row(pool, statement, *, purpose="about_juniper", state="auto"):
    mid = str(uuid.uuid4())
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO episode_memory (memory_id, episode_id, purpose, voice, channel, statement, stakes, "
            "confirmation_state, strength, last_reinforced_at, status, created_at, updated_at) "
            "VALUES (%s, 'ep-old', %s, 'juniper_said', 'chat', %s, 'low', %s, 0.5, %s, 'active', %s, %s)",
            (mid, purpose, statement, state, T0, T0, T0))
    return mid


def _correction():
    d = EpisodeDistillationV1.model_validate({"memories": [_mem()], "questions": []})
    result = validate_distillation(d, TURNS, episode_id="ep-new")
    assert result.memories, "fixture memory must validate"
    return result


async def _persist(pool, result, links):
    return await persist_episode(
        pool, episode_id="ep-new", run_id="memdistill-ep-new", result=result, model_route="memory_distill",
        model="m", prompt_version="v", usage={}, llm_latency_ms=1, hold_wait_ms=1, coverage=1.0,
        now=T0 + timedelta(days=3), supersede_links=links)


async def _one(pool, sql, params=()):
    async with pool.connection() as conn:
        return await (await conn.execute(sql, params)).fetchone()


async def _served_ids(dsn, ids=None):
    import asyncpg

    from orion.memory.lived_memory import KIND_EPISODE, fetch_live_records

    conn = await asyncpg.connect(dsn)
    try:
        by_kind = None if ids is None else {KIND_EPISODE: list(ids)}
        return {r.record_id for r in await fetch_live_records(conn, kinds=[KIND_EPISODE], ids_by_kind=by_kind)}
    finally:
        await conn.close()


async def _fold_ids(pool):
    from orion.memory.pages.juniper import EPISODE_MEMORY_SQL

    async with pool.connection() as conn:
        return {r["memory_id"] for r in await (await conn.execute(EPISODE_MEMORY_SQL)).fetchall()}


def test_write_time_supersede_sets_all_three_marks_and_readers_drop_the_old_row():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper told me her pet Biscuit is a cat.")
        assert old_id in await _served_ids(dsn) and old_id in await _fold_ids(pool)
        result = _correction()
        new_id = result.memories[0].memory_id
        link = s.Link(old_id=old_id, new_id=new_id, reason="B says the pet is a dog.", dense=0.7,
                      shared=("pet:biscuit",)).as_dict()
        counts = await _persist(pool, result, [link])
        assert counts["superseded"] == 1

        old = await _one(pool, "SELECT status, confirmation_state FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert (old["status"], old["confirmation_state"]) == ("superseded", "corrected")
        new = await _one(pool, "SELECT supersedes_memory_id::text AS p, status FROM episode_memory WHERE memory_id = %s",
                         (new_id,))
        assert (new["p"], new["status"]) == (old_id, "active")
        ev = await _one(pool, "SELECT event_id::text AS id, actor, reason, evidence FROM episode_memory_event "
                              "WHERE op = 'superseded' AND memory_id = %s", (old_id,))
        assert ev["id"] == s.superseded_event_id(old_id, new_id) and ev["actor"] == s.ACTOR_DISTILL
        assert ev["evidence"]["superseded_by"] == new_id and ev["evidence"]["prior_confirmation_state"] == "auto"

        # Retrieval never serves it, by full read or by the id re-check; the fold input drops it.
        assert old_id not in await _served_ids(dsn) and new_id in await _served_ids(dsn)
        assert await _served_ids(dsn, [old_id]) == set()
        assert old_id not in await _fold_ids(pool)
        # The pass's candidate corpus no longer holds it, so a second pass proposes nothing for it.
        async with pool.connection() as conn:
            assert old_id not in {r.memory_id for r in (await s.load_corpus(conn)).rows}

        # Idempotent: replaying the persist, or applying the pair again, writes nothing.
        assert (await _persist(pool, result, [link]))["superseded"] == 0
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link.from_dict(link), actor=s.ACTOR_PASS) is None
        n = await _one(pool, "SELECT count(*) AS n FROM episode_memory_event WHERE op = 'superseded'")
        assert n["n"] == 1

        # Undo restores the pair; the restored row is served again; the pair is never re-applied.
        async with pool.connection() as conn:
            async with conn.transaction():
                restored = await s.undo_link(conn, old_id=old_id, new_id=new_id, actor="operator", note="test")
        assert restored["restored_confirmation_state"] == "auto"
        old = await _one(pool, "SELECT status, confirmation_state FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert (old["status"], old["confirmation_state"]) == ("active", "auto")
        new = await _one(pool, "SELECT supersedes_memory_id FROM episode_memory WHERE memory_id = %s", (new_id,))
        assert new["supersedes_memory_id"] is None
        assert old_id in await _served_ids(dsn)
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link.from_dict(link), actor=s.ACTOR_PASS) is None
                assert await s.undo_link(conn, old_id=old_id, new_id=new_id, actor="operator") is None

    asyncio.run(_with_db(body))


def test_undo_never_reverses_a_later_retirement_and_keeps_a_pointer_it_did_not_set():
    async def body(pool, dsn):
        a = await _old_row(pool, "Juniper told me Biscuit is a cat.")
        result = _correction()
        b = result.memories[0].memory_id
        await _persist(pool, result, [s.Link(a, b, "r").as_dict()])
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.undo_link(conn, old_id=a, new_id=b, actor="operator") is not None
            # A is retired again by another correction C (not this pair).
            c = await _old_row(pool, "Juniper told me Biscuit is a dog.")
            await conn.execute("UPDATE episode_memory SET created_at = %s, episode_id = 'ep-c' WHERE memory_id = %s",
                               (T0 + timedelta(days=9), c))
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link(a, c, "r2"), actor=s.ACTOR_PASS,
                                          now=T0 + timedelta(days=10)) is not None
            async with conn.transaction():
                assert await s.undo_link(conn, old_id=a, new_id=b, actor="operator") is None  # not B's to undo
        row = await _one(pool, "SELECT status FROM episode_memory WHERE memory_id = %s", (a,))
        assert row["status"] == "superseded"

        # A pointer this link did not set stays when the link is undone.
        x = await _old_row(pool, "Juniper said the cabinet was cold.")
        y = await _old_row(pool, "Juniper said the cabinet was warm.")
        async with pool.connection() as conn:
            await conn.execute("UPDATE episode_memory SET episode_id = 'ep-y', created_at = %s, "
                               "supersedes_memory_id = %s WHERE memory_id = %s", (T0 + timedelta(days=2), c, y))
            async with conn.transaction():
                ev = await s.apply_link(conn, s.Link(x, y, "r"), actor=s.ACTOR_PASS, now=T0 + timedelta(days=11))
            assert ev["pointer_set"] is False
            async with conn.transaction():
                assert await s.undo_link(conn, old_id=x, new_id=y, actor="operator") is not None
        ptr = await _one(pool, "SELECT supersedes_memory_id::text AS p FROM episode_memory WHERE memory_id = %s", (y,))
        assert ptr["p"] == c

    asyncio.run(_with_db(body))


def test_open_asks_and_pending_corrections_are_never_applied():
    async def body(pool, dsn):
        asked = await _old_row(pool, "Juniper told me she moved.", state="pending_confirmation")
        plain = await _old_row(pool, "Juniper told me Biscuit is a cat.")
        pending_fix = await _old_row(pool, "Juniper told me Biscuit is a dog.", state="pending_confirmation")
        async with pool.connection() as conn:
            await conn.execute("UPDATE episode_memory SET episode_id = 'ep-fix' WHERE memory_id = %s", (pending_fix,))
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link(asked, plain, "r"), actor=s.ACTOR_PASS) is None
                assert await s.apply_link(conn, s.Link(plain, pending_fix, "r"), actor=s.ACTOR_PASS) is None

    asyncio.run(_with_db(body))


def test_confirmed_rows_and_views_are_never_superseded_and_foreign_links_are_ignored():
    async def body(pool, dsn):
        confirmed = await _old_row(pool, "Juniper told me she lives in Millbrook.", state="confirmed")
        view = await _old_row(pool, "I think Juniper prefers mornings.", purpose="orion_view")
        other = await _old_row(pool, "Juniper said the office was loud.")
        result = _correction()
        new_id = result.memories[0].memory_id
        links = [s.Link(confirmed, new_id, "r").as_dict(), s.Link(view, new_id, "r").as_dict(),
                 s.Link(other, "not-this-episode", "r").as_dict()]
        assert (await _persist(pool, result, links))["superseded"] == 0
        for mid in (confirmed, view, other):
            row = await _one(pool, "SELECT status FROM episode_memory WHERE memory_id = %s", (mid,))
            assert row["status"] == "active"
        n = await _one(pool, "SELECT count(*) AS n FROM episode_memory_event WHERE op = 'superseded'")
        assert n["n"] == 0

    asyncio.run(_with_db(body))


# --- bi-temporal validity, run audit, eval gate (issues #3659, #3660) --------------------------

async def _valid_at_ids(dsn, as_of):
    import asyncpg

    from orion.memory.lived_memory import fetch_valid_at

    conn = await asyncpg.connect(dsn)
    try:
        return {r.record_id for r in await fetch_valid_at(conn, as_of)}
    finally:
        await conn.close()


async def _valid_during_ids(dsn, start, end, **kw):
    import asyncpg

    from orion.memory.lived_memory import fetch_valid_during

    conn = await asyncpg.connect(dsn)
    try:
        return {r.record_id: r for r in await fetch_valid_during(conn, start, end, **kw)}
    finally:
        await conn.close()


async def _set_occurred(pool, mid, when):
    async with pool.connection() as conn:
        await conn.execute("UPDATE episode_memory SET occurred_at = %s WHERE memory_id = %s", (when, mid))


def test_an_update_ends_validity_at_the_new_time_and_stays_reachable_as_history():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper lives in Fort Collins.")
        await _set_occurred(pool, old_id, T0)
        result = _correction()
        new_id = result.memories[0].memory_id
        link = s.Link(old_id=old_id, new_id=new_id, reason="moved", dense=0.7, relation="update",
                      votes=("update",) * 3).as_dict()
        assert (await _persist(pool, result, [link]))["superseded"] == 1
        await _set_occurred(pool, new_id, T0 + timedelta(days=2))
        old = await _one(pool, "SELECT status, confirmation_state, invalid_at, invalidated_by::text AS by, "
                               "invalid_reason FROM episode_memory WHERE memory_id = %s", (old_id,))
        new_time = (await _one(pool, "SELECT created_at FROM episode_memory WHERE memory_id = %s", (new_id,)))
        assert old["status"] == "superseded" and old["confirmation_state"] == "auto"   # right at its time
        assert old["invalid_reason"] == "update" and old["by"] == new_id
        assert old["invalid_at"] == new_time["created_at"]  # the new row's time (no occurred_at at apply)
        ev = await _one(pool, "SELECT evidence FROM episode_memory_event WHERE op = 'superseded' AND memory_id = %s",
                        (old_id,))
        assert ev["evidence"]["relation"] == "update" and ev["evidence"]["votes"] == ["update"] * 3

        # Default recall: currently true only. Past-tense read: true then.
        assert old_id not in await _served_ids(dsn)
        assert old_id in await _valid_at_ids(dsn, T0 + timedelta(days=1))
        assert old_id not in await _valid_at_ids(dsn, T0 + timedelta(days=10))
        assert old_id not in await _valid_at_ids(dsn, T0 - timedelta(days=1))   # not yet happened

        # Issue #3674: the same read over a window ("where was I in early October?").
        during = await _valid_during_ids(dsn, T0 + timedelta(days=1), T0 + timedelta(days=3), retired_only=True,
                                         limit=5)
        assert old_id in during and during[old_id].invalid_at is not None   # true then, labelled not now
        assert new_id not in during                                           # still true: not "retired"
        assert old_id not in await _valid_during_ids(dsn, T0 - timedelta(days=5), T0 - timedelta(days=1))
        assert set(await _valid_during_ids(dsn, T0, T0 + timedelta(days=3), ids=[old_id])) == {old_id}

        # Undo clears the validity end.
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.undo_link(conn, old_id=old_id, new_id=new_id, actor="operator")
        old = await _one(pool, "SELECT status, confirmation_state, invalid_at, invalidated_by, invalid_reason "
                               "FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert (old["status"], old["confirmation_state"], old["invalid_at"], old["invalidated_by"],
                old["invalid_reason"]) == ("active", "auto", None, None, None)

        # An expired row is not served as history either.
        async with pool.connection() as conn:
            await conn.execute("UPDATE episode_memory SET expires_at = %s WHERE memory_id = %s",
                               (T0 + timedelta(hours=12), old_id))
        assert old_id not in await _valid_at_ids(dsn, T0 + timedelta(days=1))

    asyncio.run(_with_db(body))


def test_a_correction_was_never_true_and_is_never_served_as_history():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper's cat Biscuit is grey.")
        await _set_occurred(pool, old_id, T0)
        result = _correction()
        link = s.Link(old_id=old_id, new_id=result.memories[0].memory_id, reason="wrong species").as_dict()
        assert (await _persist(pool, result, [link]))["superseded"] == 1
        old = await _one(pool, "SELECT confirmation_state, invalid_at, invalid_reason FROM episode_memory "
                               "WHERE memory_id = %s", (old_id,))
        assert (old["confirmation_state"], old["invalid_reason"], old["invalid_at"]) == ("corrected", "correction", T0)
        for days in (0, 1, 10):
            assert old_id not in await _valid_at_ids(dsn, T0 + timedelta(days=days))
        assert old_id not in await _valid_during_ids(dsn, T0 - timedelta(days=1), T0 + timedelta(days=10))

    asyncio.run(_with_db(body))


def test_every_distill_run_leaves_one_append_only_audit_row_without_text():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper's cat Biscuit is grey.")
        result = _correction()
        new_id = result.memories[0].memory_id
        link = s.Link(old_id=old_id, new_id=new_id, reason="wrong species").as_dict()
        plan = {"links": [link], "held_links": [], "apply": "on", "refused_because": [], "asked": 1,
                "check_errors": 0, "dense": "on", "check_version": s.CHECK_VERSION,
                "decisions": [{"old_id": old_id, "new_id": new_id, "legs": {"cosine": 0.7}, "relation": "correction",
                               "votes": [{"relation": "correction", "confidence": 0.9}], "linked": True}],
                "abstained": {"vote:compatible": 1}}
        for _ in range(2):  # a replayed persist writes no second row
            await persist_episode(
                pool, episode_id="ep-new", run_id="memdistill-ep-new", result=result, model_route="memory_distill",
                model="m", prompt_version="v", usage={}, llm_latency_ms=1, hold_wait_ms=1, coverage=1.0,
                now=T0 + timedelta(days=3), supersede_links=[link], supersede_plan=plan)
        rows = await _one(pool, "SELECT count(*) AS n FROM memory_supersede_run")
        r = await _one(pool, "SELECT source, apply, dense, counts, decisions, applied FROM memory_supersede_run")
        assert rows["n"] == 1 and r["source"] == "distill" and r["apply"] == "on" and r["dense"] == "on"
        assert r["counts"]["applied"] == 1 and r["applied"][0]["relation"] == "correction"
        assert r["counts"]["abstained"] == {"vote:compatible": 1}
        assert r["decisions"][0]["legs"]["cosine"] == 0.7
        assert "Biscuit" not in str(r)

    asyncio.run(_with_db(body))


def test_a_missing_audit_table_never_costs_the_memories():
    async def body(pool, dsn):
        async with pool.connection() as conn:
            await conn.execute("DROP TABLE memory_supersede_run")
        result = _correction()
        counts = await persist_episode(
            pool, episode_id="ep-new", run_id="memdistill-ep-new", result=result, model_route="memory_distill",
            model="m", prompt_version="v", usage={}, llm_latency_ms=1, hold_wait_ms=1, coverage=1.0,
            now=T0 + timedelta(days=3), supersede_plan={"links": [], "apply": "refused",
                                                         "refused_because": ["eval_gate:eval_missing"]})
        assert counts["memories"] == 1

    asyncio.run(_with_db(body))


def test_eval_gate_reads_the_latest_finished_run_for_this_check_version():
    async def body(pool, dsn):
        async def gate():
            async with pool.connection() as conn:
                return await s.eval_gate(conn)

        now = datetime.now(timezone.utc)

        async def add(run_id, passed, finished, version=s.CHECK_VERSION, votes=s.VOTES, route=s.EVAL_ROUTE):
            async with pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO memory_supersede_eval_run (run_id, check_version, started_at, finished_at, passed, "
                    "n_items, n_real, false_acts, hard_negative_false_acts, errors, votes, route) "
                    "VALUES (%s, %s, %s, %s, %s, 1, 1, 0, 0, 0, %s, %s)",
                    (run_id, version, now, finished, passed, votes, route))

        assert await gate() == (False, "eval_missing")
        await add("old", True, now - timedelta(days=s.EVAL_MAX_AGE_DAYS + 2))
        assert await gate() == (False, "eval_stale")
        await add("a", True, now - timedelta(hours=5))
        assert await gate() == (True, "eval_passed")
        await add("b", False, None)                                   # still running: ignored
        await add("c", False, now - timedelta(hours=1), version="supersede_check.v2")   # other version
        await add("v", False, now - timedelta(hours=1), votes=5)     # other vote count
        await add("r", False, now - timedelta(hours=1), route="agent")   # other route
        assert await gate() == (True, "eval_passed")
        await add("d", False, now - timedelta(hours=2))
        assert await gate() == (False, "eval_failed")

    asyncio.run(_with_db(body))


def test_pass_apply_refuses_while_the_eval_gate_is_closed(tmp_path):
    import argparse
    import importlib.util
    import json

    script = Path(__file__).resolve().parents[4] / "scripts" / "memory_supersede_pass.py"
    spec = importlib.util.spec_from_file_location("memory_supersede_pass_pg", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    async def body(pool, dsn):
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text(json.dumps({"old_id": str(uuid.uuid4()), "new_id": str(uuid.uuid4()), "reason": "r",
                                     "dense": 0.7, "check_version": s.CHECK_VERSION}) + "\n")
        args = argparse.Namespace(yes=True, pairs=str(pairs), allow_no_dense=False, allow_no_eval=False, only="",
                                  exclude="", dsn=dsn, no_manifest=True, watch=[])
        with pytest.raises(SystemExit, match="eval gate closed"):
            await mod.cmd_apply(args)

    asyncio.run(_with_db(body))


def test_an_update_that_points_backwards_is_refused():
    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper lives in Riverton.")
        await _set_occurred(pool, old_id, T0 + timedelta(days=10))
        result = _correction()
        new_id = result.memories[0].memory_id
        await _persist(pool, result, None)
        await _set_occurred(pool, new_id, T0)                       # newer row about an earlier time
        async with pool.connection() as conn:
            async with conn.transaction():
                got = await s.apply_link(conn, s.Link(old_id, new_id, "r", relation="update"), actor=s.ACTOR_PASS)
        assert got is None
        old = await _one(pool, "SELECT status FROM episode_memory WHERE memory_id = %s", (old_id,))
        assert old["status"] == "active"

    asyncio.run(_with_db(body))


def test_pass_undo_stores_the_note_and_rewrites_the_manifest(tmp_path):
    """Orion's questions on PR #3734: the undo's note lands in the event, and the manifest stops
    saying the pair is applied."""
    import argparse
    import importlib.util
    import json

    script = Path(__file__).resolve().parents[4] / "scripts" / "memory_supersede_pass.py"
    spec = importlib.util.spec_from_file_location("memory_supersede_pass_pg_undo", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    async def body(pool, dsn):
        old_id = await _old_row(pool, "Juniper told me her pet Biscuit is a cat.")
        result = _correction()
        new_id = result.memories[0].memory_id
        link = s.Link(old_id=old_id, new_id=new_id, reason="r", dense=0.7, shared=("pet:biscuit",)).as_dict()
        assert (await _persist(pool, result, [link]))["superseded"] == 1
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({"check_version": "supersede_classify.v3.1", "last_mode": "apply",
                                        "proposed": [{"old_id": old_id, "new_id": new_id}],
                                        "applied": [{"old_id": old_id, "new_id": new_id, "applied_at": "t0"}]}))
        args = argparse.Namespace(dsn=dsn, old=old_id, new=new_id, actor="juniper", note="  a different pet  ",
                                  manifest=str(manifest), no_manifest=False, watch=[])
        assert await mod.cmd_undo(args) == 0
        ev = await _one(pool, "SELECT actor, evidence FROM episode_memory_event WHERE op = 'supersede_undone' "
                              "AND memory_id = %s", (old_id,))
        assert ev["actor"] == "juniper" and ev["evidence"]["note"] == "a different pet"
        m = json.loads(manifest.read_text())
        assert m["last_mode"] == "undo" and m["check_version"] == "supersede_classify.v3.1"
        assert m["in_effect"] == [] and m["applied"][0]["undone_at"]
        # A refresh rebuilds in_effect from the database, inside a read-only transaction.
        link2_old = await _old_row(pool, "Juniper told me the shed is blue.")
        async with pool.connection() as conn:
            async with conn.transaction():
                assert await s.apply_link(conn, s.Link(old_id=link2_old, new_id=new_id, reason="r", dense=0.7),
                                          actor=s.ACTOR_PASS)
        seen = {}
        real_write = mod.write_manifest

        async def spy(conn, *a, **k):
            seen["ro"] = (await (await conn.execute("SHOW transaction_read_only")).fetchone())["transaction_read_only"]
            return await real_write(conn, *a, **k)

        mod.write_manifest = spy
        manifest.write_text("not json")
        assert await mod.cmd_manifest(argparse.Namespace(dsn=dsn, manifest=str(manifest), watch=[])) == 0
        mod.write_manifest = real_write
        assert seen["ro"] == "on"
        m = json.loads(manifest.read_text())
        assert m["last_mode"] == "refresh" and [(p["old_id"], p["new_id"]) for p in m["in_effect"]] == [(link2_old, new_id)]
        assert m["in_effect"][0]["actor"] == s.ACTOR_PASS and len(m["undone"]) == 1
        assert m["undone"] == [{"old_id": old_id, "new_id": new_id, "actor": "juniper",
                                "undone_at": m["undone"][0]["undone_at"], "note": "a different pet"}]
        # A second undo of the same pair finds nothing and leaves the manifest alone.
        written = manifest.read_text()
        assert await mod.cmd_undo(args) == 1
        assert manifest.read_text() == written

    asyncio.run(_with_db(body))
