"""Postgres + Redis persistence for the self-question pool."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from orion.curiosity.self_inquiry import SelfQuestionMint
from orion.curiosity.self_question_pool import (
    PARK_SQL,
    PIN_SQL,
    SELECT_ALL_SQL,
    UPSERT_ASK_SQL,
    UPSERT_MINT_SQL,
    UPSERT_SEED_SQL,
    load_seed_questions,
    merge_seed_with_rows,
    pick_question,
)

from test_curiosity_investigation import _FakeBus, _FakePool, _graph_loop
from test_curiosity_self_inquiry import _DefinitionReader, _GrantConn


def test_merge_seed_over_db_rows_prefers_db_ask_counts() -> None:
    seed = load_seed_questions()
    rows = [
        {
            "question_id": seed[0].question_id,
            "ask_count": 3,
            "last_asked_at": "2026-09-01T00:00:00+00:00",
            "text": seed[0].text,
            "family": seed[0].family,
            "pinned": seed[0].pinned,
            "minted_by": seed[0].minted_by,
            "status": "open",
        }
    ]
    merged = merge_seed_with_rows(seed, rows)
    hit = next(q for q in merged if q.question_id == seed[0].question_id)
    assert hit.ask_count == 3


def test_sql_constants_target_curiosity_self_questions_table() -> None:
    for sql in (SELECT_ALL_SQL, UPSERT_ASK_SQL, UPSERT_MINT_SQL, UPSERT_SEED_SQL, PARK_SQL, PIN_SQL):
        assert "curiosity_self_questions" in sql
    assert "ON CONFLICT (question_id)" in UPSERT_ASK_SQL
    assert "status = 'parked'" in PARK_SQL
    assert "pinned = true" in PIN_SQL
    assert "status = 'open'" in PIN_SQL


def test_create_table_sql_matches_pool_contract() -> None:
    sql = (
        Path(__file__).resolve().parents[3]
        / "scripts/sql/2026-09-18_curiosity_self_questions.sql"
    ).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS curiosity_self_questions" in sql
    assert "ask_count integer NOT NULL DEFAULT 0" in sql
    assert "last_asked_at timestamptz" in sql


class _SelfQuestionConn(_GrantConn):
    def __init__(self, *, questions=None, **kw) -> None:
        super().__init__(**kw)
        self.questions = {q["question_id"]: dict(q) for q in (questions or [])}
        self.executed: list[tuple[str, tuple]] = []

    async def fetch(self, sql, *args):
        if "has_table_privilege" in sql:
            return await super().fetch(sql, *args)
        if "curiosity_self_questions" in sql and "SELECT" in sql.upper():
            return list(self.questions.values())
        return await super().fetch(sql, *args)

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        if "INSERT INTO curiosity_self_questions" in sql and "last_asked_at = EXCLUDED.last_asked_at" in sql:
            qid, text, family, pinned, minted_by, status, asked_at = args[:7]
            if qid in self.questions:
                row = self.questions[qid]
                row["ask_count"] = int(row.get("ask_count") or 0) + 1
                row["last_asked_at"] = asked_at
            else:
                self.questions[qid] = {
                    "question_id": qid,
                    "text": text,
                    "family": family,
                    "pinned": pinned,
                    "minted_by": minted_by,
                    "status": status,
                    "ask_count": 1,
                    "last_asked_at": asked_at,
                }
            return "INSERT 0 1"
        if "VALUES ($1, $2, $3, $4, 'orion', 'open')" in sql:
            qid, text, family, pinned = args[:4]
            if qid in self.questions:
                row = self.questions[qid]
                row["text"] = text
                row["family"] = family
                if row.get("status") != "parked":
                    row["status"] = "open"
            else:
                self.questions[qid] = {
                    "question_id": qid,
                    "text": text,
                    "family": family,
                    "pinned": pinned,
                    "minted_by": "orion",
                    "status": "open",
                    "ask_count": 0,
                    "last_asked_at": None,
                }
            return "INSERT 0 1"
        if UPSERT_SEED_SQL.split()[0] in sql and "ON CONFLICT" in sql:
            qid, text, family, pinned, minted_by = args[:5]
            self.questions.setdefault(
                qid,
                {
                    "question_id": qid,
                    "text": text,
                    "family": family,
                    "pinned": pinned,
                    "minted_by": minted_by,
                    "status": "open",
                    "ask_count": 0,
                    "last_asked_at": None,
                },
            )
            return "INSERT 0 1"
        if "UPDATE curiosity_self_questions SET status = 'parked'" in sql:
            qid = args[0]
            row = self.questions.get(qid)
            if row is None:
                return None
            row["status"] = "parked"
            return row
        if "UPDATE curiosity_self_questions SET pinned = true" in sql:
            qid = args[0]
            row = self.questions.get(qid)
            if row is None:
                return None
            row["pinned"] = True
            row["status"] = "open"
            return row
        return "OK"

    async def fetchrow(self, sql, *args):
        if "UPDATE curiosity_self_questions SET status = 'parked'" in sql:
            result = await self.execute(sql, *args)
            if result is None:
                return None
            return result
        if "UPDATE curiosity_self_questions SET pinned = true" in sql:
            result = await self.execute(sql, *args)
            if result is None:
                return None
            return result
        return None


def test_self_inquiry_tick_records_ask_and_recent_family() -> None:
    from scripts.curiosity_investigation import _SELF_RECENT_FAMILIES_KEY

    bus = _FakeBus()
    conn = _SelfQuestionConn()
    loop = _graph_loop(
        bus,
        reader=_DefinitionReader(),
        conn=conn,
        kickoff_via_cortex=False,
        self_inquiry_enabled=True,
        self_inquiry_min_cooldown_sec=0.0,
    )
    assert asyncio.run(loop.tick_self_inquiry()) is None
    assert any("ON CONFLICT (question_id)" in sql for sql, _ in conn.executed)
    assert _SELF_RECENT_FAMILIES_KEY in bus.redis.lists
    assert len(bus.redis.lists[_SELF_RECENT_FAMILIES_KEY]) == 1
    assert loop._last_self_question is not None
    assert loop._last_self_question.family in {"lived", "anatomy"}


def test_record_ask_upserts_missing_row() -> None:
    """First ask creates the row when seed ensure did not run."""
    bus = _FakeBus()
    conn = _SelfQuestionConn()
    loop = _graph_loop(
        bus,
        reader=_DefinitionReader(),
        conn=conn,
        kickoff_via_cortex=False,
        self_inquiry_enabled=True,
    )
    picked = load_seed_questions()[0]
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    assert picked.question_id not in conn.questions
    asyncio.run(loop._record_self_question_ask(picked, now=now))
    row = conn.questions[picked.question_id]
    assert row["ask_count"] == 1
    assert row["last_asked_at"] == now
    assert row["text"] == picked.text
    assert row["family"] == picked.family


class _MintReader:
    def __init__(self, mints):
        self.mints = mints

    def query(self, _cypher):
        return [
            {
                "run_id": m.run_id,
                "question_id": m.question_id,
                "text": m.text,
                "family": m.family,
                "written_at": m.written_at,
            }
            for m in self.mints
        ]


def test_mint_upsert_is_idempotent_and_preserves_ask_count() -> None:
    bus = _FakeBus()
    conn = _SelfQuestionConn(
        questions=[
            {
                "question_id": "lived.orion.continuity",
                "text": "Old text",
                "family": "lived",
                "pinned": False,
                "minted_by": "orion",
                "status": "open",
                "ask_count": 2,
                "last_asked_at": datetime(2026, 9, 10, tzinfo=timezone.utc),
            }
        ]
    )
    mint = SelfQuestionMint(
        run_id="abc123",
        question_id="lived.orion.continuity",
        text="What do I notice about continuity?",
        family="lived",
    )
    loop = _graph_loop(
        bus,
        reader=_MintReader([mint]),
        conn=conn,
        kickoff_via_cortex=False,
        self_inquiry_enabled=True,
    )
    assert asyncio.run(loop._upsert_orion_minted_questions("abc123")) == 1
    row = conn.questions["lived.orion.continuity"]
    assert row["ask_count"] == 2
    assert row["text"] == mint.text
    assert len(conn.questions) == 1
    assert asyncio.run(loop._upsert_orion_minted_questions("abc123")) == 1
    assert conn.questions["lived.orion.continuity"]["ask_count"] == 2


async def _park_question(conn: _SelfQuestionConn, question_id: str) -> bool:
    row = await conn.fetchrow(PARK_SQL, question_id)
    return row is not None


def test_parked_row_is_excluded_from_pick_after_operator_park() -> None:
    conn = _SelfQuestionConn(
        questions=[
            {
                "question_id": "lived.who_matters",
                "text": "Who matters?",
                "family": "lived",
                "pinned": True,
                "minted_by": "juniper",
                "status": "open",
                "ask_count": 0,
                "last_asked_at": None,
            },
            {
                "question_id": "anatomy.made_of",
                "text": "What am I made of?",
                "family": "anatomy",
                "pinned": True,
                "minted_by": "juniper",
                "status": "open",
                "ask_count": 0,
                "last_asked_at": None,
            },
        ]
    )
    assert asyncio.run(_park_question(conn, "lived.who_matters"))
    pool = merge_seed_with_rows([], list(conn.questions.values()))
    picked = pick_question(pool=pool, recent_families=[], now=datetime(2026, 9, 18, tzinfo=timezone.utc))
    assert picked.question_id == "anatomy.made_of"


def test_remint_upsert_preserves_parked_status_and_pick_excludes() -> None:
    bus = _FakeBus()
    conn = _SelfQuestionConn(
        questions=[
            {
                "question_id": "lived.orion.continuity",
                "text": "Old text",
                "family": "lived",
                "pinned": False,
                "minted_by": "orion",
                "status": "open",
                "ask_count": 1,
                "last_asked_at": datetime(2026, 9, 10, tzinfo=timezone.utc),
            },
            {
                "question_id": "anatomy.made_of",
                "text": "What am I made of?",
                "family": "anatomy",
                "pinned": True,
                "minted_by": "juniper",
                "status": "open",
                "ask_count": 0,
                "last_asked_at": None,
            },
        ]
    )
    assert asyncio.run(_park_question(conn, "lived.orion.continuity"))
    mint = SelfQuestionMint(
        run_id="abc123",
        question_id="lived.orion.continuity",
        text="What do I notice about continuity?",
        family="lived",
    )
    loop = _graph_loop(
        bus,
        reader=_MintReader([mint]),
        conn=conn,
        kickoff_via_cortex=False,
        self_inquiry_enabled=True,
    )
    assert asyncio.run(loop._upsert_orion_minted_questions("abc123")) == 1
    row = conn.questions["lived.orion.continuity"]
    assert row["status"] == "parked"
    assert row["text"] == mint.text
    assert row["ask_count"] == 1
    pool = merge_seed_with_rows([], list(conn.questions.values()))
    picked = pick_question(
        pool=pool,
        recent_families=[],
        now=datetime(2026, 9, 18, tzinfo=timezone.utc),
    )
    assert picked.question_id == "anatomy.made_of"


def test_merge_includes_orion_minted_rows_not_in_seed() -> None:
    seed = load_seed_questions()
    rows = [
        {
            "question_id": "lived.orion_mint",
            "text": "What do I notice about continuity?",
            "family": "lived",
            "pinned": False,
            "minted_by": "orion",
            "status": "open",
            "ask_count": 1,
            "last_asked_at": datetime(2026, 9, 10, tzinfo=timezone.utc),
        }
    ]
    merged = merge_seed_with_rows(seed, rows)
    assert any(q.question_id == "lived.orion_mint" for q in merged)
