"""Postgres + Redis persistence for the self-question pool."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from orion.curiosity.self_question_pool import (
    SELECT_ALL_SQL,
    UPSERT_ASK_SQL,
    UPSERT_MINT_SQL,
    UPSERT_SEED_SQL,
    load_seed_questions,
    merge_seed_with_rows,
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
    for sql in (SELECT_ALL_SQL, UPSERT_ASK_SQL, UPSERT_MINT_SQL, UPSERT_SEED_SQL):
        assert "curiosity_self_questions" in sql


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
        if "ask_count = ask_count + 1" in sql:
            qid, asked_at = args[:2]
            row = self.questions.setdefault(
                qid,
                {
                    "question_id": qid,
                    "text": "",
                    "family": "lived",
                    "pinned": False,
                    "minted_by": "juniper",
                    "status": "open",
                    "ask_count": 0,
                    "last_asked_at": None,
                },
            )
            row["ask_count"] = int(row.get("ask_count") or 0) + 1
            row["last_asked_at"] = asked_at
            return "UPDATE 1"
        return "OK"


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
    assert any("ask_count = ask_count + 1" in sql for sql, _ in conn.executed)
    assert _SELF_RECENT_FAMILIES_KEY in bus.redis.lists
    assert len(bus.redis.lists[_SELF_RECENT_FAMILIES_KEY]) == 1
    assert loop._last_self_question is not None
    assert loop._last_self_question.family in {"lived", "anatomy"}


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
