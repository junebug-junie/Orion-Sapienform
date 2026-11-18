"""By-id reads behind orion/refs.py: GET /acts/{act_id} and GET /temporal-self/arcs/{arc_id}.

The arc read's SQL runs against Postgres in test_temporal_self_chronicle_postgres.py; here the route
contract: id shape checked before any query, 404 when absent, timestamps as ISO text.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

ACT = "act_42c0e5f55fbd5d881786ebd4"


class _Cur:
    def __init__(self, row):
        self.row = row

    async def fetchone(self):
        return self.row


class _Conn:
    def __init__(self, rows, seen):
        self.rows, self.seen = rows, seen

    async def execute(self, sql, params):
        self.seen.append((sql, params))
        return _Cur(self.rows.get(params.get("a")))


class _Pool:
    def __init__(self, rows):
        self.rows, self.seen = rows, []

    @asynccontextmanager
    async def connection(self):
        yield _Conn(self.rows, self.seen)


def test_act_by_id_reads_one_row_and_refuses_a_bad_id_before_querying(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app import main as app_main

    opened = datetime(2026, 10, 30, 9, 26, tzinfo=timezone.utc)
    pool = _Pool({ACT: {"act_id": ACT, "actor": "prime", "verb": "ask", "opened_at": opened, "outcome_value": 0.0}})
    monkeypatch.setattr(app_main, "_checkpointer_cm", pool)
    out = asyncio.run(app_main.act_by_id(ACT))
    assert out == {"act_id": ACT, "actor": "prime", "verb": "ask", "opened_at": opened.isoformat(), "outcome_value": 0.0}
    assert "FROM grammar_acts WHERE act_id = %(a)s" in pool.seen[0][0]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_main.act_by_id("act_" + "0" * 24))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_main.act_by_id("x'; drop table grammar_acts; --"))
    assert exc.value.status_code == 422 and len(pool.seen) == 2


def test_arc_by_id_refuses_a_bad_id_before_any_store_read(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app import main as app_main

    def boom():
        raise AssertionError("no store read for a malformed id")

    monkeypatch.setattr(app_main, "_chronicle_store", boom)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_main.temporal_self_arc("../frame"))
    assert exc.value.status_code == 422
