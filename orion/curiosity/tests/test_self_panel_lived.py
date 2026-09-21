"""Self panel lived-answer ledger read (orion/curiosity/self_panel.py)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.curiosity.self_panel import read_self_panel, to_payload


class _FakeConn:
    def __init__(self, *, history=None, lived=None, journals=None, eval_run_id=None, eval_rows=None):
        self.history = history if history is not None else []
        self.lived = lived if lived is not None else []
        self.journals = journals if journals is not None else []
        self.eval_run_id = eval_run_id
        self.eval_rows = eval_rows if eval_rows is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if "concept_id = $1" in sql:
            return self.history
        if "concept_id LIKE 'self:lived:%'" in sql:
            assert "ORDER BY created_at DESC" in sql
            return self.lived
        if "journal_entries" in sql:
            return self.journals
        if "self_sense_eval_log" in sql and "WHERE run_id" in sql:
            return self.eval_rows
        raise AssertionError(f"unexpected fetch: {sql}")

    async def fetchrow(self, sql, *args):
        if "self_sense_eval_log" in sql:
            return {"run_id": self.eval_run_id} if self.eval_run_id else None
        raise AssertionError(f"unexpected fetchrow: {sql}")


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


def _lived_row(question_id: str, content: str, *, version: int = 1, created_at=None):
    return {
        "concept_id": f"self:lived:{question_id}",
        "version": version,
        "created_at": created_at or datetime(2026, 9, 18, tzinfo=timezone.utc),
        "content": content,
        "evidence_refs": ["chat_message:1"],
        "produced_by": "curiosity_self_inquiry",
    }


def test_lived_answers_collapse_to_latest_per_concept_id() -> None:
    conn = _FakeConn(
        lived=[
            _lived_row("lived.who_matters", "Juniper matters most.", version=2),
            _lived_row("lived.who_matters", "stale answer", version=1,
                       created_at=datetime(2026, 9, 1, tzinfo=timezone.utc)),
            _lived_row("lived.care", "I care about continuity."),
        ]
    )
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert len(view.lived_answers) == 2
    by_id = {la.question_id: la for la in view.lived_answers}
    assert by_id["lived.who_matters"].content == "Juniper matters most."
    assert by_id["lived.who_matters"].version == 2
    assert by_id["lived.care"].content == "I care about continuity."


def test_to_payload_includes_lived_answers() -> None:
    conn = _FakeConn(lived=[_lived_row("lived.who_matters", "Juniper matters most.")])
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    payload = to_payload(view)
    assert payload["available"] is True
    assert len(payload["lived_answers"]) == 1
    assert payload["lived_answers"][0]["question_id"] == "lived.who_matters"
    assert payload["lived_answers"][0]["content"] == "Juniper matters most."
