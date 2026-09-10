"""orion/curiosity/self_panel.py -- the Postgres-only reader behind the Hub's
"Self" panel. Pins: an unreadable pool is `unavailable_reason`, never an empty
result (same rule every reader in this package follows); the current
definition is always `history[0]`; `evidence_refs` tolerates the JSON-string
shape a raw-SQL read can hand back; a store with no rows yet is a real,
distinct state from a broken one.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from orion.curiosity.self_panel import (
    SELF_DEFINITION_CONCEPT_ID,
    SELF_INQUIRY_JOURNAL_TITLE,
    read_self_panel,
    to_payload,
)


class _FakeConn:
    """Matches asyncpg's surface for the three queries this module runs."""

    def __init__(self, *, history=None, journals=None, eval_run_id=None, eval_rows=None, raises=False):
        self.history = history if history is not None else []
        self.journals = journals if journals is not None else []
        self.eval_run_id = eval_run_id
        self.eval_rows = eval_rows if eval_rows is not None else []
        self.raises = raises
        self.fetch_calls: list[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        self.fetch_calls.append((sql, args))
        if self.raises:
            raise RuntimeError("connection refused")
        if "self_concept_history" in sql:
            assert args[0] == SELF_DEFINITION_CONCEPT_ID
            assert "ORDER BY created_at DESC" in sql, "current must be created_at, not version -- see review finding"
            return self.history
        if "journal_entries" in sql:
            assert args[0] == SELF_INQUIRY_JOURNAL_TITLE
            return self.journals
        if "self_sense_eval_log" in sql and "WHERE run_id" in sql:
            return self.eval_rows
        raise AssertionError(f"unexpected fetch: {sql}")

    async def fetchrow(self, sql, *args):
        if self.raises:
            raise RuntimeError("connection refused")
        if "self_sense_eval_log" in sql:
            return {"run_id": self.eval_run_id} if self.eval_run_id else None
        raise AssertionError(f"unexpected fetchrow: {sql}")


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return self._conn


def _row(version: int, evidence, created_at=None):
    return {
        "version": version,
        "created_at": created_at or datetime(2026, 9, 8, tzinfo=timezone.utc),
        "content": f"definition v{version}",
        "evidence_refs": evidence,
        "produced_by": "curiosity_self_inquiry",
    }


def test_no_pool_is_unavailable_not_empty() -> None:
    view = asyncio.run(read_self_panel(None))
    assert view.is_unavailable
    assert view.unavailable_reason == "no_pool"
    assert view.current is None
    payload = to_payload(view)
    assert payload == {"available": False, "reason": "no_pool"}


def test_a_failed_query_is_unavailable_not_a_crash() -> None:
    conn = _FakeConn(raises=True)
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.is_unavailable
    assert "RuntimeError" in view.unavailable_reason
    assert to_payload(view)["available"] is False


def test_no_rows_yet_is_available_with_no_current_definition() -> None:
    """A store with zero rows is a real state -- 'not written yet' -- and must
    be distinguishable from a broken read (previous test)."""
    view = asyncio.run(read_self_panel(_FakePool(_FakeConn())))
    assert not view.is_unavailable
    assert view.current is None
    assert view.history == []
    payload = to_payload(view)
    assert payload["available"] is True
    assert payload["current"] is None


def test_current_is_the_query_order_not_a_re_derived_max_version() -> None:
    """`current` trusts the SQL's ORDER BY rather than re-deriving max(version)
    in Python, so the two cannot disagree. The row order here is deliberately
    NOT version-sorted -- version is non-authoritative (self_study.py's
    version bump is a non-transactional MAX+1 read) -- to prove `current`
    does not silently re-sort by version itself."""
    conn = _FakeConn(history=[_row(2, ["a"]), _row(3, ["b"]), _row(1, ["c"])])
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.current.version == 2, "first row from the query, not the highest version"
    assert [h.version for h in view.history] == [2, 3, 1]


def test_history_query_is_bounded(monkeypatch) -> None:
    """Self-inquiry caps at 3 runs/day; the query still takes an explicit
    LIMIT rather than growing unbounded over months. Review finding 2026-09-09."""
    from orion.curiosity import self_panel as mod

    conn = _FakeConn(history=[_row(1, [])])
    asyncio.run(read_self_panel(_FakePool(conn)))
    sql, args = conn.fetch_calls[0]
    assert "LIMIT" in sql
    assert args[1] == mod._HISTORY_LIMIT


def test_evidence_refs_json_string_is_tolerated() -> None:
    """A raw-SQL read can hand back a generic JSON column as a string rather
    than a deserialized list -- same tolerance as
    self_atlas_cluster_history._coerce_evidence_refs, kept local here."""
    conn = _FakeConn(history=[_row(1, '["x", "y"]')])
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.current.evidence_refs == ["x", "y"]


def test_evidence_refs_garbage_becomes_empty_list() -> None:
    conn = _FakeConn(history=[_row(1, "not json")])
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.current.evidence_refs == []


def test_journal_entries_are_read_by_title_independent_of_the_graph() -> None:
    conn = _FakeConn(journals=[{"created_at": datetime(2026, 9, 8, tzinfo=timezone.utc), "body": "I looked."}])
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert len(view.journal_entries) == 1
    assert view.journal_entries[0].body == "I looked."
    # The query filters on title, never on any graph-derived run id.
    sql, args = conn.fetch_calls[1]
    assert args[0] == "Self-inquiry"


def test_eval_rows_are_scoped_to_the_single_latest_run() -> None:
    conn = _FakeConn(
        eval_run_id="20260909T051243Z-aa07f6",
        eval_rows=[
            {"question_key": "what_are_you", "question": "q1", "self_label_score": 1, "grounded_record_score": 3, "answer_source": "harness_trace"},
        ],
    )
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.latest_eval_run_id == "20260909T051243Z-aa07f6"
    assert len(view.latest_eval) == 1
    assert view.latest_eval[0].question_key == "what_are_you"


def test_no_eval_rows_ever_run_yet() -> None:
    conn = _FakeConn(eval_run_id=None)
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    assert view.latest_eval_run_id is None
    assert view.latest_eval == []


def test_to_payload_round_trips_every_field() -> None:
    conn = _FakeConn(
        history=[_row(2, ["ev1", "ev2"])],
        journals=[{"created_at": datetime(2026, 9, 8, tzinfo=timezone.utc), "body": "wrote this"}],
        eval_run_id="r1",
        eval_rows=[{"question_key": "k", "question": "q", "self_label_score": 0, "grounded_record_score": 3, "answer_source": "harness_trace"}],
    )
    view = asyncio.run(read_self_panel(_FakePool(conn)))
    payload = to_payload(view)
    assert payload["available"] is True
    assert payload["current"]["version"] == 2
    assert payload["current"]["evidence_refs"] == ["ev1", "ev2"]
    assert payload["history"][0]["version"] == 2
    assert payload["journal_entries"][0]["body"] == "wrote this"
    assert payload["latest_eval_run_id"] == "r1"
    assert payload["latest_eval"][0]["question_key"] == "k"
