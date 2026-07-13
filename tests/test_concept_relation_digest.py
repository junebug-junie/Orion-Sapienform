from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import concept_relation_digest as digest  # noqa: E402


class _NullAsyncCtx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeConn:
    """Minimal in-process double for the handful of asyncpg.Connection calls this
    script + repository.py::insert_crystallization() make. Dispatches on substrings
    in the SQL text rather than emulating a real query engine -- enough to prove the
    digest script's read/report/write/mark-digested behavior without a live Postgres,
    matching this repo's existing convention of mocking at the pool/connection
    boundary (see tests/test_encode_reinforce_not_duplicate.py)."""

    def __init__(self, decisions: list[dict]) -> None:
        self.decisions = decisions
        self.crystallizations: dict[str, dict] = {}
        self.sources: list[dict] = []
        self.closed = False

    def transaction(self):
        return _NullAsyncCtx()

    async def close(self) -> None:
        self.closed = True

    async def fetch(self, sql: str, *args):
        if "memory_concept_relation_decisions" in sql and "digested = false" in sql:
            pending = [dict(r) for r in self.decisions if not r["digested"]]
            pending.sort(key=lambda r: r["decided_at"])
            return pending
        raise AssertionError(f"FakeConn.fetch: unexpected query: {sql}")

    async def fetchrow(self, sql: str, *args):
        if "INSERT INTO memory_crystallizations" in sql:
            crystallization_id = args[0] if args and args[0] else str(uuid4())
            self.crystallizations[crystallization_id] = {"args": args}
            return {"crystallization_id": crystallization_id}
        raise AssertionError(f"FakeConn.fetchrow: unexpected query: {sql}")

    async def execute(self, sql: str, *args):
        if "INSERT INTO memory_crystallization_sources" in sql:
            self.sources.append({"args": args})
            return "INSERT 0 1"
        if "UPDATE memory_concept_relation_decisions" in sql and "SET digested = true" in sql:
            ids = set(args[0])
            for row in self.decisions:
                if row["decision_id"] in ids:
                    row["digested"] = True
            return f"UPDATE {len(ids)}"
        raise AssertionError(f"FakeConn.execute: unexpected query: {sql}")


def _row(*, relation: str, floor_cleared: bool, confidence: float, offset_sec: int, target: str | None = "crys_target") -> dict:
    return {
        "decision_id": str(uuid4()),
        "candidate_crystallization_id": "crys_candidate",
        "target_crystallization_id": target,
        "relation": relation,
        "confidence": confidence,
        "floor_cleared": floor_cleared,
        "decided_at": datetime(2026, 7, 13, tzinfo=timezone.utc) + timedelta(seconds=offset_sec),
        "digested": False,
    }


def _seeded_rows() -> list[dict]:
    return [
        _row(relation="same", floor_cleared=True, confidence=0.9, offset_sec=1),
        _row(relation="refines", floor_cleared=True, confidence=0.8, offset_sec=2),
        _row(relation="contradicts", floor_cleared=True, confidence=0.75, offset_sec=3),
        # Near-misses: relation judged, confidence landed under the floor.
        _row(relation="contradicts", floor_cleared=False, confidence=0.4, offset_sec=4),
        _row(relation="refines", floor_cleared=False, confidence=0.5, offset_sec=5),
        # unrelated is never floor_cleared (resolve_concept_relation degrades confidence to 0.0).
        _row(relation="unrelated", floor_cleared=False, confidence=0.0, offset_sec=6, target=None),
    ]


@pytest.mark.asyncio
async def test_digest_report_counts_and_reflection_creation():
    conn = FakeConn(_seeded_rows())

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest("postgresql://fake/db")

    assert report.decisions_seen == 6
    assert report.relation_counts == {"same": 1, "refines": 2, "contradicts": 2, "unrelated": 1}
    assert report.near_miss_counts == {"contradicts": 1, "refines": 1}

    # Only the floor_cleared=True same/refines/contradicts rows (3 of them) produce a
    # reflection crystallization -- near-misses and "unrelated" do not.
    assert len(report.reflections_created) == 3
    assert len(conn.crystallizations) == 3
    assert len(conn.sources) == 3

    # Every row seen this run is now marked digested, including the non-actionable ones.
    assert all(row["digested"] for row in conn.decisions)


@pytest.mark.asyncio
async def test_digest_reflection_crystallization_shape():
    conn = FakeConn([_row(relation="contradicts", floor_cleared=True, confidence=0.82, offset_sec=1)])

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest("postgresql://fake/db")

    assert len(report.reflections_created) == 1
    stored = next(iter(conn.crystallizations.values()))
    # args order from repository.py::insert_crystallization's INSERT INTO
    # memory_crystallizations statement: (crystallization_id, kind, subject, summary, ...)
    args = stored["args"]
    assert args[1] == "reflection"
    assert "contradicts" in args[3]  # summary mentions the relation
    assert "0.82" in args[3]  # summary embeds the confidence

    assert len(conn.sources) == 1
    source_args = conn.sources[0]["args"]
    # (crystallization_id, source_kind, source_id, excerpt, strength, note)
    assert source_args[1] == "concept_relation_decision"


@pytest.mark.asyncio
async def test_digest_second_run_with_no_new_rows_is_empty_not_duplicated():
    conn = FakeConn(_seeded_rows())

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        first = await digest._run_digest("postgresql://fake/db")
        second = await digest._run_digest("postgresql://fake/db")

    assert first.decisions_seen == 6
    assert len(first.reflections_created) == 3

    assert second.decisions_seen == 0
    assert second.relation_counts == {r: 0 for r in digest._RELATIONS}
    assert second.reflections_created == []

    # No duplicate reflections were created on the second, empty run.
    assert len(conn.crystallizations) == 3


@pytest.mark.asyncio
async def test_digest_no_pending_rows_reports_cleanly():
    conn = FakeConn([])

    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest("postgresql://fake/db")

    assert report.decisions_seen == 0
    assert report.reflections_created == []


def test_main_reports_missing_postgres_uri(capsys):
    exit_code = digest.main(["--postgres-uri", ""])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "POSTGRES_URI" in captured.err
