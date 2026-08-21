from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import attention_loop_decay_digest as digest  # noqa: E402


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeConn:
    def __init__(self, *, trace_rows: list[dict], verdict_rows: list[dict]) -> None:
        self._trace_rows = trace_rows
        self._verdict_rows = verdict_rows
        self.executed: list[tuple] = []
        self.closed = False

    async def fetch(self, sql: str, *args):
        if "attention_salience_trace" in sql:
            return self._trace_rows
        if "attention_loop_outcome" in sql:
            return self._verdict_rows
        raise AssertionError(f"unexpected query: {sql}")

    async def execute(self, sql: str, *args):
        self.executed.append((sql, args))

    def transaction(self):
        return _FakeTxn()

    async def close(self) -> None:
        self.closed = True


def _rows(loop_id: str, times: list[datetime]) -> list[dict]:
    return [
        {"theme_key": loop_id, "loop_id": loop_id, "salience": 0.3, "features": {}, "created_at": t}
        for t in times
    ]


@pytest.mark.asyncio
async def test_dry_run_reports_but_writes_nothing():
    now = datetime.now(timezone.utc)
    conn = FakeConn(trace_rows=_rows("loop-a", [now - timedelta(hours=48)]), verdict_rows=[])
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest(
            "postgresql://fake/db", now=now, min_silence=timedelta(hours=24), dry_run=True
        )
    assert report.themes_scanned == 1
    assert len(report.decayed) == 1
    assert report.decayed[0]["theme_key"] == "loop-a"
    assert conn.executed == []  # dry-run writes nothing
    assert conn.closed


@pytest.mark.asyncio
async def test_live_run_writes_outcome_and_refractory():
    now = datetime.now(timezone.utc)
    conn = FakeConn(trace_rows=_rows("loop-a", [now - timedelta(hours=48)]), verdict_rows=[])
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest(
            "postgresql://fake/db", now=now, min_silence=timedelta(hours=24), dry_run=False
        )
    assert len(report.decayed) == 1
    assert len(conn.executed) == 2  # one outcome insert + one refractory upsert
    outcome_sql, outcome_args = conn.executed[0]
    assert "attention_loop_outcome" in outcome_sql
    assert "decayed_unattended" in outcome_sql
    assert "system:implicit_decay" in outcome_sql
    assert outcome_args[1] == "loop-a"  # loop_id positional arg
    refractory_sql, refractory_args = conn.executed[1]
    assert "substrate_reverie_refractory" in refractory_sql
    assert refractory_args[0] == "loop-a"


@pytest.mark.asyncio
async def test_loop_already_decayed_once_is_not_reported_or_rewritten_on_rerun():
    # Regression: derive_implicit_verdicts() keeps decayed_unattended out of its
    # own TERMINAL_VERDICTS by design, so a repeat run without this filter would
    # re-report (and, before the ON CONFLICT DO NOTHING, could re-write) the
    # same loop as "decayed" forever. Confirmed live 2026-08-21.
    now = datetime.now(timezone.utc)
    conn = FakeConn(
        trace_rows=_rows("loop-a", [now - timedelta(hours=48)]),
        verdict_rows=[{"loop_id": "loop-a", "verdict": "decayed_unattended"}],
    )
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest(
            "postgresql://fake/db", now=now, min_silence=timedelta(hours=24), dry_run=False
        )
    assert report.decayed == []
    assert conn.executed == []


@pytest.mark.asyncio
async def test_loop_with_terminal_verdict_is_never_decayed():
    now = datetime.now(timezone.utc)
    conn = FakeConn(
        trace_rows=_rows("loop-a", [now - timedelta(hours=48)]),
        verdict_rows=[{"loop_id": "loop-a", "verdict": "resolved"}],
    )
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        report = await digest._run_digest(
            "postgresql://fake/db", now=now, min_silence=timedelta(hours=24), dry_run=False
        )
    assert report.decayed == []
    assert conn.executed == []


def test_outcome_id_is_stable_and_episode_scoped():
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    a = digest._outcome_id("loop-a", t1)
    b = digest._outcome_id("loop-a", t1)
    c = digest._outcome_id("loop-a", t2)
    assert a == b  # same loop, same episode -> idempotent re-run
    assert a != c  # same loop, different episode -> distinct row


def test_main_requires_postgres_uri():
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("POSTGRES_URI", None)
        exit_code = digest.main([])
    assert exit_code == 2


def test_main_exits_two_on_run_failure():
    with patch.object(digest, "_run_digest", new=AsyncMock(side_effect=RuntimeError("boom"))):
        exit_code = digest.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 2


def test_main_exits_zero_on_success():
    report = digest.DigestReport(themes_scanned=3, decayed=[])
    with patch.object(digest, "_run_digest", new=AsyncMock(return_value=report)):
        exit_code = digest.main(["--postgres-uri", "postgresql://fake/db", "--dry-run"])
    assert exit_code == 0
