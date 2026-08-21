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

import check_attention_loop_decay_liveness as liveness  # noqa: E402


class FakeConn:
    def __init__(self, *, trace_rows: list[dict], verdict_rows: list[dict]) -> None:
        self._trace_rows = trace_rows
        self._verdict_rows = verdict_rows
        self.closed = False

    async def fetch(self, sql: str, *args):
        if "attention_salience_trace" in sql:
            return self._trace_rows
        if "attention_loop_outcome" in sql:
            return self._verdict_rows
        raise AssertionError(f"unexpected query: {sql}")

    async def close(self) -> None:
        self.closed = True


def _rows(loop_id: str, times: list[datetime]) -> list[dict]:
    return [{"theme_key": loop_id, "loop_id": loop_id, "created_at": t} for t in times]


@pytest.mark.asyncio
async def test_no_traces_no_backlog():
    conn = FakeConn(trace_rows=[], verdict_rows=[])
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, overshoot, worst = await liveness._query_overshoot(
            "postgresql://fake/db", min_silence=timedelta(hours=24)
        )
    assert (backlog, overshoot, worst) == (0, 0.0, None)
    assert conn.closed


@pytest.mark.asyncio
async def test_loop_silent_under_floor_is_not_eligible():
    now = datetime.now(timezone.utc)
    conn = FakeConn(trace_rows=_rows("loop-a", [now - timedelta(hours=2)]), verdict_rows=[])
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, overshoot, worst = await liveness._query_overshoot(
            "postgresql://fake/db", min_silence=timedelta(hours=24)
        )
    assert backlog == 0


@pytest.mark.asyncio
async def test_loop_silent_past_floor_reports_overshoot():
    now = datetime.now(timezone.utc)
    silent_since = now - timedelta(hours=30)  # 6h past the 24h floor
    conn = FakeConn(trace_rows=_rows("loop-a", [silent_since]), verdict_rows=[])
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, overshoot, worst = await liveness._query_overshoot(
            "postgresql://fake/db", min_silence=timedelta(hours=24)
        )
    assert backlog == 1
    assert overshoot == pytest.approx(6.0, abs=0.1)
    assert worst == "loop-a"


@pytest.mark.asyncio
async def test_loop_with_terminal_verdict_is_excluded():
    now = datetime.now(timezone.utc)
    silent_since = now - timedelta(hours=48)
    conn = FakeConn(
        trace_rows=_rows("loop-a", [silent_since]),
        verdict_rows=[{"loop_id": "loop-a", "verdict": "resolved"}],
    )
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, overshoot, worst = await liveness._query_overshoot(
            "postgresql://fake/db", min_silence=timedelta(hours=24)
        )
    assert backlog == 0


@pytest.mark.asyncio
async def test_loop_already_decayed_once_does_not_count_as_ongoing_backlog():
    # Regression: derive_implicit_verdicts() deliberately keeps decayed_unattended
    # OUT of TERMINAL_VERDICTS (a real design choice for the label stream) --
    # confirmed live 2026-08-21, this made the gate report STALE forever, even
    # immediately after a successful digest run, because a loop it had just
    # labelled never stopped being "eligible". For liveness, a loop that already
    # has ANY outcome (including its own prior decayed_unattended) is handled.
    now = datetime.now(timezone.utc)
    silent_since = now - timedelta(hours=48)
    conn = FakeConn(
        trace_rows=_rows("loop-a", [silent_since]),
        verdict_rows=[{"loop_id": "loop-a", "verdict": "decayed_unattended"}],
    )
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, overshoot, worst = await liveness._query_overshoot(
            "postgresql://fake/db", min_silence=timedelta(hours=24)
        )
    assert backlog == 0


def test_main_exits_zero_when_no_backlog():
    with patch.object(liveness, "_query_overshoot", new=AsyncMock(return_value=(0, 0.0, None))):
        assert liveness.main(["--postgres-uri", "postgresql://fake/db"]) == 0


def test_main_exits_one_when_overshoot_exceeds_threshold():
    with patch.object(liveness, "_query_overshoot", new=AsyncMock(return_value=(1, 10.0, "loop-a"))):
        assert liveness.main(["--postgres-uri", "postgresql://fake/db", "--max-overshoot-hours", "3"]) == 1


def test_main_exits_zero_when_overshoot_within_threshold():
    with patch.object(liveness, "_query_overshoot", new=AsyncMock(return_value=(1, 1.0, "loop-a"))):
        assert liveness.main(["--postgres-uri", "postgresql://fake/db", "--max-overshoot-hours", "3"]) == 0


def test_main_requires_postgres_uri():
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("POSTGRES_URI", None)
        exit_code = liveness.main([])
    assert exit_code == 2


def test_main_exits_two_on_query_failure():
    with patch.object(liveness, "_query_overshoot", new=AsyncMock(side_effect=RuntimeError("connection refused"))):
        exit_code = liveness.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 2
