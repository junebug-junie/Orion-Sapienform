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

import check_concept_relation_digest_liveness as liveness  # noqa: E402


class FakeConn:
    def __init__(self, *, backlog: int, oldest_pending: datetime | None) -> None:
        self._backlog = backlog
        self._oldest_pending = oldest_pending
        self.closed = False

    async def fetchrow(self, sql: str, *args):
        assert "memory_concept_relation_decisions" in sql
        assert "digested = false" in sql
        return {"backlog": self._backlog, "oldest_pending": self._oldest_pending}

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_no_backlog_is_healthy():
    conn = FakeConn(backlog=0, oldest_pending=None)
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, oldest_pending = await liveness._query_backlog("postgresql://fake/db")
    assert backlog == 0
    assert oldest_pending is None
    assert conn.closed


@pytest.mark.asyncio
async def test_fresh_backlog_within_threshold_is_healthy():
    oldest = datetime.now(timezone.utc) - timedelta(minutes=10)
    conn = FakeConn(backlog=3, oldest_pending=oldest)
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        backlog, oldest_pending = await liveness._query_backlog("postgresql://fake/db")
    assert backlog == 3
    assert oldest_pending == oldest


def test_main_exits_zero_when_no_backlog():
    with patch.object(liveness, "_query_backlog", new=AsyncMock(return_value=(0, None))):
        exit_code = liveness.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 0


def test_main_exits_zero_when_backlog_fresh():
    oldest = datetime.now(timezone.utc) - timedelta(minutes=10)
    with patch.object(liveness, "_query_backlog", new=AsyncMock(return_value=(2, oldest))):
        exit_code = liveness.main(["--postgres-uri", "postgresql://fake/db", "--max-age-hours", "3"])
    assert exit_code == 0


def test_main_exits_one_when_backlog_stale():
    # Older than the default 3h threshold -- simulates a dead/dropped cron entry.
    oldest = datetime.now(timezone.utc) - timedelta(hours=6)
    with patch.object(liveness, "_query_backlog", new=AsyncMock(return_value=(5, oldest))):
        exit_code = liveness.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 1


def test_main_respects_custom_max_age_hours():
    oldest = datetime.now(timezone.utc) - timedelta(hours=2)
    # 2h old backlog is stale under a 1h threshold...
    with patch.object(liveness, "_query_backlog", new=AsyncMock(return_value=(1, oldest))):
        assert liveness.main(["--postgres-uri", "postgresql://fake/db", "--max-age-hours", "1"]) == 1
    # ...but healthy under a 3h threshold.
    with patch.object(liveness, "_query_backlog", new=AsyncMock(return_value=(1, oldest))):
        assert liveness.main(["--postgres-uri", "postgresql://fake/db", "--max-age-hours", "3"]) == 0


def test_main_requires_postgres_uri():
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("POSTGRES_URI", None)
        exit_code = liveness.main([])
    assert exit_code == 2


def test_main_exits_two_on_query_failure():
    with patch.object(liveness, "_query_backlog", new=AsyncMock(side_effect=RuntimeError("connection refused"))):
        exit_code = liveness.main(["--postgres-uri", "postgresql://fake/db"])
    assert exit_code == 2
