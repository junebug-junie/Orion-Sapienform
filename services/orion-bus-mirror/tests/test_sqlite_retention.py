from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from app import main as main_module
from app.main import _ensure_schema, _prune_old_bus_events, _run_sqlite_retention_loop


async def _insert_row(conn: aiosqlite.Connection, *, timestamp_iso: str) -> None:
    await conn.execute(
        "INSERT INTO bus_events(timestamp, channel, envelope_json) VALUES (?, ?, ?)",
        (timestamp_iso, "orion:test", "{}"),
    )
    await conn.commit()


class TestPruneOldBusEvents:
    @pytest.mark.asyncio
    async def test_deletes_rows_older_than_retention_window(self) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            old = (now - timedelta(hours=48)).isoformat()
            recent = (now - timedelta(hours=1)).isoformat()
            await _insert_row(conn, timestamp_iso=old)
            await _insert_row(conn, timestamp_iso=recent)

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now)

            assert deleted == 1
            cursor = await conn.execute("SELECT timestamp FROM bus_events")
            remaining = [row[0] for row in await cursor.fetchall()]
            assert remaining == [recent]

    @pytest.mark.asyncio
    async def test_no_rows_older_than_window_deletes_nothing(self) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            recent = (now - timedelta(hours=1)).isoformat()
            await _insert_row(conn, timestamp_iso=recent)

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now)

            assert deleted == 0
            cursor = await conn.execute("SELECT count(*) FROM bus_events")
            assert (await cursor.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_empty_table_prunes_zero_rows(self) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0)

            assert deleted == 0

    @pytest.mark.asyncio
    async def test_boundary_row_exactly_at_cutoff_is_not_deleted(self) -> None:
        # Strict "<" comparison against the cutoff -- a row exactly at the
        # retention boundary is kept, not deleted (avoids off-by-one pruning
        # of a row that just barely still qualifies as "within retention").
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            exactly_at_cutoff = (now - timedelta(hours=24)).isoformat()
            await _insert_row(conn, timestamp_iso=exactly_at_cutoff)

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now)

            assert deleted == 0

    @pytest.mark.asyncio
    async def test_multiple_old_rows_all_pruned_in_one_pass(self) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            for hours_ago in (25, 30, 100, 1000):
                await _insert_row(conn, timestamp_iso=(now - timedelta(hours=hours_ago)).isoformat())
            await _insert_row(conn, timestamp_iso=(now - timedelta(hours=1)).isoformat())

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now)

            assert deleted == 4
            cursor = await conn.execute("SELECT count(*) FROM bus_events")
            assert (await cursor.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_prunes_across_multiple_batches(self) -> None:
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            for hours_ago in range(25, 32):
                await _insert_row(conn, timestamp_iso=(now - timedelta(hours=hours_ago)).isoformat())
            await _insert_row(conn, timestamp_iso=(now - timedelta(hours=1)).isoformat())

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now, batch_size=3)

            assert deleted == 7
            cursor = await conn.execute("SELECT count(*) FROM bus_events")
            assert (await cursor.fetchone())[0] == 1


class TestRetentionLoop:
    @pytest.mark.asyncio
    async def test_prunes_immediately_on_start_not_after_first_interval(self, monkeypatch) -> None:
        # Regression: sleep-first meant a process restarting more often than
        # the interval never pruned at all (live 2026-10-02: 8 days of rows,
        # 30.5GB file, against a 24h retention).
        monkeypatch.setattr(main_module.settings, "MIRROR_SQLITE_PRUNE_INTERVAL_SEC", 3600.0)
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            await _insert_row(conn, timestamp_iso=(datetime.now(timezone.utc) - timedelta(days=8)).isoformat())

            task = asyncio.create_task(_run_sqlite_retention_loop(conn))
            for _ in range(50):
                await asyncio.sleep(0.01)
                cursor = await conn.execute("SELECT count(*) FROM bus_events")
                if (await cursor.fetchone())[0] == 0:
                    break
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            cursor = await conn.execute("SELECT count(*) FROM bus_events")
            assert (await cursor.fetchone())[0] == 0


class TestPruneStopsEarly:
    @pytest.mark.asyncio
    async def test_does_not_scan_recent_rows_once_old_ones_are_gone(self) -> None:
        # The stop test must read only the oldest row, not walk the retained
        # 24h of rows (no timestamp index) on the connection inserts share.
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            await _insert_row(conn, timestamp_iso=(now - timedelta(hours=30)).isoformat())
            for minutes_ago in range(50):
                await _insert_row(conn, timestamp_iso=(now - timedelta(minutes=minutes_ago + 1)).isoformat())

            statements: list[str] = []
            await conn.set_trace_callback(statements.append)
            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now, batch_size=1)
            await conn.set_trace_callback(None)

            assert deleted == 1
            assert not any("WHERE timestamp <" in s for s in statements)
            cursor = await conn.execute("SELECT count(*) FROM bus_events")
            assert (await cursor.fetchone())[0] == 50


class TestPruneYieldsToTheMessageLoop:
    @pytest.mark.asyncio
    async def test_sleeps_between_batches_in_proportion_to_batch_time(self, monkeypatch) -> None:
        # A back-to-back prune starved inserts on the shared connection
        # (live 2026-10-02: 77 msg/s -> 4.7 msg/s). Each batch must be
        # followed by a real pause, not asyncio.sleep(0).
        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def recording_sleep(delay: float) -> None:
            sleeps.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(main_module.asyncio, "sleep", recording_sleep)
        async with aiosqlite.connect(":memory:") as conn:
            await _ensure_schema(conn)
            now = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
            for hours_ago in range(25, 31):
                await _insert_row(conn, timestamp_iso=(now - timedelta(hours=hours_ago)).isoformat())

            deleted = await _prune_old_bus_events(conn, retention_hours=24.0, now=now, batch_size=2, yield_ratio=3.0)

        assert deleted == 6
        assert len(sleeps) == 3
        assert all(delay > 0 for delay in sleeps)
