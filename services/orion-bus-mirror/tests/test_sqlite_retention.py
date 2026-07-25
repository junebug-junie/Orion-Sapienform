from __future__ import annotations

from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest

from app.main import _ensure_schema, _prune_old_bus_events


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
