"""append_turn() must persist each turn's source_platform into the window row.

This is the link in the chain between the bus event and the formation gate: if
the platform is dropped here, _window_source_platform() sees only Nones, every
window reads as "direct", and the ai-town gate silently never fires while every
unit test above it still passes on hand-built turn dicts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402


def _load_window_state():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    path = SERVICE_ROOT / "app" / "window_state.py"
    spec = importlib.util.spec_from_file_location("memory_consolidation_window_state", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


window_state = _load_window_state()
WindowStore = window_state.WindowStore


def _turn(platform: str | None) -> MemoryTurnPersistedV1:
    return MemoryTurnPersistedV1(
        correlation_id="c1",
        prompt="you're not the flame, Orion",
        response="understood",
        source_platform=platform,
    )


def _written_turns(pool: AsyncMock) -> list[dict]:
    """The JSON turn blob append_turn handed to the INSERT/UPDATE.

    Both branches bind it as $2 -- INSERT is (sql, window_id, turns_json,
    created_at), UPDATE is (sql, window_id, turns_json) -- so it is positional
    arg 2 either way.
    """
    return json.loads(pool.execute.await_args.args[2])


@pytest.mark.asyncio
async def test_first_turn_in_new_window_carries_platform():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("aitown"), scores={})

    written = _written_turns(pool)
    assert len(written) == 1
    assert written[0]["source_platform"] == "aitown"


@pytest.mark.asyncio
async def test_appended_turn_carries_platform_into_existing_window():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(
                [{"correlation_id": "c0", "source_platform": "aitown"}]
            ),
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("aitown"), scores={})

    written = _written_turns(pool)
    assert [t["source_platform"] for t in written] == ["aitown", "aitown"]


# --------------------------------------------------------------------------
# the cursor is partitioned by platform
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_window_lookup_is_scoped_to_the_platform():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()

    await WindowStore(pool)._get_open_window("aitown")

    sql, platform = pool.fetchrow.await_args.args
    assert "source_platform IS NOT DISTINCT FROM $1" in sql
    assert platform == "aitown"


@pytest.mark.asyncio
async def test_direct_partition_is_keyed_by_null_not_by_equality():
    """`NULL = NULL` is NULL, so a plain `=` would make every direct turn open a
    brand-new window and never find its own. IS NOT DISTINCT FROM is load-bearing."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)

    await WindowStore(pool)._get_open_window(None)

    sql, platform = pool.fetchrow.await_args.args
    assert "IS NOT DISTINCT FROM" in sql
    assert "source_platform = $1" not in sql
    assert platform is None


@pytest.mark.asyncio
async def test_new_window_records_its_platform():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("aitown"), scores={})

    args = pool.execute.await_args.args
    assert "source_platform" in args[0]
    assert args[4] == "aitown"


@pytest.mark.asyncio
async def test_appended_turn_looks_up_its_own_platform_window():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("aitown"), scores={})

    assert pool.fetchrow.await_args.args[1] == "aitown"


@pytest.mark.asyncio
async def test_closing_seeds_the_next_window_in_the_same_partition():
    """The closing turn is carried into the next window. Before partitioning, a
    direct turn from Juniper seeded the next window and made it permanently
    mixed, so NPC dialogue right after she spoke still reached the queue."""
    closing = {"correlation_id": "c2", "source_platform": "aitown"}
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(
                [{"correlation_id": "c1", "source_platform": "aitown"}, closing]
            ),
            "source_platform": "aitown",
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).close_current_window("c2", source_platform="aitown")

    insert = pool.execute.await_args.args
    assert "INSERT INTO memory_consolidation_windows" in insert[0]
    assert insert[4] == "aitown"
    assert json.loads(insert[2])[0]["correlation_id"] == "c2"


@pytest.mark.asyncio
async def test_close_tolerates_a_row_predating_the_platform_column():
    """asyncpg Records raise KeyError for an absent column; a legacy open window
    selected before the migration must not crash the close path."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-legacy",
            "turn_correlation_ids": json.dumps([{"correlation_id": "c1"}]),
        }
    )
    pool.execute = AsyncMock()

    closed = await WindowStore(pool).close_current_window("c1", source_platform=None)

    assert closed["memory_window_id"] == "win-legacy"
    assert pool.execute.await_args.args[4] is None


@pytest.mark.asyncio
async def test_direct_turn_records_none_not_a_missing_key():
    """None must be written explicitly. A missing key would still read as
    "direct" today, but only by accident of dict.get -- pin the real shape."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn(None), scores={})

    written = _written_turns(pool)
    assert "source_platform" in written[0]
    assert written[0]["source_platform"] is None
