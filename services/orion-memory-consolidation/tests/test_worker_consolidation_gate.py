import pytest
from unittest.mock import AsyncMock

from app.window_state import WindowStore


@pytest.mark.asyncio
async def test_mark_consolidated_skipped():
    pool = AsyncMock()
    pool.execute = AsyncMock()
    store = WindowStore(pool)
    await store.mark_consolidated_skipped("win-1", reasons=["low_info_social"])
    sql = pool.execute.await_args.args[0]
    assert "skipped" in sql
