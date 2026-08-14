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
