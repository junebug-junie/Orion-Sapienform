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


@pytest.mark.asyncio
async def test_close_current_window_returns_all_turns_including_closer():
    turns = [
        {"correlation_id": "c1", "prompt": "hi", "response": "hello"},
        {"correlation_id": "c2", "prompt": "more", "response": "sure"},
        {
            "correlation_id": "c3",
            "prompt": "bye",
            "response": "see you",
            "phase_change": {"phase": "closing"},
        },
    ]
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(turns),
        }
    )
    pool.execute = AsyncMock()

    store = WindowStore(pool)
    closed = await store.close_current_window("c3")

    assert closed["memory_window_id"] == "win-1"
    assert closed["turn_correlation_ids"] == ["c1", "c2", "c3"]
    assert len(closed["turns"]) == 3
    assert closed["turns"][-1]["correlation_id"] == "c3"
    assert closed["turns"][-1]["prompt"] == "bye"
