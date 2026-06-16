import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    path = SERVICE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retry_mod = _load("app/retry_failed_windows.py", "memory_consolidation_retry")
retry_failed_windows = retry_mod.retry_failed_windows


@pytest.mark.asyncio
async def test_retry_failed_windows_runs_suggest(monkeypatch):
    window_store = AsyncMock()
    window_store.list_failed_windows = AsyncMock(
        return_value=[{"memory_window_id": "w1", "turn_correlation_ids": "[]"}]
    )
    window_store.get_window_turns = AsyncMock(return_value=[{"correlation_id": "c1", "prompt": "p", "response": "r"}])
    suggest_runner = AsyncMock()
    suggest_runner.consolidate_window = AsyncMock()
    bus = AsyncMock()

    await retry_failed_windows(
        pool=AsyncMock(),
        bus=bus,
        window_store=window_store,
        suggest_runner=suggest_runner,
    )

    suggest_runner.consolidate_window.assert_awaited_once()
