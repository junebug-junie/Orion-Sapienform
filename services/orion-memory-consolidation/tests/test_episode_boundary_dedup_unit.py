"""Boundary Fix 2 without Postgres: a turn already in a window is not re-classified.

The Postgres-backed version (test_episode_boundary_fix2_pg.py) proves the score
equality end to end; this one runs everywhere, including CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    spec = importlib.util.spec_from_file_location(name, SERVICE_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load("app/worker.py", "mc_worker_dedup_unit")

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402


def _env(corr: str) -> BaseEnvelope:
    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hey, which queue?", response="the reading queue")
    return BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="t", node="t"),
        payload=turn.model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_second_delivery_of_a_windowed_turn_is_skipped(monkeypatch):
    corr = str(uuid4())
    classify = AsyncMock()
    monkeypatch.setattr(worker, "classify_turn", classify)
    bus = AsyncMock()
    window_store = AsyncMock()
    window_store.find_windowed_turn = AsyncMock(
        return_value={"memory_window_id": "w1", "correlation_id": corr, "conversation_boundary_score": 0.98}
    )
    episode_store = AsyncMock()

    await worker.handle_memory_turn_persisted(
        _env(corr), bus=bus, window_store=window_store, suggest_runner=AsyncMock(), episode_store=episode_store
    )

    classify.assert_not_awaited()
    bus.publish.assert_not_awaited()  # no second spark_meta patch to overwrite the first
    window_store.append_turn.assert_not_awaited()
    episode_store.observe_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_dedup_lookup_failure_falls_back_to_classifying(monkeypatch):
    """A broken lookup must not drop turns; it degrades to the old behavior."""
    corr = str(uuid4())

    async def _classify(bus, *, turn, prior_turns, settings):
        return {"conversation_boundary_score": 0.1, "turn_change_appraisal": {"turn_change_status": "ok"}}

    monkeypatch.setattr(worker, "classify_turn", _classify)
    bus = AsyncMock()
    window_store = AsyncMock()
    window_store.find_windowed_turn = AsyncMock(side_effect=RuntimeError("db"))
    window_store._get_open_window = AsyncMock(return_value=None)
    window_store.get_window_turns = AsyncMock(return_value=[])

    await worker.handle_memory_turn_persisted(
        _env(corr), bus=bus, window_store=window_store, suggest_runner=AsyncMock(), episode_store=None
    )
    window_store.append_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_shadow_kill_switch_and_discard_platforms(monkeypatch):
    turn = MemoryTurnPersistedV1(correlation_id="c", prompt="p", response="r", source_platform="aitown")
    store = AsyncMock()
    assert await worker.observe_shadow_episode(AsyncMock(), store, turn=turn, scores={}, legacy_close_reason=None) is None
    store.observe_turn.assert_not_awaited()

    monkeypatch.setattr(worker.settings, "MEMORY_EPISODE_SHADOW_ENABLED", False)
    direct = MemoryTurnPersistedV1(correlation_id="d", prompt="p", response="r")
    assert await worker.observe_shadow_episode(AsyncMock(), store, turn=direct, scores={}, legacy_close_reason=None) is None
    store.observe_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_shadow_failure_never_raises_into_the_live_path():
    store = AsyncMock()
    store.observe_turn = AsyncMock(side_effect=RuntimeError("relation memory_episode_shadow does not exist"))
    direct = MemoryTurnPersistedV1(correlation_id="d", prompt="p", response="r")
    assert await worker.observe_shadow_episode(AsyncMock(), store, turn=direct, scores={}, legacy_close_reason=None) is None
