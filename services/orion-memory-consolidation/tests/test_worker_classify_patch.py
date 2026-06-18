from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

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


worker = _load("app/worker.py", "memory_consolidation_worker")
ConsolidationSuggestRunner = worker.ConsolidationSuggestRunner
handle_memory_turn_persisted = worker.handle_memory_turn_persisted

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.memory_graph.dto import SuggestDraftV1
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "memory_graph" / "joey_cats_draft.json"


@pytest.mark.asyncio
async def test_handle_memory_turn_persisted_classify_and_patch(monkeypatch):
    corr = str(uuid4())
    published: list[tuple[str, BaseEnvelope]] = []

    bus = AsyncMock()

    async def _publish(channel, env):
        published.append((channel, env))

    bus.publish = _publish

    async def _fake_classify(bus, *, turn, settings):
        return {
            "memory_significance_score": 0.8,
            "conversation_boundary_score": 0.2,
            "memory_classify_status": "ok",
        }

    monkeypatch.setattr(worker, "classify_turn", _fake_classify)

    window_store = AsyncMock()
    window_store.append_turn = AsyncMock()
    window_store.close_current_window = AsyncMock()
    suggest_runner = AsyncMock()

    turn = MemoryTurnPersistedV1(correlation_id=corr, prompt="hi", response="hello", spark_meta={})
    env = BaseEnvelope(
        kind="memory.turn.persisted.v1",
        correlation_id=corr,
        source=ServiceRef(name="sql-writer", version="0.1", node="local"),
        payload=turn.model_dump(mode="json"),
    )

    await handle_memory_turn_persisted(
        env,
        bus=bus,
        window_store=window_store,
        suggest_runner=suggest_runner,
    )

    assert len(published) == 1
    patch_env = published[0][1]
    assert patch_env.kind == "chat.history.spark_meta.patch.v1"
    window_store.append_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_consolidate_window_persists_draft(monkeypatch):
    draft_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pool = AsyncMock()
    window_store = AsyncMock()
    window_store.mark_consolidated = AsyncMock()
    window_store.mark_failed = AsyncMock()

    bus = AsyncMock()
    bus.publish = AsyncMock()

    async def _fake_suggest_with_escalation(*args, **kwargs):
        return {"draft": draft_data}

    monkeypatch.setattr(worker, "suggest_with_escalation", _fake_suggest_with_escalation)
    monkeypatch.setattr(
        worker,
        "insert_pending_draft",
        AsyncMock(return_value="draft-123"),
    )

    runner = ConsolidationSuggestRunner(pool, window_store)
    c1, c2 = str(uuid4()), str(uuid4())
    window = {
        "memory_window_id": "win-1",
        "turn_correlation_ids": [c1, c2],
        "turns": [
            {"correlation_id": c1, "prompt": "p1", "response": "r1", "memory_significance_score": 0.9},
            {"correlation_id": c2, "prompt": "p2", "response": "r2", "memory_significance_score": 0.8},
        ],
    }

    await runner.consolidate_window(window, bus=bus)

    SuggestDraftV1.model_validate(draft_data)
    window_store.mark_consolidated.assert_awaited_once_with("win-1", draft_id="draft-123")
