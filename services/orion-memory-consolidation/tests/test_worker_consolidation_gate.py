from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
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


worker = _load("app/worker.py", "memory_consolidation_worker_gate")
ConsolidationSuggestRunner = worker.ConsolidationSuggestRunner

from app.window_state import WindowStore
from orion.memory.consolidation_gate import ConsolidationGateResult

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "memory_graph" / "joey_cats_draft.json"


def _greeting_window():
    return {
        "memory_window_id": "win-greet",
        "turn_correlation_ids": ["corr-1", "corr-2"],
        "turns": [
            {
                "correlation_id": "corr-1",
                "prompt": "hey",
                "response": "hi there",
                "spark_meta": {
                    "turn_change_appraisal": {
                        "novelty_score": 0.1,
                        "shift_kind": "NONE",
                        "turn_change_status": "ok",
                    },
                    "memory_significance_score": 0.2,
                },
            },
            {
                "correlation_id": "corr-2",
                "prompt": "thanks",
                "response": "anytime",
                "spark_meta": {
                    "turn_change_appraisal": {
                        "novelty_score": 0.1,
                        "shift_kind": "NONE",
                        "turn_change_status": "ok",
                    },
                    "memory_significance_score": 0.2,
                },
            },
        ],
    }


@pytest.mark.asyncio
async def test_mark_consolidated_skipped():
    pool = AsyncMock()
    pool.execute = AsyncMock()
    store = WindowStore(pool)
    await store.mark_consolidated_skipped("win-1", reasons=["low_info_social"])
    sql = pool.execute.await_args.args[0]
    assert "skipped" in sql


@pytest.mark.asyncio
async def test_consolidate_window_skips_greeting_without_draft(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_CONSOLIDATION_OUTPUT", "crystallization_propose")
    pool = AsyncMock()
    window_store = AsyncMock()
    runner = ConsolidationSuggestRunner(pool, window_store)
    bus = AsyncMock()
    bus.publish = AsyncMock()

    suggest_mock = AsyncMock()
    monkeypatch.setattr(worker, "suggest_with_escalation", suggest_mock)
    insert_draft_mock = AsyncMock()
    monkeypatch.setattr(worker, "insert_pending_draft", insert_draft_mock)

    with patch(
        "orion.memory.consolidation_grammar.fetch_grammar_evidence_for_window",
        new=AsyncMock(return_value=(False, [])),
    ), patch(
        "orion.memory.crystallization.intake_pipeline.process_consolidation_crystallization",
        new=AsyncMock(),
    ) as pipeline_mock:
        await runner.consolidate_window(_greeting_window(), bus=bus)

    window_store.mark_consolidated_skipped.assert_awaited_once()
    suggest_mock.assert_not_awaited()
    insert_draft_mock.assert_not_awaited()
    pipeline_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_consolidate_window_proposes_crystallization(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_CONSOLIDATION_OUTPUT", "crystallization_propose")
    gate = ConsolidationGateResult(action="propose", reasons=["substantive_shift"], dominant_shift="TOPIC")
    crystallization = MagicMock(crystallization_id="crys-1")

    pool = AsyncMock()
    window_store = AsyncMock()
    runner = ConsolidationSuggestRunner(pool, window_store)
    bus = AsyncMock()
    bus.publish = AsyncMock()

    suggest_mock = AsyncMock()
    monkeypatch.setattr(worker, "suggest_with_escalation", suggest_mock)

    with patch(
        "orion.memory.consolidation_grammar.fetch_grammar_evidence_for_window",
        new=AsyncMock(return_value=(False, [])),
    ), patch(
        "orion.memory.consolidation_gate.consolidation_memory_gate",
        return_value=gate,
    ), patch(
        "orion.memory.crystallization.intake_consolidation_window.build_crystallization_from_window",
        return_value=crystallization,
    ), patch(
        "orion.memory.crystallization.intake_pipeline.process_consolidation_crystallization",
        new=AsyncMock(return_value=("crys-1", crystallization, "proposed")),
    ) as pipeline_mock:
        window = {
            "memory_window_id": "win-topic",
            "turn_correlation_ids": ["corr-1"],
            "turns": [{"correlation_id": "corr-1", "prompt": "topic", "response": "reply", "spark_meta": {}}],
        }
        await runner.consolidate_window(window, bus=bus)

    pipeline_mock.assert_awaited_once()
    window_store.mark_crystallization_proposed.assert_awaited_once_with(
        "win-topic",
        crystallization_id="crys-1",
    )
    suggest_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_window_auto_activate_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_CONSOLIDATION_OUTPUT", "crystallization_propose")
    monkeypatch.setattr(worker.settings, "MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED", True)
    gate = ConsolidationGateResult(action="propose", reasons=["substantive_shift"], dominant_shift="TOPIC")
    crystallization = MagicMock(crystallization_id="crys-auto")

    pool = AsyncMock()
    window_store = AsyncMock()
    runner = ConsolidationSuggestRunner(pool, window_store)
    bus = AsyncMock()
    bus.publish = AsyncMock()

    pipeline_mock = AsyncMock(return_value=("crys-auto", crystallization, "auto_activated"))

    with patch(
        "orion.memory.consolidation_grammar.fetch_grammar_evidence_for_window",
        new=AsyncMock(return_value=(False, [])),
    ), patch(
        "orion.memory.consolidation_gate.consolidation_memory_gate",
        return_value=gate,
    ), patch(
        "orion.memory.crystallization.intake_consolidation_window.build_crystallization_from_window",
        return_value=crystallization,
    ), patch(
        "orion.memory.crystallization.intake_pipeline.process_consolidation_crystallization",
        new=pipeline_mock,
    ), patch.object(worker, "publish_spark_meta_patch", new=AsyncMock()) as patch_mock:
        window = {
            "memory_window_id": "win-auto",
            "turn_correlation_ids": ["corr-auto"],
            "turns": [{"correlation_id": "corr-auto", "prompt": "topic", "response": "reply", "spark_meta": {}}],
        }
        await runner.consolidate_window(window, bus=bus)

    pipeline_mock.assert_awaited_once()
    patch_mock.assert_awaited()
    patch_fields = patch_mock.await_args.args[2]
    assert patch_fields["consolidation_gate"]["formation_outcome"] == "auto_activated"
    assert patch_fields["consolidation_gate"]["crystallization_id"] == "crys-auto"


@pytest.mark.asyncio
async def test_graph_draft_legacy_mode_still_inserts_draft(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_CONSOLIDATION_OUTPUT", "graph_draft")
    draft_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pool = AsyncMock()
    window_store = AsyncMock()
    window_store.mark_consolidated = AsyncMock()
    runner = ConsolidationSuggestRunner(pool, window_store)
    bus = AsyncMock()
    bus.publish = AsyncMock()

    suggest_mock = AsyncMock(return_value={"draft": draft_data})
    monkeypatch.setattr(worker, "suggest_with_escalation", suggest_mock)
    insert_draft_mock = AsyncMock(return_value="draft-legacy")
    monkeypatch.setattr(worker, "insert_pending_draft", insert_draft_mock)

    c1 = str(uuid4())
    window = {
        "memory_window_id": "win-legacy",
        "turn_correlation_ids": [c1],
        "turns": [
            {
                "correlation_id": c1,
                "prompt": "p",
                "response": "r",
                "memory_significance_score": 0.9,
            }
        ],
    }
    await runner.consolidate_window(window, bus=bus)

    suggest_mock.assert_awaited_once()
    insert_draft_mock.assert_awaited_once()
    window_store.mark_consolidated.assert_awaited_once_with("win-legacy", draft_id="draft-legacy")


@pytest.mark.asyncio
async def test_skip_only_skips_even_when_gate_proposes(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_CONSOLIDATION_OUTPUT", "skip_only")
    gate = ConsolidationGateResult(
        action="propose",
        reasons=["substantive_shift"],
        dominant_shift="TOPIC",
    )
    pool = AsyncMock()
    grammar_pool = AsyncMock()
    window_store = AsyncMock()
    runner = ConsolidationSuggestRunner(pool, window_store, grammar_pool=grammar_pool)
    bus = AsyncMock()
    bus.publish = AsyncMock()

    fetch_mock = AsyncMock(return_value=(False, []))
    with patch(
        "orion.memory.consolidation_grammar.fetch_grammar_evidence_for_window",
        new=fetch_mock,
    ) as fetch_fn, patch(
        "orion.memory.consolidation_gate.consolidation_memory_gate",
        return_value=gate,
    ), patch(
        "orion.memory.crystallization.intake_pipeline.process_consolidation_crystallization",
        new=AsyncMock(),
    ) as pipeline_mock:
        window = {
            "memory_window_id": "win-skip-only",
            "turn_correlation_ids": [str(uuid4())],
            "turns": [{"correlation_id": "c", "prompt": "topic", "response": "r", "spark_meta": {}}],
        }
        window["turns"][0]["correlation_id"] = window["turn_correlation_ids"][0]
        await runner.consolidate_window(window, bus=bus)

    fetch_fn.assert_awaited_once()
    assert fetch_fn.await_args.args[0] is grammar_pool
    window_store.mark_consolidated_skipped.assert_awaited_once()
    pipeline_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_append_turn_persists_turn_change_appraisal_in_spark_meta():
    import json
    from datetime import datetime, timezone

    from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()
    store = WindowStore(pool)
    turn = MemoryTurnPersistedV1(
        correlation_id="corr-appraisal",
        prompt="move logistics alone",
        response="that sounds heavy",
        spark_meta={},
        created_at=datetime.now(timezone.utc),
    )
    scores = {
        "memory_significance_score": 0.55,
        "turn_change_appraisal": {
            "turn_change_status": "ok",
            "novelty_score": 0.72,
            "shift_kind": "TOPIC",
        },
    }
    await store.append_turn(turn, scores=scores)
    turns_json = pool.execute.await_args.args[2]
    turns = json.loads(turns_json)
    assert turns[0]["spark_meta"]["turn_change_appraisal"]["shift_kind"] == "TOPIC"
    assert turns[0]["spark_meta"]["memory_significance_score"] == 0.55
