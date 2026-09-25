"""llm_inference reducer lane wiring in the substrate-runtime worker."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.reducer_health import clear_health_for_tests
from app.worker import BiometricsSubstrateWorker, REDUCER_SPECS
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.llm_inference_projection import ROLE_NODE_WINDOW, ROLE_WINDOW_COMPLETED
from orion.substrate.llm_inference_loop.constants import (
    LLM_INFERENCE_GRAMMAR_CURSOR_NAME,
    LLM_INFERENCE_SOURCE_SERVICE,
    LLM_INFERENCE_TRACE_PREFIX,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_health() -> None:
    clear_health_for_tests()


def _ev(idx: int, role: str, summary: str) -> GrammarEventV1:
    trace = f"{LLM_INFERENCE_TRACE_PREFIX}athena:20260925T120000Z"
    eid = f"{trace}:{idx:02d}:{role}"
    return GrammarEventV1(
        event_id=eid,
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id=eid, trace_id=trace, atom_type="observation", semantic_role=role, layer="inference", summary=summary
        ),
        provenance=GrammarProvenanceV1(source_service=LLM_INFERENCE_SOURCE_SERVICE),
    )


def test_spec_is_registered_last_and_default_off():
    spec = REDUCER_SPECS[5]
    assert spec.reducer_key == "llm_inference"
    assert spec.cursor_name == LLM_INFERENCE_GRAMMAR_CURSOR_NAME
    assert spec.source_service == LLM_INFERENCE_SOURCE_SERVICE
    assert spec.enabled(SimpleNamespace(enable_llm_inference_reducer=False)) is False
    assert spec.enabled(SimpleNamespace(enable_llm_inference_reducer=True)) is True


def test_settings_default_is_off(monkeypatch):
    import app.settings as settings_mod

    monkeypatch.setenv("POSTGRES_URI", "postgresql://u:p@unused/db")
    monkeypatch.delenv("ENABLE_LLM_INFERENCE_REDUCER", raising=False)
    s = settings_mod.Settings()
    assert s.enable_llm_inference_reducer is False
    assert s.llm_inference_grammar_batch_limit == 200


def test_cursor_is_known_to_truth_and_registry():
    import app.grammar_truth as gt
    from app.store import GRAMMAR_CURSOR_REGISTRY

    assert GRAMMAR_CURSOR_REGISTRY[LLM_INFERENCE_GRAMMAR_CURSOR_NAME] == (
        (LLM_INFERENCE_SOURCE_SERVICE,),
        LLM_INFERENCE_TRACE_PREFIX,
    )
    assert gt.REDUCER_KEY_BY_CURSOR[LLM_INFERENCE_GRAMMAR_CURSOR_NAME] == "llm_inference"
    assert gt.ENABLED_BY_REDUCER_KEY["llm_inference"](SimpleNamespace(enable_llm_inference_reducer=True)) is True


def test_tick_reduces_a_window_and_returns_last_event_id():
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_llm_inference_reducer = True
    worker._settings.llm_inference_grammar_batch_limit = 200
    worker._settings.reducer_poison_max_retries = 99
    worker._store = MagicMock()
    events = [
        _ev(0, ROLE_NODE_WINDOW, "node=circe calls=4 served=3 upstream_failed=1 refused=0 request_invalid=0 workers=circe-worker-2 classes=served:3|upstream_timeout:1"),
        _ev(1, ROLE_WINDOW_COMPLETED, "gateway=athena calls=4 nodes=1 window_sec=60.0"),
    ]
    worker._store.fetch_llm_inference_grammar_events.return_value = events
    worker._store.load_llm_inference_projection.return_value = None

    assert worker._llm_inference_tick() == events[-1].event_id

    worker._store.fetch_llm_inference_grammar_events.assert_called_once_with(limit=200)
    receipt = worker._store.save_receipt.call_args.args[0]
    assert [d.target_id for d in receipt.state_deltas] == ["llm_node:circe"]
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.25}
    saved = worker._store.save_llm_inference_projection.call_args.args[0]
    assert saved.nodes["llm_node:circe"].upstream_failed == 1


def test_tick_with_no_events_does_nothing():
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.llm_inference_grammar_batch_limit = 200
    worker._store = MagicMock()
    worker._store.fetch_llm_inference_grammar_events.return_value = []
    assert worker._llm_inference_tick() is None
    worker._store.save_receipt.assert_not_called()
