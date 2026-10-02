"""storage_write reducer lane wiring in the substrate-runtime worker."""

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
from orion.schemas.storage_write_projection import ROLE_FAMILY_WINDOW, ROLE_WINDOW_COMPLETED
from orion.substrate.storage_write_loop.constants import (
    STORAGE_WRITE_GRAMMAR_CURSOR_NAME,
    STORAGE_WRITE_NODE_ID,
    STORAGE_WRITE_SOURCE_SERVICE,
    STORAGE_WRITE_TRACE_PREFIX,
)

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clear_health() -> None:
    clear_health_for_tests()


def _ev(idx: int, role: str, summary: str) -> GrammarEventV1:
    trace = f"{STORAGE_WRITE_TRACE_PREFIX}athena:20261002T120000Z"
    eid = f"{trace}:{idx:02d}:{role}"
    return GrammarEventV1(
        event_id=eid,
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id=eid, trace_id=trace, atom_type="observation", semantic_role=role, layer="storage", summary=summary
        ),
        provenance=GrammarProvenanceV1(source_service=STORAGE_WRITE_SOURCE_SERVICE),
    )


def test_spec_is_registered_last_and_default_off():
    spec = REDUCER_SPECS[6]
    assert spec is REDUCER_SPECS[-1]
    assert spec.reducer_key == "storage_write"
    assert spec.cursor_name == STORAGE_WRITE_GRAMMAR_CURSOR_NAME
    assert spec.source_service == STORAGE_WRITE_SOURCE_SERVICE
    assert spec.enabled(SimpleNamespace(enable_storage_write_reducer=False)) is False
    assert spec.enabled(SimpleNamespace(enable_storage_write_reducer=True)) is True


def test_settings_default_is_off(monkeypatch):
    import app.settings as settings_mod

    monkeypatch.setenv("POSTGRES_URI", "postgresql://u:p@unused/db")
    monkeypatch.delenv("ENABLE_STORAGE_WRITE_REDUCER", raising=False)
    s = settings_mod.Settings()
    assert s.enable_storage_write_reducer is False
    assert s.storage_write_grammar_batch_limit == 200


def test_cursor_is_known_to_truth_and_registry():
    import app.grammar_truth as gt
    from app.store import GRAMMAR_CURSOR_REGISTRY

    assert GRAMMAR_CURSOR_REGISTRY[STORAGE_WRITE_GRAMMAR_CURSOR_NAME] == (
        (STORAGE_WRITE_SOURCE_SERVICE,),
        STORAGE_WRITE_TRACE_PREFIX,
    )
    assert gt.REDUCER_KEY_BY_CURSOR[STORAGE_WRITE_GRAMMAR_CURSOR_NAME] == "storage_write"
    assert gt.ENABLED_BY_REDUCER_KEY["storage_write"](SimpleNamespace(enable_storage_write_reducer=True)) is True


def _worker() -> BiometricsSubstrateWorker:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = MagicMock()
    worker._settings.enable_storage_write_reducer = True
    worker._settings.storage_write_grammar_batch_limit = 200
    worker._settings.reducer_poison_max_retries = 99
    worker._store = MagicMock()
    return worker


def test_tick_reduces_a_window_and_returns_last_event_id():
    worker = _worker()
    events = [
        _ev(0, ROLE_FAMILY_WINDOW,
            "family=home_cooling_sample attempted=7 committed=0 duplicate=0 failed=7 skipped=0 "
            "unrouted=0 classes=serialization:7 p50_ms=none p95_ms=none"),
        _ev(1, ROLE_FAMILY_WINDOW,
            "family=grammar_events attempted=265 committed=265 duplicate=0 failed=0 skipped=0 "
            "unrouted=0 classes=none p50_ms=9 p95_ms=40"),
        _ev(2, ROLE_WINDOW_COMPLETED,
            "writer=athena attempted=272 committed=265 failed=7 unrouted=0 families=2 "
            "grammar_queue_max=3 window_sec=60.0"),
    ]
    worker._store.fetch_storage_write_grammar_events.return_value = events
    worker._store.load_storage_write_projection.return_value = None

    assert worker._storage_write_tick() == events[-1].event_id

    worker._store.fetch_storage_write_grammar_events.assert_called_once_with(limit=200)
    receipt = worker._store.save_receipt.call_args.args[0]
    assert [d.target_id for d in receipt.state_deltas] == [STORAGE_WRITE_NODE_ID]
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "storage_write"
    # 7 failures / max(7 attempts, 10): the worst family, not the pooled 7/272
    assert delta.after["pressure_hints"] == {"write_failure_pressure": 0.7}
    assert delta.after["failure_window"]["scope"] == "home_cooling_sample"
    saved = worker._store.save_storage_write_projection.call_args.args[0]
    assert saved.families["home_cooling_sample"].failure_classes == {"serialization": 7}
    assert saved.grammar_queue_max == 3


def test_tick_with_no_events_does_nothing():
    worker = _worker()
    worker._store.fetch_storage_write_grammar_events.return_value = []
    assert worker._storage_write_tick() is None
    worker._store.save_receipt.assert_not_called()
