"""Unit tests for the AST/HOT self-model live tick, appended to the tail of
`_attention_broadcast_tick()` (docs/superpowers/specs/2026-07-29-ast-hot-
reducer-live-ticking-design.md).

Verifies: disabled by default (no-op, matching every other tick in this
file), fails open without breaking the broadcast tick's own existing writes,
persists a real AttentionSelfModelV1 built from the same graph snapshot the
broadcast tick already fetched (zero extra FalkorDB round-trip) plus one
cross-service field-lane read, and accumulates its own in-process trend
buffer across ticks.
"""

from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _make_worker(
    monkeypatch, *, broadcast_enabled: bool = True, self_model_enabled: bool = True
) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "ORION_ATTENTION_BROADCAST_ENABLED", "true" if broadcast_enabled else "false"
    )
    monkeypatch.setenv(
        "SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED",
        "true" if self_model_enabled else "false",
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._substrate_graph_store = None
    worker._store = MagicMock()
    worker._store.get_latest_field_attention_frame.return_value = None
    worker._attention_self_model_trend_buffer = deque(
        maxlen=max(2, int(worker._settings.attention_self_model_trend_window_ticks))
    )
    return worker


def _pe_node(node_id: str, error: float) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, metadata={"prediction_error": error})


def _fake_store(*extra_nodes: SimpleNamespace) -> MagicMock:
    nodes = {
        "node:hot": SimpleNamespace(
            node_id="node:hot",
            label="unresolved contradiction",
            metadata={"dynamic_pressure": 0.9},
            signals=SimpleNamespace(confidence=0.8),
        ),
    }
    for node in extra_nodes:
        nodes[node.node_id] = node
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(nodes=nodes)
    return fake_store


def test_self_model_tick_disabled_by_default_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, self_model_enabled=False)
    fake_store = _fake_store()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    worker._store.save_attention_broadcast.assert_called_once()
    worker._store.save_attention_self_model.assert_not_called()


def test_self_model_tick_persists_real_model(monkeypatch):
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    fake_store = _fake_store(_pe_node("node:substrate.biometrics", 0.42))
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    worker._store.save_attention_self_model.assert_called_once()
    call = worker._store.save_attention_self_model.call_args
    model = call.args[0]
    assert model.attention_reason in {"bottom_up_salience", "field_salience_only"}
    assert model.prediction_error_confidence is not None
    assert call.kwargs["retention_hours"] == (
        worker._settings.attention_self_model_log_retention_hours
    )


def test_self_model_generated_at_is_fresh_even_with_stale_field_frame(monkeypatch):
    """Regression, review finding 2026-07-29: reduce_attention_self_model()
    falls back to field_frame.generated_at for the model's own generated_at
    unless `now=` is passed explicitly. save_attention_self_model()'s dedup
    key is derived from generated_at (ON CONFLICT (model_id) DO NOTHING) --
    if the field lane ever stalls and get_latest_field_attention_frame()
    keeps returning the exact same stale row, an unfixed caller would
    compute the identical model_id every tick and every insert after the
    first would silently no-op, while this tick's own completion log kept
    firing every 30s looking perfectly healthy. Asserts two consecutive
    ticks against an IDENTICAL field_frame each get a distinct, wall-clock
    generated_at -- not the field lane's own frozen timestamp."""
    import time

    worker = _make_worker(monkeypatch, self_model_enabled=True)
    stale_field_frame = SimpleNamespace(
        generated_at=datetime(2000, 1, 1, tzinfo=timezone.utc),  # deliberately ancient/frozen
        overall_salience=0.4,
        dominant_targets=[],
    )
    worker._store.get_latest_field_attention_frame.return_value = stale_field_frame

    fake_store = _fake_store(_pe_node("node:substrate.biometrics", 0.3))
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()
        time.sleep(0.01)
        worker._attention_broadcast_tick()

    assert worker._store.save_attention_self_model.call_count == 2
    first_model = worker._store.save_attention_self_model.call_args_list[0].args[0]
    second_model = worker._store.save_attention_self_model.call_args_list[1].args[0]
    assert first_model.generated_at != second_model.generated_at
    assert first_model.generated_at.year > 2020  # not the stale 2000-01-01 field_frame timestamp
    assert second_model.generated_at.year > 2020


def test_self_model_tick_reads_broadcast_lane_field_frame(monkeypatch):
    """The reducer's `broadcast` argument must be the exact same projection
    object the broadcast tick just persisted -- no second construction, no
    drift between what gets saved to the broadcast lane and what the
    self-model reducer reasons about."""
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    fake_store = _fake_store()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    broadcast_arg = worker._store.save_attention_broadcast.call_args.args[0]
    self_model_arg = worker._store.save_attention_self_model.call_args.args[0]
    assert self_model_arg.broadcast_selected_open_loop_id == broadcast_arg.selected_open_loop_id
    worker._store.get_latest_field_attention_frame.assert_called_once()


def test_self_model_tick_fails_open_without_breaking_broadcast_tick(monkeypatch):
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    worker._store.get_latest_field_attention_frame.side_effect = RuntimeError("db down")
    fake_store = _fake_store()
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()  # must not raise

    worker._store.save_attention_broadcast.assert_called_once()
    worker._store.save_coalition_dwell.assert_called_once()
    worker._store.save_attention_broadcast_history.assert_called_once()
    worker._store.save_attention_self_model.assert_not_called()


def test_self_model_tick_never_calls_a_foreign_writer(monkeypatch):
    """Structural check for this doc's core safety property: the self-model
    tick must be read-only against every existing live input. Asserts the
    only new store call beyond the existing broadcast-tick writes is the one
    it exclusively owns."""
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    fake_store = _fake_store(_pe_node("node:substrate.execution", 0.1))
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    existing_broadcast_writers = {
        "save_attention_broadcast",
        "save_coalition_dwell",
        "save_attention_broadcast_history",
    }
    called_methods = {
        name
        for name, args, kwargs in worker._store.mock_calls
        if not name.startswith("get_")
    }
    assert called_methods <= existing_broadcast_writers | {"save_attention_self_model"}


def test_trend_buffer_accumulates_across_ticks(monkeypatch):
    """`_get_substrate_graph_store()` caches the store object across ticks
    (matching real production behavior -- one store instance, many calls to
    its own `.snapshot()`), so this must patch `build_substrate_store_from_env`
    once and vary `.snapshot()`'s return value per call, not re-patch with a
    fresh store each iteration (which would only ever take effect on the
    first call, since later calls hit the already-cached store)."""
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    assert len(worker._attention_self_model_trend_buffer) == 0

    fake_store = MagicMock()
    fake_store.snapshot.side_effect = [
        _fake_store(_pe_node("node:substrate.biometrics", error)).snapshot.return_value
        for error in (0.1, 0.5, 0.9, 0.2)
    ]
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        for _ in range(4):
            worker._attention_broadcast_tick()

    assert len(worker._attention_self_model_trend_buffer) == 4
    assert list(worker._attention_self_model_trend_buffer) == [
        {"biometrics": 0.1},
        {"biometrics": 0.5},
        {"biometrics": 0.9},
        {"biometrics": 0.2},
    ]
    # 4th call's model should carry a real, non-degenerate predicted_shift --
    # at least 2 snapshots must exist in the window before a trend is
    # computed at all (compute_prediction_error_trend's own contract).
    last_model = worker._store.save_attention_self_model.call_args.args[0]
    assert last_model.predicted_shift is not None
    assert "biometrics" in last_model.predicted_shift


def test_trend_buffer_respects_configured_maxlen(monkeypatch):
    monkeypatch.setenv("SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS", "3")
    worker = _make_worker(monkeypatch, self_model_enabled=True)
    assert worker._attention_self_model_trend_buffer.maxlen == 3

    fake_store = MagicMock()
    fake_store.snapshot.side_effect = [
        _fake_store(_pe_node("node:substrate.route", error)).snapshot.return_value
        for error in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]
    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        for _ in range(5):
            worker._attention_broadcast_tick()

    assert len(worker._attention_self_model_trend_buffer) == 3
    assert list(worker._attention_self_model_trend_buffer) == [
        {"route": 0.3},
        {"route": 0.4},
        {"route": 0.5},
    ]
