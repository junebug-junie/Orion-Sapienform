from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-attention-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.store import AttentionRuntimeStore  # noqa: E402
from orion.schemas.field_attention_frame import FieldAttentionFrameV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:tick_a:field_attention_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        overall_salience=0.5,
    )


def _mock_engine_for_frames(*, latest: dict | None, by_tick: dict | None = None):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "substrate_field_state" in sql:
            result.mappings.return_value.first.return_value = None
        elif "source_field_tick_id" in sql:
            row = {"frame_json": by_tick} if by_tick else None
            result.mappings.return_value.first.return_value = row
        else:
            row = {"frame_json": latest} if latest else None
            result.mappings.return_value.first.return_value = row
        return result

    conn.execute.side_effect = execute_side_effect
    return fake_engine


def test_load_latest_field(monkeypatch) -> None:
    from orion.schemas.field_state import FieldStateV1

    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_field",
        node_vectors={"node:athena": {"cortex_exec_step_load": 0.5}},
    )
    payload = field.model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute_side_effect(stmt, params=None):
        result = MagicMock()
        result.mappings.return_value.first.return_value = {"field_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    loaded = store.load_latest_field()
    assert loaded is not None
    assert loaded.tick_id == "tick_field"


def test_load_latest_attention_frame(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    monkeypatch.setattr(store, "_engine", _mock_engine_for_frames(latest=payload))
    loaded = store.load_latest_attention_frame()
    assert loaded is not None
    assert loaded.frame_id == _frame().frame_id


def test_load_attention_frame_for_field_tick(monkeypatch) -> None:
    payload = _frame().model_dump(mode="json")
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    monkeypatch.setattr(store, "_engine", _mock_engine_for_frames(latest=None, by_tick=payload))
    loaded = store.load_attention_frame_for_field_tick("tick_a")
    assert loaded is not None
    assert loaded.source_field_tick_id == "tick_a"


def test_save_attention_frame_idempotent(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = _mock_engine_for_frames(latest=None)
    store._engine = fake_engine
    store.save_attention_frame(_frame())
    conn = fake_engine.begin.return_value.__enter__.return_value
    assert conn.execute.called
    sql = str(conn.execute.call_args[0][0])
    assert "ON CONFLICT (frame_id)" in sql


# -- advance_node_prediction_error_baseline (2026-07-30 fix) -------------------
# See orion/sentience_striving_program/README.md §12 for the live incident this
# method exists to fix: the old per-tick rolling-window recompute
# (load_prediction_error_history) let a target's real n_samples reset to
# whatever survived a ~30-minute retention window. This method persists a
# cumulative baseline instead.


def _mock_engine_for_baseline(*, existing_row: dict | None, new_rows: list[dict]):
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    calls: list[str] = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "substrate_node_prediction_error_baseline" in sql and "SELECT" in sql:
            calls.append("read_baseline")
            result.mappings.return_value.first.return_value = existing_row
        elif "substrate_reduction_receipts" in sql:
            calls.append("fetch_new_rows")
            result.mappings.return_value.all.return_value = new_rows
        elif "INSERT INTO substrate_node_prediction_error_baseline" in sql:
            calls.append("persist_baseline")
        else:
            raise AssertionError(f"unexpected SQL in mock: {sql}")
        return result

    conn.execute.side_effect = execute_side_effect
    return fake_engine, conn, calls


def test_advance_node_prediction_error_baseline_cold_start_no_prior_row(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=None,
        new_rows=[
            {"error": "0.05", "created_at": now},
            {"error": "0.9", "created_at": now},
        ],
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.execution",
        reducer_key="execution_trajectory",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    assert baseline.observation_count == 2
    assert baseline.last_value == pytest.approx(0.9)
    assert calls == ["read_baseline", "fetch_new_rows", "persist_baseline"]


def test_advance_node_prediction_error_baseline_no_new_rows_skips_write(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    existing = {
        "ewma": 0.1,
        "variance": 0.02,
        "observation_count": 10,
        "last_value": 0.15,
        "last_receipt_created_at": datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
    }
    fake_engine, conn, calls = _mock_engine_for_baseline(existing_row=existing, new_rows=[])
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.chat",
        reducer_key="chat_session",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # A true no-op tick: the persisted state is returned unchanged and no
    # write is issued -- "nothing new landed this tick" must not be
    # misrepresented as "no real history."
    assert baseline.observation_count == 10
    assert baseline.last_value == pytest.approx(0.15)
    assert calls == ["read_baseline", "fetch_new_rows"]  # no persist_baseline call


def test_advance_node_prediction_error_baseline_accumulates_on_existing_row(monkeypatch) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    existing = {
        "ewma": 0.05,
        "variance": 0.0,
        "observation_count": 2,
        "last_value": 0.0053,
        "last_receipt_created_at": datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
    }
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=existing, new_rows=[{"error": "0.006", "created_at": now}]
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.chat",
        reducer_key="chat_session",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # Real cumulative count: 2 (persisted) + 1 (new this tick) = 3, not reset.
    assert baseline.observation_count == 3
    assert baseline.last_value == pytest.approx(0.006)
    assert "persist_baseline" in calls


def test_advance_node_prediction_error_baseline_skips_malformed_rows_but_advances_cursor(
    monkeypatch,
) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    fake_engine, conn, calls = _mock_engine_for_baseline(
        existing_row=None,
        new_rows=[
            {"error": None, "created_at": now},
            {"error": "not-a-number", "created_at": now},
            {"error": "0.3", "created_at": now},
        ],
    )
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.route",
        reducer_key="route_arbitration",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    # Only the one real parseable value is folded -- malformed rows are
    # skipped for the fold, but the batch (and thus the cursor) is still
    # consumed and persisted, per the method's own no-permanent-wedge contract.
    assert baseline.observation_count == 1
    assert baseline.last_value == pytest.approx(0.3)
    assert "persist_baseline" in calls


def test_advance_node_prediction_error_baseline_degrades_to_cold_start_on_error(
    monkeypatch,
) -> None:
    store = AttentionRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    fake_engine.begin.side_effect = RuntimeError("boom")
    store._engine = fake_engine

    baseline = store.advance_node_prediction_error_baseline(
        target_id="node:substrate.bus_synaptic",
        reducer_key="bus_synaptic",
        alpha=0.2,
        min_variance=1e-5,
        fetch_limit=200,
    )

    assert baseline.observation_count == 0
    assert baseline.last_value is None
