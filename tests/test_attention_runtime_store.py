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
        node_vectors={"node:athena": {"execution_load": 0.5}},
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
