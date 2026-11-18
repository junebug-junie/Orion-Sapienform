"""#2534 decision 2: the worker hands build_attention_frame the field tick the
previous frame was built from -- from its own cache, else one primary-key read."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1

from app.store import AttentionRuntimeStore
from app.worker import AttentionRuntimeWorker

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)


def _frame(tick_id: str) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id=f"f:{tick_id}", generated_at=NOW, source_field_tick_id=tick_id,
        source_field_generated_at=NOW, overall_salience=0.0,
    )


def _worker(cached: FieldStateV1 | None) -> AttentionRuntimeWorker:
    w = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    w._store = MagicMock(spec=AttentionRuntimeStore)
    w._last_field = cached
    return w


def test_cache_hit_needs_no_query() -> None:
    field = FieldStateV1(tick_id="t1", generated_at=NOW)
    w = _worker(field)
    assert w._previous_field_for(_frame("t1")) is field
    w._store.load_field_for_tick.assert_not_called()


def test_mismatch_or_restart_reads_the_previous_frames_own_tick() -> None:
    loaded = FieldStateV1(tick_id="t1", generated_at=NOW)
    for cached in (None, FieldStateV1(tick_id="t0", generated_at=NOW)):
        w = _worker(cached)
        w._store.load_field_for_tick.return_value = loaded
        assert w._previous_field_for(_frame("t1")) is loaded
        w._store.load_field_for_tick.assert_called_once_with("t1")


def test_no_previous_frame_and_load_failure_fall_back_to_none() -> None:
    w = _worker(None)
    assert w._previous_field_for(None) is None
    w._store.load_field_for_tick.side_effect = RuntimeError("db down")
    assert w._previous_field_for(_frame("t1")) is None


def test_load_field_for_tick_reads_by_primary_key() -> None:
    import json

    store = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    conn = MagicMock()
    conn.execute.return_value.mappings.return_value.first.return_value = {
        "field_json": json.dumps({"tick_id": "t9", "generated_at": NOW.isoformat()})
    }
    store._engine = MagicMock()
    store._engine.connect.return_value.__enter__.return_value = conn
    out = store.load_field_for_tick("t9")
    assert out is not None and out.tick_id == "t9"
    sql = str(conn.execute.call_args[0][0])
    assert "WHERE tick_id = :tick_id" in sql
    assert conn.execute.call_args[0][1] == {"tick_id": "t9"}
