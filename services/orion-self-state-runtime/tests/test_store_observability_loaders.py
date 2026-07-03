from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
)

from app.store import SelfStateRuntimeStore


def _store_with_row(row) -> SelfStateRuntimeStore:
    store = SelfStateRuntimeStore.__new__(SelfStateRuntimeStore)
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.first.return_value = row
    store._engine = engine
    return store


def _store_raising(exc: Exception) -> SelfStateRuntimeStore:
    store = SelfStateRuntimeStore.__new__(SelfStateRuntimeStore)
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.side_effect = exc
    store._engine = engine
    return store


def _projection_payload(**overrides) -> dict:
    projection = AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(),
        attended_node_ids=["node:a", "node:b"],
        dwell_ticks=3,
        **overrides,
    )
    return projection.model_dump(mode="json")


# --- load_latest_attention_broadcast ---


def test_attention_broadcast_fresh_row_parses():
    now = datetime.now(timezone.utc)
    store = _store_with_row(
        {"projection_json": _projection_payload(), "generated_at": now}
    )
    projection = store.load_latest_attention_broadcast()
    assert projection is not None
    assert projection.attended_node_ids == ["node:a", "node:b"]
    assert projection.dwell_ticks == 3


def test_attention_broadcast_naive_timestamp_treated_as_utc():
    naive_now = datetime.now(timezone.utc).replace(tzinfo=None)
    store = _store_with_row(
        {"projection_json": _projection_payload(), "generated_at": naive_now}
    )
    assert store.load_latest_attention_broadcast() is not None


def test_attention_broadcast_stale_row_returns_none():
    stale = datetime.now(timezone.utc) - timedelta(seconds=301)
    store = _store_with_row(
        {"projection_json": _projection_payload(), "generated_at": stale}
    )
    assert store.load_latest_attention_broadcast() is None


def test_attention_broadcast_no_row_returns_none():
    store = _store_with_row(None)
    assert store.load_latest_attention_broadcast() is None


def test_attention_broadcast_missing_table_returns_none():
    store = _store_raising(Exception("relation does not exist"))
    assert store.load_latest_attention_broadcast() is None


def test_attention_broadcast_bad_payload_returns_none():
    store = _store_with_row(
        {
            "projection_json": {"not": "a projection"},
            "generated_at": datetime.now(timezone.utc),
        }
    )
    assert store.load_latest_attention_broadcast() is None


# --- load_hub_presence ---


def test_hub_presence_fresh_row_parses_with_as_of():
    now = datetime.now(timezone.utc)
    presence = {"connection_health": "fresh", "last_turn_age_sec": 5}
    store = _store_with_row({"presence_json": dict(presence), "generated_at": now})
    loaded = store.load_hub_presence()
    assert loaded is not None
    assert loaded["connection_health"] == "fresh"
    assert loaded["last_turn_age_sec"] == 5
    assert loaded["as_of"] == now.isoformat()


def test_hub_presence_string_payload_parses():
    now = datetime.now(timezone.utc)
    store = _store_with_row(
        {"presence_json": '{"turns_per_minute": 2.0}', "generated_at": now}
    )
    loaded = store.load_hub_presence()
    assert loaded is not None
    assert loaded["turns_per_minute"] == 2.0
    assert loaded["as_of"] == now.isoformat()


def test_hub_presence_stale_row_returns_none():
    stale = datetime.now(timezone.utc) - timedelta(seconds=601)
    store = _store_with_row(
        {"presence_json": {"connection_health": "fresh"}, "generated_at": stale}
    )
    assert store.load_hub_presence() is None


def test_hub_presence_no_row_returns_none():
    store = _store_with_row(None)
    assert store.load_hub_presence() is None


def test_hub_presence_empty_payload_returns_none():
    store = _store_with_row(
        {"presence_json": {}, "generated_at": datetime.now(timezone.utc)}
    )
    assert store.load_hub_presence() is None


def test_hub_presence_missing_table_returns_none():
    store = _store_raising(Exception("relation does not exist"))
    assert store.load_hub_presence() is None
