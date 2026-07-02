from __future__ import annotations

import json

from orion.substrate.relational.adapters.attention_ctx import (
    map_attention_broadcast_ctx_to_substrate,
)


def _payload(**overrides):
    payload = {
        "selected_action_type": "invoke",
        "selected_open_loop_id": "loop_1",
        "selected_description": "sustained execution prediction error",
        "attended_node_ids": [f"node:{i}" for i in range(12)],
    }
    payload.update(overrides)
    return payload


def test_maps_broadcast_to_single_attending_node():
    record = map_attention_broadcast_ctx_to_substrate({"attention_broadcast": _payload()})
    assert record is not None
    assert record.anchor_scope == "orion"
    assert len(record.nodes) == 1
    node = record.nodes[0]
    assert node.label == "attending:current_focus"
    assert node.metadata["selected_action_type"] == "invoke"
    assert node.metadata["selected_description"] == "sustained execution prediction error"
    # Hard cap: at most 8 attended node ids in metadata.
    assert node.metadata["attended_node_ids"] == [f"node:{i}" for i in range(8)]


def test_accepts_json_string_payload():
    record = map_attention_broadcast_ctx_to_substrate(
        {"attention_broadcast": json.dumps(_payload())}
    )
    assert record is not None
    assert record.nodes[0].metadata["selected_action_type"] == "invoke"


def test_returns_none_when_nothing_attended():
    record = map_attention_broadcast_ctx_to_substrate(
        {"attention_broadcast": _payload(selected_action_type="none", attended_node_ids=[])}
    )
    assert record is None


def test_returns_none_on_missing_or_garbage_ctx():
    assert map_attention_broadcast_ctx_to_substrate({}) is None
    assert map_attention_broadcast_ctx_to_substrate({"attention_broadcast": "not json"}) is None
    assert map_attention_broadcast_ctx_to_substrate(None) is None
