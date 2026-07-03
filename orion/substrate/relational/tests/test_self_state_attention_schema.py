"""Tests for attention schema derivation in the self-state build path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateV1
from orion.self_state.builder import build_self_state, derive_attention_schema
from orion.self_state.policy import load_self_state_policy

REPO = Path(__file__).resolve().parents[4]
ATTENTION_POLICY = load_attention_policy(
    REPO / "config" / "attention" / "field_attention_policy.v1.yaml"
)
SELF_POLICY = load_self_state_policy(
    REPO / "config" / "self_state" / "self_state_policy.v1.yaml"
)
NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def _projection(
    *,
    attended_node_ids: list[str],
    dwell_ticks: int = 0,
    open_loops: list[OpenLoopV1] | None = None,
) -> AttentionBroadcastProjectionV1:
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=open_loops or []),
        attended_node_ids=attended_node_ids,
        dwell_ticks=dwell_ticks,
    )


def _open_loop(loop_id: str = "loop:1") -> OpenLoopV1:
    return OpenLoopV1(id=loop_id, description="unresolved thread")


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_attention_schema",
        node_vectors={
            "node:athena": {
                "execution_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1"],
    )


def _build_state(**kwargs) -> SelfStateV1:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    return build_self_state(
        field=field,
        attention=attention,
        policy=SELF_POLICY,
        now=NOW,
        **kwargs,
    )


# --- derive_attention_schema: derivation rules ---


def test_derive_absent_projection_returns_defaults():
    assert derive_attention_schema(None) == (None, 0, 0)


def test_derive_focused_single_for_one_node():
    projection = _projection(attended_node_ids=["node:a"], dwell_ticks=4)
    assert derive_attention_schema(projection) == ("focused_single", 4, 1)


def test_derive_distributed_for_two_nodes():
    projection = _projection(attended_node_ids=["node:a", "node:b"], dwell_ticks=2)
    assert derive_attention_schema(projection) == ("distributed", 2, 2)


def test_derive_distributed_for_many_nodes():
    projection = _projection(
        attended_node_ids=[f"node:{i}" for i in range(5)], dwell_ticks=7
    )
    assert derive_attention_schema(projection) == ("distributed", 7, 5)


def test_derive_open_loop_for_zero_nodes_with_open_loops():
    projection = _projection(
        attended_node_ids=[], dwell_ticks=1, open_loops=[_open_loop()]
    )
    assert derive_attention_schema(projection) == ("open_loop", 1, 0)


def test_derive_none_for_zero_nodes_without_open_loops():
    projection = _projection(attended_node_ids=[], dwell_ticks=0)
    assert derive_attention_schema(projection) == ("none", 0, 0)


def test_derive_clamps_negative_dwell_ticks():
    projection = _projection(attended_node_ids=["node:a"], dwell_ticks=-3)
    assert derive_attention_schema(projection) == ("focused_single", 0, 1)


# --- build_self_state: population and graceful degradation ---


def test_build_self_state_without_broadcast_keeps_schema_defaults():
    state = _build_state()
    assert state.attention_schema_type is None
    assert state.attention_dwell_ticks == 0
    assert state.attention_node_count == 0


def test_build_self_state_populates_focused_single():
    projection = _projection(attended_node_ids=["node:athena"], dwell_ticks=6)
    state = _build_state(attention_broadcast=projection)
    assert state.attention_schema_type == "focused_single"
    assert state.attention_dwell_ticks == 6
    assert state.attention_node_count == 1


def test_build_self_state_populates_distributed():
    projection = _projection(
        attended_node_ids=["node:a", "node:b", "node:c"], dwell_ticks=3
    )
    state = _build_state(attention_broadcast=projection)
    assert state.attention_schema_type == "distributed"
    assert state.attention_dwell_ticks == 3
    assert state.attention_node_count == 3


def test_build_self_state_populates_open_loop():
    projection = _projection(attended_node_ids=[], open_loops=[_open_loop()])
    state = _build_state(attention_broadcast=projection)
    assert state.attention_schema_type == "open_loop"
    assert state.attention_node_count == 0


def test_build_self_state_populates_none_when_idle():
    projection = _projection(attended_node_ids=[])
    state = _build_state(attention_broadcast=projection)
    assert state.attention_schema_type == "none"
    assert state.attention_dwell_ticks == 0
    assert state.attention_node_count == 0


# --- schema-level checks (retained from v1) ---


def test_attention_schema_all_types_valid():
    """All attention schema type values are valid on SelfStateV1."""
    now = datetime.now(timezone.utc)
    for schema_type in ["focused_single", "distributed", "open_loop", "none", "unknown"]:
        state = SelfStateV1(
            self_state_id="s1",
            generated_at=now,
            source_field_tick_id="t1",
            source_field_generated_at=now,
            source_attention_frame_id="f1",
            source_attention_generated_at=now,
            overall_intensity=0.5,
            overall_confidence=0.5,
            attention_schema_type=schema_type,
        )
        assert state.attention_schema_type == schema_type
