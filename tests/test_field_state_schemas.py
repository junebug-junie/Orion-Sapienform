from datetime import datetime, timezone

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1


def test_field_state_v1_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    state = FieldStateV1(
        generated_at=now,
        tick_id="tick_abc123",
        node_vectors={
            "node:atlas": {
                "availability": 1.0,
                "gpu_pressure": 0.72,
                "memory_pressure": 0.31,
            }
        },
        capability_vectors={
            "capability:llm_inference": {
                "pressure": 0.61,
                "confidence": 0.78,
                "available_capacity": 0.39,
            }
        },
        edges=[
            FieldEdgeV1(
                source_id="node:atlas",
                target_id="capability:llm_inference",
                edge_type="node_capability",
                weight=0.85,
                channel_map={"gpu_pressure": "pressure"},
            )
        ],
        recent_perturbations=["state_delta:atlas_gpu_pressure_reinforced"],
    )
    payload = state.model_dump(mode="json")
    restored = FieldStateV1.model_validate(payload)
    assert restored.schema_version == "field.state.v1"
    assert restored.node_vectors["node:atlas"]["gpu_pressure"] == 0.72
    assert restored.edges[0].weight == 0.85


def test_field_state_accepts_topology_metadata() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    state = FieldStateV1(
        generated_at=now,
        tick_id="tick_meta",
        topology_id="orion_field_topology",
        topology_version="v1",
        topology_loaded_from="config/field/orion_field_topology.v1.yaml",
    )
    restored = FieldStateV1.model_validate(state.model_dump(mode="json"))
    assert restored.topology_id == "orion_field_topology"


def test_field_state_v1_carries_sustained_load_pressure_identity() -> None:
    """2026-09-07: `sustained_load_pressure_channel`/`_node_id` round-trip
    alongside the pre-existing scalar."""
    now = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
    state = FieldStateV1(
        generated_at=now,
        tick_id="tick_identity",
        sustained_load_pressure=0.71,
        sustained_load_pressure_channel="disk_capacity_pressure",
        sustained_load_pressure_node_id="node:athena",
    )
    restored = FieldStateV1.model_validate(state.model_dump(mode="json"))
    assert restored.sustained_load_pressure == 0.71
    assert restored.sustained_load_pressure_channel == "disk_capacity_pressure"
    assert restored.sustained_load_pressure_node_id == "node:athena"


def test_field_state_v1_pre_migration_payload_reads_with_no_fabricated_identity() -> None:
    """Backward-read compat: a real pre-2026-09-07 payload has
    `sustained_load_pressure` but no identity keys at all in the JSON blob
    (`model_config = ConfigDict(extra="forbid")` only rejects UNKNOWN
    fields, not missing ones, so this must validate cleanly). The new
    fields must default to `None`, not raise and not invent a value."""
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    pre_migration_payload = {
        "schema_version": "field.state.v1",
        "generated_at": now.isoformat(),
        "tick_id": "tick_pre_migration",
        "sustained_load_pressure": 0.71,
        # No sustained_load_pressure_channel / _node_id keys at all.
    }
    restored = FieldStateV1.model_validate(pre_migration_payload)
    assert restored.sustained_load_pressure == 0.71
    assert restored.sustained_load_pressure_channel is None
    assert restored.sustained_load_pressure_node_id is None
