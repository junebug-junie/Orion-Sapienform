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
