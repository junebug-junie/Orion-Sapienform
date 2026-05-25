from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

from app.graph.lattice import load_lattice
from app.tensor.field_state import empty_field_state


def test_empty_field_state_has_all_lattice_nodes() -> None:
    lattice_path = Path("config/field/biometrics_lattice.yaml")
    lattice = load_lattice(lattice_path)
    now = datetime(2026, 5, 24, tzinfo=timezone.utc)
    state = empty_field_state(lattice=lattice, now=now, tick_id="tick_test")
    assert "node:atlas" in state.node_vectors
    assert state.node_vectors["node:atlas"]["availability"] == 1.0
    assert "capability:llm_inference" in state.capability_vectors
    assert len(state.edges) == len(lattice.edges)


def test_decay_fades_pressure_channels() -> None:
    from app.digestion.decay import apply_decay

    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_decay",
        node_vectors={"node:atlas": {"gpu_pressure": 0.8, "availability": 1.0}},
        capability_vectors={},
        edges=[],
    )
    apply_decay(state, decay_rate=0.5)
    assert state.node_vectors["node:atlas"]["gpu_pressure"] == 0.4
    assert state.node_vectors["node:atlas"]["availability"] == 1.0


def test_diffusion_spreads_gpu_pressure_to_capability() -> None:
    from app.digestion.diffusion import apply_diffusion

    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.85,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_diff",
        node_vectors={"node:atlas": {"gpu_pressure": 0.8}},
        capability_vectors={
            "capability:llm_inference": {
                "pressure": 0.0,
                "confidence": 1.0,
                "available_capacity": 1.0,
            }
        },
        edges=[edge],
    )
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.68


def test_suppression_blocks_availability_panic_for_circe() -> None:
    from app.digestion.suppression import apply_suppression

    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_sup",
        node_vectors={
            "node:circe": {
                "availability": 0.2,
                "expected_offline_suppression": 1.0,
                "staleness": 0.9,
            }
        },
        capability_vectors={},
        edges=[],
    )
    apply_suppression(state)
    vec = state.node_vectors["node:circe"]
    assert vec["availability"] >= 0.8
    assert vec["staleness"] == 0.0
