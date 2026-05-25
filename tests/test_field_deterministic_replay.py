from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1

from app.graph.lattice import load_lattice
from app.ingest.state_deltas import delta_to_perturbations
from app.tensor.field_state import empty_field_state
from app.tensor.update_rules import run_digestion_tick


def _replay(receipts: list[ReductionReceiptV1], *, decay_rate: float, diffusion_rate: float):
    lattice = load_lattice(Path("config/field/biometrics_lattice.yaml"))
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    state = empty_field_state(lattice=lattice, now=now, tick_id="tick_replay")
    seen: set[str] = set()
    for receipt in receipts:
        perturbations = []
        for delta in receipt.state_deltas:
            if delta.delta_id in seen:
                continue
            seen.add(delta.delta_id)
            perturbations.extend(delta_to_perturbations(delta))
        run_digestion_tick(
            state,
            perturbations=perturbations,
            decay_rate=decay_rate,
            diffusion_rate=diffusion_rate,
        )
    return state


def _atlas_strain_delta(*, delta_id: str = "delta_replay_atlas") -> StateDeltaV1:
    return StateDeltaV1(
        delta_id=delta_id,
        target_projection="active_node_pressure_projection",
        target_kind="active_node_pressure",
        target_id="atlas",
        operation="reinforce",
        after={
            "node_id": "atlas",
            "active_pressures": ["strain"],
            "pressure_score": 0.72,
            "availability_status": "online",
            "suppressed_pressures": [],
            "capability_impacts": [],
            "evidence_event_ids": [],
            "last_updated_at": "2026-05-24T12:00:00+00:00",
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="node_pressure_reducer",
    )


def test_replay_is_deterministic_for_same_receipts() -> None:
    delta = _atlas_strain_delta()
    receipt = ReductionReceiptV1(
        receipt_id="rcpt_replay",
        accepted_event_ids=["gev_1"],
        state_deltas=[delta],
        created_at=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )
    a = _replay([receipt], decay_rate=0.92, diffusion_rate=1.0)
    b = _replay([receipt], decay_rate=0.92, diffusion_rate=1.0)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    assert a.node_vectors["node:atlas"]["gpu_pressure"] > 0.0
    assert a.capability_vectors["capability:llm_inference"]["pressure"] > 0.0


def test_duplicate_delta_id_skipped_on_replay() -> None:
    delta = _atlas_strain_delta(delta_id="delta_dup")
    receipt_a = ReductionReceiptV1(
        receipt_id="rcpt_a",
        accepted_event_ids=["gev_1"],
        state_deltas=[delta],
        created_at=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )
    receipt_b = ReductionReceiptV1(
        receipt_id="rcpt_b",
        accepted_event_ids=["gev_1"],
        state_deltas=[delta],
        created_at=datetime(2026, 5, 24, 12, 1, tzinfo=timezone.utc),
    )
    single = _replay([receipt_a], decay_rate=1.0, diffusion_rate=1.0)
    duplicate = _replay([receipt_a, receipt_b], decay_rate=1.0, diffusion_rate=1.0)
    assert (
        duplicate.node_vectors["node:atlas"]["gpu_pressure"]
        == single.node_vectors["node:atlas"]["gpu_pressure"]
    )
