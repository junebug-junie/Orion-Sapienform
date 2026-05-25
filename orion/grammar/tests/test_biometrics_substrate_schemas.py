from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1


def test_organ_emission_roundtrip() -> None:
    now = datetime.now(timezone.utc)
    raw = OrganEmissionV1(
        emission_id="oem_test",
        organ_id="biometrics_pressure",
        invocation_id="inv_test",
        triggered_by_event_ids=["gev_1"],
        inspected_projection_ids=["proj_node_bio"],
        candidate_events=[],
        created_at=now,
    ).model_dump(mode="json")
    assert OrganEmissionV1.model_validate(raw).organ_id == "biometrics_pressure"


def test_reduction_receipt_requires_schema_version() -> None:
    now = datetime.now(timezone.utc)
    r = ReductionReceiptV1(
        receipt_id="rcpt_test",
        accepted_event_ids=[],
        rejected_event_ids=[],
        merged_event_ids=[],
        noop_event_ids=[],
        state_deltas=[],
        projection_updates=[],
        created_at=now,
    )
    assert r.schema_version == "substrate.reduction_receipt.v1"


def test_node_biometrics_projection_defaults() -> None:
    now = datetime.now(timezone.utc)
    p = NodeBiometricsProjectionV1(
        projection_id="proj_node_bio",
        generated_at=now,
        nodes={
            "atlas": NodeBiometricsStateV1(node_id="atlas"),
        },
    )
    assert p.nodes["atlas"].availability_status == "unknown"


def test_state_delta_roundtrip() -> None:
    d = StateDeltaV1(
        delta_id="delta_1",
        target_projection="node_biometrics",
        target_kind="node",
        target_id="atlas",
        operation="update",
        caused_by_event_ids=["gev_1"],
        reducer_id="biometrics_node_reducer",
    )
    assert StateDeltaV1.model_validate(d.model_dump(mode="json")).operation == "update"


def test_projection_update_roundtrip() -> None:
    u = ProjectionUpdateV1(
        projection_kind="active_node_pressure",
        projection_id="proj_pressure",
        node_id="atlas",
        operation="reinforce",
    )
    assert ProjectionUpdateV1.model_validate(u.model_dump(mode="json")).node_id == "atlas"
