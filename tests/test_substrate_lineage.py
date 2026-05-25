from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.biometrics_loop.lineage import (
    emission_touches_node,
    receipt_touches_node,
    state_deltas_for_node,
)


FIXED_TS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)


def test_receipt_touches_node_via_state_delta() -> None:
    receipt = ReductionReceiptV1(
        receipt_id="rcpt_1",
        created_at=FIXED_TS,
        state_deltas=[
            StateDeltaV1(
                delta_id="delta_1",
                target_projection="proj",
                target_kind="active_node_pressure",
                target_id="atlas",
                operation="create",
                caused_by_event_ids=["gev_1"],
                reducer_id="node_pressure_reducer",
            )
        ],
    )
    assert receipt_touches_node(receipt, "atlas")
    assert not receipt_touches_node(receipt, "circe")


def test_emission_touches_node_via_trace() -> None:
    emission = OrganEmissionV1.model_validate(
        {
            "emission_id": "oem_1",
            "organ_id": "biometrics_pressure",
            "invocation_id": "inv_1",
            "created_at": FIXED_TS.isoformat(),
            "candidate_events": [
                {
                    "event_id": "gev_1",
                    "trace_id": "substrate.pressure:atlas:2026-05-24T12:00:00Z",
                    "event_kind": "atom_emitted",
                    "emitted_at": FIXED_TS.isoformat(),
                    "provenance": {
                        "source_service": "orion-substrate-runtime",
                        "source_component": "biometrics_pressure",
                    },
                }
            ],
        }
    )
    assert emission_touches_node(emission, "atlas")


def test_state_deltas_for_node_filters() -> None:
    receipt = ReductionReceiptV1(
        receipt_id="rcpt_1",
        created_at=FIXED_TS,
        state_deltas=[
            StateDeltaV1(
                delta_id="delta_atlas",
                target_projection="proj",
                target_kind="node_biometrics",
                target_id="atlas",
                operation="update",
                caused_by_event_ids=["gev_1"],
                reducer_id="biometrics_node_reducer",
            ),
            StateDeltaV1(
                delta_id="delta_circe",
                target_projection="proj",
                target_kind="node_biometrics",
                target_id="circe",
                operation="update",
                caused_by_event_ids=["gev_2"],
                reducer_id="biometrics_node_reducer",
            ),
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="node_biometrics",
                projection_id="proj",
                node_id="atlas",
                operation="update",
            )
        ],
    )
    deltas = state_deltas_for_node(receipt, "atlas")
    assert len(deltas) == 1
    assert deltas[0]["delta_id"] == "delta_atlas"
