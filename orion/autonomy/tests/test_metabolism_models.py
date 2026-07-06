from orion.autonomy.models import MetabolismResultV1
from orion.core.schemas.drives import TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def test_metabolism_result_v1_accepts_tensions_and_curiosity() -> None:
    tension = TensionEventV1(
        artifact_id="tension-gap-1",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="substrate.world_coverage_gap",
        magnitude=0.65,
        drive_impacts={"predictive": 0.15},
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    signal = FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="hardware_compute_gpu had zero digest items",
        confidence=0.7,
        notes=["run_id:wp-run-gap-1"],
    )
    result = MetabolismResultV1(
        drive_deltas={"predictive": 0.15},
        tensions=[tension],
        curiosity_signals=[signal],
    )
    assert result.drive_deltas["predictive"] == 0.15
    assert result.tensions[0].kind == "substrate.world_coverage_gap"
    assert result.curiosity_signals[0].signal_type == "world_coverage_gap"
