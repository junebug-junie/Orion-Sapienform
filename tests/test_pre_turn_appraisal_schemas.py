from __future__ import annotations

from orion.schemas.pre_turn_appraisal import (
    PreTurnAppraisalRequestV1,
    TurnAppraisalBundleV1,
    TurnAppraisalParadigmSliceV1,
    TurnWindowMessageV1,
)


def test_request_round_trip_minimal() -> None:
    req = PreTurnAppraisalRequestV1(
        correlation_id="corr-1",
        session_id="sess-1",
        turn_window=[
            TurnWindowMessageV1(role="user", content="give me nuts and bolts"),
            TurnWindowMessageV1(role="assistant", content="here is a high level plan"),
        ],
        paradigms_requested=["repair_pressure"],
        contract_before={"mode": "default"},
    )
    data = req.model_dump(mode="json")
    assert PreTurnAppraisalRequestV1.model_validate(data).correlation_id == "corr-1"


def test_bundle_carries_metadata_attachments() -> None:
    bundle = TurnAppraisalBundleV1(
        correlation_id="corr-1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.82,
                confidence=0.71,
                dimensions={"specificity_demand": 0.91},
                evidence=[],
                contract_delta={"mode": "repair_concrete", "rules": ["include file/module boundaries"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        },
        grammar_scalars={"repair_pressure": {"level": 0.82, "confidence": 0.71}},
    )
    assert bundle.metadata_attachments["repair_pressure_contract"]["mode"] == "repair_concrete"
