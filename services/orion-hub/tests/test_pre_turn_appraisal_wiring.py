from __future__ import annotations

from orion.schemas.cortex.contracts import CortexChatRequest
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1, TurnAppraisalParadigmSliceV1
from orion.substrate.appraisal import REPAIR_PRESSURE_CONTRACT_METADATA_KEY
from scripts.pre_turn_appraisal_wiring import apply_pre_turn_appraisal_bundle


def test_apply_bundle_attaches_metadata_when_mode_changes() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(
        correlation_id="c1",
        paradigms={
            "repair_pressure": TurnAppraisalParadigmSliceV1(
                appraisal_kind="repair_pressure",
                level=0.82,
                confidence=0.71,
                contract_delta={"mode": "repair_concrete", "rules": ["include file/module boundaries"]},
            )
        },
        metadata_attachments={
            "repair_pressure_contract": {"mode": "repair_concrete", "rules": ["include file/module boundaries"]}
        },
        grammar_scalars={"repair_pressure": {"level": 0.82, "confidence": 0.71}},
    )
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=True)
    assert req.metadata[REPAIR_PRESSURE_CONTRACT_METADATA_KEY]["mode"] == "repair_concrete"
    assert summary is not None
    assert summary["level"] == 0.82
    assert summary["changed_behavior"] == "repair_concrete"


def test_apply_bundle_skips_when_disabled() -> None:
    req = CortexChatRequest(prompt="hi", mode="brain")
    bundle = TurnAppraisalBundleV1(correlation_id="c1")
    summary = apply_pre_turn_appraisal_bundle(req, bundle, enabled=False)
    assert summary is None
    assert req.metadata is None
