from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
SQL_WRITER_ROOT = THIS_DIR.parent
REPO_ROOT = SQL_WRITER_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SQL_WRITER_ROOT))
sys.modules.pop("app", None)
from app.settings import DEFAULT_ROUTE_MAP
from app.worker import (
    MODEL_MAP,
    _normalize_calibration_profile_audit_payload,
    _normalize_endogenous_runtime_audit_payload,
    _normalize_endogenous_runtime_record_payload,
)
from orion.core.schemas.calibration_adoption import CalibrationProfileAuditV1
from orion.core.schemas.endogenous import EndogenousTriggerDecisionV1, EndogenousTriggerRequestV1, EndogenousWorkflowPlanV1
from orion.core.schemas.endogenous_runtime import (
    EndogenousRuntimeAuditV1,
    EndogenousRuntimeExecutionRecordV1,
    EndogenousRuntimeSignalDigestV1,
)


def _runtime_record() -> EndogenousRuntimeExecutionRecordV1:
    req = EndogenousTriggerRequestV1(subject_ref="project:orion")
    decision = EndogenousTriggerDecisionV1(
        request_id=req.request_id,
        outcome="trigger",
        workflow_type="reflective_journal",
        reasons=["bounded"],
    )
    plan = EndogenousWorkflowPlanV1(
        request_id=req.request_id,
        workflow_type="reflective_journal",
        trigger_outcome="trigger",
        reasons=["bounded"],
    )
    return EndogenousRuntimeExecutionRecordV1(
        invocation_surface="operator_review",
        correlation_id="corr-1",
        subject_ref="project:orion",
        trigger_request=req,
        signal_digest=EndogenousRuntimeSignalDigestV1(),
        decision=decision,
        plan=plan,
        mentor_invoked=False,
        materialized_artifact_ids=["a1"],
        execution_success=True,
        audit_events=["calibration_profile_id:phase12-1"],
        created_at=datetime.now(timezone.utc),
    )


def test_route_map_and_model_map_include_endogenous_operational_kinds() -> None:
    assert DEFAULT_ROUTE_MAP["endogenous.runtime.record.v1"] == "EndogenousRuntimeRecordSQL"
    assert DEFAULT_ROUTE_MAP["endogenous.runtime.audit.v1"] == "EndogenousRuntimeAuditSQL"
    assert DEFAULT_ROUTE_MAP["calibration.profile.audit.v1"] == "CalibrationProfileAuditSQL"

    assert "EndogenousRuntimeRecordSQL" in MODEL_MAP
    assert "EndogenousRuntimeAuditSQL" in MODEL_MAP
    assert "CalibrationProfileAuditSQL" in MODEL_MAP


def test_runtime_and_calibration_payload_normalizers_extract_filter_fields() -> None:
    record = _runtime_record()
    norm_record = _normalize_endogenous_runtime_record_payload(record.model_dump(mode="json"))
    assert norm_record["runtime_record_id"] == record.runtime_record_id
    assert norm_record["trigger_outcome"] == "trigger"
    assert norm_record["workflow_type"] == "reflective_journal"
    assert norm_record["calibration_profile_id"] == "phase12-1"

    audit = EndogenousRuntimeAuditV1(
        enabled=True,
        invocation_surface="operator_review",
        status="ok",
        allow_mentor_branch=False,
        allowed_workflow_types=["reflective_journal"],
        runtime_record_id=record.runtime_record_id,
        decision_outcome="trigger",
        workflow_type="reflective_journal",
        actions=["calibration_profile_id:phase12-1"],
    )
    norm_audit = _normalize_endogenous_runtime_audit_payload(audit.model_dump(mode="json"))
    assert norm_audit["runtime_record_id"] == record.runtime_record_id
    assert norm_audit["status"] == "ok"
    assert norm_audit["calibration_profile_id"] == "phase12-1"

    calibration_audit = CalibrationProfileAuditV1(
        event_type="activated",
        operator_id="operator:qa",
        rationale="activate canary",
        profile_id="phase12-1",
    )
    norm_cal = _normalize_calibration_profile_audit_payload(calibration_audit.model_dump(mode="json"))
    assert norm_cal["audit_id"] == calibration_audit.audit_id
    assert norm_cal["event_type"] == "activated"
    assert norm_cal["profile_id"] == "phase12-1"
