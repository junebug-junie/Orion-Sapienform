"""Schema tests for investigation_v2 mode and profile permissions."""

from __future__ import annotations

from orion.schemas.context_exec import (
    ContextExecRequestV1,
    EvidenceBundle,
    InvestigationReportV2,
    SourceResult,
    SourceStatus,
    context_exec_permissions_for_llm_profile,
)


def test_context_exec_request_accepts_investigation_v2_mode() -> None:
    req = ContextExecRequestV1(text="probe", mode="investigation_v2")
    assert req.mode == "investigation_v2"


def test_context_exec_permissions_for_llm_profile_agent_read_repo() -> None:
    perms = context_exec_permissions_for_llm_profile("agent")
    assert perms.read_repo is True


def test_evidence_bundle_and_source_result_schema() -> None:
    bundle = EvidenceBundle(
        repo=SourceResult(source="repo", status=SourceStatus.hit, findings=[{"claim": "x"}]),
        recall=SourceResult(source="recall", status=SourceStatus.unavailable, error="timeout"),
    )
    report = InvestigationReportV2(
        answer_status="partial_grounding",
        summary="test",
        sources={"repo": "hit", "recall": "unavailable"},
        evidence=bundle,
    )
    data = report.model_dump(mode="json")
    restored = InvestigationReportV2.model_validate(data)
    assert restored.evidence.repo is not None
    assert restored.evidence.repo.status == SourceStatus.hit

