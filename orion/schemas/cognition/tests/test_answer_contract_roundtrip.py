from __future__ import annotations

from orion.schemas.cognition.answer_contract import (
    AnswerContract,
    AnswerContractDraft,
    AnswerGroundingStatus,
    Finding,
    FindingsBundle,
    GroundingStatus,
    InvestigationState,
    RenderedAnswer,
)


def test_answer_contract_json_roundtrip() -> None:
    obj = AnswerContract(
        request_kind="repo_technical",
        requires_repo_grounding=True,
        preferred_render_style="steps",
    )
    raw = obj.model_dump(mode="json")
    back = AnswerContract.model_validate(raw)
    assert back == obj


def test_findings_bundle_roundtrip() -> None:
    fb = FindingsBundle(
        findings=[Finding(claim="x", evidence_type="repo_file", verified=True, confidence=0.9)],
        missing_evidence=["y"],
        grounded_status="grounded_partial",
    )
    back = FindingsBundle.model_validate(fb.model_dump(mode="json"))
    assert back.findings[0].claim == "x"


def test_grounding_status_alias() -> None:
    g = GroundingStatus(status="insufficient_grounding", reason="x")
    assert isinstance(g, AnswerGroundingStatus)


def test_draft_extra_ignored() -> None:
    d = AnswerContractDraft.model_validate({"request_kind": "conceptual", "unknown_future": 1})
    assert d.request_kind == "conceptual"


def test_rendered_answer_roundtrip() -> None:
    r = RenderedAnswer(text="hi", grounded_status="grounded_complete", claims_used=1)
    assert RenderedAnswer.model_validate(r.model_dump(mode="json")).text == "hi"


def test_investigation_state_roundtrip() -> None:
    inv = InvestigationState(status="repo_required", evidence_acquired=False, findings_count=0)
    assert InvestigationState.model_validate(inv.model_dump(mode="json")).status == "repo_required"
