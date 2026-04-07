from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.core.schemas.mentor import (
    MentorProposalItemV1,
    MentorRequestV1,
    MentorResponseV1,
)
from orion.core.schemas.reasoning import ClaimV1, MentorProposalV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_policy import PromotionEvaluationRequestV1
from orion.reasoning.materializer import ReasoningMaterializer
from orion.reasoning.mentor_context import build_mentor_context
from orion.reasoning.mentor_gateway import MentorGateway
from orion.reasoning.mentor_mapper import map_mentor_response_to_proposals
from orion.reasoning.promotion import PromotionEngine
from orion.reasoning.repository import InMemoryReasoningRepository


def _seed_repo() -> InMemoryReasoningRepository:
    repo = InMemoryReasoningRepository()
    claim = ClaimV1(
        anchor_scope="orion",
        subject_ref="concept:continuity",
        status="provisional",
        authority="local_inferred",
        confidence=0.8,
        salience=0.7,
        novelty=0.3,
        risk_tier="low",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["chat:1"],
            "source_channel": "orion:test",
            "source_kind": "unit",
            "producer": "pytest",
        },
        claim_text="continuity pressure",
        claim_kind="assertion",
    )
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(
                source_family="manual",
                source_kind="seed",
                source_channel="orion:test",
                producer="pytest",
            ),
            artifacts=[claim],
        )
    )
    return repo


def test_mentor_contracts_validate_and_reject_malformed() -> None:
    req = MentorRequestV1(
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="ontology_cleanup",
        anchor_scope="orion",
    )
    assert req.constraints.output_json_only is True

    ok = MentorResponseV1(
        proposal_batch_id="batch-1",
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="ontology_cleanup",
        proposals=[
            MentorProposalItemV1(
                proposal_id="p1",
                proposal_type="merge",
                confidence=0.7,
                rationale="duplicate concept",
            )
        ],
    )
    assert ok.proposals

    with pytest.raises(ValidationError):
        MentorResponseV1(
            proposal_batch_id="batch-bad",
            mentor_provider="openai",
            mentor_model="gpt-test",
            task_type="ontology_cleanup",
            proposals=[{"proposal_id": "p1"}],
        )


def test_context_packaging_is_bounded_and_inspectable() -> None:
    repo = _seed_repo()
    req = MentorRequestV1(
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="missing_evidence_scan",
        anchor_scope="orion",
    )
    context, packet = build_mentor_context(req, repo, max_artifacts=5)
    assert len(context.artifact_ids) <= 5
    assert packet
    assert set(packet[0].keys()) >= {"artifact_id", "artifact_type", "anchor_scope", "status"}


def test_mapping_to_mentor_proposals_preserves_advisory_provenance() -> None:
    req = MentorRequestV1(
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="contradiction_review",
        anchor_scope="orion",
        correlation_id="corr-1",
    )
    req = req.model_copy(update={"context": req.context.model_copy(update={"artifact_ids": ["artifact-a"]})})
    response = MentorResponseV1(
        proposal_batch_id="batch-1",
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="contradiction_review",
        proposals=[
            MentorProposalItemV1(
                proposal_id="proposal-1",
                proposal_type="flag_contradiction",
                confidence=0.8,
                rationale="claims conflict",
                evidence_refs=["artifact-a"],
                suggested_payload={"contradiction_flags": ["c1"]},
                risk_tier="high",
            )
        ],
    )
    mapped = map_mentor_response_to_proposals(req, response)
    assert mapped and isinstance(mapped[0], MentorProposalV1)
    assert mapped[0].status == "proposed"
    assert mapped[0].authority == "mentor_inferred"
    assert mapped[0].provenance.model == "gpt-test"


def test_promotion_handles_mentor_proposals_conservatively() -> None:
    repo = _seed_repo()
    mentor = MentorProposalV1(
        anchor_scope="orion",
        subject_ref="self:orion",
        status="proposed",
        authority="mentor_inferred",
        confidence=0.8,
        salience=0.7,
        novelty=0.4,
        risk_tier="high",
        observed_at=datetime.now(timezone.utc),
        provenance={
            "evidence_refs": ["artifact-1"],
            "source_channel": "orion:mentor:gateway",
            "source_kind": "MentorResponse:contradiction_review",
            "producer": "mentor_gateway",
        },
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="contradiction_review",
        proposal_type="identity_revision",
        rationale="identity conflict",
        suggested_payload={"contradiction_flags": ["c1"]},
    )
    repo.write_artifacts(
        ReasoningWriteRequestV1(
            context=ReasoningWriteContextV1(source_family="other", source_kind="mentor", source_channel="orion:test", producer="pytest"),
            artifacts=[mentor],
        )
    )
    engine = PromotionEngine(repo)
    canonical_attempt = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[mentor.artifact_id], target_status="canonical"))
    provisional_attempt = engine.evaluate(PromotionEvaluationRequestV1(artifact_ids=[mentor.artifact_id], target_status="provisional"))
    assert canonical_attempt.items[0].outcome == "blocked"
    assert provisional_attempt.items[0].outcome == "blocked"


class _GoodProvider:
    def run(self, request: MentorRequestV1, *, context_packet: list[dict]) -> MentorResponseV1:
        assert context_packet is not None
        return MentorResponseV1(
            proposal_batch_id="batch-good",
            mentor_provider=request.mentor_provider,
            mentor_model=request.mentor_model,
            task_type=request.task_type,
            proposals=[
                MentorProposalItemV1(
                    proposal_id="proposal-good",
                    proposal_type="missing_evidence",
                    confidence=0.6,
                    rationale="need more evidence",
                    evidence_refs=request.context.artifact_ids[:1],
                    suggested_payload={"missing_evidence": ["event_x"]},
                    risk_tier="low",
                )
            ],
        )


class _FailProvider:
    def run(self, request: MentorRequestV1, *, context_packet: list[dict]) -> MentorResponseV1:
        raise RuntimeError("provider_down")


def test_gateway_request_to_materialized_proposals_flow_and_failure_safety() -> None:
    repo = _seed_repo()
    gateway = MentorGateway(repository=repo, materializer=ReasoningMaterializer(repo), provider=_GoodProvider())

    req = MentorRequestV1(
        mentor_provider="openai",
        mentor_model="gpt-test",
        task_type="missing_evidence_scan",
        anchor_scope="orion",
    )
    result = gateway.execute(req)
    assert result.success is True
    assert result.materialized_count == 1
    stored = repo.list_by_type("mentor_proposal")
    assert stored and stored[0].status == "proposed"

    failing = MentorGateway(repository=repo, materializer=ReasoningMaterializer(repo), provider=_FailProvider())
    fail_result = failing.execute(req)
    assert fail_result.success is False
    assert fail_result.failure_reason == "provider_down"
