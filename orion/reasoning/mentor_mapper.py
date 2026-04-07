from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.mentor import MentorRequestV1, MentorResponseV1
from orion.core.schemas.reasoning import MentorProposalV1, ReasoningProvenanceV1


def map_mentor_response_to_proposals(
    request: MentorRequestV1,
    response: MentorResponseV1,
    *,
    producer: str = "mentor_gateway",
) -> list[MentorProposalV1]:
    """Deterministically map mentor response items into advisory MentorProposalV1 artifacts."""

    observed_at = datetime.now(timezone.utc)
    proposals: list[MentorProposalV1] = []
    for item in response.proposals:
        payload = dict(item.suggested_payload or {})
        payload.setdefault("proposal_id", item.proposal_id)
        payload.setdefault("proposal_type", item.proposal_type)

        proposal = MentorProposalV1(
            artifact_id=item.proposal_id,
            anchor_scope=request.anchor_scope,
            subject_ref=request.subject_ref,
            status="proposed",
            authority="mentor_inferred",
            confidence=item.confidence,
            salience=min(1.0, max(0.0, item.confidence)),
            novelty=0.4,
            risk_tier=item.risk_tier,
            observed_at=observed_at,
            provenance=ReasoningProvenanceV1(
                evidence_refs=(request.context.artifact_ids + item.evidence_refs)[:40],
                source_channel="orion:mentor:gateway",
                source_kind=f"MentorResponse:{response.task_type}",
                producer=producer,
                model=response.mentor_model,
                correlation_id=request.correlation_id,
                trace_id=request.request_id,
            ),
            mentor_provider=response.mentor_provider,
            mentor_model=response.mentor_model,
            task_type=response.task_type,
            proposal_type=item.proposal_type,
            rationale=item.rationale,
            suggested_payload=payload,
        )
        proposals.append(proposal)

    return proposals
