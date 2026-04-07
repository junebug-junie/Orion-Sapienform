from __future__ import annotations

from typing import Any

from orion.core.schemas.mentor import MentorContextSliceV1, MentorRequestV1
from orion.reasoning.repository import InMemoryReasoningRepository


def build_mentor_context(
    request: MentorRequestV1,
    repository: InMemoryReasoningRepository,
    *,
    max_artifacts: int = 12,
) -> tuple[MentorContextSliceV1, list[dict[str, Any]]]:
    """Build bounded, inspectable context packet for mentor tasks."""

    selected = []
    artifacts = repository.list_latest(limit=max_artifacts * 2)
    for artifact in artifacts:
        if len(selected) >= max_artifacts:
            break
        if request.anchor_scope != "world" and artifact.anchor_scope != request.anchor_scope:
            continue
        if request.subject_ref and artifact.subject_ref and artifact.subject_ref != request.subject_ref:
            continue
        selected.append(artifact)

    artifact_ids = [a.artifact_id for a in selected]
    evidence_refs: list[str] = []
    for artifact in selected:
        evidence_refs.extend([ref for ref in artifact.provenance.evidence_refs if ref])

    context = MentorContextSliceV1(
        artifact_ids=artifact_ids,
        evidence_refs=list(dict.fromkeys(evidence_refs))[:40],
        summary_refs=[f"scope:{request.anchor_scope}", f"task:{request.task_type}"],
    )

    packet = [
        {
            "artifact_id": a.artifact_id,
            "artifact_type": a.artifact_type,
            "anchor_scope": a.anchor_scope,
            "subject_ref": a.subject_ref,
            "status": a.status,
            "risk_tier": a.risk_tier,
            "confidence": a.confidence,
            "salience": a.salience,
            "source_kind": a.provenance.source_kind,
            "evidence_refs": a.provenance.evidence_refs[:8],
        }
        for a in selected
    ]
    return context, packet
