from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.core.schemas.reasoning import ClaimV1, ConceptV1, ReasoningProvenanceV1, RelationV1


def _evidence_refs_from_delta(delta: ConceptProfileDelta) -> list[str]:
    refs: list[str] = []
    for evidence in delta.evidence:
        refs.append(f"message:{evidence.message_id}")
    return refs


def map_concept_profile_to_reasoning(
    profile: ConceptProfile,
    *,
    producer: str = "concept_adapter",
    include_legacy_claims: bool = True,
) -> list[ConceptV1 | ClaimV1 | RelationV1]:
    """Translate concept profile artifacts into canonical concept artifacts (+ bounded legacy compatibility)."""

    artifacts: list[ConceptV1 | ClaimV1 | RelationV1] = []

    for item in profile.concepts:
        concept = ConceptV1(
            anchor_scope=profile.subject,
            subject_ref=f"concept:{item.concept_id}",
            status="provisional",
            authority="local_inferred",
            confidence=item.confidence,
            salience=min(max(item.salience, 0.0), 1.0),
            novelty=0.0,
            risk_tier="low",
            observed_at=profile.created_at,
            provenance=ReasoningProvenanceV1(
                evidence_refs=[f"message:{ev.message_id}" for ev in item.evidence],
                source_channel="orion:concept_induction",
                source_kind="ConceptProfile",
                producer=producer,
                correlation_id=str(item.evidence[0].correlation_id) if item.evidence and item.evidence[0].correlation_id else None,
                trace_id=item.evidence[0].trace_id if item.evidence else None,
            ),
            concept_id=item.concept_id,
            label=item.label,
            aliases=item.aliases,
            concept_type=item.type,
            source_family="concept_induction",
            source_artifact_ref=profile.profile_id,
            metadata={"embedding_ref": item.embedding_ref, **(item.metadata or {})},
        )
        artifacts.append(concept)

        if include_legacy_claims:
            artifacts.append(
                ClaimV1(
                    anchor_scope=profile.subject,
                    subject_ref=f"concept:{item.concept_id}",
                    status="provisional",
                    authority="local_inferred",
                    confidence=item.confidence,
                    salience=min(max(item.salience, 0.0), 1.0),
                    novelty=0.0,
                    risk_tier="low",
                    observed_at=profile.created_at,
                    provenance=concept.provenance,
                    claim_text=f"Concept '{item.label}' ({item.type}) is salient in current profile.",
                    claim_kind="concept_item",
                    qualifiers={"aliases": item.aliases, "embedding_ref": item.embedding_ref},
                )
            )

    cluster_map: dict[str, list[str]] = {cluster.cluster_id: list(cluster.concept_ids) for cluster in profile.clusters}
    for artifact in artifacts:
        if isinstance(artifact, ConceptV1):
            for cluster_id, concept_ids in cluster_map.items():
                if artifact.concept_id in concept_ids and cluster_id not in artifact.cluster_refs:
                    artifact.cluster_refs.append(cluster_id)

    for cluster in profile.clusters:
        for concept_id in cluster.concept_ids:
            artifacts.append(
                RelationV1(
                    anchor_scope=profile.subject,
                    subject_ref=f"cluster:{cluster.cluster_id}",
                    status="proposed",
                    authority="local_inferred",
                    confidence=cluster.cohesion_score,
                    salience=cluster.cohesion_score,
                    novelty=0.0,
                    risk_tier="low",
                    observed_at=profile.created_at,
                    provenance=ReasoningProvenanceV1(
                        evidence_refs=[f"profile:{profile.profile_id}"],
                        source_channel="orion:concept_induction",
                        source_kind="ConceptProfileCluster",
                        producer=producer,
                    ),
                    source_ref=f"cluster:{cluster.cluster_id}",
                    target_ref=f"concept:{concept_id}",
                    relation_type="related_to",
                    directed=True,
                    metadata={"cluster_label": cluster.label, "cohesion_score": cluster.cohesion_score},
                )
            )

    return artifacts


def map_concept_delta_to_reasoning(
    delta: ConceptProfileDelta,
    *,
    subject: str,
    observed_at: datetime | None = None,
    producer: str = "concept_adapter",
) -> list[ClaimV1]:
    observed = observed_at or datetime.now(timezone.utc)
    refs = _evidence_refs_from_delta(delta)

    claims: list[ClaimV1] = []
    for change_kind, concepts in (("added", delta.added), ("removed", delta.removed), ("updated", delta.updated)):
        for concept_id in concepts:
            claims.append(
                ClaimV1(
                    anchor_scope=subject,
                    subject_ref=f"concept:{concept_id}",
                    status="proposed",
                    authority="local_inferred",
                    confidence=0.6,
                    salience=0.5,
                    novelty=0.5,
                    risk_tier="low",
                    observed_at=observed,
                    provenance=ReasoningProvenanceV1(
                        evidence_refs=refs,
                        source_channel="orion:concept_induction",
                        source_kind="ConceptProfileDelta",
                        producer=producer,
                    ),
                    claim_text=f"Concept delta indicates '{change_kind}' for {concept_id}.",
                    claim_kind="concept_delta",
                    qualifiers={"delta_id": delta.delta_id, "from_rev": delta.from_rev, "to_rev": delta.to_rev},
                )
            )
    return claims
