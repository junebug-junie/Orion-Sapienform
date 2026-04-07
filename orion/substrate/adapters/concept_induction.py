from __future__ import annotations

from collections import defaultdict

from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    ContradictionNodeV1,
    EvidenceNodeV1,
    HypothesisNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
)

from ._common import make_provenance, make_temporal


def map_concept_profile_to_substrate(
    *,
    profile: ConceptProfile,
    anchor_scope: str = "orion",
    subject_ref: str | None = None,
) -> SubstrateGraphRecordV1:
    resolved_subject_ref = subject_ref or profile.metadata.get("subject_ref") or profile.subject
    nodes = []
    edges = []

    evidence_to_concepts: dict[str, list[str]] = defaultdict(list)
    for concept in profile.concepts:
        concept_node_id = f"sub-concept-{concept.concept_id}"
        nodes.append(
            ConceptNodeV1(
                node_id=concept_node_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=profile.created_at, valid_from=profile.window_start, valid_to=profile.window_end),
                provenance=make_provenance(
                    source_kind="concept_induction.profile",
                    source_channel="orion:concept_induction",
                    producer="concept_induction_adapter",
                    evidence_refs=[str(item.message_id) for item in concept.evidence],
                ),
                label=concept.label,
                definition=str(concept.metadata.get("definition") or "") or None,
                taxonomy_path=[str(concept.type)] if concept.type else [],
                signals={
                    "confidence": concept.confidence,
                    "salience": min(1.0, max(0.0, concept.salience)),
                },
                metadata={"concept_id": concept.concept_id, "aliases": list(concept.aliases), "embedding_ref": concept.embedding_ref},
            )
        )
        for evidence in concept.evidence:
            evidence_node_id = f"sub-evidence-msg-{evidence.message_id}"
            evidence_to_concepts[evidence_node_id].append(concept_node_id)
            nodes.append(
                EvidenceNodeV1(
                    node_id=evidence_node_id,
                    anchor_scope=anchor_scope,
                    subject_ref=resolved_subject_ref,
                    temporal=make_temporal(observed_at=evidence.timestamp),
                    provenance=make_provenance(
                        source_kind="concept_induction.evidence_ref",
                        source_channel=evidence.channel,
                        producer="concept_induction_adapter",
                        correlation_id=str(evidence.correlation_id) if evidence.correlation_id else None,
                        trace_id=evidence.trace_id,
                    ),
                    evidence_type="message_ref",
                    content_ref=str(evidence.message_id),
                    signals={"confidence": concept.confidence, "salience": min(1.0, max(0.0, concept.salience))},
                )
            )

    # Conservative cluster mapping: cluster summaries become hypotheses, not ontology branches.
    for cluster in profile.clusters:
        if not cluster.summary and not cluster.label:
            continue
        hypothesis_node_id = f"sub-hypothesis-cluster-{cluster.cluster_id}"
        nodes.append(
            HypothesisNodeV1(
                node_id=hypothesis_node_id,
                anchor_scope=anchor_scope,
                subject_ref=resolved_subject_ref,
                temporal=make_temporal(observed_at=profile.created_at, valid_from=profile.window_start, valid_to=profile.window_end),
                provenance=make_provenance(
                    source_kind="concept_induction.cluster",
                    source_channel="orion:concept_induction",
                    producer="concept_induction_adapter",
                ),
                hypothesis_text=cluster.summary or f"Cluster: {cluster.label}",
                signals={"confidence": cluster.cohesion_score, "salience": cluster.cohesion_score},
                metadata={"cluster_id": cluster.cluster_id, "cluster_label": cluster.label, "concept_ids": list(cluster.concept_ids)},
            )
        )
        for concept_id in cluster.concept_ids:
            edges.append(
                SubstrateEdgeV1(
                    source=NodeRefV1(node_id=f"sub-concept-{concept_id}", node_kind="concept"),
                    target=NodeRefV1(node_id=hypothesis_node_id, node_kind="hypothesis"),
                    predicate="co_occurs_with",
                    temporal=make_temporal(observed_at=profile.created_at),
                    provenance=make_provenance(
                        source_kind="concept_induction.cluster",
                        source_channel="orion:concept_induction",
                        producer="concept_induction_adapter",
                    ),
                    confidence=cluster.cohesion_score,
                    salience=cluster.cohesion_score,
                )
            )

    for evidence_node_id, concept_node_ids in evidence_to_concepts.items():
        for concept_node_id in concept_node_ids:
            edges.append(
                SubstrateEdgeV1(
                    source=NodeRefV1(node_id=evidence_node_id, node_kind="evidence"),
                    target=NodeRefV1(node_id=concept_node_id, node_kind="concept"),
                    predicate="supports",
                    temporal=make_temporal(observed_at=profile.created_at),
                    provenance=make_provenance(
                        source_kind="concept_induction.support",
                        source_channel="orion:concept_induction",
                        producer="concept_induction_adapter",
                    ),
                )
            )

    return SubstrateGraphRecordV1(
        graph_id=f"sub-graph-concept-profile-{profile.profile_id}",
        anchor_scope=anchor_scope,
        subject_ref=resolved_subject_ref,
        nodes=nodes,
        edges=edges,
        created_at=profile.created_at,
    )


def map_concept_delta_to_substrate(
    *,
    delta: ConceptProfileDelta,
    observed_at,
    anchor_scope: str = "orion",
    subject_ref: str | None = None,
) -> SubstrateGraphRecordV1:
    nodes = []
    edges = []
    if delta.rationale and delta.added and delta.removed:
        involved = [f"sub-concept-{delta.added[0]}", f"sub-concept-{delta.removed[0]}"]
        contradiction_id = f"sub-contradiction-delta-{delta.delta_id}"
        nodes.append(
            ContradictionNodeV1(
                node_id=contradiction_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                temporal=make_temporal(observed_at=observed_at),
                provenance=make_provenance(
                    source_kind="concept_induction.delta",
                    source_channel="orion:concept_induction",
                    producer="concept_induction_adapter",
                    evidence_refs=[str(item.message_id) for item in delta.evidence],
                ),
                summary=delta.rationale,
                involved_node_ids=involved,
                signals={"confidence": 0.6, "salience": 0.5},
                metadata={"delta_id": delta.delta_id, "from_rev": delta.from_rev, "to_rev": delta.to_rev},
            )
        )
        for concept_ref in involved:
            edges.append(
                SubstrateEdgeV1(
                    source=NodeRefV1(node_id=contradiction_id, node_kind="contradiction"),
                    target=NodeRefV1(node_id=concept_ref, node_kind="concept"),
                    predicate="contradicts",
                    temporal=make_temporal(observed_at=observed_at),
                    provenance=make_provenance(
                        source_kind="concept_induction.delta",
                        source_channel="orion:concept_induction",
                        producer="concept_induction_adapter",
                    ),
                )
            )

    return SubstrateGraphRecordV1(
        graph_id=f"sub-graph-concept-delta-{delta.delta_id}",
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        nodes=nodes,
        edges=edges,
        created_at=make_temporal(observed_at=observed_at).observed_at,
    )
