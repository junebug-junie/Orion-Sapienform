from __future__ import annotations

from orion.core.schemas.cognitive_substrate import BaseSubstrateNodeV1
from orion.core.schemas.frontier_expansion import (
    FrontierDeltaItemV1,
    FrontierExpansionRequestV1,
    FrontierExpansionResponseV1,
    FrontierGraphDeltaBundleV1,
    FrontierSourceProvenanceV1,
    LandingPostureV1,
)


_ZONE_POSTURE: dict[str, LandingPostureV1] = {
    "world_ontology": "fast_track_proposal",
    "concept_graph": "moderate_proposal",
    "autonomy_graph": "conservative_proposal",
    "self_relationship_graph": "strict_proposal_only",
}


def _proposalize_node(node: BaseSubstrateNodeV1, *, strict: bool) -> BaseSubstrateNodeV1:
    metadata = dict(node.metadata)
    metadata["frontier_proposal_only"] = True
    metadata["frontier_identity_protected"] = strict
    update = {
        "promotion_state": "proposed",
        "metadata": metadata,
    }
    if strict:
        update["risk_tier"] = "high"
    return node.model_copy(update=update)


class FrontierDeltaMapper:
    """Deterministic mapping from structured frontier response into substrate-native graph delta bundle."""

    def map_response(
        self,
        *,
        request: FrontierExpansionRequestV1,
        response: FrontierExpansionResponseV1,
    ) -> FrontierGraphDeltaBundleV1:
        strict_zone = request.target_zone in {"autonomy_graph", "self_relationship_graph"}
        candidate_nodes: list[BaseSubstrateNodeV1] = []
        candidate_edges = []
        contradiction_candidates: list[str] = []
        evidence_gaps: list[str] = []

        for item in response.delta_items:
            self._validate_item_zone_safety(item=item, strict_zone=strict_zone)
            if item.candidate_node is not None:
                candidate_nodes.append(_proposalize_node(item.candidate_node, strict=strict_zone))
            if item.candidate_edge is not None:
                candidate_edges.append(item.candidate_edge)
            if item.item_kind == "contradiction_flag" and item.contradiction_summary:
                contradiction_candidates.append(item.contradiction_summary)
            if item.item_kind == "evidence_gap" and item.evidence_gap_question:
                evidence_gaps.append(item.evidence_gap_question)

        if strict_zone:
            for edge in candidate_edges:
                edge.metadata["frontier_proposal_only"] = True
                edge.metadata["frontier_identity_protected"] = True

        notes = list(response.rationale_notes)
        notes.append(f"zone:{request.target_zone}")
        notes.append(f"posture:{_ZONE_POSTURE[request.target_zone]}")
        if strict_zone:
            notes.append("strict_zone_identity_protection_applied")

        return FrontierGraphDeltaBundleV1(
            request_id=request.request_id,
            response_id=response.response_id,
            target_zone=request.target_zone,
            task_type=request.task_type,
            suggested_landing_posture=_ZONE_POSTURE[request.target_zone],
            candidate_nodes=candidate_nodes,
            candidate_edges=candidate_edges,
            contradiction_candidates=contradiction_candidates,
            evidence_gap_candidates=evidence_gaps,
            source_provenance=FrontierSourceProvenanceV1(provider=response.provider, model=response.model),
            confidence=response.confidence,
            notes=notes,
        )

    @staticmethod
    def _validate_item_zone_safety(*, item: FrontierDeltaItemV1, strict_zone: bool) -> None:
        if not strict_zone or item.candidate_node is None:
            return
        if item.candidate_node.promotion_state == "canonical":
            raise ValueError("frontier expansion cannot write canonical nodes in autonomy/self-relationship zones")
