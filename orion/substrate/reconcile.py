from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    BaseSubstrateNodeV1,
    SubstrateEdgeV1,
)
from .store import SubstrateGraphStore


_TIER_RANK_NAMES: dict[int, str] = {
    1: "operator_static",
    2: "graphdb_durable",
    3: "concept_induced",
    4: "snapshot_ephemeral",
}

# Fixed contract with the Phase 2 concept-embedding producer: an incoming concept
# node may carry a plain list[float] embedding at this exact metadata key. Do not
# rename -- do not promote to a typed schema field (see
# docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md Phase 3).
_CONCEPT_EMBEDDING_METADATA_KEY = "concept_embedding"

# Mirrors ConceptClusterer.threshold in orion/spark/concept_induction/clusterer.py --
# reused deliberately for consistency with that already-validated behavior, not
# re-derived.
_CONCEPT_EMBEDDING_SIMILARITY_THRESHOLD = 0.8

# How many existing concept nodes to scan for an embedding match per incoming node.
# Mirrors the 500-node cap GraphDBSubstrateStore.snapshot() already uses -- a
# familiar order of magnitude for this store, not a new tuning knob.
_CONCEPT_REGION_SCAN_LIMIT = 500


def _cosine(a: Any, b: Any) -> float:
    """Plain-Python cosine similarity -- copied from
    ConceptClusterer._cosine (orion/spark/concept_induction/clusterer.py) rather than
    imported, to avoid coupling orion.substrate to orion.spark.concept_induction for a
    ten-line helper. Keep in sync if that implementation changes."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _is_valid_embedding(value: Any) -> bool:
    """A concept embedding must be a non-empty list/tuple of real numbers. Anything
    else (missing, wrong type, empty, non-numeric elements) is treated as "no
    embedding available" so callers degrade to string match instead of raising."""
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(component, (int, float)) and not isinstance(component, bool) for component in value)


@dataclass(frozen=True)
class NodeMergeDecision:
    canonical_node_id: str
    merged: bool
    reason: str
    tier_conflict: bool = False
    tier_outcome: str = ""


@dataclass(frozen=True)
class EdgeMergeDecision:
    canonical_edge_id: str
    merged: bool
    reason: str


class SubstrateIdentityResolver:
    """Conservative deterministic identity resolver for substrate materialization.

    Concept identity has two layers:

    1. Embedding similarity (when the incoming node and a same-scope/subject
       existing concept node both carry ``metadata["concept_embedding"]`` and
       cosine similarity is >= ``_CONCEPT_EMBEDDING_SIMILARITY_THRESHOLD``): the
       incoming node resolves to the *existing* node's identity, so paraphrases
       ("surface encodings" vs "surface-level representations") merge instead of
       becoming permanent duplicates.
    2. The original exact-string match on ``concept_id`` then ``label`` -- used
       whenever no embedding is available on the incoming node, an existing
       match, or this resolver was constructed without a ``store`` (the default,
       matching today's behavior for every existing caller).

    ``store`` is optional and defaults to ``None`` so existing callers (e.g.
    ``SubstrateGraphMaterializer``'s ``identity_resolver or
    SubstrateIdentityResolver()`` default construction) see zero behavior change
    unless a store is explicitly wired in.
    """

    def __init__(self, *, store: SubstrateGraphStore | None = None, concept_region_scan_limit: int = _CONCEPT_REGION_SCAN_LIMIT) -> None:
        self._store = store
        self._concept_region_scan_limit = concept_region_scan_limit

    def canonical_node_key(self, node: BaseSubstrateNodeV1) -> str | None:
        subject = str(node.subject_ref or "")
        scope = node.anchor_scope
        if node.node_kind == "concept":
            embedding_match_key = self._concept_embedding_match_key(node, scope=scope, subject=subject)
            if embedding_match_key is not None:
                return embedding_match_key
            return self._legacy_concept_key(node, scope=scope, subject=subject)
        if node.node_kind == "drive":
            drive_kind = str(getattr(node, "drive_kind", "")).strip().lower()
            return f"drive|{scope}|{subject}|{drive_kind}" if drive_kind else None
        if node.node_kind == "goal":
            signature = str(node.metadata.get("proposal_signature") or "").strip().lower()
            return f"goal|{scope}|{subject}|sig:{signature}" if signature else None
        if node.node_kind == "evidence":
            content_ref = str(getattr(node, "content_ref", "")).strip().lower()
            evidence_type = str(getattr(node, "evidence_type", "")).strip().lower()
            return f"evidence|{scope}|{subject}|{evidence_type}|{content_ref}" if content_ref else None
        if node.node_kind == "tension":
            artifact_id = str(node.metadata.get("artifact_id") or "").strip().lower()
            if artifact_id:
                return f"tension|{scope}|{subject}|artifact:{artifact_id}"
            return None
        if node.node_kind == "contradiction":
            delta_id = str(node.metadata.get("delta_id") or "").strip().lower()
            if delta_id:
                return f"contradiction|{scope}|{subject}|delta:{delta_id}"
            summary = str(getattr(node, "summary", "")).strip().lower()
            involved = ",".join(sorted(getattr(node, "involved_node_ids", [])))
            if summary and involved:
                digest = hashlib.sha256(f"{summary}|{involved}".encode("utf-8", errors="ignore")).hexdigest()[:16]
                return f"contradiction|{scope}|{subject}|hash:{digest}"
            return None
        if node.node_kind == "state_snapshot":
            # snapshots are distinct events by default; do not over-collapse
            return None
        if node.node_kind == "hypothesis":
            return None
        return None

    def _concept_embedding_match_key(self, node: BaseSubstrateNodeV1, *, scope: str, subject: str) -> str | None:
        """If ``node`` carries a usable embedding and a same-scope/subject existing
        concept node in the store is similar enough (cosine >= threshold), return
        THAT existing node's own identity key (recomputed via
        ``_legacy_concept_key`` from its own concept_id/label) so the store's
        identity-index lookup collides with it and the two merge. Returns ``None``
        -- never raises -- for any reason this can't be determined: no store wired,
        no/invalid embedding, or no candidate met the threshold."""
        if self._store is None:
            return None
        embedding = node.metadata.get(_CONCEPT_EMBEDDING_METADATA_KEY)
        if not _is_valid_embedding(embedding):
            return None
        try:
            region = self._store.read_concept_region(limit_nodes=self._concept_region_scan_limit)
            candidates = region.nodes if region is not None else []
        except Exception:
            return None

        best_candidate: BaseSubstrateNodeV1 | None = None
        best_similarity = 0.0
        for candidate in candidates:
            if candidate is None or candidate.node_kind != "concept":
                continue
            if candidate.node_id == node.node_id:
                continue
            if candidate.anchor_scope != scope or str(candidate.subject_ref or "") != subject:
                continue
            candidate_embedding = candidate.metadata.get(_CONCEPT_EMBEDDING_METADATA_KEY)
            if not _is_valid_embedding(candidate_embedding):
                continue
            try:
                similarity = _cosine(embedding, candidate_embedding)
            except Exception:
                continue
            if similarity >= _CONCEPT_EMBEDDING_SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_candidate = candidate

        if best_candidate is None:
            return None
        return self._legacy_concept_key(
            best_candidate,
            scope=best_candidate.anchor_scope,
            subject=str(best_candidate.subject_ref or ""),
        )

    @staticmethod
    def _legacy_concept_key(node: BaseSubstrateNodeV1, *, scope: str, subject: str) -> str | None:
        """The original exact-string concept identity: ``concept_id`` first, then
        ``label``. Extracted unchanged so both the plain string-match path and the
        embedding-match path (which recomputes an existing node's own key) share
        exactly one implementation."""
        concept_id = str(node.metadata.get("concept_id") or "").strip().lower()
        if concept_id:
            return f"concept|{scope}|{subject}|{concept_id}"
        label = str(getattr(node, "label", "")).strip().lower()
        if label:
            return f"concept|{scope}|{subject}|label:{label}"
        return None

    @staticmethod
    def canonical_edge_key(edge: SubstrateEdgeV1) -> str:
        return f"{edge.source.node_id}|{edge.predicate}|{edge.target.node_id}"


def tier_rank_decision(existing: BaseSubstrateNodeV1, incoming: BaseSubstrateNodeV1) -> tuple[bool, str]:
    """Return (tier_conflict, tier_outcome) for a proposed merge.

    ``tier_conflict`` is True when the incoming node has lower authority (higher rank number)
    than the existing node, meaning the existing node is protected.
    """
    existing_rank: int | None = existing.provenance.tier_rank
    incoming_rank: int | None = incoming.provenance.tier_rank
    if existing_rank is not None and incoming_rank is not None:
        if incoming_rank > existing_rank:
            tier_name = _TIER_RANK_NAMES.get(existing_rank, f"tier{existing_rank}")
            return True, f"{tier_name}_protected"
        if incoming_rank < existing_rank:
            tier_name = _TIER_RANK_NAMES.get(incoming_rank, f"tier{incoming_rank}")
            return False, f"{tier_name}_accepted"
        return False, "symmetric_tier"
    return False, ""


def merge_node(existing: BaseSubstrateNodeV1, incoming: BaseSubstrateNodeV1, *, source_graph_id: str) -> BaseSubstrateNodeV1:
    # Tier-rank policy:
    #   incoming_rank > existing_rank  → lower authority: existing is protected, keep existing signals
    #   incoming_rank < existing_rank  → higher authority: incoming wins on confidence/salience
    #   same or no tier info           → take max (existing behaviour)
    existing_rank: int | None = existing.provenance.tier_rank
    incoming_rank: int | None = incoming.provenance.tier_rank

    tier_protected, _ = tier_rank_decision(existing, incoming)
    tier_promoted = (
        existing_rank is not None
        and incoming_rank is not None
        and incoming_rank < existing_rank
    )

    lineage = list(existing.metadata.get("materialization_lineage") or [])
    lineage.append(
        {
            "source_graph_id": source_graph_id,
            "source_node_id": incoming.node_id,
            "observed_at": incoming.temporal.observed_at.isoformat(),
            "provenance": incoming.provenance.model_dump(mode="json"),
        }
    )
    lineage = lineage[-50:]

    merged_evidence_refs = sorted({*existing.provenance.evidence_refs, *incoming.provenance.evidence_refs})
    # When incoming has higher authority (lower tier rank), adopt incoming provenance as base.
    prov_base = incoming.provenance if tier_promoted else existing.provenance
    merged_provenance = prov_base.model_copy(update={"evidence_refs": merged_evidence_refs})

    valid_from_candidates = [value for value in (existing.temporal.valid_from, incoming.temporal.valid_from) if value is not None]
    valid_to_candidates = [value for value in (existing.temporal.valid_to, incoming.temporal.valid_to) if value is not None]

    # Tier-rank policy on valid_from: higher-authority wins (same cases as confidence).
    if tier_protected:
        merged_valid_from = existing.temporal.valid_from
    elif tier_promoted:
        merged_valid_from = incoming.temporal.valid_from
    else:
        merged_valid_from = min(valid_from_candidates) if valid_from_candidates else None

    merged_temporal = existing.temporal.model_copy(
        update={
            "observed_at": max(existing.temporal.observed_at, incoming.temporal.observed_at),
            "valid_from": merged_valid_from,
            "valid_to": max(valid_to_candidates) if valid_to_candidates else None,
        }
    )

    merged_metadata: dict[str, Any] = {**incoming.metadata, **existing.metadata}
    merged_metadata["materialization_lineage"] = lineage

    merged_signals = existing.signals.model_copy(
        update={
            "confidence": (
                existing.signals.confidence if tier_protected
                else incoming.signals.confidence if tier_promoted
                else max(existing.signals.confidence, incoming.signals.confidence)
            ),
            "salience": (
                existing.signals.salience if tier_protected
                else incoming.signals.salience if tier_promoted
                else max(existing.signals.salience, incoming.signals.salience)
            ),
            "activation": existing.signals.activation.model_copy(
                update={
                    "activation": existing.signals.activation.activation if tier_protected else max(existing.signals.activation.activation, incoming.signals.activation.activation),
                    "recency_score": max(existing.signals.activation.recency_score, incoming.signals.activation.recency_score),
                    "decay_half_life_seconds": existing.signals.activation.decay_half_life_seconds
                    or incoming.signals.activation.decay_half_life_seconds,
                    "decay_floor": max(existing.signals.activation.decay_floor, incoming.signals.activation.decay_floor),
                }
            ),
        }
    )

    return existing.model_copy(
        update={
            "temporal": merged_temporal,
            "provenance": merged_provenance,
            "signals": merged_signals,
            "metadata": merged_metadata,
        }
    )


def merge_edge(existing: SubstrateEdgeV1, incoming: SubstrateEdgeV1, *, source_graph_id: str) -> SubstrateEdgeV1:
    lineage = list(existing.metadata.get("materialization_lineage") or [])
    lineage.append(
        {
            "source_graph_id": source_graph_id,
            "source_edge_id": incoming.edge_id,
            "observed_at": incoming.temporal.observed_at.isoformat(),
        }
    )
    lineage = lineage[-100:]
    merged_metadata = {**incoming.metadata, **existing.metadata, "materialization_lineage": lineage}
    merged_evidence_refs = sorted({*existing.provenance.evidence_refs, *incoming.provenance.evidence_refs})
    return existing.model_copy(
        update={
            "confidence": max(existing.confidence, incoming.confidence),
            "salience": max(existing.salience, incoming.salience),
            "provenance": existing.provenance.model_copy(update={"evidence_refs": merged_evidence_refs}),
            "temporal": existing.temporal.model_copy(update={"observed_at": max(existing.temporal.observed_at, incoming.temporal.observed_at)}),
            "metadata": merged_metadata,
        }
    )
