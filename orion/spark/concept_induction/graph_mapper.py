from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.concept_induction import ConceptCluster, ConceptItem, ConceptProfile, StateEstimate


def _binding_value(binding: dict[str, dict[str, str]], key: str) -> str | None:
    raw = binding.get(key)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _as_int(raw: str | None, default: int = 0) -> int:
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _as_dt(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return datetime.now(timezone.utc)


def _as_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class _ConceptAccumulator:
    concept_id: str
    label: str
    concept_type: str
    salience: float
    confidence: float
    embedding_ref: str | None = None
    aliases: set[str] = field(default_factory=set)
    metadata_json: str | None = None


@dataclass
class _ClusterAccumulator:
    cluster_id: str
    label: str
    summary: str
    cohesion_score: float
    concept_ids: set[str] = field(default_factory=set)
    metadata_json: str | None = None


@dataclass
class GraphProfileSummary:
    profile_uri: str
    profile_id: str
    subject: str
    revision: int
    created_at: datetime
    window_start: datetime
    window_end: datetime
    profile_metadata_json: str | None = None


@dataclass
class GraphProfileDetails:
    concepts: dict[str, _ConceptAccumulator]
    clusters: dict[str, _ClusterAccumulator]
    state_estimate: StateEstimate | None
    provenance: dict[str, str]


def map_latest_profile_rows(rows: list[dict[str, dict[str, str]]]) -> dict[str, GraphProfileSummary]:
    out: dict[str, GraphProfileSummary] = {}
    for row in rows:
        subject = _binding_value(row, "subject")
        profile_uri = _binding_value(row, "profile")
        profile_id = _binding_value(row, "profile_id")
        if not subject or not profile_uri or not profile_id:
            continue
        out[subject] = GraphProfileSummary(
            profile_uri=profile_uri,
            profile_id=profile_id,
            subject=subject,
            revision=_as_int(_binding_value(row, "revision"), default=1),
            created_at=_as_dt(_binding_value(row, "created_at")),
            window_start=_as_dt(_binding_value(row, "window_start")),
            window_end=_as_dt(_binding_value(row, "window_end")),
            profile_metadata_json=_binding_value(row, "profile_metadata_json"),
        )
    return out


def map_profile_details_rows(rows: list[dict[str, dict[str, str]]]) -> dict[str, GraphProfileDetails]:
    concepts_by_profile: dict[str, dict[str, _ConceptAccumulator]] = {}
    clusters_by_profile: dict[str, dict[str, _ClusterAccumulator]] = {}
    state_by_profile: dict[str, StateEstimate | None] = {}
    prov_by_profile: dict[str, dict[str, str]] = {}

    for row in rows:
        profile_uri = _binding_value(row, "profile")
        if not profile_uri:
            continue

        concepts = concepts_by_profile.setdefault(profile_uri, {})
        clusters = clusters_by_profile.setdefault(profile_uri, {})
        prov = prov_by_profile.setdefault(profile_uri, {})

        concept_id = _binding_value(row, "concept_id")
        if concept_id:
            concept = concepts.get(concept_id)
            if concept is None:
                concept = _ConceptAccumulator(
                    concept_id=concept_id,
                    label=_binding_value(row, "concept_label") or concept_id,
                    concept_type=_binding_value(row, "concept_type") or "unknown",
                    salience=_as_float(_binding_value(row, "concept_salience"), 0.0),
                    confidence=_as_float(_binding_value(row, "concept_confidence"), 0.5),
                    embedding_ref=_binding_value(row, "concept_embedding_ref"),
                    metadata_json=_binding_value(row, "concept_metadata_json"),
                )
                concepts[concept_id] = concept
            alias = _binding_value(row, "concept_alias")
            if alias:
                concept.aliases.add(alias)

        cluster_id = _binding_value(row, "cluster_id")
        if cluster_id:
            cluster = clusters.get(cluster_id)
            if cluster is None:
                cluster = _ClusterAccumulator(
                    cluster_id=cluster_id,
                    label=_binding_value(row, "cluster_label") or cluster_id,
                    summary=_binding_value(row, "cluster_summary") or "",
                    cohesion_score=_as_float(_binding_value(row, "cluster_cohesion_score"), 0.0),
                    metadata_json=_binding_value(row, "cluster_metadata_json"),
                )
                clusters[cluster_id] = cluster
            cluster_concept = _binding_value(row, "cluster_concept")
            if cluster_concept:
                cluster.concept_ids.add(cluster_concept)

        state_confidence = _binding_value(row, "state_confidence")
        if state_confidence and profile_uri not in state_by_profile:
            state_by_profile[profile_uri] = StateEstimate(
                dimensions=_as_json_dict(_binding_value(row, "state_dimensions_json")),
                trend=_as_json_dict(_binding_value(row, "state_trend_json")),
                confidence=_as_float(state_confidence, 0.5),
                window_start=_as_dt(_binding_value(row, "state_window_start")),
                window_end=_as_dt(_binding_value(row, "state_window_end")),
            )

        writer_service = _binding_value(row, "writer_service")
        writer_version = _binding_value(row, "writer_version")
        correlation_id = _binding_value(row, "correlation_id")
        if writer_service:
            prov["writer_service"] = writer_service
        if writer_version:
            prov["writer_version"] = writer_version
        if correlation_id:
            prov["correlation_id"] = correlation_id

    out: dict[str, GraphProfileDetails] = {}
    for profile_uri in set(concepts_by_profile) | set(clusters_by_profile) | set(state_by_profile) | set(prov_by_profile):
        out[profile_uri] = GraphProfileDetails(
            concepts=concepts_by_profile.get(profile_uri, {}),
            clusters=clusters_by_profile.get(profile_uri, {}),
            state_estimate=state_by_profile.get(profile_uri),
            provenance=prov_by_profile.get(profile_uri, {}),
        )
    return out


def build_concept_profile(summary: GraphProfileSummary, details: GraphProfileDetails) -> ConceptProfile:
    concepts = [
        ConceptItem(
            concept_id=item.concept_id,
            label=item.label,
            aliases=sorted(item.aliases),
            type=item.concept_type,
            salience=item.salience,
            confidence=item.confidence,
            embedding_ref=item.embedding_ref,
            evidence=[],
            metadata=_as_json_dict(item.metadata_json),
        )
        for item in details.concepts.values()
    ]
    concepts.sort(key=lambda item: item.concept_id)

    clusters = [
        ConceptCluster(
            cluster_id=item.cluster_id,
            label=item.label,
            summary=item.summary,
            concept_ids=sorted(item.concept_ids),
            cohesion_score=item.cohesion_score,
            metadata=_as_json_dict(item.metadata_json),
        )
        for item in details.clusters.values()
    ]
    clusters.sort(key=lambda item: item.cluster_id)

    metadata = _as_json_dict(summary.profile_metadata_json)
    if details.provenance:
        metadata.setdefault("graph_provenance", details.provenance)

    return ConceptProfile(
        profile_id=summary.profile_id,
        subject=summary.subject,
        revision=summary.revision,
        created_at=summary.created_at,
        window_start=summary.window_start,
        window_end=summary.window_end,
        concepts=concepts,
        clusters=clusters,
        state_estimate=details.state_estimate,
        metadata=metadata,
    )
