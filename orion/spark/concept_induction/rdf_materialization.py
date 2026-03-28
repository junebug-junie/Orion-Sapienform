from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from orion.core.schemas.concept_induction import ConceptProfile
from orion.schemas.rdf import RdfWriteRequest
from orion.schemas.spark_concept_graph import SparkConceptProfileGraphMaterializationV1

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
XSD_INT = "http://www.w3.org/2001/XMLSchema#integer"
XSD_FLOAT = "http://www.w3.org/2001/XMLSchema#float"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
XSD_BOOL = "http://www.w3.org/2001/XMLSchema#boolean"
XSD_DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"

ORION_NS = "http://conjourney.net/orion#"
SPARK_PROFILE_GRAPH = "http://conjourney.net/graph/spark/concept-profile"


def _iri(value: str) -> str:
    return f"<{value}>"


def _escape_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _literal(value: Any, datatype: str = XSD_STRING) -> str:
    if isinstance(value, bool):
        lex = "true" if value else "false"
    elif isinstance(value, float):
        lex = format(value, ".12g")
    elif isinstance(value, datetime):
        ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        lex = ts.isoformat()
        datatype = XSD_DATETIME
    else:
        lex = str(value)
    return f'"{_escape_literal(lex)}"^^<{datatype}>'


def _uri_for_profile(subject: str, profile_id: str) -> str:
    return f"http://conjourney.net/orion/spark/concept-profile/{quote(subject, safe='')}/{quote(profile_id, safe='')}"


def _uri_for_subject(subject: str) -> str:
    return f"http://conjourney.net/orion/entity/{quote(subject.strip().lower().replace(' ', '-'), safe='')}"


def build_concept_profile_rdf_request(
    *,
    profile: ConceptProfile,
    correlation_id: str | None,
    writer_service: str,
    writer_version: str,
) -> RdfWriteRequest:
    profile_uri = _uri_for_profile(profile.subject, profile.profile_id)
    subject_uri = _uri_for_subject(profile.subject)
    provenance_uri = f"{profile_uri}/provenance"
    state_uri = f"{profile_uri}/state-estimate"

    triples: list[str] = []

    def add(subject: str, predicate: str, obj: str) -> None:
        triples.append(f"{_iri(subject)} {_iri(predicate)} {obj} .")

    add(profile_uri, RDF_TYPE, _iri(f"{ORION_NS}SparkConceptProfile"))
    add(profile_uri, f"{ORION_NS}profileId", _literal(profile.profile_id))
    add(profile_uri, f"{ORION_NS}subject", _literal(profile.subject))
    add(profile_uri, f"{ORION_NS}revision", _literal(profile.revision, XSD_INT))
    add(profile_uri, f"{ORION_NS}createdAt", _literal(profile.created_at))
    add(profile_uri, f"{ORION_NS}windowStart", _literal(profile.window_start))
    add(profile_uri, f"{ORION_NS}windowEnd", _literal(profile.window_end))
    add(profile_uri, f"{ORION_NS}aboutEntity", _iri(subject_uri))
    add(profile_uri, f"{ORION_NS}hasProvenance", _iri(provenance_uri))
    add(profile_uri, f"{ORION_NS}sourceKind", _literal("spark_concept_profile"))

    add(subject_uri, RDF_TYPE, _iri(f"{ORION_NS}SparkConceptProfileSubject"))
    add(subject_uri, f"{ORION_NS}subjectName", _literal(profile.subject))

    if profile.metadata:
        add(
            profile_uri,
            f"{ORION_NS}profileMetadataJson",
            _literal(json.dumps(profile.metadata, sort_keys=True), XSD_STRING),
        )

    add(provenance_uri, RDF_TYPE, _iri(f"{ORION_NS}MaterializationProvenance"))
    add(provenance_uri, f"{ORION_NS}writerService", _literal(writer_service))
    add(provenance_uri, f"{ORION_NS}writerVersion", _literal(writer_version))
    add(provenance_uri, f"{ORION_NS}producedAt", _literal(profile.created_at))
    if correlation_id:
        add(provenance_uri, f"{ORION_NS}correlationId", _literal(correlation_id))

    for concept in profile.concepts:
        concept_uri = f"{profile_uri}/concept/{quote(concept.concept_id, safe='')}"
        add(profile_uri, f"{ORION_NS}hasConcept", _iri(concept_uri))
        add(concept_uri, RDF_TYPE, _iri(f"{ORION_NS}SparkConcept"))
        add(concept_uri, f"{ORION_NS}conceptId", _literal(concept.concept_id))
        add(concept_uri, f"{ORION_NS}label", _literal(concept.label))
        add(concept_uri, f"{ORION_NS}conceptType", _literal(concept.type))
        add(concept_uri, f"{ORION_NS}salience", _literal(concept.salience, XSD_DOUBLE))
        add(concept_uri, f"{ORION_NS}confidence", _literal(concept.confidence, XSD_DOUBLE))
        if concept.embedding_ref:
            add(concept_uri, f"{ORION_NS}embeddingRef", _literal(concept.embedding_ref))
        for alias in concept.aliases:
            add(concept_uri, f"{ORION_NS}alias", _literal(alias))
        if concept.metadata:
            add(
                concept_uri,
                f"{ORION_NS}conceptMetadataJson",
                _literal(json.dumps(concept.metadata, sort_keys=True), XSD_STRING),
            )

    for cluster in profile.clusters:
        cluster_uri = f"{profile_uri}/cluster/{quote(cluster.cluster_id, safe='')}"
        add(profile_uri, f"{ORION_NS}hasCluster", _iri(cluster_uri))
        add(cluster_uri, RDF_TYPE, _iri(f"{ORION_NS}SparkConceptCluster"))
        add(cluster_uri, f"{ORION_NS}clusterId", _literal(cluster.cluster_id))
        add(cluster_uri, f"{ORION_NS}label", _literal(cluster.label))
        add(cluster_uri, f"{ORION_NS}summary", _literal(cluster.summary))
        add(cluster_uri, f"{ORION_NS}cohesionScore", _literal(cluster.cohesion_score, XSD_DOUBLE))
        for concept_id in cluster.concept_ids:
            concept_uri = f"{profile_uri}/concept/{quote(concept_id, safe='')}"
            add(cluster_uri, f"{ORION_NS}includesConcept", _iri(concept_uri))
        if cluster.metadata:
            add(
                cluster_uri,
                f"{ORION_NS}clusterMetadataJson",
                _literal(json.dumps(cluster.metadata, sort_keys=True), XSD_STRING),
            )

    if profile.state_estimate is not None:
        estimate = profile.state_estimate
        add(profile_uri, f"{ORION_NS}hasStateEstimate", _iri(state_uri))
        add(state_uri, RDF_TYPE, _iri(f"{ORION_NS}SparkStateEstimate"))
        add(state_uri, f"{ORION_NS}estimateConfidence", _literal(estimate.confidence, XSD_DOUBLE))
        add(state_uri, f"{ORION_NS}estimateWindowStart", _literal(estimate.window_start))
        add(state_uri, f"{ORION_NS}estimateWindowEnd", _literal(estimate.window_end))
        add(
            state_uri,
            f"{ORION_NS}dimensionsJson",
            _literal(json.dumps(estimate.dimensions, sort_keys=True), XSD_STRING),
        )
        add(
            state_uri,
            f"{ORION_NS}trendJson",
            _literal(json.dumps(estimate.trend, sort_keys=True), XSD_STRING),
        )

    payload = SparkConceptProfileGraphMaterializationV1(
        profile_id=profile.profile_id,
        subject=profile.subject,
        revision=profile.revision,
        produced_at=profile.created_at,
        window_start=profile.window_start,
        window_end=profile.window_end,
        concept_count=len(profile.concepts),
        cluster_count=len(profile.clusters),
        state_estimate_present=profile.state_estimate is not None,
        correlation_id=correlation_id,
        writer_service=writer_service,
        writer_version=writer_version,
    )

    return RdfWriteRequest(
        id=f"spark-profile-{profile.subject}-{profile.revision}",
        source=writer_service,
        graph=SPARK_PROFILE_GRAPH,
        triples="\n".join(triples),
        kind="spark.concept_profile.graph.v1",
        payload=payload.model_dump(mode="json"),
    )
