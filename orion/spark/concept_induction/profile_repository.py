from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from orion.core.schemas.concept_induction import ConceptProfile

from .graph_mapper import (
    GraphProfileDetails,
    build_concept_profile,
    map_latest_profile_rows,
    map_profile_details_rows,
)
from .graph_query import (
    GraphQueryClient,
    GraphQueryConfig,
    GraphQueryError,
    build_latest_profile_query,
    build_profile_details_query,
)
from .parity_evidence import (
    ParityReadinessThresholds,
    configure_parity_evidence_store,
    get_parity_evidence_snapshot,
    record_parity_evidence,
)
from .settings import DEFAULT_CONCEPT_STORE_PATH, ConceptSettings, get_settings
from .store import LocalProfileStore


logger = logging.getLogger("orion.spark.concept.profile_repository")

AvailabilityKind = Literal["available", "empty", "unavailable"]
RepositoryBackendKind = Literal["local", "graph", "shadow"]


@dataclass(frozen=True)
class ConceptProfileRepositoryStatus:
    backend: str
    source_path: str
    placeholder_default_in_use: bool
    source_available: bool


@dataclass(frozen=True)
class ConceptProfileLookupV1:
    subject: str
    profile: ConceptProfile | None
    availability: AvailabilityKind
    unavailable_reason: str | None = None


class ConceptProfileRepository(Protocol):
    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> ConceptProfileLookupV1:
        ...

    def list_latest(
        self, subjects: Sequence[str], *, observer: dict[str, str] | None = None
    ) -> list[ConceptProfileLookupV1]:
        ...

    def status(self) -> ConceptProfileRepositoryStatus:
        ...


class LocalConceptProfileRepository:
    """Repository seam adapter backed by LocalProfileStore."""

    def __init__(self, *, store_path: str) -> None:
        self._store_path = store_path
        self._store = LocalProfileStore(store_path)

    def status(self) -> ConceptProfileRepositoryStatus:
        return ConceptProfileRepositoryStatus(
            backend="local",
            source_path=self._store_path,
            placeholder_default_in_use=self._store_path == DEFAULT_CONCEPT_STORE_PATH,
            source_available=Path(self._store_path).exists(),
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> ConceptProfileLookupV1:
        del observer
        try:
            profile = self._store.load(subject)
        except Exception:
            return ConceptProfileLookupV1(
                subject=subject,
                profile=None,
                availability="unavailable",
                unavailable_reason="read_error",
            )

        if profile is not None:
            return ConceptProfileLookupV1(subject=subject, profile=profile, availability="available")

        if not Path(self._store_path).exists():
            return ConceptProfileLookupV1(
                subject=subject,
                profile=None,
                availability="unavailable",
                unavailable_reason="source_missing",
            )

        return ConceptProfileLookupV1(subject=subject, profile=None, availability="empty")

    def list_latest(
        self, subjects: Sequence[str], *, observer: dict[str, str] | None = None
    ) -> list[ConceptProfileLookupV1]:
        return [self.get_latest(subject, observer=observer) for subject in subjects]


class GraphConceptProfileRepository:
    """Repository adapter backed by GraphDB SPARQL reads."""

    def __init__(
        self,
        *,
        endpoint: str | None,
        graph_uri: str,
        timeout_sec: float,
        user: str | None = None,
        password: str | None = None,
        query_client: GraphQueryClient | None = None,
    ) -> None:
        self._endpoint = (endpoint or "").strip()
        self._graph_uri = graph_uri
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._query_client = query_client or (
            GraphQueryClient(
                GraphQueryConfig(
                    endpoint=self._endpoint,
                    graph_uri=self._graph_uri,
                    timeout_sec=self._timeout_sec,
                    user=self._user,
                    password=self._password,
                )
            )
            if self._endpoint
            else None
        )

    def status(self) -> ConceptProfileRepositoryStatus:
        return ConceptProfileRepositoryStatus(
            backend="graph",
            source_path=self._endpoint or "graphdb:unconfigured",
            placeholder_default_in_use=False,
            source_available=bool(self._endpoint),
        )

    def _query_profiles(self, subjects: Sequence[str]) -> list[ConceptProfileLookupV1]:
        if not subjects:
            return []
        if self._query_client is None:
            return [
                ConceptProfileLookupV1(
                    subject=subject,
                    profile=None,
                    availability="unavailable",
                    unavailable_reason="graph_not_configured",
                )
                for subject in subjects
            ]

        try:
            summary_rows = self._query_client.select(
                build_latest_profile_query(subjects=subjects, graph_uri=self._graph_uri)
            )
            summaries = map_latest_profile_rows(summary_rows)
            profile_uris = [item.profile_uri for item in summaries.values()]
            details: dict[str, GraphProfileDetails] = {}
            if profile_uris:
                detail_rows = self._query_client.select(
                    build_profile_details_query(profile_uris=profile_uris, graph_uri=self._graph_uri)
                )
                details = map_profile_details_rows(detail_rows)
        except (GraphQueryError, Exception):
            return [
                ConceptProfileLookupV1(
                    subject=subject,
                    profile=None,
                    availability="unavailable",
                    unavailable_reason="query_error",
                )
                for subject in subjects
            ]

        results: list[ConceptProfileLookupV1] = []
        for subject in subjects:
            summary = summaries.get(subject)
            if summary is None:
                results.append(ConceptProfileLookupV1(subject=subject, profile=None, availability="empty"))
                continue
            profile = build_concept_profile(
                summary,
                details.get(
                    summary.profile_uri,
                    GraphProfileDetails(concepts={}, clusters={}, state_estimate=None, provenance={}),
                ),
            )
            results.append(ConceptProfileLookupV1(subject=subject, profile=profile, availability="available"))
        return results

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> ConceptProfileLookupV1:
        result = self._query_profiles([subject])[0]
        logger.info(
            "concept_profile_repository_status %s",
            _status_json(
                backend="graph",
                subjects=[subject],
                results=[result],
                observer=observer,
            ),
        )
        return result

    def list_latest(
        self, subjects: Sequence[str], *, observer: dict[str, str] | None = None
    ) -> list[ConceptProfileLookupV1]:
        results = self._query_profiles(subjects)
        logger.info(
            "concept_profile_repository_status %s",
            _status_json(
                backend="graph",
                subjects=subjects,
                results=results,
                observer=observer,
            ),
        )
        return results


class ShadowConceptProfileRepository:
    """Local-primary repository with graph parity diagnostics."""

    def __init__(self, *, local: LocalConceptProfileRepository, graph: GraphConceptProfileRepository) -> None:
        self._local = local
        self._graph = graph

    def status(self) -> ConceptProfileRepositoryStatus:
        local_status = self._local.status()
        return ConceptProfileRepositoryStatus(
            backend="shadow",
            source_path=local_status.source_path,
            placeholder_default_in_use=local_status.placeholder_default_in_use,
            source_available=local_status.source_available,
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> ConceptProfileLookupV1:
        local_result = self._local.get_latest(subject, observer=observer)
        graph_result = self._graph.get_latest(subject, observer=observer)
        _log_parity(subjects=[subject], local_results=[local_result], graph_results=[graph_result], observer=observer)
        logger.info(
            "concept_profile_repository_status %s",
            _status_json(
                backend="shadow",
                subjects=[subject],
                results=[local_result],
                observer=observer,
            ),
        )
        return local_result

    def list_latest(
        self, subjects: Sequence[str], *, observer: dict[str, str] | None = None
    ) -> list[ConceptProfileLookupV1]:
        local_results = self._local.list_latest(subjects, observer=observer)
        graph_results = self._graph.list_latest(subjects, observer=observer)
        _log_parity(subjects=subjects, local_results=local_results, graph_results=graph_results, observer=observer)
        logger.info(
            "concept_profile_repository_status %s",
            _status_json(
                backend="shadow",
                subjects=subjects,
                results=local_results,
                observer=observer,
            ),
        )
        return local_results


def _mismatch_fields(local: ConceptProfile | None, graph: ConceptProfile | None) -> list[str]:
    if local is None and graph is None:
        return []
    if local is None or graph is None:
        return ["profile_missing"]
    fields: list[str] = []
    if local.profile_id != graph.profile_id:
        fields.append("profile_id")
    if local.revision != graph.revision:
        fields.append("revision")
    if local.window_start != graph.window_start or local.window_end != graph.window_end:
        fields.append("freshness_window")
    if len(local.concepts) != len(graph.concepts):
        fields.append("concept_count")
    if len(local.clusters) != len(graph.clusters):
        fields.append("cluster_count")
    local_concepts = {(c.concept_id, c.label) for c in local.concepts}
    graph_concepts = {(c.concept_id, c.label) for c in graph.concepts}
    if local_concepts != graph_concepts:
        fields.append("concept_ids_labels")
    local_clusters = {cluster.cluster_id for cluster in local.clusters}
    graph_clusters = {cluster.cluster_id for cluster in graph.clusters}
    if local_clusters != graph_clusters:
        fields.append("cluster_ids")
    if bool(local.state_estimate) != bool(graph.state_estimate):
        fields.append("state_estimate_presence")
    elif local.state_estimate and graph.state_estimate:
        if (
            local.state_estimate.window_start != graph.state_estimate.window_start
            or local.state_estimate.window_end != graph.state_estimate.window_end
            or local.state_estimate.dimensions != graph.state_estimate.dimensions
            or local.state_estimate.trend != graph.state_estimate.trend
        ):
            fields.append("state_estimate")
    return fields


def _mismatch_classes(
    *,
    local: ConceptProfileLookupV1,
    graph: ConceptProfileLookupV1,
    mismatch_fields: list[str],
) -> list[str]:
    classes: list[str] = []
    if graph.availability == "unavailable":
        if graph.unavailable_reason == "query_error":
            classes.append("query_error")
        classes.append("graph_unavailable")
        return classes
    if local.availability == "available" and graph.availability == "empty":
        classes.append("profile_missing_on_graph")
    if local.availability == "empty" and graph.availability == "available":
        classes.append("profile_missing_on_local")

    mapping = {
        "revision": "revision_mismatch",
        "freshness_window": "freshness_window_mismatch",
        "concept_count": "concept_count_mismatch",
        "cluster_count": "cluster_count_mismatch",
        "concept_ids_labels": "concept_identity_mismatch",
        "cluster_ids": "cluster_identity_mismatch",
        "state_estimate": "state_estimate_mismatch",
        "state_estimate_presence": "state_estimate_mismatch",
        "profile_id": "profile_identity_mismatch",
    }
    for field in mismatch_fields:
        mapped = mapping.get(field)
        if mapped:
            classes.append(mapped)
    return sorted(set(classes))


def _status_json(
    *,
    backend: str,
    subjects: Sequence[str],
    results: Sequence[ConceptProfileLookupV1],
    observer: dict[str, str] | None = None,
) -> str:
    unavailable = [item.unavailable_reason for item in results if item.availability == "unavailable"]
    payload = {
        "backend": backend,
        "consumer": (observer or {}).get("consumer"),
        "correlation_id": (observer or {}).get("correlation_id"),
        "session_id": (observer or {}).get("session_id"),
        "subjects_requested": list(subjects),
        "profiles_returned": sum(1 for item in results if item.availability == "available"),
        "unavailable_reason": unavailable[0] if unavailable else None,
    }
    return str(payload)


def _log_parity(
    *,
    subjects: Sequence[str],
    local_results: Sequence[ConceptProfileLookupV1],
    graph_results: Sequence[ConceptProfileLookupV1],
    observer: dict[str, str] | None = None,
) -> None:
    graph_by_subject = {item.subject: item for item in graph_results}
    mismatch_fields: list[str] = []
    mismatches = 0
    unavailable_reasons: list[str] = []
    subject_outcomes: list[dict[str, object]] = []
    for local in local_results:
        graph = graph_by_subject.get(local.subject)
        if graph is None:
            mismatches += 1
            mismatch_fields.append("subject_missing")
            subject_outcomes.append(
                {
                    "subject": local.subject,
                    "graph_unavailable": False,
                    "empty_on_local_only": False,
                    "empty_on_graph_only": False,
                    "mismatch_classes": ["profile_missing_on_graph"],
                }
            )
            continue
        if graph.availability == "unavailable" and graph.unavailable_reason:
            unavailable_reasons.append(graph.unavailable_reason)
        if local.availability != graph.availability:
            mismatches += 1
            mismatch_fields.append("availability")
            subject_outcomes.append(
                {
                    "subject": local.subject,
                    "graph_unavailable": graph.availability == "unavailable",
                    "empty_on_local_only": local.availability == "empty" and graph.availability == "available",
                    "empty_on_graph_only": local.availability == "available" and graph.availability == "empty",
                    "mismatch_classes": _mismatch_classes(local=local, graph=graph, mismatch_fields=[]),
                }
            )
            continue
        fields = _mismatch_fields(local.profile, graph.profile)
        if fields:
            mismatches += 1
            mismatch_fields.extend(fields)
        subject_outcomes.append(
            {
                "subject": local.subject,
                "graph_unavailable": graph.availability == "unavailable",
                "empty_on_local_only": local.availability == "empty" and graph.availability == "available",
                "empty_on_graph_only": local.availability == "available" and graph.availability == "empty",
                "mismatch_classes": _mismatch_classes(local=local, graph=graph, mismatch_fields=fields),
            }
        )

    consumer = (observer or {}).get("consumer") or "unknown_consumer"
    evidence_summary = record_parity_evidence(consumer=consumer, subject_outcomes=subject_outcomes)

    logger.info(
        "concept_profile_repository_parity %s",
        str(
            {
                "consumer": (observer or {}).get("consumer"),
                "backend": "shadow",
                "correlation_id": (observer or {}).get("correlation_id"),
                "session_id": (observer or {}).get("session_id"),
                "subjects_requested": list(subjects),
                "local_profiles_returned": sum(1 for item in local_results if item.availability == "available"),
                "graph_profiles_returned": sum(1 for item in graph_results if item.availability == "available"),
                "mismatch_count": mismatches,
                "mismatch_fields": sorted(set(mismatch_fields)) if mismatch_fields else [],
                "unavailable_reason": unavailable_reasons[0] if unavailable_reasons else None,
            }
        ),
    )
    if evidence_summary.get("should_emit_summary"):
        logger.info("concept_profile_parity_evidence %s", str(evidence_summary))


def build_concept_profile_repository(
    settings: ConceptSettings | None = None,
    *,
    backend_override: RepositoryBackendKind | None = None,
) -> ConceptProfileRepository:
    cfg = settings or get_settings()
    configure_parity_evidence_store(
        thresholds=ParityReadinessThresholds(
            min_comparisons=max(1, int(getattr(cfg, "concept_profile_parity_min_comparisons", 50))),
            max_mismatch_rate=float(getattr(cfg, "concept_profile_parity_max_mismatch_rate", 0.05)),
            max_unavailable_rate=float(getattr(cfg, "concept_profile_parity_max_unavailable_rate", 0.02)),
            critical_mismatch_classes=tuple(
                item.strip()
                for item in str(
                    getattr(
                        cfg,
                        "concept_profile_parity_critical_mismatch_classes",
                        "profile_missing_on_graph,profile_missing_on_local,query_error",
                    )
                ).split(",")
                if item.strip()
            ),
        ),
        summary_interval=max(1, int(getattr(cfg, "concept_profile_parity_summary_interval", 25))),
    )
    local_repo = LocalConceptProfileRepository(store_path=cfg.store_path)

    backend: RepositoryBackendKind = backend_override or getattr(cfg, "concept_profile_repository_backend", "local")
    if backend == "local":
        return local_repo

    graph_repo = GraphConceptProfileRepository(
        endpoint=getattr(cfg, "concept_profile_graphdb_endpoint", ""),
        graph_uri=getattr(cfg, "concept_profile_graph_uri", "http://conjourney.net/graph/spark/concept-profile"),
        timeout_sec=getattr(cfg, "concept_profile_graph_timeout_sec", 6.0),
        user=getattr(cfg, "concept_profile_graphdb_user", ""),
        password=getattr(cfg, "concept_profile_graphdb_pass", ""),
    )

    if backend == "graph":
        return graph_repo

    return ShadowConceptProfileRepository(local=local_repo, graph=graph_repo)


def concept_profile_parity_evidence_snapshot() -> dict:
    """Inspectable bounded parity evidence surface (process-local)."""
    return get_parity_evidence_snapshot()
