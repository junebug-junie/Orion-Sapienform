from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1
from orion.spark.concept_induction.graph_query import GraphQueryClient, GraphQueryConfig, GraphQueryError

logger = logging.getLogger("orion.autonomy.repository")

AvailabilityKind = Literal["available", "empty", "unavailable"]
RepositoryBackendKind = Literal["local", "graph", "shadow"]

AUTONOMY_IDENTITY_GRAPH = "http://conjourney.net/graph/autonomy/identity"
AUTONOMY_DRIVES_GRAPH = "http://conjourney.net/graph/autonomy/drives"
AUTONOMY_GOALS_GRAPH = "http://conjourney.net/graph/autonomy/goals"


@dataclass(frozen=True)
class AutonomyRepositoryStatus:
    backend: str
    source_path: str
    source_available: bool


@dataclass(frozen=True)
class AutonomyLookupV1:
    subject: str
    state: AutonomyStateV1 | None
    availability: AvailabilityKind
    unavailable_reason: str | None = None
    subquery_diagnostics: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class SubjectBinding:
    model_layer: str
    entity_id: str


SUBJECT_BINDINGS: dict[str, SubjectBinding] = {
    "orion": SubjectBinding(model_layer="self-model", entity_id="self:orion"),
    "juniper": SubjectBinding(model_layer="user-model", entity_id="user:juniper"),
    "relationship": SubjectBinding(model_layer="relationship-model", entity_id="relationship:orion|juniper"),
}


class AutonomyRepository(Protocol):
    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        ...

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        ...

    def status(self) -> AutonomyRepositoryStatus:
        ...


def _escape_sparql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _literal(binding: dict[str, dict[str, str]], key: str) -> str | None:
    val = binding.get(key, {}).get("value")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _bounded_reason(exc: Exception, *, limit: int = 180) -> str:
    reason = " ".join(str(exc or "").split()).strip() or exc.__class__.__name__
    return reason[:limit]


def _classify_query_error(exc: Exception) -> str:
    if isinstance(exc, GraphQueryError):
        if exc.error_type and exc.error_type != "query_error":
            return exc.error_type
    lowered = str(exc or "").lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if any(token in lowered for token in ("connection refused", "name or service not known", "max retries exceeded")):
        return "connection_error"
    if "400" in lowered or "malformed" in lowered or "parse" in lowered:
        return "malformed_query"
    return "query_error"


def _dominant_drive_from_evidence(
    *,
    explicit: str | None,
    drive_pressures: dict[str, float],
    active_drives: Sequence[str],
) -> str | None:
    explicit_text = " ".join(str(explicit or "").split()).strip()
    if explicit_text:
        return explicit_text
    if drive_pressures:
        return max(
            drive_pressures.items(),
            key=lambda item: (float(item[1]), item[0]),
        )[0]
    for drive in active_drives:
        drive_text = " ".join(str(drive or "").split()).strip()
        if drive_text:
            return drive_text
    return None


def _status_json(*, backend: str, subjects: Sequence[str], results: Sequence[AutonomyLookupV1], observer: dict[str, str] | None) -> str:
    unavailable = [item.unavailable_reason for item in results if item.availability == "unavailable"]
    payload = {
        "backend": backend,
        "consumer": (observer or {}).get("consumer"),
        "correlation_id": (observer or {}).get("correlation_id"),
        "session_id": (observer or {}).get("session_id"),
        "subjects_requested": list(subjects),
        "states_returned": sum(1 for item in results if item.availability == "available"),
        "unavailable_reason": unavailable[0] if unavailable else None,
    }
    return json.dumps(payload, sort_keys=True)


class LocalAutonomyRepository:
    """Tiny local fallback seam for tests/dev; defaults to empty."""

    def __init__(self, *, source_path: str | None = None) -> None:
        self._source_path = source_path or ""

    def status(self) -> AutonomyRepositoryStatus:
        return AutonomyRepositoryStatus(
            backend="local",
            source_path=self._source_path or "autonomy:local:empty",
            source_available=bool(self._source_path and Path(self._source_path).exists()),
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        del observer
        if subject not in SUBJECT_BINDINGS:
            return AutonomyLookupV1(subject=subject, state=None, availability="empty")
        return AutonomyLookupV1(subject=subject, state=None, availability="empty")

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        return [self.get_latest(subject, observer=observer) for subject in subjects]


class GraphAutonomyRepository:
    def __init__(
        self,
        *,
        endpoint: str | None,
        timeout_sec: float,
        user: str | None = None,
        password: str | None = None,
        query_client: GraphQueryClient | None = None,
        goals_limit: int = 3,
        subject_max_workers: int | None = None,
    ) -> None:
        self._endpoint = (endpoint or "").strip()
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._goals_limit = max(1, min(int(goals_limit), 5))
        self._subquery_max_workers = 3
        self._subject_max_workers = (
            max(1, int(subject_max_workers))
            if subject_max_workers is not None
            else max(1, int(os.getenv("AUTONOMY_SUBJECT_MAX_WORKERS", "3")))
        )
        self._chat_stance_short_circuit = str(os.getenv("AUTONOMY_CHAT_STANCE_SHORT_CIRCUIT", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._query_client = query_client or (
            GraphQueryClient(
                GraphQueryConfig(
                    endpoint=self._endpoint,
                    graph_uri=AUTONOMY_IDENTITY_GRAPH,
                    timeout_sec=self._timeout_sec,
                    user=self._user,
                    password=self._password,
                )
            )
            if self._endpoint
            else None
        )

    def status(self) -> AutonomyRepositoryStatus:
        return AutonomyRepositoryStatus(
            backend="graph",
            source_path=self._endpoint or "graphdb:unconfigured",
            source_available=bool(self._endpoint),
        )

    def _select_rows(self, sparql: str) -> list[dict[str, dict[str, str]]]:
        if self._query_client is None:
            raise GraphQueryError("graph_not_configured")
        return self._query_client.select(sparql)

    def _fetch_identity(self, *, subject: str, model_layer: str, entity_id: str) -> tuple[dict[str, str] | None, int]:
        sparql = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?summary ?anchor_strategy ?created_at
WHERE {{
  GRAPH <{AUTONOMY_IDENTITY_GRAPH}> {{
    ?artifact a orion:IdentitySnapshot ;
      orion:subjectKey \"{_escape_sparql(subject)}\" ;
      orion:modelLayerKey \"{_escape_sparql(model_layer)}\" ;
      orion:entityId \"{_escape_sparql(entity_id)}\" ;
      orion:artifactId ?artifact_id ;
      orion:timestamp ?created_at .
    OPTIONAL {{ ?artifact orion:snapshotSummary ?summary . }}
    OPTIONAL {{ ?artifact orion:anchorStrategy ?anchor_strategy . }}
  }}
}}
ORDER BY DESC(?created_at) DESC(STR(?artifact_id))
LIMIT 1
""".strip()
        rows = self._select_rows(sparql)
        if not rows:
            return None, 0
        row = rows[0]
        return ({
            "artifact_id": _literal(row, "artifact_id") or "",
            "summary": _literal(row, "summary") or "",
            "anchor_strategy": _literal(row, "anchor_strategy") or "",
            "created_at": _literal(row, "created_at") or "",
        }, len(rows))

    def _fetch_drive_audit(self, *, subject: str, model_layer: str, entity_id: str) -> tuple[dict[str, object] | None, int]:
        sparql = f"""
PREFIX orion: <http://conjourney.net/orion#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?artifact_id ?dominant_drive ?created_at ?active_drive ?drive_name ?drive_pressure ?tension_kind
WHERE {{
  GRAPH <{AUTONOMY_DRIVES_GRAPH}> {{
    ?artifact a orion:DriveAudit ;
      orion:subjectKey \"{_escape_sparql(subject)}\" ;
      orion:modelLayerKey \"{_escape_sparql(model_layer)}\" ;
      orion:entityId \"{_escape_sparql(entity_id)}\" ;
      orion:artifactId ?artifact_id ;
      orion:timestamp ?created_at .
    OPTIONAL {{ ?artifact orion:dominantDriveName ?dominant_drive . }}
    OPTIONAL {{
      ?artifact orion:highlightsActiveDrive ?active_drive_ref .
      OPTIONAL {{
        FILTER(isIRI(?active_drive_ref))
        ?active_drive_ref rdfs:label ?active_drive_label .
      }}
      BIND(COALESCE(?active_drive_label, STR(?active_drive_ref)) AS ?active_drive)
    }}
    OPTIONAL {{
      ?artifact orion:hasDriveAssessment ?assessment .
      ?assessment orion:driveDimension ?drive_ref ;
        orion:drivePressure ?drive_pressure .
      OPTIONAL {{
        FILTER(isIRI(?drive_ref))
        ?drive_ref rdfs:label ?drive_name_label .
      }}
      BIND(COALESCE(?drive_name_label, STR(?drive_ref)) AS ?drive_name)
    }}
    OPTIONAL {{ ?artifact orion:derivedFromTension ?tension_ref . ?tension_ref orion:tensionKind ?tension_kind . }}
  }}
}}
ORDER BY DESC(?created_at) DESC(STR(?artifact_id))
LIMIT 80
""".strip()
        rows = self._select_rows(sparql)
        if not rows:
            return None, 0

        latest_id = _literal(rows[0], "artifact_id")
        if not latest_id:
            return None, len(rows)
        filtered = [r for r in rows if _literal(r, "artifact_id") == latest_id]
        dominant_drive = _literal(filtered[0], "dominant_drive")
        created_at = _literal(filtered[0], "created_at")
        active_drives: list[str] = []
        drive_pressures: dict[str, float] = {}
        tensions: list[str] = []

        for row in filtered:
            active = _literal(row, "active_drive")
            if active and active not in active_drives:
                active_drives.append(active)
            dname = _literal(row, "drive_name")
            dpressure = _literal(row, "drive_pressure")
            if dname and dpressure is not None:
                try:
                    drive_pressures[dname] = float(dpressure)
                except Exception:
                    pass
            tension_kind = _literal(row, "tension_kind")
            if tension_kind and tension_kind not in tensions:
                tensions.append(tension_kind)

        return ({
            "artifact_id": latest_id,
            "dominant_drive": _dominant_drive_from_evidence(
                explicit=dominant_drive,
                drive_pressures=drive_pressures,
                active_drives=active_drives,
            ),
            "created_at": created_at,
            "active_drives": active_drives,
            "drive_pressures": drive_pressures,
            "tension_kinds": tensions,
        }, len(rows))

    def _fetch_goals(self, *, subject: str, model_layer: str, entity_id: str) -> tuple[list[AutonomyGoalHeadlineV1], int]:
        sparql = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?goal_statement ?drive_origin ?priority ?cooldown_until ?proposal_signature ?created_at
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH}> {{
    ?artifact a orion:ProposedGoal ;
      orion:subjectKey \"{_escape_sparql(subject)}\" ;
      orion:modelLayerKey \"{_escape_sparql(model_layer)}\" ;
      orion:entityId \"{_escape_sparql(entity_id)}\" ;
      orion:artifactId ?artifact_id ;
      orion:timestamp ?created_at ;
      orion:goalStatement ?goal_statement ;
      orion:driveOrigin ?drive_origin ;
      orion:proposalPriority ?priority ;
      orion:proposalSignature ?proposal_signature .
    OPTIONAL {{ ?artifact orion:cooldownUntil ?cooldown_until . }}
  }}
}}
ORDER BY DESC(?created_at) DESC(STR(?artifact_id))
LIMIT {self._goals_limit}
""".strip()
        rows = self._select_rows(sparql)
        out: list[AutonomyGoalHeadlineV1] = []
        for row in rows:
            try:
                out.append(
                    AutonomyGoalHeadlineV1(
                        artifact_id=_literal(row, "artifact_id") or "",
                        goal_statement=_literal(row, "goal_statement") or "",
                        drive_origin=_literal(row, "drive_origin") or "",
                        priority=float(_literal(row, "priority") or 0.0),
                        cooldown_until=_literal(row, "cooldown_until"),
                        proposal_signature=_literal(row, "proposal_signature") or "",
                    )
                )
            except Exception:
                continue
        return out, len(rows)

    def _query_subject(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        binding = SUBJECT_BINDINGS.get(subject)
        if binding is None:
            return AutonomyLookupV1(subject=subject, state=None, availability="empty")

        correlation_id = str((observer or {}).get("correlation_id") or "")

        if self._query_client is None:
            logger.warning(
                "autonomy_graph_lookup subject=%s model_layer=%s entity_id=%s query_ok=false identity_rows=0 drives_rows=0 goals_rows=0 availability=unavailable unavailable_reason=graph_not_configured",
                subject,
                binding.model_layer,
                binding.entity_id,
            )
            return AutonomyLookupV1(subject=subject, state=None, availability="unavailable", unavailable_reason="graph_not_configured")

        identity: dict[str, str] | None = None
        audit: dict[str, object] | None = None
        goals: list[AutonomyGoalHeadlineV1] = []
        identity_rows = 0
        drives_rows = 0
        goals_rows = 0
        failed_subquery: str | None = None
        failure_reason: str | None = None
        failure_type: str | None = None
        subquery_diagnostics: dict[str, dict[str, object]] = {}
        subqueries = (
            ("identity", self._fetch_identity),
            ("drives", self._fetch_drive_audit),
            ("goals", self._fetch_goals),
        )
        logger.info(
            "autonomy_graph_subject_start subject=%s execution_mode=concurrent subquery_workers=%s correlation_id=%s",
            subject,
            self._subquery_max_workers,
            correlation_id or "-",
        )
        with ThreadPoolExecutor(max_workers=self._subquery_max_workers) as executor:
            future_map = {
                executor.submit(fn, subject=subject, model_layer=binding.model_layer, entity_id=binding.entity_id): (name, time.perf_counter())
                for name, fn in subqueries
            }
            for future in as_completed(future_map):
                name, started_at = future_map[future]
                try:
                    value, row_count = future.result()
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    if name == "identity":
                        identity = value  # type: ignore[assignment]
                        identity_rows = row_count
                    elif name == "drives":
                        audit = value  # type: ignore[assignment]
                        drives_rows = row_count
                    else:
                        goals = value  # type: ignore[assignment]
                        goals_rows = row_count
                    status = "empty" if row_count == 0 else "ok"
                    subquery_diagnostics[name] = {
                        "status": status,
                        "row_count": row_count,
                        "elapsed_ms": elapsed_ms,
                    }
                    logger.info(
                        "autonomy_graph_subquery subject=%s subquery=%s status=%s rows=%s repo=%s elapsed_ms=%s timeout_sec=%s correlation_id=%s",
                        subject,
                        name,
                        status,
                        row_count,
                        self._endpoint or "graphdb:unconfigured",
                        elapsed_ms,
                        self._timeout_sec,
                        correlation_id or "-",
                    )
                except (GraphQueryError, Exception) as exc:
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    failure_type = _classify_query_error(exc)
                    if failed_subquery is None:
                        failed_subquery = name
                        failure_reason = _bounded_reason(exc)
                    logger.warning(
                        "autonomy_graph_subquery subject=%s subquery=%s status=%s repo=%s elapsed_ms=%s timeout_sec=%s error_type=%s reason=%s correlation_id=%s",
                        subject,
                        name,
                        failure_type,
                        self._endpoint or "graphdb:unconfigured",
                        elapsed_ms,
                        self._timeout_sec,
                        failure_type,
                        _bounded_reason(exc),
                        correlation_id or "-",
                    )
                    subquery_diagnostics[name] = {
                        "status": failure_type,
                        "row_count": 0,
                        "elapsed_ms": elapsed_ms,
                        "error_type": failure_type,
                        "reason": _bounded_reason(exc),
                    }

        if failed_subquery is not None and not identity and not audit and not goals:
            logger.warning(
                "autonomy_graph_lookup subject=%s model_layer=%s entity_id=%s query_ok=false failed_subquery=%s failure_reason=%s failure_type=%s identity_rows=%s drives_rows=%s goals_rows=%s availability=unavailable unavailable_reason=%s",
                subject,
                binding.model_layer,
                binding.entity_id,
                failed_subquery,
                failure_reason or "unknown",
                failure_type or "query_error",
                identity_rows,
                drives_rows,
                goals_rows,
                failure_type or "query_error",
            )
            return AutonomyLookupV1(
                subject=subject,
                state=None,
                availability="unavailable",
                unavailable_reason=failure_type or "query_error",
                subquery_diagnostics=subquery_diagnostics,
            )

        if not identity and not audit and not goals:
            logger.info(
                "autonomy_graph_lookup subject=%s model_layer=%s entity_id=%s query_ok=true identity_rows=%s drives_rows=%s goals_rows=%s availability=empty empty_reason=no_rows",
                subject,
                binding.model_layer,
                binding.entity_id,
                identity_rows,
                drives_rows,
                goals_rows,
            )
            return AutonomyLookupV1(subject=subject, state=None, availability="empty", subquery_diagnostics=subquery_diagnostics)

        generated_at = None
        if identity and identity.get("created_at"):
            generated_at = identity.get("created_at")
        elif audit and audit.get("created_at"):
            generated_at = audit.get("created_at")

        state = AutonomyStateV1(
            subject=subject,
            model_layer=binding.model_layer,
            entity_id=binding.entity_id,
            latest_identity_snapshot_id=(identity or {}).get("artifact_id") or None,
            latest_drive_audit_id=(audit or {}).get("artifact_id") or None,
            latest_goal_ids=[goal.artifact_id for goal in goals],
            identity_summary=(identity or {}).get("summary") or None,
            anchor_strategy=(identity or {}).get("anchor_strategy") or None,
            dominant_drive=(audit or {}).get("dominant_drive") if isinstance(audit, dict) else None,
            active_drives=list((audit or {}).get("active_drives") or []),
            drive_pressures=dict((audit or {}).get("drive_pressures") or {}),
            tension_kinds=list((audit or {}).get("tension_kinds") or []),
            goal_headlines=goals,
            source="graph",
            generated_at=generated_at,
        )
        partial = failed_subquery is not None
        logger.info(
            "autonomy_graph_lookup subject=%s model_layer=%s entity_id=%s query_ok=%s identity_rows=%s drives_rows=%s goals_rows=%s availability=available mapped_state=true summary_present=%s partial=%s",
            subject,
            binding.model_layer,
            binding.entity_id,
            "false" if partial else "true",
            identity_rows,
            drives_rows,
            goals_rows,
            "yes" if bool(state.identity_summary) else "no",
            "yes" if partial else "no",
        )
        return AutonomyLookupV1(
            subject=subject,
            state=state,
            availability="available",
            unavailable_reason=(failure_type if partial else None),
            subquery_diagnostics=subquery_diagnostics,
        )

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        result = self._query_subject(subject, observer=observer)
        logger.info("autonomy_repository_status %s", _status_json(backend="graph", subjects=[subject], results=[result], observer=observer))
        return result

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        corr = str((observer or {}).get("correlation_id") or "-")
        consumer = str((observer or {}).get("consumer") or "")
        ordered_subjects = list(subjects)
        preferred_order = [name for name in ("orion", "relationship", "juniper") if name in ordered_subjects]
        short_circuit_allowed = self._chat_stance_short_circuit and consumer == "chat_stance" and bool(preferred_order)
        started = time.perf_counter()
        logger.info(
            "autonomy_graph_subject_fanout_start subjects=%s execution_mode=concurrent workers=%s correlation_id=%s short_circuit_allowed=%s",
            list(subjects),
            self._subject_max_workers,
            corr,
            short_circuit_allowed,
        )
        results_by_subject: dict[str, AutonomyLookupV1] = {}
        executor = ThreadPoolExecutor(max_workers=max(1, min(self._subject_max_workers, len(subjects) or 1)))
        try:
            future_map = {executor.submit(self._query_subject, subject, observer=observer): subject for subject in subjects}
            if short_circuit_allowed:
                pending = set(future_map.keys())
                short_circuit_triggered = False
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        subject = future_map[future]
                        results_by_subject[subject] = future.result()
                    selected = next(
                        (
                            results_by_subject.get(name)
                            for name in preferred_order
                            if results_by_subject.get(name) is not None and results_by_subject.get(name).availability == "available"
                        ),
                        None,
                    )
                    if selected is not None:
                        short_circuit_triggered = True
                        for future in pending:
                            future.cancel()
                        logger.info(
                            "autonomy_graph_subject_fanout_short_circuit correlation_id=%s selected_subject=%s selected_availability=%s selected_unavailable_reason=%s",
                            corr,
                            selected.subject,
                            selected.availability,
                            selected.unavailable_reason,
                        )
                        break
                if short_circuit_triggered:
                    for subject in ordered_subjects:
                        if subject not in results_by_subject:
                            results_by_subject[subject] = AutonomyLookupV1(
                                subject=subject,
                                state=None,
                                availability="unavailable",
                                unavailable_reason="short_circuited_after_preferred_subject",
                            )
            else:
                for future in as_completed(future_map):
                    subject = future_map[future]
                    results_by_subject[subject] = future.result()
        finally:
            executor.shutdown(wait=not short_circuit_allowed, cancel_futures=short_circuit_allowed)
        results = [results_by_subject.get(subject, AutonomyLookupV1(subject=subject, state=None, availability="empty")) for subject in subjects]
        logger.info(
            "autonomy_graph_subject_fanout_end subjects=%s elapsed_ms=%s correlation_id=%s",
            list(subjects),
            round((time.perf_counter() - started) * 1000.0, 2),
            corr,
        )
        logger.info("autonomy_repository_status %s", _status_json(backend="graph", subjects=subjects, results=results, observer=observer))
        return results


class ShadowAutonomyRepository:
    def __init__(self, *, local: LocalAutonomyRepository, graph: GraphAutonomyRepository) -> None:
        self._local = local
        self._graph = graph

    def status(self) -> AutonomyRepositoryStatus:
        local_status = self._local.status()
        return AutonomyRepositoryStatus(backend="shadow", source_path=local_status.source_path, source_available=local_status.source_available)

    def get_latest(self, subject: str, *, observer: dict[str, str] | None = None) -> AutonomyLookupV1:
        local_result = self._local.get_latest(subject, observer=observer)
        graph_result = self._graph.get_latest(subject, observer=observer)
        result = local_result if local_result.availability == "available" else graph_result
        logger.info("autonomy_repository_status %s", _status_json(backend="shadow", subjects=[subject], results=[result], observer=observer))
        return result

    def list_latest(self, subjects: Sequence[str], *, observer: dict[str, str] | None = None) -> list[AutonomyLookupV1]:
        local_results = {item.subject: item for item in self._local.list_latest(subjects, observer=observer)}
        graph_results = {item.subject: item for item in self._graph.list_latest(subjects, observer=observer)}
        merged = [
            local_results[subject] if local_results[subject].availability == "available" else graph_results.get(subject, local_results[subject])
            for subject in subjects
        ]
        logger.info("autonomy_repository_status %s", _status_json(backend="shadow", subjects=subjects, results=merged, observer=observer))
        return merged


def build_autonomy_repository(
    *,
    backend: RepositoryBackendKind = "graph",
    endpoint: str | None = None,
    timeout_sec: float = 5.0,
    user: str | None = None,
    password: str | None = None,
    goals_limit: int = 3,
    subject_max_workers: int | None = None,
) -> AutonomyRepository:
    local_repo = LocalAutonomyRepository()
    if backend == "local":
        return local_repo

    graph_repo = GraphAutonomyRepository(
        endpoint=endpoint,
        timeout_sec=timeout_sec,
        user=user,
        password=password,
        goals_limit=goals_limit,
        subject_max_workers=subject_max_workers,
    )
    if backend == "graph":
        return graph_repo
    return ShadowAutonomyRepository(local=local_repo, graph=graph_repo)
