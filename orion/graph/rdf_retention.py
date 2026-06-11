"""SPARQL retention/pruning for Fuseki / SPARQL RDF stores."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import httpx

logger = logging.getLogger(__name__)

ORION = "http://conjourney.net/orion#"
CM = "http://orion.ai/collapse#"

# Default: no automatic deletion. Memory graphs (autonomy, chat, collapse, …) are never
# pruned unless RDF_RETENTION_POLICIES is set explicitly. TDB disk bloat is reclaimed
# via offline compact (services/orion-rdf-store: make compact), not SPARQL DELETE.
DEFAULT_POLICIES: list[dict[str, Any]] = []

# Optional telemetry-only example (enable via RDF_RETENTION_POLICIES if desired):
# TELEMETRY_RETENTION_EXAMPLE = [
#     {"graph": "http://conjourney.net/graph/orion/cognition", "max_age_days": 30},
#     {"graph": "http://conjourney.net/graph/orion/metacog", "max_age_days": 30},
# ]

ARTIFACT_CHILD_PREDICATES = (
    f"{ORION}hasDriveAssessment",
    f"{ORION}hasProvenance",
    f"{ORION}supportedByEvidence",
    f"{ORION}hasCorrelation",
    f"{ORION}hasTrace",
    f"{ORION}hasTurnContext",
    f"{ORION}referencesSourceEvent",
    f"{ORION}derivedFromTension",
)


@dataclass(frozen=True)
class GraphRetentionPolicy:
    graph: str
    max_age_days: int | None = None
    max_artifacts: int | None = None
    timestamp_predicate: str = f"{ORION}timestamp"
    artifact_type: str | None = f"{ORION}AutonomyArtifact"
    batch_size: int = 200

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GraphRetentionPolicy:
        graph = str(raw.get("graph") or "").strip()
        if not graph:
            raise ValueError("retention policy requires graph")
        max_age = raw.get("max_age_days")
        max_artifacts = raw.get("max_artifacts")
        return cls(
            graph=graph,
            max_age_days=int(max_age) if max_age is not None else None,
            max_artifacts=int(max_artifacts) if max_artifacts is not None else None,
            timestamp_predicate=str(raw.get("timestamp_predicate") or f"{ORION}timestamp"),
            artifact_type=str(raw["artifact_type"]) if raw.get("artifact_type") else None,
            batch_size=int(raw.get("batch_size") or 200),
        )


@dataclass
class PruneGraphResult:
    graph: str
    deleted_by_age: int = 0
    deleted_by_cap: int = 0
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def parse_retention_policies(raw: str | None) -> list[GraphRetentionPolicy]:
    if not raw or not str(raw).strip():
        return [GraphRetentionPolicy.from_dict(item) for item in DEFAULT_POLICIES]
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("RDF_RETENTION_POLICIES must be a JSON array")
    return [GraphRetentionPolicy.from_dict(item) for item in parsed]


def cutoff_literal(days: int, *, now: datetime | None = None) -> str:
    ref = now or datetime.now(timezone.utc)
    cutoff = ref - timedelta(days=days)
    return cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_subject_age_delete(
    policy: GraphRetentionPolicy,
    cutoff_iso: str,
    *,
    artifact_type: str | None = None,
) -> str:
    g = policy.graph
    ts = policy.timestamp_predicate
    type_clause = ""
    if artifact_type:
        type_clause = f"    ?s a <{artifact_type}> .\n"
    return f"""\
DELETE {{
  GRAPH <{g}> {{ ?s ?p ?o }}
}}
WHERE {{
  GRAPH <{g}> {{
{type_clause}    ?s <{ts}> ?ts .
    FILTER(?ts < "{cutoff_iso}"^^<http://www.w3.org/2001/XMLSchema#dateTime>)
    ?s ?p ?o .
  }}
}}
"""


def build_artifact_child_delete(graph: str, artifact_uri: str) -> str:
    preds = ">|".join(f"<{p}>" for p in ARTIFACT_CHILD_PREDICATES)
    return f"""\
DELETE {{
  GRAPH <{graph}> {{ ?s ?p ?o }}
}}
WHERE {{
  GRAPH <{graph}> {{
    <{artifact_uri}> ({preds}) ?s .
    ?s ?p ?o .
  }}
}}
"""


def build_artifact_cap_select(policy: GraphRetentionPolicy) -> str:
    if not policy.max_artifacts or policy.max_artifacts <= 0:
        raise ValueError("max_artifacts required")
    g = policy.graph
    ts = policy.timestamp_predicate
    artifact_type = policy.artifact_type or f"{ORION}AutonomyArtifact"
    return f"""\
SELECT ?artifact WHERE {{
  GRAPH <{g}> {{
    ?artifact a <{artifact_type}> ;
              <{ts}> ?ts .
  }}
}}
ORDER BY DESC(?ts)
OFFSET {policy.max_artifacts}
LIMIT {policy.batch_size}
"""


class SparqlRetentionClient:
    def __init__(
        self,
        *,
        query_url: str,
        update_url: str,
        user: str | None = None,
        password: str | None = None,
        timeout_sec: float = 120.0,
    ) -> None:
        self._query_url = query_url
        self._update_url = update_url
        self._auth = (user, password) if user and password is not None else None
        self._timeout = timeout_sec

    def _post_query(self, sparql: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._query_url,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
                auth=self._auth,
            )
            resp.raise_for_status()
            return resp.json()

    def _post_update(self, sparql: str, *, dry_run: bool) -> None:
        if dry_run:
            logger.info("retention_dry_run update_chars=%s", len(sparql))
            return
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._update_url,
                content=sparql,
                headers={"Content-Type": "application/sparql-update"},
                auth=self._auth,
            )
            resp.raise_for_status()

    def select_artifact_uris(self, policy: GraphRetentionPolicy) -> list[str]:
        payload = self._post_query(build_artifact_cap_select(policy))
        bindings = payload.get("results", {}).get("bindings", [])
        out: list[str] = []
        for row in bindings:
            val = row.get("artifact", {}).get("value")
            if val:
                out.append(val)
        return out

    def prune_graph(
        self,
        policy: GraphRetentionPolicy,
        *,
        dry_run: bool = False,
        now: datetime | None = None,
    ) -> PruneGraphResult:
        result = PruneGraphResult(graph=policy.graph, dry_run=dry_run)
        artifact_type = policy.artifact_type

        if policy.max_age_days and policy.max_age_days > 0:
            cutoff = cutoff_literal(policy.max_age_days, now=now)
            update = build_subject_age_delete(policy, cutoff, artifact_type=artifact_type)
            try:
                self._post_update(update, dry_run=dry_run)
                result.deleted_by_age += 1
                logger.info(
                    "retention_age_prune graph=%s max_age_days=%s cutoff=%s dry_run=%s",
                    policy.graph,
                    policy.max_age_days,
                    cutoff,
                    dry_run,
                )
            except httpx.HTTPError as exc:
                msg = f"age prune failed graph={policy.graph}: {exc}"
                logger.error(msg)
                result.errors.append(msg)

        if policy.max_artifacts and policy.max_artifacts > 0 and artifact_type:
            stale: list[str] = []
            if dry_run:
                logger.info(
                    "retention_cap_prune_dry_run graph=%s max_artifacts=%s batch_size=%s",
                    policy.graph,
                    policy.max_artifacts,
                    policy.batch_size,
                )
            else:
                try:
                    stale = self.select_artifact_uris(policy)
                except httpx.HTTPError as exc:
                    msg = f"artifact cap select failed graph={policy.graph}: {exc}"
                    logger.error(msg)
                    result.errors.append(msg)

            for artifact_uri in stale:
                subject_delete = f"""\
DELETE {{
  GRAPH <{policy.graph}> {{ ?s ?p ?o }}
}}
WHERE {{
  GRAPH <{policy.graph}> {{
    BIND(<{artifact_uri}> AS ?s)
    ?s ?p ?o .
  }}
}}
"""
                child_delete = build_artifact_child_delete(policy.graph, artifact_uri)
                try:
                    self._post_update(subject_delete, dry_run=dry_run)
                    self._post_update(child_delete, dry_run=dry_run)
                    result.deleted_by_cap += 1
                except httpx.HTTPError as exc:
                    msg = f"cap prune failed graph={policy.graph} artifact={artifact_uri}: {exc}"
                    logger.error(msg)
                    result.errors.append(msg)

            logger.info(
                "retention_cap_prune graph=%s max_artifacts=%s stale=%s dry_run=%s",
                policy.graph,
                policy.max_artifacts,
                len(stale),
                dry_run,
            )

        return result


def run_retention_pass(
    *,
    policies: Sequence[GraphRetentionPolicy],
    query_url: str,
    update_url: str,
    user: str | None = None,
    password: str | None = None,
    dry_run: bool = False,
    timeout_sec: float = 120.0,
) -> list[PruneGraphResult]:
    client = SparqlRetentionClient(
        query_url=query_url,
        update_url=update_url,
        user=user,
        password=password,
        timeout_sec=timeout_sec,
    )
    results: list[PruneGraphResult] = []
    for policy in policies:
        results.append(client.prune_graph(policy, dry_run=dry_run))
    return results
