from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import requests


ORION_NS = "http://conjourney.net/orion#"
SPARK_PROFILE_GRAPH = "http://conjourney.net/graph/spark/concept-profile"


class GraphQueryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = "query_error",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


@dataclass(frozen=True)
class GraphQueryConfig:
    endpoint: str
    graph_uri: str = SPARK_PROFILE_GRAPH
    timeout_sec: float = 5.0
    user: str | None = None
    password: str | None = None


class GraphQueryClient:
    def __init__(self, cfg: GraphQueryConfig) -> None:
        self._cfg = cfg

    def select(self, sparql: str) -> list[dict[str, dict[str, str]]]:
        auth = None
        if self._cfg.user and self._cfg.password:
            auth = (self._cfg.user, self._cfg.password)

        try:
            response = requests.post(
                self._cfg.endpoint,
                data=sparql,
                headers={
                    "Content-Type": "application/sparql-query",
                    "Accept": "application/sparql-results+json",
                },
                auth=auth,
                timeout=self._cfg.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            return list(payload.get("results", {}).get("bindings", []))
        except requests.exceptions.Timeout as exc:
            raise GraphQueryError(str(exc), error_type="timeout") from exc
        except requests.exceptions.ConnectionError as exc:
            raise GraphQueryError(str(exc), error_type="connection_error") from exc
        except requests.exceptions.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            malformed_statuses = {400, 422}
            error_type = "malformed_query" if status_code in malformed_statuses else "http_error"
            raise GraphQueryError(str(exc), error_type=error_type, status_code=status_code) from exc
        except Exception as exc:  # noqa: BLE001
            raise GraphQueryError(str(exc), error_type="query_error") from exc


def _escape_sparql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _sparql_values_str(var_name: str, values: Sequence[str]) -> str:
    if not values:
        return ""
    rendered = " ".join(f'"{_escape_sparql(v)}"' for v in values)
    return f"VALUES ?{var_name} {{ {rendered} }}"


def _sparql_values_uri(var_name: str, values: Sequence[str]) -> str:
    if not values:
        return ""
    rendered = " ".join(f"<{v}>" for v in values)
    return f"VALUES ?{var_name} {{ {rendered} }}"


def build_latest_profile_query(*, subjects: Sequence[str], graph_uri: str = SPARK_PROFILE_GRAPH) -> str:
    subject_values = _sparql_values_str("subject", subjects)
    return f"""
PREFIX orion: <{ORION_NS}>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?profile ?subject ?profile_id ?revision ?created_at ?window_start ?window_end ?profile_metadata_json
WHERE {{
  GRAPH <{graph_uri}> {{
    {subject_values}
    ?profile a orion:SparkConceptProfile ;
      orion:subject ?subject ;
      orion:profileId ?profile_id ;
      orion:revision ?revision ;
      orion:createdAt ?created_at ;
      orion:windowStart ?window_start ;
      orion:windowEnd ?window_end .
    OPTIONAL {{ ?profile orion:profileMetadataJson ?profile_metadata_json . }}

    FILTER NOT EXISTS {{
      ?other a orion:SparkConceptProfile ;
        orion:subject ?subject ;
        orion:revision ?other_revision ;
        orion:createdAt ?other_created_at ;
        orion:profileId ?other_profile_id .
      FILTER (
        xsd:integer(?other_revision) > xsd:integer(?revision) ||
        (xsd:integer(?other_revision) = xsd:integer(?revision) && ?other_created_at > ?created_at) ||
        (xsd:integer(?other_revision) = xsd:integer(?revision) && ?other_created_at = ?created_at && STR(?other_profile_id) > STR(?profile_id))
      )
    }}
  }}
}}
""".strip()


def build_profile_details_query(*, profile_uris: Sequence[str], graph_uri: str = SPARK_PROFILE_GRAPH) -> str:
    profile_values = _sparql_values_uri("profile", profile_uris)
    return f"""
PREFIX orion: <{ORION_NS}>
SELECT
  ?profile
  ?concept ?concept_id ?concept_label ?concept_type ?concept_salience ?concept_confidence
  ?concept_embedding_ref ?concept_alias ?concept_metadata_json
  ?cluster ?cluster_id ?cluster_label ?cluster_summary ?cluster_cohesion_score
  ?cluster_concept ?cluster_metadata_json
  ?state ?state_confidence ?state_window_start ?state_window_end ?state_dimensions_json ?state_trend_json
  ?provenance ?writer_service ?writer_version ?correlation_id
WHERE {{
  GRAPH <{graph_uri}> {{
    {profile_values}
    OPTIONAL {{
      ?profile orion:hasConcept ?concept .
      ?concept orion:conceptId ?concept_id ;
        orion:label ?concept_label ;
        orion:conceptType ?concept_type ;
        orion:salience ?concept_salience ;
        orion:confidence ?concept_confidence .
      OPTIONAL {{ ?concept orion:embeddingRef ?concept_embedding_ref . }}
      OPTIONAL {{ ?concept orion:alias ?concept_alias . }}
      OPTIONAL {{ ?concept orion:conceptMetadataJson ?concept_metadata_json . }}
    }}

    OPTIONAL {{
      ?profile orion:hasCluster ?cluster .
      ?cluster orion:clusterId ?cluster_id ;
        orion:label ?cluster_label ;
        orion:summary ?cluster_summary ;
        orion:cohesionScore ?cluster_cohesion_score .
      OPTIONAL {{ ?cluster orion:includesConcept ?cluster_concept_uri . ?cluster_concept_uri orion:conceptId ?cluster_concept . }}
      OPTIONAL {{ ?cluster orion:clusterMetadataJson ?cluster_metadata_json . }}
    }}

    OPTIONAL {{
      ?profile orion:hasStateEstimate ?state .
      ?state orion:estimateConfidence ?state_confidence ;
        orion:estimateWindowStart ?state_window_start ;
        orion:estimateWindowEnd ?state_window_end ;
        orion:dimensionsJson ?state_dimensions_json ;
        orion:trendJson ?state_trend_json .
    }}

    OPTIONAL {{
      ?profile orion:hasProvenance ?provenance .
      OPTIONAL {{ ?provenance orion:writerService ?writer_service . }}
      OPTIONAL {{ ?provenance orion:writerVersion ?writer_version . }}
      OPTIONAL {{ ?provenance orion:correlationId ?correlation_id . }}
    }}
  }}
}}
""".strip()
