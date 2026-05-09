"""Self-study RDF adapter — graphdb_durable tier.

Queries the self-study named graph in GraphDB and maps induced concepts into
ConceptNodeV1 nodes anchored to 'orion'.  Uses the same endpoint/credentials
as the existing autonomy and orionmem adapters (GRAPHDB_QUERY_ENDPOINT etc.).

Controlled by env vars:
  SELF_STUDY_NAMED_GRAPH  — named graph URI (comma-separated if multiple)
  GRAPHDB_QUERY_ENDPOINT  — SPARQL endpoint (same as autonomy)
  GRAPHDB_REPO            — repository name
  GRAPHDB_USER / GRAPHDB_PASS — optional basic auth
  SELF_STUDY_GRAPHDB_TIMEOUT_SEC — request timeout (default 5.0)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)

from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.self_study")

_TIER_RANK = 2  # graphdb_durable


def _make_prov() -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="self_study",
        source_channel="graphdb.sparql",
        producer="self_study_adapter",
        tier_rank=_TIER_RANK,
    )


def _resolve_config() -> dict[str, Any]:
    endpoint_raw = (
        os.getenv("GRAPHDB_QUERY_ENDPOINT")
        or os.getenv("GRAPHDB_URL")
        or ""
    ).strip()
    repo = (os.getenv("GRAPHDB_REPO") or "collapse").strip() or "collapse"
    user = (os.getenv("GRAPHDB_USER") or "").strip() or None
    password = (os.getenv("GRAPHDB_PASS") or "").strip() or None

    endpoint = endpoint_raw
    if endpoint and endpoint.rstrip("/").endswith("/repositories"):
        endpoint = f"{endpoint.rstrip('/')}/{repo}"
    elif endpoint and "/repositories/" not in endpoint:
        endpoint = f"{endpoint.rstrip('/')}/repositories/{repo}"

    graphs_raw = (os.getenv("SELF_STUDY_NAMED_GRAPH") or "").strip()
    graphs = [g.strip() for g in graphs_raw.split(",") if g.strip()]

    timeout = float(os.getenv("SELF_STUDY_GRAPHDB_TIMEOUT_SEC") or "5.0")

    return {"endpoint": endpoint or None, "user": user, "password": password, "graphs": graphs, "timeout": timeout}


def map_self_study_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:  # noqa: ARG001
    """Query GraphDB self-study named graph → ConceptNodeV1 nodes (anchor=orion)."""
    cfg = _resolve_config()
    endpoint = cfg["endpoint"]
    graphs = cfg["graphs"]

    if not endpoint or not graphs:
        logger.debug("self_study_adapter_skipped reason=no_endpoint_or_graphs")
        return None

    vals = " ".join(f"<{g}>" for g in graphs)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?concept ?label ?confidence WHERE {{
  VALUES ?g {{ {vals} }}
  GRAPH ?g {{
    ?concept a ?type .
    OPTIONAL {{ ?concept rdfs:label ?label . }}
    OPTIONAL {{ ?concept skos:prefLabel ?label . }}
    OPTIONAL {{ ?concept <https://orion.local/ns/self/v1#confidence> ?confidence . }}
  }}
}}
LIMIT 32
""".strip()

    try:
        data = urllib.parse.urlencode({"query": sparql}).encode("utf-8")
        req = urllib.request.Request(
            str(endpoint),
            data=data,
            headers={
                "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        if cfg["user"] or cfg["password"]:
            tok = base64.b64encode(f"{cfg['user'] or ''}:{cfg['password'] or ''}".encode("utf-8")).decode("ascii")
            req.add_header("Authorization", f"Basic {tok}")
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("self_study_adapter_failed error=%s", exc)
        return None

    bindings = (payload.get("results") or {}).get("bindings") or []
    if not bindings:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    prov = _make_prov()
    nodes: list[Any] = []

    seen_labels: set[str] = set()
    for b in bindings:
        uri = (b.get("concept") or {}).get("value") or ""
        label_raw = (b.get("label") or {}).get("value") or ""
        label = label_raw.strip()[:120] or uri.rsplit("/", 1)[-1][:120]
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)

        try:
            confidence = float((b.get("confidence") or {}).get("value") or "0.6")
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.6

        nodes.append(
            ConceptNodeV1(
                anchor_scope="orion",
                label=label,
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=confidence, salience=0.5),
                metadata={"concept_uri": uri, "source_kind": "self_study"},
            )
        )

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes) if nodes else None
