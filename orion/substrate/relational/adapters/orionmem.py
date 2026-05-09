"""Orionmem AffectiveDisposition adapter — snapshot_ephemeral tier.

Queries orionmem named graphs for AffectiveDisposition entries and maps each
into a StateSnapshotNodeV1.  Replaces the inline SPARQL block
``fetch_chat_stance_memory_graph_hints`` in chat_stance.py with a substrate-
backed read that routes through the unified layer.

Env vars:
  CHAT_STANCE_MEMORY_GRAPH_GRAPHS  — comma-separated named graph URIs
  GRAPHDB_QUERY_ENDPOINT / GRAPHDB_URL
  GRAPHDB_REPO / GRAPHDB_USER / GRAPHDB_PASS
  CHAT_STANCE_MEMORY_GRAPH_TIMEOUT_SEC — request timeout (default 2.0)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    StateSnapshotNodeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateAnchorScopeV1,
)

from orion.substrate.adapters._common import make_temporal

logger = logging.getLogger("orion.substrate.relational.adapters.orionmem")

_TIER_RANK = 4  # snapshot_ephemeral
_RELATIONSHIP_GRAPH_TOKENS = ("relationship", "social", "juniper")


def _make_prov(*, graph_uri: str) -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="orionmem.affective_disposition",
        source_channel="graphdb.sparql",
        producer="orionmem_adapter",
        evidence_refs=[graph_uri],
        tier_rank=_TIER_RANK,
    )


def _resolve_config() -> dict[str, Any]:
    endpoint_raw = (
        os.getenv("GRAPHDB_QUERY_ENDPOINT")
        or os.getenv("GRAPHDB_URL")
        or os.getenv("CONCEPT_PROFILE_GRAPHDB_ENDPOINT")
        or os.getenv("CONCEPT_PROFILE_GRAPHDB_URL")
        or ""
    ).strip()
    repo = (os.getenv("GRAPHDB_REPO") or os.getenv("CONCEPT_PROFILE_GRAPHDB_REPO") or "collapse").strip() or "collapse"
    user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
    password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None

    endpoint = endpoint_raw
    if endpoint and endpoint.rstrip("/").endswith("/repositories"):
        endpoint = f"{endpoint.rstrip('/')}/{repo}"
    elif endpoint and "/repositories/" not in endpoint:
        endpoint = f"{endpoint.rstrip('/')}/repositories/{repo}"

    graphs_raw = (os.getenv("CHAT_STANCE_MEMORY_GRAPH_GRAPHS") or "").strip()
    graphs = [g.strip() for g in graphs_raw.split(",") if g.strip()]
    timeout = float(os.getenv("CHAT_STANCE_MEMORY_GRAPH_TIMEOUT_SEC") or "2.0")

    return {"endpoint": endpoint or None, "user": user, "password": password, "graphs": graphs, "timeout": timeout}


def _anchor_for_graph(graph_uri: str) -> SubstrateAnchorScopeV1:
    lower = graph_uri.lower()
    if any(tok in lower for tok in _RELATIONSHIP_GRAPH_TOKENS):
        return "relationship"
    return "orion"


def map_orionmem_to_substrate(ctx: dict[str, Any]) -> SubstrateGraphRecordV1 | None:  # noqa: ARG001
    """Query orionmem AffectiveDisposition graphs → StateSnapshotNodeV1 nodes."""
    cfg = _resolve_config()
    endpoint = cfg["endpoint"]
    graphs = cfg["graphs"]

    if not endpoint or not graphs:
        logger.debug("orionmem_adapter_skipped reason=no_endpoint_or_graphs")
        return None

    vals = " ".join(f"<{g}>" for g in graphs)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX orionmem: <https://orion.local/ns/mem/v2026-05#>
SELECT DISTINCT ?g ?s ?lab ?tp WHERE {{
  VALUES ?g {{ {vals} }}
  GRAPH ?g {{
    ?s a orionmem:AffectiveDisposition .
    OPTIONAL {{ ?s rdfs:label ?lab . }}
    OPTIONAL {{ ?s orionmem:trustPolarity ?tp . }}
  }}
}}
LIMIT 12
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
        logger.debug("orionmem_adapter_failed error=%s", exc)
        return None

    bindings = (payload.get("results") or {}).get("bindings") or []
    if not bindings:
        return None

    now = datetime.now(timezone.utc)
    temporal = make_temporal(observed_at=now)
    nodes: list[Any] = []

    for b in bindings:
        graph_uri = (b.get("g") or {}).get("value") or ""
        subject_uri = (b.get("s") or {}).get("value") or ""
        label = ((b.get("lab") or {}).get("value") or "").strip()[:120]
        trust_polarity = ((b.get("tp") or {}).get("value") or "").strip()

        if not label:
            continue

        anchor = _anchor_for_graph(graph_uri)
        prov = _make_prov(graph_uri=graph_uri)

        polarity_dim = 0.0
        if "positive" in trust_polarity.lower():
            polarity_dim = 1.0
        elif "negative" in trust_polarity.lower():
            polarity_dim = -1.0

        nodes.append(
            StateSnapshotNodeV1(
                anchor_scope=anchor,
                temporal=temporal,
                provenance=prov,
                signals=SubstrateSignalBundleV1(confidence=0.7, salience=0.5),
                snapshot_source="orionmem",
                dimensions={"trust_polarity": polarity_dim},
                metadata={
                    "label": label,
                    "trustPolarity": trust_polarity,
                    "subject_uri": subject_uri,
                    "graph_uri": graph_uri,
                },
            )
        )

    return SubstrateGraphRecordV1(anchor_scope="orion", nodes=nodes) if nodes else None
