from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)


def fetch_memory_graph_sparql_candidates(query_text: str, settings: Any) -> List[Dict[str, Any]]:
    """Optional SPARQL expansion from configured operator memory named graphs (flag-gated)."""
    endpoint = str(getattr(settings, "RECALL_RDF_ENDPOINT_URL", "") or "").strip().rstrip("/")
    if not endpoint:
        return []
    raw_graphs = str(getattr(settings, "RECALL_MEMORY_GRAPH_NAMED_GRAPHS", "") or "").strip()
    graphs = [g.strip() for g in raw_graphs.split(",") if g.strip()]
    if not graphs:
        return []
    timeout = float(getattr(settings, "RECALL_MEMORY_GRAPH_SPARQL_TIMEOUT_SEC", 2.0) or 2.0)
    vals = " ".join(f"<{g}>" for g in graphs)
    sparql = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX orionmem: <https://orion.local/ns/mem/v2026-05#>
SELECT DISTINCT ?lab ?tp ?tgt WHERE {{
  VALUES ?g {{ {vals} }}
  GRAPH ?g {{
    ?s a orionmem:AffectiveDisposition .
    OPTIONAL {{ ?s rdfs:label ?lab . }}
    OPTIONAL {{ ?s orionmem:trustPolarity ?tp . }}
    OPTIONAL {{
      ?s orionmem:dispositionTarget ?tg .
      ?tg rdfs:label ?tgt .
    }}
  }}
}}
LIMIT 12
""".strip()
    user = getattr(settings, "RECALL_RDF_USER", None) or getattr(settings, "GRAPHDB_USER", "")
    password = getattr(settings, "RECALL_RDF_PASS", None) or getattr(settings, "GRAPHDB_PASS", "")
    auth = (user, password) if user or password else None
    try:
        r = requests.post(
            endpoint,
            data=urlencode({"query": sparql}),
            headers={"Accept": "application/sparql-results+json", "Content-Type": "application/x-www-form-urlencoded"},
            auth=auth,
            timeout=timeout,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        logger.debug("memory_graph_sparql_query_failed error=%s", exc)
        return []
    bindings = (payload.get("results") or {}).get("bindings") or []
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(bindings):
        lab = (b.get("lab") or {}).get("value") or ""
        tp = (b.get("tp") or {}).get("value") or ""
        tgt = (b.get("tgt") or {}).get("value") or ""
        snippet = " ".join(x for x in [lab, tp, tgt] if x).strip() or "memory_graph"
        out.append(
            {
                "id": f"mgsparql:{i}",
                "source": "memory_graph_sparql",
                "text": snippet,
                "score": 0.55,
                "tags": ["memory_graph_sparql"],
            }
        )
    return out
