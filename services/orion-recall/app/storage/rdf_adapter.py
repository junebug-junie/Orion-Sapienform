from __future__ import annotations

import re
from typing import Dict, List

import requests

from app.settings import settings


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")


def _extract_keywords(query_text: str, *, max_keywords: int = 6) -> List[str]:
    tokens = _TOKEN_RE.findall(query_text.lower())
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _escape_sparql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_sparql_query(keywords: List[str], *, max_nodes: int, max_results: int) -> str:
    filters = " || ".join(
        f'CONTAINS(LCASE(STR(?node)), "{_escape_sparql(keyword)}") || '
        f'CONTAINS(LCASE(STR(?o)), "{_escape_sparql(keyword)}")'
        for keyword in keywords
    )
    return f"""
    SELECT ?node ?p ?neighbor
    WHERE {{
      {{
        SELECT DISTINCT ?node
        WHERE {{
          ?node ?p0 ?o .
          FILTER({filters})
        }}
        LIMIT {max_nodes}
      }}
      {{
        ?node ?p ?neighbor .
      }} UNION {{
        ?neighbor ?p ?node .
      }}
    }}
    LIMIT {max_results}
    """


def fetch_rdf_fragments(
    *,
    query_text: str,
    max_items: int = 8,
) -> List[Dict[str, str]]:
    """
    Minimal GraphDB lookup used for recall fusion.
    Returns a list of dicts that can be transformed downstream.
    """
    if not query_text:
        return []

    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint:
        return []

    q = query_text.strip()
    if not q:
        return []

    keywords = _extract_keywords(q)
    if not keywords:
        keywords = [q[:80].lower()]

    max_nodes = max(1, min(max_items, 8))
    max_results = max_items * 4
    sparql = _build_sparql_query(keywords, max_nodes=max_nodes, max_results=max_results)

    try:
        resp = requests.post(
            endpoint,
            data=sparql,
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
            auth=(settings.RECALL_RDF_USER, settings.RECALL_RDF_PASS),
            timeout=settings.RECALL_RDF_TIMEOUT_SEC,
        )
    except Exception:
        return []

    if resp.status_code != 200:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    bindings = data.get("results", {}).get("bindings", [])
    by_subject: Dict[str, List[str]] = {}
    for b in bindings:
        node = b.get("node", {}).get("value")
        predicate = b.get("p", {}).get("value")
        neighbor = b.get("neighbor", {}).get("value")
        if not node or not predicate or not neighbor:
            continue
        by_subject.setdefault(node, []).append(f"{predicate} {neighbor}")

    frags: List[Dict[str, str]] = []
    for s, texts in by_subject.items():
        text = " | ".join(texts)[:1500]
        frags.append(
            {
                "id": s,
                "source": "rdf",
                "source_ref": "graphdb",
                "uri": s,
                "text": text,
                "ts": 0.0,
                "tags": ["rdf"],
                "score": 0.6,
                "meta": {"subject": s},
            }
        )

    return frags[:max_items]
