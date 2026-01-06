from __future__ import annotations

import requests
from typing import Dict, List

from app.settings import settings


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

    q = query_text.strip().lower().replace('"', '\\"')[:80]
    if not q:
        return []

    sparql = f"""
    SELECT ?s ?p ?o
    WHERE {{
      ?s ?p ?o .
      FILTER(CONTAINS(LCASE(STR(?o)), "{q}"))
    }}
    LIMIT {max_items}
    """

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
        s = b.get("s", {}).get("value")
        o = b.get("o", {}).get("value")
        if not s or not o:
            continue
        by_subject.setdefault(s, []).append(str(o))

    frags: List[Dict[str, str]] = []
    for s, texts in by_subject.items():
        text = " ".join(texts)[:1500]
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
