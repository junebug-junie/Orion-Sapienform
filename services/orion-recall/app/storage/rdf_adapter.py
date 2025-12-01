# app/storage/rdf_adapater.py
from __future__ import annotations

import requests
from typing import Any, Dict, List

from app.settings import settings
from app.types import Fragment


def fetch_rdf_fragments(
    *,
    query_text: str,
    max_items: int = 8,
) -> List[Fragment]:
    """
    Lightweight GraphDB search.

    NOTE: This is a generic "contains text anywhere" search – since your
    schema is custom, this is intentionally conservative and is disabled
    by default (RECALL_ENABLE_RDF=false).
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

    frags: List[Fragment] = []
    for s, texts in by_subject.items():
        text = " ".join(texts)[:1500]
        frags.append(
            Fragment(
                id=s,
                kind="rdf",
                source="rdf",
                text=text,
                ts=0.0,
                tags=["rdf"],
                salience=0.6,
                meta={"subject": s},
            )
        )

    return frags[:max_items]
