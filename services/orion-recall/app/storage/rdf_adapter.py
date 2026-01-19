from __future__ import annotations

import re
from typing import Dict, List, Tuple

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


def _extract_labels(values: List[str], *, max_items: int) -> List[str]:
    seen = set()
    labels: List[str] = []
    for value in values:
        if not value:
            continue
        tail = value.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        candidate = tail.replace("_", " ").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        labels.append(candidate)
        if len(labels) >= max_items:
            break
    return labels


def _is_claim_node(node: str) -> bool:
    return "/claim/" in node


def _looks_like_uri(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _build_graphtri_anchor_sparql(
    session_id: str,
    keywords: List[str],
    *,
    max_turns: int,
    max_terms: int,
    filtered: bool,
) -> str:
    filter_clause = ""
    if filtered and keywords:
        filters = " || ".join(
            f'CONTAINS(LCASE(STR(?term)), "{_escape_sparql(keyword)}")' for keyword in keywords
        )
        filter_clause = f"FILTER({filters})"

    return f"""
    SELECT DISTINCT ?term
    WHERE {{
      GRAPH <orion:chat> {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#sessionId> "{_escape_sparql(session_id)}" .
      }}
      GRAPH <orion:enrichment> {{
        {{ ?turn <http://orion.ai/collapse#hasTag> ?term . }}
        UNION {{ ?turn <http://orion.ai/collapse#hasEntity> ?term . }}
        UNION {{
          ?claim a <http://conjourney.net/orion#Claim> ;
                 <http://conjourney.net/orion#subject> ?turn ;
                 <http://conjourney.net/orion#obj> ?term .
        }}
      }}
      {filter_clause}
    }}
    LIMIT {max_terms}
    """


def _build_graphtri_anchor_kind_sparql(
    session_id: str,
    query_terms: List[str],
    *,
    max_terms: int,
    filtered: bool,
) -> str:
    filter_clause = ""
    if filtered and query_terms:
        filters = " || ".join(
            f'CONTAINS(LCASE(STR(?term)), "{_escape_sparql(term)}")' for term in query_terms
        )
        filter_clause = f"FILTER({filters})"

    return f"""
    SELECT ?kind ?term
    WHERE {{
      GRAPH <orion:chat> {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#sessionId> "{_escape_sparql(session_id)}" .
      }}
      GRAPH <orion:enrichment> {{
        {{ ?turn <http://orion.ai/collapse#hasTag> ?term .
           BIND("tag" AS ?kind) }}
        UNION {{ ?turn <http://orion.ai/collapse#hasEntity> ?term .
           BIND("entity" AS ?kind) }}
        UNION {{
          ?claim a <http://conjourney.net/orion#Claim> ;
                 <http://conjourney.net/orion#subject> ?turn ;
                 <http://conjourney.net/orion#obj> ?term .
          BIND("claim" AS ?kind)
        }}
      }}
      {filter_clause}
    }}
    LIMIT {max_terms}
    """


def fetch_graphtri_anchors(
    *,
    session_id: str,
    query_terms: List[str],
    max_terms: int = 12,
) -> Dict[str, List[str]]:
    if not session_id:
        return {"entities_terms": [], "tags_terms": [], "claim_objs": [], "related_terms": []}

    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint:
        return {"entities_terms": [], "tags_terms": [], "claim_objs": [], "related_terms": []}

    def _run_sparql(filtered: bool) -> List[Dict[str, str]]:
        sparql = _build_graphtri_anchor_kind_sparql(
            session_id,
            query_terms,
            max_terms=max_terms,
            filtered=filtered,
        )
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
        results: List[Dict[str, str]] = []
        for b in bindings:
            kind = b.get("kind", {}).get("value")
            term = b.get("term", {}).get("value")
            if not kind or not term:
                continue
            results.append({"kind": kind, "term": term})
        return results

    results = _run_sparql(filtered=True)
    if len(results) < 3:
        fallback = _run_sparql(filtered=False)
        if fallback:
            results = fallback

    tags_terms: List[str] = []
    entities_terms: List[str] = []
    claim_objs: List[str] = []
    related_terms: List[str] = []
    seen = set()
    for item in results:
        term = item["term"]
        if _looks_like_uri(term):
            term_values = _extract_labels([term], max_items=1)
            term = term_values[0] if term_values else term
        term = term.strip()
        if not term:
            continue
        if item["kind"] == "tag":
            tags_terms.append(term)
        elif item["kind"] == "entity":
            entities_terms.append(term)
        elif item["kind"] == "claim":
            claim_objs.append(term)
        if term not in seen:
            seen.add(term)
            related_terms.append(term)
        if len(related_terms) >= max_terms:
            break

    return {
        "entities_terms": entities_terms[:max_terms],
        "tags_terms": tags_terms[:max_terms],
        "claim_objs": claim_objs[:max_terms],
        "related_terms": related_terms[:max_terms],
    }


def fetch_rdf_graphtri_anchor_terms(
    *,
    query_text: str,
    session_id: str,
    max_items: int = 10,
) -> List[str]:
    if not query_text or not session_id:
        return []

    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint:
        return []

    q = query_text.strip()
    if not q:
        return []

    keywords = _extract_keywords(q)
    max_turns = max(1, min(max_items, 12))
    max_terms = max_items

    def _run_sparql(filtered: bool) -> List[str]:
        sparql = _build_graphtri_anchor_sparql(
            session_id,
            keywords,
            max_turns=max_turns,
            max_terms=max_terms,
            filtered=filtered,
        )
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
        raw_terms = [b.get("term", {}).get("value") for b in bindings if b.get("term")]
        literals = [t for t in raw_terms if isinstance(t, str) and not _looks_like_uri(t)]
        uris = [t for t in raw_terms if isinstance(t, str) and _looks_like_uri(t)]
        labels = _extract_labels(uris, max_items=max_items)
        combined = [t.strip() for t in literals if t and t.strip()]
        combined.extend(labels)
        seen = set()
        deduped: List[str] = []
        for term in combined:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
            if len(deduped) >= max_items:
                break
        return deduped

    results = _run_sparql(filtered=True)
    if len(results) < 3:
        fallback = _run_sparql(filtered=False)
        if fallback:
            return fallback[:max_items]
    return results[:max_items]


def fetch_rdf_graphtri_fragments(
    *,
    query_text: str,
    session_id: str,
    max_items: int = 8,
) -> List[Dict[str, str]]:
    if not query_text or not session_id:
        return []

    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint:
        return []

    sparql = f"""
    SELECT ?turn ?claim ?pred ?obj ?conf ?sal
    WHERE {{
      GRAPH <orion:chat> {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#sessionId> "{_escape_sparql(session_id)}" .
      }}
      GRAPH <orion:enrichment> {{
        ?claim a <http://conjourney.net/orion#Claim> ;
               <http://conjourney.net/orion#subject> ?turn ;
               <http://conjourney.net/orion#predicate> ?pred ;
               <http://conjourney.net/orion#obj> ?obj .
        OPTIONAL {{ ?claim <http://conjourney.net/orion#confidence> ?conf }}
        OPTIONAL {{ ?claim <http://conjourney.net/orion#salience> ?sal }}
      }}
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
    frags: List[Dict[str, str]] = []
    for b in bindings:
        turn = b.get("turn", {}).get("value")
        claim = b.get("claim", {}).get("value")
        pred = b.get("pred", {}).get("value")
        obj = b.get("obj", {}).get("value")
        conf = b.get("conf", {}).get("value")
        sal = b.get("sal", {}).get("value")
        if not turn or not pred or not obj:
            continue
        pred_tail = pred.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        parts = [f"{pred_tail} {obj}"]
        if conf:
            parts.append(f"confidence={conf}")
        if sal:
            parts.append(f"salience={sal}")
        parts.append(f"evidence={turn}")
        text = "Claim: " + " | ".join(parts)
        frags.append(
            {
                "id": claim or turn,
                "source": "rdf",
                "source_ref": "graphdb",
                "uri": claim or turn,
                "text": text,
                "ts": 0.0,
                "tags": ["rdf", "claim"],
                "score": 0.6,
                "meta": {"subject": turn},
            }
        )
        if len(frags) >= max_items:
            break
    return frags


def _build_claim_snippet(pairs: List[Tuple[str, str]]) -> str:
    data: Dict[str, str] = {}
    for predicate, neighbor in pairs:
        key = predicate.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        data[key] = neighbor
    pred = data.get("predicate")
    obj = data.get("obj")
    confidence = data.get("confidence")
    salience = data.get("salience")
    timestamp = data.get("timestamp")
    subject = data.get("subject")
    parts = []
    if pred and obj:
        parts.append(f"{pred} {obj}")
    if confidence:
        parts.append(f"confidence={confidence}")
    if salience:
        parts.append(f"salience={salience}")
    if timestamp:
        parts.append(f"ts={timestamp}")
    if subject:
        parts.append(f"evidence={subject}")
    if not parts:
        return ""
    return f"Claim: {' | '.join(parts)}"


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
    by_subject: Dict[str, List[Tuple[str, str]]] = {}
    for b in bindings:
        node = b.get("node", {}).get("value")
        predicate = b.get("p", {}).get("value")
        neighbor = b.get("neighbor", {}).get("value")
        if not node or not predicate or not neighbor:
            continue
        by_subject.setdefault(node, []).append((predicate, neighbor))

    frags: List[Dict[str, str]] = []
    for s, pairs in by_subject.items():
        if _is_claim_node(s):
            text = _build_claim_snippet(pairs)
        else:
            text = " | ".join([f"{predicate} {neighbor}" for predicate, neighbor in pairs])[:1500]
        if not text:
            continue
        tags = ["rdf", "claim"] if _is_claim_node(s) else ["rdf"]
        frags.append(
            {
                "id": s,
                "source": "rdf",
                "source_ref": "graphdb",
                "uri": s,
                "text": text,
                "ts": 0.0,
                "tags": tags,
                "score": 0.6,
                "meta": {"subject": s},
            }
        )

    return frags[:max_items]


def fetch_rdf_expansion_terms(
    *,
    query_text: str,
    max_items: int = 6,
) -> List[str]:
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

    max_nodes = max(1, min(max_items, 6))
    max_results = max_items * 6
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
    neighbors: List[str] = []
    for b in bindings:
        neighbor = b.get("neighbor", {}).get("value")
        if neighbor:
            neighbors.append(neighbor)

    return _extract_labels(neighbors, max_items=max_items)
