from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

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

def fetch_rdf_chatturn_fragments(
    *,
    query_text: str,
    session_id: str | None,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    """
    Pull recent ChatTurns from GRAPH <orion:chat> for the session.
    NO keyword filtering at SPARQL layer (sustainable).
    Ranking happens later (vector / lexical) in fusion.
    """
    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint or not session_id:
        return []

    # If you have a timestamp predicate, add it here and ORDER BY DESC(?ts).
    # If not, we order by the turn URI string as a stable proxy.
    sparql = f"""
    SELECT ?turn ?prompt ?response
    WHERE {{
      GRAPH <orion:chat> {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#sessionId> "{_escape_sparql(session_id)}" ;
              <http://conjourney.net/orion#prompt> ?prompt ;
              <http://conjourney.net/orion#response> ?response .
      }}
    }}
    ORDER BY DESC(STR(?turn))
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
    out: List[Dict[str, Any]] = []
    for b in bindings:
        turn = b.get("turn", {}).get("value")
        prompt = (b.get("prompt", {}).get("value") or "").strip()
        response = (b.get("response", {}).get("value") or "").strip()
        if not turn:
            continue

        # Keep the exact text accessible for quoting.
        text = f'ExactUserText: "{prompt}"\nOrionResponse: "{response}"'.strip()

        out.append(
            {
                "id": turn,
                "source": "rdf_chat",
                "source_ref": "graphdb",
                "uri": turn,
                "text": text[:1800],
                "ts": 0.0,
                "tags": ["rdf", "chat", "chatturn"],
                # Base score is neutral; ranking should happen later.
                "score": 0.50,
                "meta": {"session_id": session_id},
            }
        )
    return out

# ---------------------------------------------------------------------
# Backwards-compat exports
# These are imported by legacy pipeline/worker codepaths.
# Even if you don't use them, they must exist to prevent boot cascades.
# ---------------------------------------------------------------------

def fetch_rdf_fragments(*, query_text: str, max_items: int = 8):
    """
    Legacy API: return generic RDF fragments for query_text.
    Prefer the real implementation if present; otherwise degrade safely.
    """
    impl = globals().get("_fetch_rdf_fragments_impl") or globals().get("fetch_rdf_fragments_impl")
    if callable(impl):
        return impl(query_text=query_text, max_items=max_items)
    # If you have fetch_rdf_graphtri_fragments, use it as a weak fallback
    try:
        return _fetch_rdf_neighborhood_fragments(query_text=query_text, max_items=max_items)
    except Exception:
        return []


def fetch_rdf_expansion_terms(*, query_text: str, max_items: int = 6):
    """
    Legacy API: return related terms for query expansion.
    If not implemented, return [] (safe).
    """
    impl = globals().get("_fetch_rdf_expansion_terms_impl") or globals().get("fetch_rdf_expansion_terms_impl")
    if callable(impl):
        return impl(query_text=query_text, max_items=max_items)
    try:
        # If you have a neighborhood scan, reuse labels as expansion terms.
        items = _fetch_rdf_neighborhood_fragments(query_text=query_text, max_items=max_items * 3)
        terms = []
        seen = set()
        for it in items:
            txt = str(it.get("text") or "")
            for token in re.findall(r"[A-Za-z0-9_]{3,}", txt):
                t = token.strip()
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                terms.append(t)
                if len(terms) >= max_items:
                    break
            if len(terms) >= max_items:
                break
        return terms
    except Exception:
        return []


def _fetch_rdf_neighborhood_fragments(*, query_text: str, max_items: int = 8):
    """
    Minimal, safe RDF neighborhood lookup.
    This exists only to satisfy legacy callers if the richer functions were removed.
    """
    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint or not query_text:
        return []
    q = (query_text or "").strip()
    if not q:
        return []

    keywords = _extract_keywords(q, max_keywords=6) if "max_keywords" in _extract_keywords.__code__.co_varnames else _extract_keywords(q)

    # Fallback: reuse your existing _build_sparql_query if present
    if "_build_sparql_query" in globals():
        sparql = _build_sparql_query(keywords, max_nodes=max(1, min(max_items, 6)), max_results=max_items * 4)
    else:
        # extremely basic SPARQL
        filters = " || ".join(f'CONTAINS(LCASE(STR(?o)), "{_escape_sparql(k)}")' for k in keywords)
        sparql = f"""
        SELECT ?s ?p ?o WHERE {{
          ?s ?p ?o .
          FILTER({filters})
        }}
        LIMIT {max_items * 4}
        """

    try:
        resp = requests.post(
            endpoint,
            data=sparql,
            headers={"Content-Type": "application/sparql-query", "Accept": "application/sparql-results+json"},
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
    out = []
    for b in bindings:
        s = b.get("s", {}).get("value") or b.get("node", {}).get("value")
        p = b.get("p", {}).get("value")
        o = b.get("o", {}).get("value") or b.get("neighbor", {}).get("value")
        if not s or not p or not o:
            continue
        out.append(
            {
                "id": s,
                "source": "rdf",
                "source_ref": "graphdb",
                "uri": s,
                "text": f"{p} {o}"[:1500],
                "ts": 0.0,
                "tags": ["rdf"],
                "score": 0.5,
                "meta": {"subject": s},
            }
        )
        if len(out) >= max_items:
            break
    return out
