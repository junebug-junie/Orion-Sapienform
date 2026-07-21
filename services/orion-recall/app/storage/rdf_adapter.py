from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests

from app.settings import settings


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")

# Predicate ChatTurn/Claim timestamps are written under (services/orion-rdf-writer/app/rdf_builder.py:377,608).
# rdf_builder writes it as an xsd:string literal (ISO-8601-ish, sometimes space- instead of "T"-separated,
# per services/orion-vector-writer/app/chat_history.py:46-48); some other node types reuse the same
# predicate with an xsd:double epoch literal (e.g. CognitiveTrace at rdf_builder.py:260), so parsing here
# tolerates both shapes.
_RDF_TIMESTAMP_PREDICATE = "http://conjourney.net/orion#timestamp"


def _parse_rdf_timestamp(value: Any) -> float:
    """Parse a ChatTurn/Claim ``ORION.timestamp`` literal into a recall-scoring epoch float.

    Returns 0.0 (the historical placeholder) when the value is missing or unparseable, so
    ``scoring._compute_recency_factor`` falls back to its neutral 0.5 recency weight rather than
    exploding on a bad literal.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    normalized = text
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    normalized = normalized.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except (TypeError, ValueError):
        return 0.0


_ORION_GRAPH_IRIS: Dict[str, str] = {
    "orion:chat": "http://conjourney.net/graph/orion/chat",
    "orion:collapse": "http://conjourney.net/graph/orion/collapse",
    "orion:enrichment": "http://conjourney.net/graph/orion/enrichment",
    "orion:cognition": "http://conjourney.net/graph/orion/cognition",
    "orion:metacog": "http://conjourney.net/graph/orion/metacog",
    "orion:chat:social": "http://conjourney.net/graph/orion/chat/social",
    "orion:default": "http://conjourney.net/graph/orion/default",
    "orion:self": "http://conjourney.net/graph/orion/self",
    "orion:self:induced": "http://conjourney.net/graph/orion/self/induced",
    "orion:self:reflective": "http://conjourney.net/graph/orion/self/reflective",
    "orion:compressions": "http://conjourney.net/graph/orion/compressions",
}


def _infer_rdf_store_backend() -> str:
    explicit = (os.getenv("GRAPH_BACKEND") or os.getenv("RDF_STORE_BACKEND") or "").strip().lower()
    if explicit:
        return explicit
    endpoint = (settings.RECALL_RDF_ENDPOINT_URL or "").lower()
    if "/repositories/" in endpoint or ":7200" in endpoint:
        return "graphdb"
    return "fuseki"


def graph_iri_for_sparql(graph_name: str) -> str:
    """Map compact Orion graph names to SPARQL GRAPH IRIs for the active RDF backend."""
    raw = str(graph_name or "").strip()
    if not raw:
        return raw
    if _infer_rdf_store_backend() == "graphdb":
        normalize = os.getenv("RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not normalize:
            return raw
    lower = raw.lower()
    if lower.startswith(("http://", "https://", "urn:")):
        return raw
    if raw in _ORION_GRAPH_IRIS:
        return _ORION_GRAPH_IRIS[raw]
    safe = re.sub(r"[^A-Za-z0-9._:/-]+", "_", raw)
    safe = safe.replace(":", "/").strip("/")
    if not safe:
        safe = "unknown"
    return f"http://conjourney.net/graph/{safe}"


def _sparql_graph(graph_name: str) -> str:
    return f"GRAPH <{graph_iri_for_sparql(graph_name)}>"


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


def _build_chatturn_keyword_filter(keywords: List[str]) -> str:
    if not keywords:
        return ""
    filters = " || ".join(
        f'CONTAINS(LCASE(STR(?prompt)), "{_escape_sparql(keyword)}") || '
        f'CONTAINS(LCASE(STR(?response)), "{_escape_sparql(keyword)}")'
        for keyword in keywords
    )
    return f"FILTER({filters})"


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
    Pull recent ChatTurns from the Orion chat graph for the session.
    NO keyword filtering at SPARQL layer (sustainable).
    Ranking happens later (vector / lexical) in fusion.
    """
    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint or not query_text:
        return []

    keywords = _extract_keywords(query_text)
    filter_clause = _build_chatturn_keyword_filter(keywords) if keywords else ""

    # rdf_builder.py writes ORION.timestamp on every ChatTurn (rdf_builder.py:377); select and
    # order on it (real recency) instead of the turn URI string (an arbitrary, permanently-stable
    # UUID sort that used to masquerade as recency).
    chat_graph = _sparql_graph("orion:chat")
    sparql = f"""
    SELECT ?turn ?prompt ?response ?ts
    WHERE {{
      {chat_graph} {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#prompt> ?prompt ;
              <http://conjourney.net/orion#response> ?response .
        OPTIONAL {{ ?turn <{_RDF_TIMESTAMP_PREDICATE}> ?ts }}
      }}
      {filter_clause}
    }}
    ORDER BY DESC(?ts)
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
        ts_raw = b.get("ts", {}).get("value")
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
                "ts": _parse_rdf_timestamp(ts_raw),
                "tags": ["rdf", "chat", "chatturn"],
                # Base score is neutral; ranking should happen later.
                "score": 0.50,
                "meta": {},
            }
        )
    return out


def fetch_rdf_chatturn_exact_matches(
    *,
    tokens: List[str],
    session_id: str | None,
    max_items: int = 20,
) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    endpoint = settings.RECALL_RDF_ENDPOINT_URL
    if not endpoint:
        return []

    filters = []
    for token in tokens:
        escaped = _escape_sparql(token.lower())
        filters.append(f'CONTAINS(LCASE(STR(?prompt)), "{escaped}")')
        filters.append(f'CONTAINS(LCASE(STR(?response)), "{escaped}")')
    filter_clause = " || ".join(filters) if filters else "TRUE"

    chat_graph = _sparql_graph("orion:chat")
    sparql = f"""
    SELECT ?turn ?prompt ?response
    WHERE {{
      {chat_graph} {{
        ?turn a <http://conjourney.net/orion#ChatTurn> ;
              <http://conjourney.net/orion#prompt> ?prompt ;
              <http://conjourney.net/orion#response> ?response .
      }}
      FILTER({filter_clause})
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
                "score": 0.7,
                "meta": {},
            }
        )
        if len(out) >= max_items:
            break
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
    try:
        return _fetch_rdf_neighborhood_fragments(query_text=query_text, max_items=max_items)
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
    def _fetch_subject_literals(subject: str) -> str:
        if not subject:
            return ""
        sparql = f"""
        SELECT ?label ?prompt ?response
        WHERE {{
          OPTIONAL {{ <{_escape_sparql(subject)}> <http://www.w3.org/2000/01/rdf-schema#label> ?label }}
          OPTIONAL {{ <{_escape_sparql(subject)}> <http://conjourney.net/orion#prompt> ?prompt }}
          OPTIONAL {{ <{_escape_sparql(subject)}> <http://conjourney.net/orion#response> ?response }}
        }}
        LIMIT 1
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
            return ""
        if resp.status_code != 200:
            return ""
        try:
            data = resp.json()
        except Exception:
            return ""
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return ""
        row = bindings[0]
        label = (row.get("label", {}).get("value") or "").strip()
        prompt = (row.get("prompt", {}).get("value") or "").strip()
        response = (row.get("response", {}).get("value") or "").strip()
        parts: List[str] = []
        if label:
            parts.append(f"label: {label}")
        if prompt or response:
            parts.append(f'ExactUserText: "{prompt}"')
            parts.append(f'OrionResponse: "{response}"')
        return "\n".join(p for p in parts if p)

    out = []
    for b in bindings:
        s = b.get("s", {}).get("value") or b.get("node", {}).get("value")
        p = b.get("p", {}).get("value")
        o = b.get("o", {}).get("value") or b.get("neighbor", {}).get("value")
        if not s or not p or not o:
            continue
        text = f"{p} {o}"[:1500]
        if p.endswith("rdf-syntax-ns#type") and (
            o.endswith("ChatTurn") or o.endswith("Entity")
        ):
            literal = _fetch_subject_literals(s)
            if literal:
                text = literal[:1500]
        out.append(
            {
                "id": s,
                "source": "rdf",
                "source_ref": "graphdb",
                "uri": s,
                "text": text,
                "ts": 0.0,
                "tags": ["rdf"],
                "score": 0.5,
                "meta": {"subject": s},
            }
        )
        if len(out) >= max_items:
            break
    return out
