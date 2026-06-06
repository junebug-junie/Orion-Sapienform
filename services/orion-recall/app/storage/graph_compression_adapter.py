from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

import httpx
from sqlalchemy import create_engine, text

logger = logging.getLogger("orion-recall.graph_compression_adapter")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_COMPRESSIONS_GRAPH_URI = "http://conjourney.net/graph/orion/compressions"
_ORN_NS = "http://orion.conjourney.net/ns/compression#"


def _extract_keywords(query_text: str, max_keywords: int = 6) -> list[str]:
    tokens = _TOKEN_RE.findall(query_text.lower())
    seen: set[str] = set()
    kw: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            kw.append(t)
        if len(kw) >= max_keywords:
            break
    return kw


def _score_artifact(
    row: Dict[str, Any],
    keywords: list[str],
) -> float:
    """Simple salience + keyword hit scoring."""
    base = float(row.get("salience") or 0.0)
    region_id = str(row.get("region_id") or "").lower()
    scope = str(row.get("scope") or "").lower()
    hits = sum(1 for kw in keywords if kw in region_id or kw in scope)
    return base + 0.1 * hits


def _fetch_summary_from_fuseki(
    region_id: str,
    rdf_query_url: str,
    rdf_user: str,
    rdf_pass: str,
    timeout_sec: float,
) -> str:
    query = f"""
SELECT ?summary WHERE {{
  GRAPH <{_COMPRESSIONS_GRAPH_URI}> {{
    <{region_id}> <{_ORN_NS}summary> ?summary .
  }}
}}
LIMIT 1
"""
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(
            rdf_query_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            auth=(rdf_user, rdf_pass),
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
    if bindings:
        return bindings[0].get("summary", {}).get("value", "")
    return ""


def fetch_graph_compression_fragments(
    *,
    query_text: str,
    mode: Literal["global", "local", "unified"],
    max_global: int = 5,
    max_local: int = 5,
    scopes: list[str],
    pg_dsn: str,
    rdf_query_url: Optional[str],
    rdf_user: str,
    rdf_pass: str,
    timeout_sec: float,
) -> List[Dict[str, Any]]:
    """
    Query Postgres artifact index, rank by salience + keyword relevance,
    fetch summaries from Fuseki, and return recall fragments.
    Returns [] on any error (never raises).
    """
    try:
        engine = create_engine(pg_dsn, pool_pre_ping=True)
        scope_filter = ", ".join(f"'{s}'" for s in scopes)
        limit = max(max_global, max_local) * 2  # over-fetch then rank

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT region_id, scope, kind, summary_kind, salience, trust_tier,"
                    f" compression_version, generated_at, stale"
                    f" FROM compression_artifacts"
                    f" WHERE scope IN ({scope_filter})"
                    f" AND stale = false"
                    f" ORDER BY salience DESC NULLS LAST"
                    f" LIMIT :limit"
                ),
                {"limit": limit},
            ).mappings().fetchall()

        rows = [dict(r) for r in rows]
        # Defensive Python-side stale filter (the SQL WHERE handles this against a
        # real DB; this guard ensures stale artifacts never leak through).
        rows = [r for r in rows if not r.get("stale")]
        if not rows:
            return []

        keywords = _extract_keywords(query_text)
        scored = sorted(rows, key=lambda r: _score_artifact(r, keywords), reverse=True)

        if mode == "global":
            ranked = scored[:max_global]
        elif mode == "local":
            ranked = scored[:max_local]
        else:  # unified
            ranked = scored[:max_global + max_local]

        fragments: List[Dict[str, Any]] = []
        for row in ranked:
            region_id = row["region_id"]
            summary = ""
            if rdf_query_url:
                try:
                    summary = _fetch_summary_from_fuseki(
                        region_id, rdf_query_url, rdf_user, rdf_pass, timeout_sec
                    )
                except Exception as exc:
                    logger.debug("compression_summary_fetch_failed region_id=%s reason=%s", region_id, exc)
            if not summary:
                summary = (
                    f"[graph compression] {row.get('scope')} {row.get('kind')} region "
                    f"(salience={row.get('salience', 0):.2f}, kind={row.get('summary_kind')})"
                )

            fragments.append(
                {
                    "source": "graph_compression",
                    "source_ref": region_id,
                    "text": summary,
                    "tags": [
                        f"scope:{row.get('scope')}",
                        f"kind:{row.get('kind')}",
                        f"trust:{row.get('trust_tier')}",
                        f"summary_kind:{row.get('summary_kind')}",
                    ],
                    "salience": float(row.get("salience") or 0.0),
                    "compression_version": row.get("compression_version"),
                    "score": _score_artifact(row, keywords),
                }
            )
        return fragments
    except Exception as exc:
        logger.warning("fetch_graph_compression_fragments_failed reason=%s", exc)
        return []
