from __future__ import annotations

import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger("orion.graph-compression.federator.episodic")

EPISODIC_GRAPHS = [
    "http://conjourney.net/graph/orion/chat",
    "http://conjourney.net/graph/orion/enrichment",
    "http://conjourney.net/graph/orion/collapse",
    "http://conjourney.net/graph/orion/cognition",
    "http://conjourney.net/graph/orion/metacog",
    "http://conjourney.net/graph/orion/chat/social",
    # Autonomy graphs are written under graph/autonomy/* (NOT graph/orion/autonomy/*)
    # — see orion/autonomy/constants.py.
    "http://conjourney.net/graph/autonomy/identity",
    "http://conjourney.net/graph/autonomy/drives",
    "http://conjourney.net/graph/autonomy/goals",
]

Triple = Tuple[str, str, str]


class EpisodicFederator:
    def __init__(
        self,
        *,
        query_url: str,
        user: str,
        password: str,
        timeout_sec: float,
    ) -> None:
        self._query_url = query_url
        self._auth = (user, password)
        self._timeout = timeout_sec

    def _build_sparql(self, max_nodes: int = 2000) -> str:
        # UNION (not conjunction) across graphs: a subject/triple may live in any
        # one of the episodic graphs. Concatenating the GRAPH clauses would require
        # the same triple to exist in ALL graphs simultaneously (≈ empty result).
        union_block = "\n      UNION\n      ".join(
            f"{{ GRAPH <{g}> {{ ?s ?p ?o }} }}" for g in EPISODIC_GRAPHS
        )
        return f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      {union_block}
    }}
    LIMIT {max_nodes}
  }}
  {union_block}
}}
"""

    def fetch(self, *, max_nodes: int = 2000) -> List[Triple]:
        query = self._build_sparql(max_nodes)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._query_url,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                    auth=self._auth,
                )
                resp.raise_for_status()
                bindings = resp.json().get("results", {}).get("bindings", [])
        except Exception as exc:
            logger.warning("episodic_federator_fetch_failed reason=%s", exc)
            return []
        # Keep only object terms that are graph nodes (IRIs/bnodes). Literal
        # objects (chat text, timestamps) are not entities and must not become
        # cluster nodes — they would later be serialized as invalid <IRI>s.
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
            and b["o"].get("type") in ("uri", "bnode")
        ]
