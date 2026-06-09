from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("orion.graph-compression.federator.self_study")

SELF_STUDY_GRAPHS = [
    "http://conjourney.net/graph/orion/self",
    "http://conjourney.net/graph/orion/self/induced",
    "http://conjourney.net/graph/orion/self/reflective",
]

Triple = tuple[str, str, str]


class SelfStudyFederator:
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

    def fetch(self, *, max_nodes: int = 500) -> list[Triple]:
        # Inner subquery uses ?p0 ?o0 to only bind ?s, avoiding AND semantics
        inner_clauses = "\n  ".join(
            f"GRAPH <{g}> {{ ?s ?p0 ?o0 }}" for g in SELF_STUDY_GRAPHS
        )
        outer_clauses = "\n  ".join(
            f"GRAPH <{g}> {{ ?s ?p ?o }}" for g in SELF_STUDY_GRAPHS
        )
        query = f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      {{ {inner_clauses} }}
    }}
    LIMIT {max_nodes}
  }}
  {{ {outer_clauses} }}
}}
"""
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
            logger.warning("self_study_federator_fetch_failed reason=%s", exc)
            return []
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
        ]
