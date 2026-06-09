from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("orion.graph-compression.federator.substrate")

SUBSTRATE_GRAPH = "http://conjourney.net/graph/orion/substrate"
Triple = tuple[str, str, str]


class SubstrateFederator:
    """
    Fetches triples from the substrate graph using bounded SPARQL.
    Degraded to empty list on any error.
    """

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
        query = f"""
SELECT ?s ?p ?o WHERE {{
  {{
    SELECT DISTINCT ?s WHERE {{
      GRAPH <{SUBSTRATE_GRAPH}> {{ ?s ?p0 ?o0 }}
    }}
    LIMIT {max_nodes}
  }}
  GRAPH <{SUBSTRATE_GRAPH}> {{ ?s ?p ?o }}
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
            logger.warning("substrate_federator_fetch_failed reason=%s", exc)
            return []
        return [
            (b["s"]["value"], b["p"]["value"], b["o"]["value"])
            for b in bindings
            if "s" in b and "p" in b and "o" in b
        ]
