from __future__ import annotations

from typing import Optional

import requests
from rdflib import Graph


def graphdb_statements_url(base_url: str, repo: str) -> str:
    b = base_url.rstrip("/")
    return f"{b}/repositories/{repo}/statements"


def insert_batch(
    graph: Graph,
    *,
    named_graph: str,
    graphdb_url: str,
    repo: str,
    user: str = "",
    password: str = "",
    session: Optional[requests.Session] = None,
) -> None:
    """POST Turtle into GraphDB ``/repositories/{repo}/statements`` with ``named-graph-uri`` query param."""
    from urllib.parse import quote

    url = graphdb_statements_url(graphdb_url, repo)
    ng = quote(named_graph, safe="")
    full_url = f"{url}?named-graph-uri={ng}"
    ttl = graph.serialize(format="turtle")
    sess = session or requests.Session()
    auth = (user, password) if user or password else None
    headers = {"Content-Type": "application/x-turtle"}
    r = sess.post(full_url, data=ttl.encode("utf-8"), headers=headers, auth=auth, timeout=60)
    r.raise_for_status()


def compensate_batch(
    batch_id: str,
    *,
    graphdb_url: str,
    repo: str,
    user: str = "",
    password: str = "",
    session: Optional[requests.Session] = None,
) -> None:
    """DELETE all triples on subjects tagged with ``orionmem:revisionBatch`` for ``batch_id``."""
    url = graphdb_statements_url(graphdb_url, repo)
    # revisionBatch values are operator-controlled batch UUIDs from approve path
    sparql = f"""PREFIX orionmem: <https://orion.local/ns/mem/v2026-05#>
DELETE {{
  ?s ?p ?o .
}}
WHERE {{
  ?s orionmem:revisionBatch "{batch_id}" .
  ?s ?p ?o .
}}
"""
    sess = session or requests.Session()
    auth = (user, password) if user or password else None
    r = sess.post(
        url,
        data=sparql.encode("utf-8"),
        headers={"Content-Type": "application/sparql-update"},
        auth=auth,
        timeout=120,
    )
    r.raise_for_status()
