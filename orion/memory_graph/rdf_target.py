"""Resolve memory-graph RDF write target: Fuseki graph store + SPARQL update vs legacy GraphDB."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryGraphGraphDbTarget:
    kind: Literal["graphdb"]
    graphdb_url: str
    repo: str
    user: str
    password: str


@dataclass(frozen=True)
class MemoryGraphSparqlTarget:
    kind: Literal["sparql"]
    graph_store_url: str
    update_url: str
    user: str
    password: str


def resolve_memory_graph_rdf_target() -> MemoryGraphGraphDbTarget | MemoryGraphSparqlTarget | None:
    """Prefer RDF store graph HTTP when configured; legacy GraphDB only when ``MEMORY_GRAPH_APPROVAL_BACKEND=graphdb``."""
    mode = (os.getenv("MEMORY_GRAPH_APPROVAL_BACKEND") or "auto").strip().lower()
    gs = (os.getenv("RDF_STORE_GRAPH_STORE_URL") or "").strip()
    up = (os.getenv("RDF_STORE_UPDATE_URL") or "").strip()
    u = (os.getenv("RDF_STORE_USER") or "").strip()
    p = (os.getenv("RDF_STORE_PASS") or "").strip()

    if mode == "graphdb":
        base = (os.getenv("GRAPHDB_URL") or "").strip().rstrip("/")
        repo = (os.getenv("GRAPHDB_REPO") or "collapse").strip() or "collapse"
        if not base:
            return None
        return MemoryGraphGraphDbTarget(
            kind="graphdb",
            graphdb_url=base,
            repo=repo,
            user=(os.getenv("GRAPHDB_USER") or "").strip(),
            password=(os.getenv("GRAPHDB_PASS") or "").strip(),
        )

    if mode in {"auto", "sparql", "fuseki", ""} and gs and up:
        return MemoryGraphSparqlTarget(kind="sparql", graph_store_url=gs, update_url=up, user=u, password=p)

    return None
