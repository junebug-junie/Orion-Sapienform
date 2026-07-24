"""Shared graph backend resolution and SPARQL HTTP clients."""

from orion.graph.persistence_router import GraphPersistenceRouter
from orion.graph.persistence_routes import (
    RouteTarget,
    WorkloadRoute,
    load_persistence_routes,
    resolve_workload_route,
)
from orion.graph.backend_config import (
    GraphBackendConfig,
    is_legacy_graphdb_enabled,
    resolve_autonomy_read_query_url,
    resolve_generic_sparql_read_query_url,
    resolve_graph_backend,
    resolve_graph_query_url,
    resolve_graph_store_url,
    resolve_graph_update_url,
    strip_graph_credentials,
)

__all__ = [
    "GraphPersistenceRouter",
    "GraphBackendConfig",
    "RouteTarget",
    "WorkloadRoute",
    "load_persistence_routes",
    "resolve_workload_route",
    "is_legacy_graphdb_enabled",
    "resolve_autonomy_read_query_url",
    "resolve_generic_sparql_read_query_url",
    "resolve_graph_backend",
    "resolve_graph_query_url",
    "resolve_graph_store_url",
    "resolve_graph_update_url",
    "strip_graph_credentials",
]

# orion.graph.sparql_client is Fuseki/SPARQL-specific (needs `requests`) and was
# previously re-exported here unconditionally -- meaning EVERY consumer of this
# package, including Falkor-only ones (orion-bus-mirror, orion-graph-compression,
# orion-meta-tags, orion-recall -- 6 real call sites across 4 services import
# orion.graph.falkor_client directly, none of it SPARQL-related), was forced to
# have `requests` installed just to import anything from orion.graph at all.
# orion-bus-mirror never had `requests` in its own requirements.txt and crash-
# looped on this (ModuleNotFoundError, live 2026-07-24) -- confirmed not a recent
# regression (git log on that requirements.txt shows no `requests` in the last 3
# commits touching it), a latent bug that surfaced on this restart. Made optional
# instead of adding `requests` to bus-mirror's requirements.txt as a one-off patch,
# since the same crash risk applies to any of the other 3 services too, and this
# repo is actively decommissioning Fuseki -- a Falkor-only consumer requiring
# Fuseki's own HTTP client library is exactly the wrong direction for that effort.
try:
    from orion.graph.sparql_client import (
        GraphStoreClient,
        SparqlHttpClient,
        SparqlQueryClient,
        SparqlUpdateClient,
        redact_http_url_for_log,
        resolve_substrate_sparql_http_basic_auth,
    )

    __all__ += [
        "GraphStoreClient",
        "SparqlHttpClient",
        "SparqlQueryClient",
        "SparqlUpdateClient",
        "redact_http_url_for_log",
        "resolve_substrate_sparql_http_basic_auth",
    ]
except ImportError:
    pass
