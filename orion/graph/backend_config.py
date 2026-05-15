"""Central graph backend resolution (Fuseki / SPARQL vs explicit GraphDB legacy).

Rules (summary):
- ``GRAPH_BACKEND=fuseki`` or ``RDF_STORE_BACKEND=fuseki`` → derive Fuseki URLs from base + dataset.
- ``GRAPH_BACKEND=sparql`` → use explicit ``RDF_STORE_*`` URLs.
- ``GRAPH_BACKEND=graphdb`` → legacy GraphDB only when explicitly selected.
- ``GRAPH_BACKEND=auto`` or unset → if ``RDF_STORE_QUERY_URL`` / ``AUTONOMY_GRAPH_QUERY_URL`` / Fuseki
  derivation applies, use SPARQL. **Never** enable GraphDB from ``GRAPHDB_URL`` alone.
- ``GRAPH_BACKEND=disabled`` → disabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True)
class GraphBackendConfig:
    backend: str  # fuseki | sparql | graphdb | disabled
    query_url: str | None
    update_url: str | None
    graph_store_url: str | None
    dataset: str | None
    user: str | None
    password: str | None
    legacy_graphdb: bool
    source: str


def strip_graph_credentials(url: str | None) -> str | None:
    if not url:
        return None
    try:
        p = urlparse(url)
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:  # noqa: BLE001
        return url


def _strip(s: str | None) -> str:
    return (s or "").strip()


def _fuseki_dataset() -> str:
    return (_strip(os.getenv("RDF_STORE_DATASET")) or "orion").strip("/")


def _fuseki_base() -> str:
    return _strip(os.getenv("RDF_STORE_BASE_URL")).rstrip("/")


def _derive_fuseki_urls(base: str, dataset: str) -> tuple[str, str, str, str]:
    q = f"{base}/{dataset}/query"
    u = f"{base}/{dataset}/update"
    g = f"{base}/{dataset}/data"
    return q, u, g, f"derived:{base}/{dataset}"


def resolve_graph_backend(environ: Mapping[str, str] | None = None) -> GraphBackendConfig:
    env = dict(os.environ if environ is None else environ)
    gb = _strip(env.get("GRAPH_BACKEND")).lower()
    rdf_b = _strip(env.get("RDF_STORE_BACKEND")).lower()

    if gb == "disabled":
        return GraphBackendConfig(
            backend="disabled",
            query_url=None,
            update_url=None,
            graph_store_url=None,
            dataset=None,
            user=None,
            password=None,
            legacy_graphdb=False,
            source="GRAPH_BACKEND=disabled",
        )

    if gb == "graphdb" or rdf_b == "graphdb":
        base = _strip(env.get("GRAPHDB_URL")).rstrip("/")
        repo = _strip(env.get("GRAPHDB_REPO")) or "collapse"
        user = _strip(env.get("GRAPHDB_USER")) or None
        password = _strip(env.get("GRAPHDB_PASS")) or None
        if not base:
            return GraphBackendConfig(
                backend="graphdb",
                query_url=None,
                update_url=None,
                graph_store_url=None,
                dataset=repo,
                user=user,
                password=password,
                legacy_graphdb=True,
                source="GRAPH_BACKEND=graphdb_missing_url",
            )
        endpoint = base if "/repositories/" in base else f"{base}/repositories/{repo}"
        statements = f"{endpoint}/statements"
        return GraphBackendConfig(
            backend="graphdb",
            query_url=endpoint,
            update_url=statements,
            graph_store_url=statements,
            dataset=repo,
            user=user or None,
            password=password or None,
            legacy_graphdb=True,
            source="GRAPH_BACKEND=graphdb",
        )

    if gb == "fuseki" or rdf_b == "fuseki":
        base = _fuseki_base()
        ds = _fuseki_dataset()
        if not base:
            return GraphBackendConfig(
                backend="fuseki",
                query_url=_strip(env.get("RDF_STORE_QUERY_URL")) or None,
                update_url=_strip(env.get("RDF_STORE_UPDATE_URL")) or None,
                graph_store_url=_strip(env.get("RDF_STORE_GRAPH_STORE_URL")) or None,
                dataset=ds,
                user=_strip(env.get("RDF_STORE_USER")) or None,
                password=_strip(env.get("RDF_STORE_PASS")) or None,
                legacy_graphdb=False,
                source="RDF_STORE_BACKEND=fuseki_missing_base",
            )
        q, u, g, src = _derive_fuseki_urls(base, ds)
        return GraphBackendConfig(
            backend="fuseki",
            query_url=_strip(env.get("RDF_STORE_QUERY_URL")) or q,
            update_url=_strip(env.get("RDF_STORE_UPDATE_URL")) or u,
            graph_store_url=_strip(env.get("RDF_STORE_GRAPH_STORE_URL")) or g,
            dataset=ds,
            user=_strip(env.get("RDF_STORE_USER")) or None,
            password=_strip(env.get("RDF_STORE_PASS")) or None,
            legacy_graphdb=False,
            source=src,
        )

    if gb == "sparql":
        q = _strip(env.get("RDF_STORE_QUERY_URL")) or None
        u = _strip(env.get("RDF_STORE_UPDATE_URL")) or None
        g = _strip(env.get("RDF_STORE_GRAPH_STORE_URL")) or None
        return GraphBackendConfig(
            backend="sparql",
            query_url=q,
            update_url=u,
            graph_store_url=g,
            dataset=_strip(env.get("RDF_STORE_DATASET")) or None,
            user=_strip(env.get("RDF_STORE_USER")) or None,
            password=_strip(env.get("RDF_STORE_PASS")) or None,
            legacy_graphdb=False,
            source="GRAPH_BACKEND=sparql",
        )

    # auto / unset
    q_explicit = _strip(env.get("RDF_STORE_QUERY_URL")) or None
    if q_explicit:
        return GraphBackendConfig(
            backend="sparql",
            query_url=q_explicit,
            update_url=_strip(env.get("RDF_STORE_UPDATE_URL")) or None,
            graph_store_url=_strip(env.get("RDF_STORE_GRAPH_STORE_URL")) or None,
            dataset=_strip(env.get("RDF_STORE_DATASET")) or None,
            user=_strip(env.get("RDF_STORE_USER")) or None,
            password=_strip(env.get("RDF_STORE_PASS")) or None,
            legacy_graphdb=False,
            source="RDF_STORE_QUERY_URL",
        )

    return GraphBackendConfig(
        backend="disabled",
        query_url=None,
        update_url=None,
        graph_store_url=None,
        dataset=None,
        user=None,
        password=None,
        legacy_graphdb=False,
        source="graph_backend_auto_no_sparql_config",
    )


def resolve_graph_query_url(environ: Mapping[str, str] | None = None) -> str | None:
    return resolve_graph_backend(environ).query_url


def resolve_graph_update_url(environ: Mapping[str, str] | None = None) -> str | None:
    return resolve_graph_backend(environ).update_url


def resolve_graph_store_url(environ: Mapping[str, str] | None = None) -> str | None:
    return resolve_graph_backend(environ).graph_store_url


def is_legacy_graphdb_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return resolve_graph_backend(environ).legacy_graphdb


def resolve_autonomy_read_query_url(environ: Mapping[str, str] | None = None) -> tuple[str | None, str]:
    """SPARQL query URL for autonomy reads (AUTONOMY_GRAPH_QUERY_URL wins, then RDF_STORE, then Fuseki derive)."""
    env = dict(os.environ if environ is None else environ)
    q = _strip(env.get("AUTONOMY_GRAPH_QUERY_URL"))
    if q:
        return q, "AUTONOMY_GRAPH_QUERY_URL"
    q = _strip(env.get("RDF_STORE_QUERY_URL"))
    if q:
        return q, "RDF_STORE_QUERY_URL"
    cfg = resolve_graph_backend(env)
    if cfg.query_url and cfg.backend in {"fuseki", "sparql"}:
        return cfg.query_url, cfg.source
    base = _fuseki_base()
    ds = _fuseki_dataset()
    if _strip(env.get("RDF_STORE_BACKEND")).lower() == "fuseki" and base:
        url, _, _, src = _derive_fuseki_urls(base, ds)
        return url, src
    return None, "unconfigured"


def resolve_generic_sparql_read_query_url(environ: Mapping[str, str] | None = None) -> tuple[str | None, str]:
    """Non-autonomy reads (self-study, orionmem): RDF_STORE_QUERY_URL first, then autonomy URL, then resolver."""
    env = dict(os.environ if environ is None else environ)
    q = _strip(env.get("RDF_STORE_QUERY_URL"))
    if q:
        return q, "RDF_STORE_QUERY_URL"
    q = _strip(env.get("AUTONOMY_GRAPH_QUERY_URL"))
    if q:
        return q, "AUTONOMY_GRAPH_QUERY_URL"
    cfg = resolve_graph_backend(env)
    if cfg.query_url and cfg.backend in {"fuseki", "sparql"}:
        return cfg.query_url, cfg.source
    base = _fuseki_base()
    ds = _fuseki_dataset()
    if _strip(env.get("RDF_STORE_BACKEND")).lower() == "fuseki" and base:
        url, _, _, src = _derive_fuseki_urls(base, ds)
        return url, src
    return None, "unconfigured"


def resolve_rdf_store_auth(environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    env = dict(os.environ if environ is None else environ)
    u = _strip(env.get("RDF_STORE_USER")) or None
    p = _strip(env.get("RDF_STORE_PASS")) or None
    return u, p
