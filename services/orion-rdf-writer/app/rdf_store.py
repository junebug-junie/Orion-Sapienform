from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

import httpx

from app.settings import Settings


@dataclass(frozen=True)
class RdfWriteResult:
    backend: str
    graph_name: str | None
    normalized_graph_uri: str | None
    byte_count: int
    status_code: int | None = None
    endpoint: str | None = None
    elapsed_ms: float | None = None


class RdfStoreClient(Protocol):
    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult: ...

    async def health(self) -> dict[str, Any]: ...


def normalize_graph_name(graph_name: str | None) -> str | None:
    if graph_name is None:
        return None
    raw = str(graph_name).strip()
    if not raw:
        return None
    lower = raw.lower()
    if lower.startswith("http://") or lower.startswith("https://") or lower.startswith("urn:"):
        return raw
    mapping = {
        "orion:chat": "http://conjourney.net/graph/orion/chat",
        "orion:collapse": "http://conjourney.net/graph/orion/collapse",
        "orion:enrichment": "http://conjourney.net/graph/orion/enrichment",
        "orion:cognition": "http://conjourney.net/graph/orion/cognition",
        "orion:metacog": "http://conjourney.net/graph/orion/metacog",
        "orion:chat:social": "http://conjourney.net/graph/orion/chat/social",
        "orion:default": "http://conjourney.net/graph/orion/default",
        "orion:compressions": "http://conjourney.net/graph/orion/compressions",
        "orion:self": "http://conjourney.net/graph/orion/self",
        "orion:self:induced": "http://conjourney.net/graph/orion/self/induced",
        "orion:self:reflective": "http://conjourney.net/graph/orion/self/reflective",
    }
    if raw in mapping:
        return mapping[raw]
    safe = re.sub(r"[^A-Za-z0-9._:/-]+", "_", raw)
    safe = safe.replace(":", "/").strip("/")
    if not safe:
        safe = "unknown"
    return f"http://conjourney.net/graph/{safe}"


def _strip_credentials(url: str | None) -> str | None:
    if not url:
        return None
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.port:
        netloc = f"{netloc}:{p.port}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def _httpx_auth(user: str | None, password: str | None) -> httpx.Auth | None:
    if user and password is not None:
        return (user, password)
    return None


def _httpx_limits(settings: Settings) -> httpx.Limits:
    return httpx.Limits(
        max_connections=int(settings.RDF_WRITE_HTTP_MAX_CONNECTIONS),
        max_keepalive_connections=int(settings.RDF_WRITE_HTTP_MAX_KEEPALIVE),
    )


def httpx_limits_for_settings(settings: Settings) -> httpx.Limits:
    """Used by the RDF write pipeline to size the shared AsyncClient connection pool."""
    return _httpx_limits(settings)


class GraphDbRdfStoreClient:
    # GraphDB defaults keep the legacy ?context=<compact> form (e.g. <orion:chat>) for
    # backward compatibility. Set RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT=true to emit
    # normalized absolute graph IRIs in ?context= (matching Fuseki-style naming).

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        if not settings.GRAPHDB_URL:
            raise ValueError("GRAPHDB_URL is required when RDF_STORE_BACKEND=graphdb")

    @property
    def backend(self) -> str:
        return "graphdb"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        base_url = (
            f"{self._settings.GRAPHDB_URL.rstrip('/')}"
            f"/repositories/{self._settings.GRAPHDB_REPO}/statements"
        )
        params: dict[str, str] = {}
        if graph_name:
            ctx = graph_name
            if self._settings.RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT:
                ctx = normalize_graph_name(graph_name) or graph_name
            params["context"] = f"<{ctx}>"
        headers = {"Content-Type": "text/plain"}
        auth = _httpx_auth(self._settings.GRAPHDB_USER, self._settings.GRAPHDB_PASS)
        resp = await self._client.post(
            base_url,
            content=content,
            headers=headers,
            params=params,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=normalize_graph_name(graph_name),
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=base_url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "endpoint": f"{self._settings.GRAPHDB_URL.rstrip('/')}/repositories/{self._settings.GRAPHDB_REPO}/statements",
            "repo": self._settings.GRAPHDB_REPO,
        }


class FusekiRdfStoreClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        base = (settings.RDF_STORE_BASE_URL or "http://orion-athena-fuseki:3030").rstrip("/")
        ds = settings.RDF_STORE_DATASET.strip().strip("/")
        self._query_url = settings.RDF_STORE_QUERY_URL or f"{base}/{ds}/query"
        self._update_url = settings.RDF_STORE_UPDATE_URL or f"{base}/{ds}/update"
        self._graph_store_url = settings.RDF_STORE_GRAPH_STORE_URL or f"{base}/{ds}/data"

    @property
    def backend(self) -> str:
        return "fuseki"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        url = self._graph_store_url
        params: dict[str, str] = {}
        ng = normalize_graph_name(graph_name)
        if ng:
            params["graph"] = ng
        headers = {"Content-Type": "application/n-triples"}
        auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
        resp = await self._client.post(
            url,
            content=content,
            headers=headers,
            params=params,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=ng,
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "query_url": _strip_credentials(self._query_url),
            "update_url": _strip_credentials(self._update_url),
            "graph_store_url": _strip_credentials(self._graph_store_url),
            "dataset": self._settings.RDF_STORE_DATASET,
        }


class GenericSparqlRdfStoreClient:
    """
    Conservative adapter: prefer Graph Store HTTP POST; optional SPARQL UPDATE fallback.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client
        self._graph_store_url = settings.RDF_STORE_GRAPH_STORE_URL
        self._update_url = settings.RDF_STORE_UPDATE_URL

    @property
    def backend(self) -> str:
        return "generic"

    async def write_graph(self, content: str, graph_name: str | None = None) -> RdfWriteResult:
        t0 = time.perf_counter()
        ng = normalize_graph_name(graph_name)
        if self._graph_store_url:
            url = self._graph_store_url
            params: dict[str, str] = {}
            if ng:
                params["graph"] = ng
            headers = {"Content-Type": "application/n-triples"}
            auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
            resp = await self._client.post(
                url,
                content=content,
                headers=headers,
                params=params,
                auth=auth,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return RdfWriteResult(
                backend=self.backend,
                graph_name=graph_name,
                normalized_graph_uri=ng,
                byte_count=len(content.encode("utf-8")),
                status_code=resp.status_code,
                endpoint=url,
                elapsed_ms=elapsed_ms,
            )
        if not self._update_url:
            raise ValueError("Generic RDF store requires RDF_STORE_GRAPH_STORE_URL or RDF_STORE_UPDATE_URL")
        if ng:
            body = f"INSERT DATA {{ GRAPH <{ng}> {{ {content} }} }}"
        else:
            body = f"INSERT DATA {{ {content} }}"
        headers = {"Content-Type": "application/sparql-update"}
        auth = _httpx_auth(self._settings.RDF_STORE_USER, self._settings.RDF_STORE_PASS)
        resp = await self._client.post(
            self._update_url,
            content=body,
            headers=headers,
            auth=auth,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return RdfWriteResult(
            backend=self.backend,
            graph_name=graph_name,
            normalized_graph_uri=ng,
            byte_count=len(content.encode("utf-8")),
            status_code=resp.status_code,
            endpoint=self._update_url,
            elapsed_ms=elapsed_ms,
        )

    async def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "graph_store_url": _strip_credentials(self._graph_store_url),
            "update_url": _strip_credentials(self._update_url),
            "query_url": _strip_credentials(self._settings.RDF_STORE_QUERY_URL),
        }


def build_rdf_store_client(settings: Settings, client: httpx.AsyncClient) -> RdfStoreClient:
    b = (settings.RDF_STORE_BACKEND or "fuseki").strip().lower()
    if b == "graphdb":
        return GraphDbRdfStoreClient(settings, client)
    if b == "fuseki":
        return FusekiRdfStoreClient(settings, client)
    if b == "generic":
        return GenericSparqlRdfStoreClient(settings, client)
    if b == "rdf4j":
        if not settings.RDF_STORE_GRAPH_STORE_URL and not settings.RDF_STORE_UPDATE_URL:
            raise ValueError(
                "RDF_STORE_BACKEND=rdf4j requires RDF_STORE_GRAPH_STORE_URL and/or RDF_STORE_UPDATE_URL "
                "(alias to generic adapter in this spike)."
            )
        return GenericSparqlRdfStoreClient(settings, client)
    raise ValueError(f"Unknown RDF_STORE_BACKEND={settings.RDF_STORE_BACKEND!r} (expected graphdb|fuseki|generic|rdf4j)")
