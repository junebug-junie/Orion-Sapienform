"""Minimal SPARQL Protocol HTTP clients (Fuseki + generic SPARQL endpoints)."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import requests


def redact_http_url_for_log(url: str) -> str:
    """Host + path + query + fragment only (strips userinfo from URL)."""
    p = urlparse(url)
    netloc = p.hostname or ""
    if p.port:
        netloc = f"{netloc}:{p.port}"
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def resolve_substrate_sparql_http_basic_auth() -> tuple[str | None, str | None, str]:
    """Resolve Basic Auth for substrate SPARQL HTTP (query + update).

    Precedence (first complete user+password pair wins):
    ``SUBSTRATE_GRAPH_USER`` / ``SUBSTRATE_GRAPH_PASS`` →
    ``RDF_STORE_USER`` / ``RDF_STORE_PASS`` →
    ``FUSEKI_USER`` / ``FUSEKI_PASS``.

    Returns ``(user, password, source_label)`` where ``source_label`` is one of
    ``SUBSTRATE_GRAPH_USER``, ``RDF_STORE_USER``, ``FUSEKI_USER``, or ``none``.
    """
    su, sp = os.getenv("SUBSTRATE_GRAPH_USER", "").strip(), os.getenv("SUBSTRATE_GRAPH_PASS", "").strip()
    if su and sp:
        return su, sp, "SUBSTRATE_GRAPH_USER"
    ru, rp = os.getenv("RDF_STORE_USER", "").strip(), os.getenv("RDF_STORE_PASS", "").strip()
    if ru and rp:
        return ru, rp, "RDF_STORE_USER"
    fu, fp = os.getenv("FUSEKI_USER", "").strip(), os.getenv("FUSEKI_PASS", "").strip()
    if fu and fp:
        return fu, fp, "FUSEKI_USER"
    return None, None, "none"


def _basic_auth_tuple(user: str | None, password: str | None) -> tuple[str, str] | None:
    if user and password:
        return (user, password)
    return None


class SparqlHttpClient:
    """SPARQL 1.1 Protocol over HTTP: separate query and update endpoints, optional Basic Auth."""

    def __init__(
        self,
        query_url: str,
        update_url: str,
        *,
        timeout_sec: float = 30.0,
        user: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._query_url = query_url.rstrip("/")
        self._update_url = update_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._session = session or requests.Session()

    @property
    def update_url_redacted(self) -> str:
        return redact_http_url_for_log(self._update_url)

    @property
    def query_url_redacted(self) -> str:
        return redact_http_url_for_log(self._query_url)

    def select(self, sparql: str) -> list[dict[str, Any]]:
        auth = _basic_auth_tuple(self._user, self._password)
        r = self._session.post(
            self._query_url,
            data=sparql.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
            auth=auth,
            timeout=self._timeout_sec,
        )
        r.raise_for_status()
        payload = r.json()
        return list(payload.get("results", {}).get("bindings", []))

    def update(self, sparql: str) -> None:
        auth = _basic_auth_tuple(self._user, self._password)
        r = self._session.post(
            self._update_url,
            data=sparql.encode("utf-8"),
            headers={"Content-Type": "application/sparql-update"},
            auth=auth,
            timeout=self._timeout_sec,
        )
        r.raise_for_status()


class SparqlQueryClient:
    def __init__(
        self,
        query_url: str,
        *,
        timeout_sec: float = 30.0,
        user: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._client = SparqlHttpClient(
            query_url,
            update_url=query_url,
            timeout_sec=timeout_sec,
            user=user,
            password=password,
            session=session,
        )

    def select(self, sparql: str) -> list[dict[str, Any]]:
        return self._client.select(sparql)


class SparqlUpdateClient:
    def __init__(
        self,
        update_url: str,
        *,
        timeout_sec: float = 120.0,
        user: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._client = SparqlHttpClient(
            query_url=update_url,
            update_url=update_url,
            timeout_sec=timeout_sec,
            user=user,
            password=password,
            session=session,
        )

    def update(self, sparql: str) -> None:
        return self._client.update(sparql)


class GraphStoreClient:
    def __init__(
        self,
        graph_store_url: str,
        *,
        timeout_sec: float = 60.0,
        user: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._graph_store_url = graph_store_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._session = session or requests.Session()

    def post_graph(self, content: str, *, graph_uri: str | None = None, content_type: str = "application/n-triples") -> None:
        auth = _basic_auth_tuple(self._user, self._password)
        url = self._graph_store_url
        if graph_uri:
            url = f"{url}?graph={quote(graph_uri, safe='')}"
        r = self._session.post(
            url,
            data=content.encode("utf-8") if isinstance(content, str) else content,
            headers={"Content-Type": content_type},
            auth=auth,
            timeout=self._timeout_sec,
        )
        r.raise_for_status()
