"""Minimal SPARQL Protocol HTTP clients (Fuseki + generic SPARQL endpoints)."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests


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
        self._query_url = query_url
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._session = session or requests.Session()

    def select(self, sparql: str) -> list[dict[str, Any]]:
        auth = (self._user, self._password) if self._user and self._password else None
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
        self._update_url = update_url
        self._timeout_sec = timeout_sec
        self._user = user
        self._password = password
        self._session = session or requests.Session()

    def update(self, sparql: str) -> None:
        auth = (self._user, self._password) if self._user and self._password else None
        r = self._session.post(
            self._update_url,
            data=sparql.encode("utf-8"),
            headers={"Content-Type": "application/sparql-update"},
            auth=auth,
            timeout=self._timeout_sec,
        )
        r.raise_for_status()


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
        auth = (self._user, self._password) if self._user and self._password else None
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
