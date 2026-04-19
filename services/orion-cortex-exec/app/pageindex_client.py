from __future__ import annotations

from typing import Any

import requests

from .settings import settings


class JournalPageIndexClient:
    def __init__(self, base_url: str | None = None, timeout_sec: float = 15.0) -> None:
        self._base_url = (base_url or settings.journal_pageindex_service_url).rstrip("/")
        self._timeout_sec = timeout_sec

    def rebuild_journal_corpus(self) -> dict[str, Any]:
        response = requests.post(f"{self._base_url}/corpora/journals/rebuild", timeout=self._timeout_sec)
        response.raise_for_status()
        return response.json()

    def get_journal_corpus_status(self) -> dict[str, Any]:
        response = requests.get(f"{self._base_url}/corpora/journals/status", timeout=self._timeout_sec)
        response.raise_for_status()
        return response.json()

    def query_journal_pageindex(self, query: str, *, allow_fallback: bool = False, top_k: int = 8) -> dict[str, Any]:
        payload = {"query": query, "allow_fallback": allow_fallback, "top_k": top_k}
        response = requests.post(f"{self._base_url}/corpora/journals/query", json=payload, timeout=self._timeout_sec)
        response.raise_for_status()
        return response.json()
