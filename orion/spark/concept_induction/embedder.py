from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("orion.spark.concept.embedder")


@dataclass
class EmbeddingResponse:
    embeddings: Dict[str, List[float]]
    error: Optional[str] = None


class EmbeddingClient:
    """HTTP client for external embedding host."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, phrases: List[str]) -> EmbeddingResponse:
        if not phrases:
            return EmbeddingResponse(embeddings={})
        url = f"{self.base_url}/embed"
        try:
            resp = requests.post(url, json={"items": phrases}, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            vectors = data.get("embeddings") or {}
            if not isinstance(vectors, dict):
                return EmbeddingResponse(
                    embeddings={}, error="invalid_response_structure"
                )
            return EmbeddingResponse(embeddings=vectors)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding host unavailable: %s", exc)
            return EmbeddingResponse(embeddings={}, error=str(exc))
