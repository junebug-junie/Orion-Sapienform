from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger("orion.spark.concept.embedder")


@dataclass
class EmbeddingResponse:
    embeddings: Dict[str, List[float]]
    error: Optional[str] = None


class EmbeddingClient:
    """HTTP client for the vector-host embedding endpoint."""

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request_payload(self, phrase: str) -> dict[str, object]:
        return {
            "doc_id": f"concept-{uuid4()}",
            "text": phrase,
            "embedding_profile": "default",
            "include_latent": False,
        }

    def embed(self, phrases: List[str]) -> EmbeddingResponse:
        if not phrases:
            return EmbeddingResponse(embeddings={})
        url = f"{self.base_url}/embedding"
        try:
            embeddings: Dict[str, List[float]] = {}
            for phrase in phrases:
                resp = requests.post(
                    url,
                    json=self._request_payload(phrase),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                vector = data.get("embedding")
                if not isinstance(vector, list):
                    return EmbeddingResponse(
                        embeddings={}, error="invalid_response_structure"
                    )
                embeddings[phrase] = vector
            return EmbeddingResponse(embeddings=embeddings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding host unavailable: %s", exc)
            return EmbeddingResponse(embeddings={}, error=str(exc))
