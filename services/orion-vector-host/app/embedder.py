from __future__ import annotations

from typing import List, Tuple

import httpx

from .settings import Settings


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.VECTOR_HOST_EMBED_READ_TIMEOUT_SEC,
                connect=settings.VECTOR_HOST_EMBED_CONNECT_TIMEOUT_SEC,
            )
        )

    def _embedding_url(self) -> str:
        backend = self._settings.VECTOR_HOST_EMBED_BACKEND
        if backend != "vllm":
            raise RuntimeError(f"Embedding backend {backend} is not supported for semantic embeddings")
        base_url = self._settings.ORION_LLM_VLLM_URL

        if not base_url:
            raise RuntimeError(f"Base URL not configured for embedding backend {backend}")
        return f"{base_url.rstrip('/')}/v1/embeddings"

    async def embed(self, text: str) -> Tuple[List[float], str, int]:
        url = self._embedding_url()
        payload = {"model": self._settings.VECTOR_HOST_EMBEDDING_MODEL, "input": [text]}
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not items:
            raise RuntimeError("Embedding response missing data")
        embedding = items[0].get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Embedding response missing embedding vector")
        embedding_model = data.get("model") or self._settings.VECTOR_HOST_EMBEDDING_MODEL
        return embedding, embedding_model, len(embedding)

    async def close(self) -> None:
        await self._client.aclose()
