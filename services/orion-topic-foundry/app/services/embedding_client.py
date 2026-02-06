from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests


logger = logging.getLogger("topic-foundry.embedding")


class VectorHostEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        *,
        retries: int = 2,
        backoff_sec: float = 0.75,
        timeout_sec: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = max(0, int(retries))
        self.backoff_sec = float(backoff_sec)
        self.timeout_sec = float(timeout_sec)
        self._last_model: Optional[str] = None
        self._last_dim: Optional[int] = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for idx, text in enumerate(texts):
            doc_id = f"topic-foundry-{idx}"
            embedding = self._embed_text(doc_id, text)
            embeddings.append(embedding)
        return embeddings

    def _embed_text(self, doc_id: str, text: str) -> List[float]:
        payload = {
            "doc_id": doc_id,
            "text": text,
            "embedding_profile": "default",
            "include_latent": False,
        }
        url = self.base_url
        attempt = 0
        while True:
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not isinstance(embedding, list):
                    raise RuntimeError("Embedding response missing embedding vector")
                self._log_model_info(data.get("embedding_model"), data.get("embedding_dim"))
                return embedding
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.retries:
                    logger.error("Embedding failed doc_id=%s error=%s", doc_id, exc)
                    raise
                sleep_for = self.backoff_sec * (2**attempt)
                logger.warning(
                    "Embedding retry doc_id=%s attempt=%s error=%s",
                    doc_id,
                    attempt + 1,
                    exc,
                )
                time.sleep(sleep_for)
                attempt += 1

    def _log_model_info(self, model: Optional[str], dim: Optional[int]) -> None:
        if model == self._last_model and dim == self._last_dim:
            return
        if model or dim:
            logger.info("Embedding model=%s dim=%s", model, dim)
        self._last_model = model
        self._last_dim = dim

    @property
    def embedding_model(self) -> Optional[str]:
        return self._last_model

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._last_dim
