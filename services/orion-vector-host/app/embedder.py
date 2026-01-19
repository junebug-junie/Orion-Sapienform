from __future__ import annotations

import asyncio
from typing import List, Tuple

import httpx
import torch
import torch.nn.functional as torch_nn
from transformers import AutoModel, AutoTokenizer

from .settings import Settings


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._device = torch.device(settings.VECTOR_HOST_EMBEDDING_DEVICE)

        backend = settings.VECTOR_HOST_EMBED_BACKEND
        if backend == "hf":
            self._load_hf_model()
        elif backend == "vllm":
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.VECTOR_HOST_EMBED_READ_TIMEOUT_SEC,
                    connect=settings.VECTOR_HOST_EMBED_CONNECT_TIMEOUT_SEC,
                )
            )
        else:
            raise RuntimeError(f"Embedding backend {backend} is not supported for semantic embeddings")

    def _load_hf_model(self) -> None:
        model_name = self._settings.VECTOR_HOST_EMBEDDING_MODEL
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def _embedding_url(self) -> str:
        backend = self._settings.VECTOR_HOST_EMBED_BACKEND
        if backend != "vllm":
            raise RuntimeError(f"Embedding backend {backend} is not supported for semantic embeddings")
        base_url = self._settings.ORION_LLM_VLLM_URL

        if not base_url:
            raise RuntimeError(f"Base URL not configured for embedding backend {backend}")
        return f"{base_url.rstrip('/')}/v1/embeddings"

    def _embed_hf(self, text: str) -> Tuple[List[float], str, int]:
        if not self._tokenizer or not self._model:
            raise RuntimeError("HuggingFace embedding model is not initialized")

        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = last_hidden_state * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        normalized = torch_nn.normalize(pooled, p=2, dim=1)
        embedding = normalized[0].tolist()
        return embedding, self._settings.VECTOR_HOST_EMBEDDING_MODEL, len(embedding)

    async def embed(self, text: str) -> Tuple[List[float], str, int]:
        backend = self._settings.VECTOR_HOST_EMBED_BACKEND
        if backend == "hf":
            return await asyncio.to_thread(self._embed_hf, text)

        url = self._embedding_url()
        payload = {"model": self._settings.VECTOR_HOST_EMBEDDING_MODEL, "input": [text]}
        if not self._client:
            raise RuntimeError("Embedding HTTP client is not initialized")
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
        if self._client:
            await self._client.aclose()
