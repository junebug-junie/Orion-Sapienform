"""Local cross-encoder reranker behind ``POST /rerank`` (issue #3533).

orion-recall's lived-memory collector sends one query and up to ~24 candidate memories; this
scores each (query, memory) pair with a cross-encoder that reads both texts together, which
orders candidates far better than the cosine of two separate embeddings.

Model: ``VECTOR_HOST_RERANK_MODEL``. Runs on this host only (no external inference API); the
weights come from the Hugging Face Hub on first load and are kept in the persistent HF cache
volume mounted by docker-compose.yml, so a container recreate does not download them again.
Default ``BAAI/bge-reranker-v2-m3`` on the athena Tesla P4 (``VECTOR_HOST_RERANK_DEVICE=cuda``,
the GPU pinned by ``VECTOR_HOST_RERANK_GPU_DEVICE_ID`` in docker-compose.yml). Replay
2026-11-17 (11 real referring messages): v2-m3 retrieved 9/11 targets, bge-reranker-base 7/11,
mxbai-rerank-base 7/11, ms-marco-MiniLM-L-12 6/11. On CPU v2-m3 took 2.3-3.6 s p50 per turn,
too slow for a chat turn; a missing GPU falls back to CPU rather than failing. Empty model name =
the endpoint answers 503 and recall serves no lived memories.

One forward pass at a time on its own single-thread executor, for the same reason the embedder
has one (``app/embedder.py``): parallel torch passes each start an OpenMP team and slow every
pass down. The embedder and reranker executors are separate, so a burst of bus embeds does not
queue a chat turn's rerank.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

MAX_PAIRS = 64
MAX_LENGTH = 512
# Requests allowed in flight (running + queued) before /rerank answers 503.
MAX_IN_FLIGHT = 2


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self._torch = torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"  # no GPU passed through: still serve, slower
        self.device = str(device)
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-rerank")
        self._in_flight = 0

    @property
    def busy(self) -> bool:
        return self._in_flight >= MAX_IN_FLIGHT

    def _score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        inputs = self._tokenizer(
            [query] * len(documents), list(documents), padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits.view(-1).float()
        return [float(x) for x in logits.tolist()]

    async def score(self, query: str, documents: Sequence[str]) -> List[float]:
        self._in_flight += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._score, query, list(documents)[:MAX_PAIRS],
            )
        finally:
            self._in_flight -= 1

    def close(self) -> None:
        self._executor.shutdown(wait=False)
