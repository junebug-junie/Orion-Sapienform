"""POST /rerank (issue #3533): request-order scores from the local cross-encoder, 503 without one."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("torch")

from fastapi import HTTPException  # noqa: E402

from app import main  # noqa: E402


class _FakeReranker:
    model_name = "fake-ce"

    async def score(self, query, documents):
        return [float(len(set(query.split()) & set(d.split()))) for d in documents]


def test_rerank_returns_one_score_per_document_in_order(monkeypatch):
    monkeypatch.setattr(main, "reranker", _FakeReranker())
    out = asyncio.run(main.rerank_endpoint(main.RerankRequest(query="dog stitches", documents=["cat", "dog stitches out"])))
    assert out.model == "fake-ce" and out.scores == [0.0, 2.0]


def test_rerank_without_a_model_is_503_and_rejects_oversized_requests(monkeypatch):
    monkeypatch.setattr(main, "reranker", None)
    with pytest.raises(HTTPException) as err:
        asyncio.run(main.rerank_endpoint(main.RerankRequest(query="q", documents=["d"])))
    assert err.value.status_code == 503
    monkeypatch.setattr(main, "reranker", _FakeReranker())
    with pytest.raises(HTTPException) as err:
        asyncio.run(main.rerank_endpoint(main.RerankRequest(query="q", documents=["d"] * (main.MAX_RERANK_PAIRS + 1))))
    assert err.value.status_code == 400
