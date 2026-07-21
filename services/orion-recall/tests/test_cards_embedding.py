"""Unit tests for memory cards embedding scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_cosine_similarity_identical_vectors() -> None:
    from app.cards_embedding import cosine_similarity

    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    from app.cards_embedding import cosine_similarity

    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_read_cached_embedding_requires_matching_fingerprint() -> None:
    from app.cards_embedding import read_cached_embedding, text_fingerprint

    text = "Juniper lives in Denver"
    fp = text_fingerprint(text)
    sub = {"recall_embedding": {"text_fp": fp, "vector": [0.1, 0.2, 0.3]}}
    assert read_cached_embedding(sub, text_fp=fp) == [0.1, 0.2, 0.3]
    assert read_cached_embedding(sub, text_fp="stale") is None


def test_cards_adapter_uses_embedding_scorer_not_regex_tokens() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "app" / "cards_adapter.py").read_text(encoding="utf-8")
    assert "score_cards_by_embedding" in text
    assert "_tokens" not in text
    assert "cards_embedding" in text


def test_parse_subschema_parses_json_string() -> None:
    from app.cards_embedding import parse_subschema

    assert parse_subschema('{"memory_graph": {"x": 1}}') == {"memory_graph": {"x": 1}}


def test_parse_subschema_empty_on_invalid() -> None:
    from app.cards_embedding import parse_subschema

    assert parse_subschema("not-json") == {}


def test_fetch_card_fragments_sql_bounds_candidate_set() -> None:
    """Live incident 2026-07-21: the candidate SELECT had no LIMIT, so every
    call scored the entire memory_cards table. Locks in the fix -- the query
    must bound how many rows it considers per call."""
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "app" / "cards_adapter.py").read_text(encoding="utf-8")
    fragment = text.split("async def fetch_card_fragments(", 1)[1].split(
        "async def fetch_card_fragments_guarded", 1
    )[0]
    assert "LIMIT" in fragment
    assert "RECALL_CARDS_CANDIDATE_LIMIT" in fragment


@pytest.mark.asyncio
async def test_score_cards_by_embedding_caps_new_embeds_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live incident 2026-07-21: embedding every cache-miss card in one call
    always exceeded RECALL_CARDS_TIMEOUT_SEC, so persist_card_embeddings (the
    last step) never ran and the cache never converged. This locks in the
    fix -- embed_texts must never be asked to embed more than
    RECALL_CARDS_MAX_NEW_EMBEDS_PER_CALL new cards in a single call, even
    when far more rows lack a cache hit."""
    from app import cards_embedding

    monkeypatch.setattr(cards_embedding.settings, "RECALL_CARDS_MAX_NEW_EMBEDS_PER_CALL", 3, raising=False)
    monkeypatch.setattr(cards_embedding, "resolve_cards_embedding_url", lambda: "http://fake-embed/embedding")

    seen_pending_sizes: list[int] = []

    async def fake_embed_texts(texts, *, url, timeout_sec, max_concurrency):
        seen_pending_sizes.append(len(texts))
        return {t: [0.1, 0.2, 0.3] for t in texts}, "fake-model", 3

    async def fake_persist(pool, updates):
        return None

    monkeypatch.setattr(cards_embedding, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(cards_embedding, "persist_card_embeddings", fake_persist)

    # 10 rows, none cached (distinct titles so read_cached_embedding always misses).
    rows = [
        {
            "card_id": f"card-{i}",
            "title": f"Unique title {i}",
            "summary": "summary",
            "tags": [],
            "anchors": [],
            "anchor_class": None,
            "subschema": {},
            "updated_at": None,
        }
        for i in range(10)
    ]

    scored = await cards_embedding.score_cards_by_embedding(
        pool=None,
        rows=rows,
        query_text="find something",
        min_similarity=0.0,
    )

    # One call for the query embed (1 text), one for the batch -- the batch
    # call must be capped at 3, not all 10 pending cards.
    assert seen_pending_sizes[0] == 1
    assert seen_pending_sizes[1] == 3
    assert len(scored) == 3
