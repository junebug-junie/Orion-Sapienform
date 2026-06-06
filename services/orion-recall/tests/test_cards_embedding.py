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
