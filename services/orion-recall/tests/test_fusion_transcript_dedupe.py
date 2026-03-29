from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.fusion import fuse_candidates


def _profile() -> dict:
    return {
        "profile": "reflect.v1",
        "max_per_source": 10,
        "max_total_items": 20,
        "render_budget_tokens": 256,
        "time_decay_half_life_hours": 72,
        "relevance": {
            "backend_weights": {
                "vector": 1.0,
                "sql_chat": 0.8,
                "sql_timeline": 0.8,
            }
        },
    }


def test_fusion_collapses_identical_vector_transcript_with_different_ids():
    text = 'ExactUserText: "Teddy loves Addy"\nOrionResponse: "I remember that."'
    cands = [
        {"id": "doc-a", "source": "vector", "text": text, "score": 0.9, "tags": ["vector-assoc", "collection:a"]},
        {"id": "doc-b", "source": "vector", "text": text, "score": 0.8, "tags": ["vector-assoc", "collection:b"]},
    ]

    bundle, _ = fuse_candidates(candidates=cands, profile=_profile(), query_text="Teddy", diagnostic=True)

    assert len(bundle.items) == 1


def test_fusion_collapses_duplicate_transcript_across_vector_collections():
    text = "User: Teddy loves Addy\nOrion: Yes"
    cands = [
        {"id": "vec-1", "source": "vector", "text": text, "score": 0.6, "tags": ["vector-assoc", "collection:chat_a"]},
        {"id": "vec-2", "source": "vector", "text": text, "score": 0.7, "tags": ["vector-assoc", "collection:chat_b"]},
    ]

    bundle, _ = fuse_candidates(candidates=cands, profile=_profile(), query_text="Teddy", diagnostic=True)

    assert len(bundle.items) == 1
    assert bundle.items[0].id == "vec-2"


def test_fusion_collapses_cross_source_duplicate_transcript():
    text = "User: Teddy loves Addy\nOrion: Yes"
    cands = [
        {"id": "vec-1", "source": "vector", "text": text, "score": 0.85, "tags": ["vector-assoc", "collection:chat"]},
        {"id": "sql-1", "source": "sql_chat", "text": text, "score": 0.7, "tags": ["sql", "chat"]},
    ]

    bundle, _ = fuse_candidates(candidates=cands, profile=_profile(), query_text="Teddy", diagnostic=True)

    assert len(bundle.items) == 1


def test_fusion_keeps_distinct_non_transcript_items():
    cands = [
        {"id": "fact-1", "source": "vector", "text": "GraphDB endpoint is repo collapse.", "score": 0.9, "tags": ["vector-assoc"]},
        {"id": "fact-2", "source": "vector", "text": "GraphDB endpoint is repo collapse.", "score": 0.8, "tags": ["vector-assoc"]},
    ]

    bundle, _ = fuse_candidates(candidates=cands, profile=_profile(), query_text="GraphDB", diagnostic=True)

    assert len(bundle.items) == 2
