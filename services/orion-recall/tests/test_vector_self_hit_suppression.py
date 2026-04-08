from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.storage import vector_adapter


class _Collection:
    def query(self, **kwargs):
        return {
            "ids": [["doc-1", "doc-2"]],
            "documents": [["Teddy loves Addy", "other memory"]],
            "metadatas": [[{"correlation_id": "corr-self", "timestamp": "2099-01-01T00:00:00Z"}, {"timestamp": "2099-01-01T00:00:00Z"}]],
            "distances": [[0.01, 0.2]],
        }

    def get(self, **kwargs):
        return {
            "ids": ["doc-1", "doc-2"],
            "documents": ["Teddy loves Addy", "other memory"],
            "metadatas": [{"correlation_id": "corr-self"}, {}],
        }


class _Client:
    def get_or_create_collection(self, name):
        return _Collection()


def test_fetch_vector_fragments_excludes_active_turn(monkeypatch):
    monkeypatch.setattr(vector_adapter, "_get_client", lambda: _Client())
    monkeypatch.setattr(vector_adapter, "_embed_query_text", lambda text: [0.1, 0.2])
    monkeypatch.setattr(vector_adapter.settings, "RECALL_VECTOR_COLLECTIONS", "test_collection")

    out = vector_adapter.fetch_vector_fragments(
        query_text="Teddy loves Addy",
        time_window_days=30,
        max_items=5,
        exclude_ids=["corr-self"],
        exclude_text="Teddy loves Addy",
    )

    assert len(out) == 1
    assert out[0]["id"] == "doc-2"


def test_fetch_vector_exact_matches_excludes_active_turn(monkeypatch):
    monkeypatch.setattr(vector_adapter, "_get_client", lambda: _Client())
    monkeypatch.setattr(vector_adapter.settings, "RECALL_VECTOR_COLLECTIONS", "test_collection")

    out = vector_adapter.fetch_vector_exact_matches(
        tokens=["Teddy"],
        max_items=5,
        exclude_ids=["corr-self"],
        exclude_text="Teddy loves Addy",
    )

    assert len(out) == 1
    assert out[0]["id"] == "doc-2"
