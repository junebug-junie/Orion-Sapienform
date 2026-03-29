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

from app import worker
from orion.core.contracts.recall import RecallQueryV1


def test_process_recall_excludes_active_turn_candidates(monkeypatch):
    monkeypatch.setattr(worker, "get_profile", lambda _name: {"profile": "reflect.v1", "enable_query_expansion": False, "enable_sql_timeline": True, "sql_top_k": 3, "sql_since_minutes": 60, "max_total_items": 10})
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", True)

    async def _anchor(**kwargs):
        return [], {}

    async def _query(fragment, profile, *, session_id, node_id, entities, diagnostic=False, exclusion=None):
        return [
            {
                "id": "corr-live",
                "source": "sql_chat",
                "source_ref": "chat_history_log",
                "text": 'ExactUserText: "Teddy loves Addy"\nOrionResponse: "..."',
                "ts": 1000.0,
                "score": 0.9,
                "tags": [],
            },
            {
                "id": "stable-memory-1",
                "source": "vector",
                "source_ref": "orion_chat",
                "text": "Older memory about project milestones",
                "ts": 900.0,
                "score": 0.8,
                "tags": [],
            },
        ], {"sql_chat": 1, "vector": 1}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "_query_backends", _query)

    q = RecallQueryV1(
        fragment="Teddy loves Addy",
        profile="reflect.v1",
        exclude={
            "active_turn_text": "Teddy loves Addy",
            "active_turn_ids": ["corr-live"],
            "active_turn_ts": 1000.0,
        },
    )

    bundle, decision = asyncio.run(worker.process_recall(q, corr_id="corr-live", diagnostic=True))

    ids = [item.id for item in bundle.items]
    assert "corr-live" not in ids
    assert "stable-memory-1" in ids
    assert decision.selected_ids == ids
