from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.fusion import render_continuity_bundle
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, RecallQueryV1


def test_render_continuity_bundle_user_prioritized():
    candidates = [
        {"id": "t1", "source": "sql_chat", "snippet": "User: hey\nAssistant: hi", "score": 0.5},
        {"id": "t2", "source": "sql_chat", "snippet": "User: move stress\nAssistant: I hear you", "score": 0.8},
    ]
    profile = {
        "render_budget_tokens": 96,
        "max_total_items": 6,
        "render_lane": "continuity",
        "profile": "chat.continuity.v1",
    }
    bundle, _ = render_continuity_bundle(
        candidates=candidates,
        profile=profile,
        query_text="move stress",
        latency_ms=1,
    )
    assert "move stress" in bundle.rendered


def test_worker_uses_continuity_render_when_phase_continuity(monkeypatch):
    monkeypatch.setattr(worker.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(worker.settings, "RECALL_CONTINUITY_SQL_MINUTES", 120)
    monkeypatch.setattr(worker.settings, "RECALL_CONTINUITY_RENDER_BUDGET", 96)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(worker.settings, "RECALL_INTENT_ROUTING_ENABLED", False)

    continuity_profile = {
        "profile": "chat.continuity.v1",
        "enable_query_expansion": False,
        "enable_anchor_candidates": False,
        "enable_sql_timeline": False,
        "enable_sql_chat": True,
        "sql_chat_top_k": 6,
        "sql_since_minutes": 120,
        "max_total_items": 6,
        "render_budget_tokens": 96,
        "render_lane": "continuity",
    }
    monkeypatch.setattr(worker, "get_profile", lambda _name: continuity_profile)

    fetch_pairs_called = []

    async def _fetch_pairs(**kwargs):
        fetch_pairs_called.append(kwargs)
        return []

    async def _fetch_msgs(**kwargs):
        return []

    async def _anchor(**kwargs):
        return [], {}

    fuse_called = []
    render_called = []

    def _mock_fuse(**kwargs):
        fuse_called.append(kwargs)
        return MemoryBundleV1(rendered="fuse", stats=MemoryBundleStatsV1()), []

    def _mock_render(**kwargs):
        render_called.append(kwargs)
        return MemoryBundleV1(rendered="ok", stats=MemoryBundleStatsV1()), []

    monkeypatch.setattr(worker, "fetch_chat_history_pairs", _fetch_pairs)
    monkeypatch.setattr(worker, "fetch_chat_messages", _fetch_msgs)
    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(worker, "fuse_candidates", _mock_fuse)
    monkeypatch.setattr(worker, "render_continuity_bundle", _mock_render)

    q = RecallQueryV1(
        fragment="hey",
        profile="chat.continuity.v1",
        recall_phase="continuity",
        session_id="sess-1",
    )

    import asyncio

    bundle, _decision = asyncio.run(worker.process_recall(q, corr_id="corr-pcr-continuity"))

    assert fetch_pairs_called
    assert render_called
    assert not fuse_called
    assert bundle.rendered == "ok"
    assert render_called[0]["profile"]["sql_since_minutes"] == 120
    assert render_called[0]["profile"]["render_budget_tokens"] == 96
