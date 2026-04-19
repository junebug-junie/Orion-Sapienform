from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any

from orion.core.bus.bus_schemas import ServiceRef

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import executor  # noqa: E402


@dataclass
class _FakeItem:
    id: str
    snippet: str
    source: str = "journal.entry.index"
    source_ref: str | None = None
    score: float = 0.9
    tags: list[str] | None = None
    uri: str | None = None

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "id": self.id,
            "snippet": self.snippet,
            "source": self.source,
            "source_ref": self.source_ref,
            "score": self.score,
            "tags": self.tags or [],
            "uri": self.uri,
        }


class _FakeBundle:
    def __init__(self, items: list[_FakeItem]):
        self.items = items
        self.rendered = "\n".join(item.snippet for item in items)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {"items": [item.model_dump(mode=mode) for item in self.items], "rendered": self.rendered}


class _FakeRecallReply:
    def __init__(self, items: list[_FakeItem]):
        self.bundle = _FakeBundle(items)
        self.debug = {"decision": {}}


async def _run_with_items(monkeypatch, *, user_text: str, items: list[_FakeItem], selection: dict[str, Any]) -> dict[str, Any]:
    async def _fake_query(self, **kwargs):
        return _FakeRecallReply(items)

    async def _fake_select(**kwargs):
        return selection

    monkeypatch.setattr(executor.RecallClient, "query", _fake_query)
    monkeypatch.setattr(executor, "_journal_pageindex_select_with_llm", _fake_select)

    ctx = {
        "messages": [{"role": "user", "content": user_text}],
        "verb": "chat_general",
        "intent": "reflective_depth",
        "session_id": "s1",
        "node_id": "n1",
    }
    _, _, _ = await executor.run_recall_step(
        bus=object(),  # query path is monkeypatched
        source=ServiceRef(name="test", version="0.0.1", node="local"),
        ctx=ctx,
        correlation_id="corr-test",
        recall_cfg={"profile": "reflect.v1"},
    )
    return ctx


def test_identity_query_selects_expected_entry(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-identity", snippet="Identity continuity reflection.", tags=["theme:identity"]),
        _FakeItem(id="i2", source_ref="entry-other", snippet="Unrelated operational note."),
    ]
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Can you reflect on my identity continuity?",
            items=items,
            selection={"selected_entry_ids": ["entry-identity"], "selected_block_ids": [], "reasons": {"entry-identity": "identity match"}},
        )
    )
    selected = (ctx.get("journal_pageindex_context") or {}).get("selected_entries") or []
    assert selected and selected[0]["entry_id"] == "entry-identity"


def test_tension_query_selects_entry_and_block(monkeypatch) -> None:
    items = [
        _FakeItem(
            id="i1",
            source_ref="entry-tension",
            snippet="Tension between speed and depth.\n\nI need continuity.",
            tags=["tension:speed_vs_depth"],
        )
    ]
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="What tension is recurring in my continuity lane?",
            items=items,
            selection={
                "selected_entry_ids": ["entry-tension"],
                "selected_block_ids": ["entry-tension::block::1"],
                "reasons": {"entry-tension::block::1": "explicit tension mention"},
            },
        )
    )
    pageindex = ctx.get("journal_pageindex_context") or {}
    assert [item["entry_id"] for item in pageindex.get("selected_entries") or []] == ["entry-tension"]
    assert [item["block_id"] for item in pageindex.get("selected_blocks") or []] == ["entry-tension::block::1"]


def test_dream_query_selects_dream_entry_and_block(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-dream", snippet="Dream motif of bridges returning.", tags=["dream:bridge"]),
        _FakeItem(id="i2", source_ref="entry-other", snippet="No dream content."),
    ]
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Any dream motifs repeating?",
            items=items,
            selection={"selected_entry_ids": ["entry-dream"], "selected_block_ids": ["entry-dream::block::1"], "reasons": {}},
        )
    )
    pageindex = ctx.get("journal_pageindex_context") or {}
    assert [item["entry_id"] for item in pageindex.get("selected_entries") or []] == ["entry-dream"]
    assert [item["block_id"] for item in pageindex.get("selected_blocks") or []] == ["entry-dream::block::1"]


def test_collapse_response_query_selects_collapse_entry(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-collapse", snippet="Collapse response journal from storage event."),
        _FakeItem(id="i2", source_ref="entry-other", snippet="General note."),
    ]
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Find my collapse response reflection.",
            items=items,
            selection={"selected_entry_ids": ["entry-collapse"], "selected_block_ids": [], "reasons": {}},
        )
    )
    selected = (ctx.get("journal_pageindex_context") or {}).get("selected_entries") or []
    assert selected and selected[0]["entry_id"] == "entry-collapse"


def test_weak_hit_falls_back_safely(monkeypatch) -> None:
    items = [_FakeItem(id="i1", source_ref="entry-fallback", snippet="Thin signal.")]
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Reflect on continuity",
            items=items,
            selection={"selected_entry_ids": [], "selected_block_ids": [], "reasons": {}},
        )
    )
    pageindex = ctx.get("journal_pageindex_context") or {}
    assert pageindex.get("fallback_invoked") is True
    selected_entries = pageindex.get("selected_entries") or []
    assert selected_entries and selected_entries[0]["entry_id"] == "entry-fallback"
