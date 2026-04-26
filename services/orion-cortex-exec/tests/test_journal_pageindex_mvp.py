from __future__ import annotations

import asyncio
import logging
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


def test_service_backed_path_is_primary(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-identity", snippet="Identity continuity reflection.", tags=["theme:identity"]),
        _FakeItem(id="i2", source_ref="entry-other", snippet="Unrelated operational note."),
    ]

    async def _native_must_not_run(**kwargs):
        raise AssertionError("native selector should not run when service succeeds")

    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "get_journal_corpus_status",
        lambda self: {"build_success": True, "corpus_exists": True},
    )
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "query_journal_pageindex",
        lambda self, query, allow_fallback=False, top_k=8: {
            "query_invoked": True,
            "query_result_count": 1,
            "fallback_invoked": False,
            "results": [
                {
                    "entry_id": "entry-service",
                    "node_id": "entry-service::node::1",
                    "heading": "Service heading",
                    "excerpt": "Service excerpt about reflective continuity.",
                    "created_at": "2026-03-10T12:00:00Z",
                    "source_kind": "journal.entry.index",
                    "provenance": {
                        "reflective_themes": ["continuity"],
                        "trigger_kind": "manual",
                        "trigger_summary": "manual summary",
                        "conversation_frame": "reflective",
                        "task_mode": "reflective_dialogue",
                        "identity_salience": "high",
                        "stance_summary": "brief reflective frame",
                        "active_identity_facets": ["identity continuity"],
                        "active_growth_axes": ["stability"],
                        "active_relationship_facets": ["trust"],
                        "social_posture": ["warm"],
                        "active_tensions": ["speed_vs_depth"],
                        "dream_motifs": ["bridge"],
                        "response_hazards": ["overgeneralization"],
                    },
                }
            ],
        },
    )
    monkeypatch.setattr(executor, "_journal_pageindex_select_with_llm", _native_must_not_run)
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Can you reflect on my identity continuity?",
            items=items,
            selection={},
        )
    )
    pageindex = ctx.get("journal_pageindex_context") or {}
    assert [item["entry_id"] for item in pageindex.get("selected_entries") or []] == ["entry-service"]
    assert [item["block_id"] for item in pageindex.get("selected_blocks") or []] == ["entry-service::node::1"]
    selected_entry = (pageindex.get("selected_entries") or [])[0]
    assert selected_entry["reflective_themes"] == ["continuity"]
    assert selected_entry["active_identity_facets"] == ["identity continuity"]
    assert selected_entry["active_tensions"] == ["speed_vs_depth"]
    assert selected_entry["dream_motifs"] == ["bridge"]
    assert selected_entry["stance_summary"] == "brief reflective frame"
    assert pageindex.get("fallback_invoked") is False


def test_service_unavailable_falls_back_to_native(monkeypatch) -> None:
    items = [
        _FakeItem(
            id="i1",
            source_ref="entry-tension",
            snippet="Tension between speed and depth.\n\nI need continuity.",
            tags=["tension:speed_vs_depth"],
        )
    ]
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "get_journal_corpus_status",
        lambda self: (_ for _ in ()).throw(RuntimeError("service unavailable")),
    )
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
    assert pageindex.get("fallback_invoked") is True


def test_service_error_falls_back_to_native(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-dream", snippet="Dream motif of bridges returning.", tags=["dream:bridge"])
    ]
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "get_journal_corpus_status",
        lambda self: {"build_success": True, "corpus_exists": True},
    )
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "query_journal_pageindex",
        lambda self, query, allow_fallback=False, top_k=8: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
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
    assert pageindex.get("fallback_invoked") is True


def test_service_shape_stays_chat_stance_compatible(monkeypatch) -> None:
    items = [
        _FakeItem(id="i1", source_ref="entry-collapse", snippet="Collapse response journal from storage event.")
    ]
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "get_journal_corpus_status",
        lambda self: {"build_success": True, "corpus_exists": True},
    )
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "query_journal_pageindex",
        lambda self, query, allow_fallback=False, top_k=8: {
            "query_invoked": True,
            "query_result_count": 1,
            "fallback_invoked": False,
            "results": [
                {
                    "entry_id": "entry-service",
                    "node_id": "entry-service::node::1",
                    "heading": "Service heading",
                    "excerpt": "Service excerpt about reflective continuity.",
                    "created_at": "2026-03-10T12:00:00Z",
                    "source_kind": "journal.entry.index",
                    "provenance": {
                        "reflective_themes": ["continuity"],
                        "active_identity_facets": ["identity continuity"],
                        "active_tensions": ["speed_vs_depth"],
                    },
                }
            ],
        },
    )
    ctx = asyncio.run(
        _run_with_items(
            monkeypatch,
            user_text="Find my collapse response reflection.",
            items=items,
            selection={},
        )
    )
    pageindex = ctx.get("journal_pageindex_context") or {}
    selected = pageindex.get("selected_entries") or []
    assert selected and selected[0]["entry_id"] == "entry-service"
    assert "body_excerpt" in selected[0]
    selected_blocks = pageindex.get("selected_blocks") or []
    assert selected_blocks and selected_blocks[0]["excerpt"] == "Service excerpt about reflective continuity."


def test_logs_report_impl_and_service_status(monkeypatch, caplog) -> None:
    items = [_FakeItem(id="i1", source_ref="entry-fallback", snippet="Thin signal.")]
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "get_journal_corpus_status",
        lambda self: {"build_success": True, "corpus_exists": True},
    )
    monkeypatch.setattr(
        executor.JournalPageIndexClient,
        "query_journal_pageindex",
        lambda self, query, allow_fallback=False, top_k=8: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
    caplog.set_level(logging.INFO, logger="orion.cortex.exec")
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
    assert any("journal_pageindex_impl=legacy_fallback" in rec.message for rec in caplog.records)
    assert any("pageindex_service_status=error" in rec.message for rec in caplog.records)
