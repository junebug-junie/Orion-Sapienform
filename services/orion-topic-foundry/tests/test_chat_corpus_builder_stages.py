from __future__ import annotations

from datetime import datetime, timezone

from app.pipelines.chat_corpus_builder.anchors import extract_anchors
from app.pipelines.chat_corpus_builder.blocks import extract_blocks
from app.pipelines.chat_corpus_builder.claims import mine_claims_and_resolutions
from app.pipelines.chat_corpus_builder.episodes import segment_episodes
from app.pipelines.chat_corpus_builder.graph import build_turn_graph
from app.pipelines.chat_corpus_builder.indexer import build_chat_turn_index


def _rows() -> list[dict]:
    return [
        {
            "id": "t1",
            "correlation_id": "c1",
            "created_at": datetime(2026, 4, 25, 14, 31, tzinfo=timezone.utc),
            "prompt": "journal_entry_index has 0 rows in orion-pageindex",
            "response": "The issue is missing backfill. Run docker compose exec sql-writer.",
            "thought_process": "",
            "source": "chat",
            "memory_status": None,
            "memory_tier": None,
            "memory_reason": None,
            "spark_meta": {"service": "orion-pageindex"},
            "client_meta": {"mode": "debug", "trace_mode": "on"},
        },
        {
            "id": "t2",
            "correlation_id": "c1",
            "created_at": datetime(2026, 4, 25, 14, 40, tzinfo=timezone.utc),
            "prompt": "ran docker compose and backfilled journal_entry_index",
            "response": "That worked. journal_entry_index now has 540 rows.",
            "thought_process": "",
            "source": "chat",
            "memory_status": None,
            "memory_tier": None,
            "memory_reason": None,
            "spark_meta": {},
            "client_meta": {"mode": "debug"},
        },
    ]


def test_stage_pipeline_outputs_expected_records() -> None:
    turns = build_chat_turn_index(_rows())
    assert len(turns) == 2
    assert turns[0].has_commands is True
    anchors = extract_anchors(turns)
    assert any("journal_entry_index" in item.anchors for item in anchors)
    edges = build_turn_graph(turns, anchors)
    assert edges
    episodes = segment_episodes(turns, anchors, edges)
    assert len(episodes) == 1
    assert episodes[0].turn_ids == ["t1", "t2"]
    blocks = extract_blocks(turns)
    assert len(blocks) == 2
    claims = mine_claims_and_resolutions(episodes, blocks)
    assert claims
    assert claims[0].status in {"confirmed", "accepted", "unresolved", "rejected"}
