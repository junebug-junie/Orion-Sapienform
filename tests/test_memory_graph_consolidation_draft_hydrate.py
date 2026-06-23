from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orion.memory_graph.consolidation_draft_hydrate import (
    format_turn_utterance_text,
    hydrate_consolidation_draft_dict,
    hydrate_draft_utterance_text,
)
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.utterance_text import ensure_draft_utterance_text


def test_format_turn_utterance_text() -> None:
    text = format_turn_utterance_text("hello", "hi there")
    assert text == "User: hello\nOrion: hi there"


def test_hydrate_maps_utterance_ids_by_turn_index() -> None:
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["turn-id-1", "turn-id-2"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {},
    }
    hydrated = hydrate_draft_utterance_text(
        draft,
        turn_correlation_ids=["corr-a", "corr-b"],
        turns_by_correlation={
            "corr-a": {"prompt": "first prompt", "response": "first response"},
            "corr-b": {"prompt": "second prompt", "response": "second response"},
        },
    )
    assert hydrated["utterance_text_by_id"]["turn-id-1"] == "User: first prompt\nOrion: first response"
    assert hydrated["utterance_text_by_id"]["turn-id-2"] == "User: second prompt\nOrion: second response"
    enriched = ensure_draft_utterance_text(SuggestDraftV1.model_validate(hydrated))
    assert enriched.utterance_text_by_id["turn-id-1"]


@pytest.mark.asyncio
async def test_hydrate_consolidation_draft_dict_fetches_chat_history() -> None:
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {"correlation_id": "corr-a", "prompt": "p1", "response": "r1"},
        ]
    )
    draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["turn-id-1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
        "utterance_text_by_id": {},
    }
    hydrated = await hydrate_consolidation_draft_dict(pool, draft, ["corr-a"])
    assert hydrated["utterance_text_by_id"]["turn-id-1"] == "User: p1\nOrion: r1"
