from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.memory_graph.approve import approve_memory_graph_draft
from orion.memory_graph.dto import SuggestDraftV1


def test_compensate_batch_on_postgres_failure() -> None:
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    pool = MagicMock()

    async def boom(*args, **kwargs):
        raise RuntimeError("pg down")

    comp = MagicMock()

    async def _run() -> None:
        await approve_memory_graph_draft(
            draft,
            pool,
            graphdb_url="http://g",
            graphdb_repo="r",
            named_graph_iri="https://x/ng",
        )

    with (
        patch("orion.memory_graph.approve.insert_batch", MagicMock()),
        patch("orion.memory_graph.approve.compensate_batch", comp),
        patch(
            "orion.memory_graph.approve.insert_cards_and_edges_batch",
            new_callable=AsyncMock,
            side_effect=boom,
        ),
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(_run())
    assert comp.called
