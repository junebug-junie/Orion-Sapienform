from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.memory_graph.approve import approve_memory_graph_draft
from orion.memory_graph.dto import SuggestDraftV1


def test_postgres_failure_propagates_no_rdf_write_to_compensate(monkeypatch) -> None:
    """approve_memory_graph_draft is Postgres-only as of 2026-07-22 (see its
    docstring) -- the RDF graph store write + compensation logic were
    removed. On a Postgres failure, the exception must simply propagate;
    there is no longer an external write for a caller to roll back."""
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    pool = MagicMock()

    async def boom(*args, **kwargs):
        raise RuntimeError("pg down")

    async def _run() -> None:
        await approve_memory_graph_draft(
            draft,
            pool,
            named_graph_iri="https://x/ng",
        )

    with patch(
        "orion.memory_graph.approve.insert_cards_and_edges_batch",
        new_callable=AsyncMock,
        side_effect=boom,
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(_run())


def test_approve_writes_only_to_postgres(monkeypatch) -> None:
    """No RDF/GraphDB module is imported or called -- approve.py has no
    dependency on Fuseki/GraphDB env config at all anymore."""
    raw = json.loads(Path("tests/fixtures/memory_graph/joey_cats_draft.json").read_text(encoding="utf-8"))
    draft = SuggestDraftV1.model_validate(raw)
    pool = MagicMock()

    async def _run():
        return await approve_memory_graph_draft(
            draft,
            pool,
            named_graph_iri="https://x/ng",
        )

    with patch(
        "orion.memory_graph.approve.insert_cards_and_edges_batch",
        new_callable=AsyncMock,
        return_value=["card-1"],
    ) as mock_insert:
        outcome = asyncio.run(_run())

    assert outcome.ok is True
    assert outcome.card_ids == ["card-1"]
    mock_insert.assert_awaited_once()
