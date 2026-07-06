from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.backends import graphiti_core as core_backend


@pytest.mark.asyncio
async def test_graphiti_core_ingest_dual_writes_postgres():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None))

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver):
        result = await core_backend.ingest_episode(
            pool,
            episode_id="gep_crys_a",
            crystallization_id="crys_a",
            kind="stance",
            subject="A",
            summary="summary",
            status="active",
            metadata={"sensitivity": "public"},
            links=[],
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    assert result["edge_ids"]
    assert mock_driver.execute_query.await_count >= 1
    assert conn.execute.await_count >= 3


@pytest.mark.asyncio
async def test_search_filters_intimate_crystallization_ids(monkeypatch):
    async def fake_filter(driver, ids):
        return [cid for cid in ids if cid != "crys_intimate"]

    monkeypatch.setattr(core_backend, "_embed_query", AsyncMock(return_value=None))
    monkeypatch.setattr(core_backend, "_filter_intimate_crystallization_ids", fake_filter)

    class FakeGraphiti:
        def __init__(self, graph_driver):
            pass

        async def search(self, **kwargs):
            return [{"crystallization_id": "crys_a"}, {"crystallization_id": "crys_intimate"}]

    fake_graphiti_mod = MagicMock()
    fake_graphiti_mod.Graphiti = FakeGraphiti

    with patch.object(core_backend, "_falkor_driver", return_value=MagicMock()), patch.dict(
        "sys.modules", {"graphiti_core": fake_graphiti_mod}
    ):
        out = await core_backend.search(
            "memory",
            seed_crystallization_id="crys_seed",
            limit=10,
            embed_url="",
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    assert "crys_a" in out["crystallization_ids"]
    assert "crys_intimate" not in out["crystallization_ids"]
