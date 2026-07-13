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

    # Regression: graphiti-core==0.19.0's FalkorDriver.execute_query(self, cypher_query_,
    # **kwargs) only accepts one positional arg beyond the query string; the previous
    # ingest_episode() called execute_query(query, {"a": 1, ...}) -- a second positional
    # dict -- which raised "takes 2 positional arguments but 3 were given" against the real
    # driver (AsyncMock doesn't care about calling convention, so only an explicit call_args
    # check catches this).
    for call in mock_driver.execute_query.await_args_list:
        assert len(call.args) == 1, f"execute_query got extra positional args: {call.args[1:]}"


@pytest.mark.asyncio
async def test_search_filters_intimate_crystallization_ids(monkeypatch):
    async def fake_filter(driver, ids):
        return [cid for cid in ids if cid != "crys_intimate"]

    monkeypatch.setattr(core_backend, "_filter_intimate_crystallization_ids", fake_filter)
    # graphiti_core.Graphiti() eagerly builds OpenAI-backed llm_client/embedder/cross_encoder
    # unless supplied; core_backend.search() passes its own no-OpenAI stubs (see
    # _no_op_llm_client/_no_op_cross_encoder/_orion_embedder_client). Those stubs do a deep
    # `from graphiti_core.<submodule>.client import ...` import, which the bare Graphiti-only
    # mock below can't satisfy -- so stub the three helper functions directly instead of trying
    # to fake graphiti_core's whole submodule tree via sys.modules. search() reads
    # embedder.used (set by the real _OrionEmbedderClient.create()) for the trace, so the
    # fake embedder needs a `.used` attribute too.
    class _FakeEmbedder:
        used = False

    monkeypatch.setattr(core_backend, "_no_op_llm_client", lambda: object())
    monkeypatch.setattr(core_backend, "_no_op_cross_encoder", lambda: object())
    monkeypatch.setattr(core_backend, "_orion_embedder_client", lambda embed_url: _FakeEmbedder())

    class FakeGraphiti:
        def __init__(self, graph_driver, llm_client=None, embedder=None, cross_encoder=None):
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
