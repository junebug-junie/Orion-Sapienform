import types
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.backends import graphiti_core as core_backend


class _FakeEntityNode:
    """Stands in for graphiti_core.nodes.EntityNode in this dev venv (graphiti-core isn't
    installed here -- it is installed in the adapter's Docker image, which is where the real
    import path is exercised live). save() mirrors the FalkorDB branch of the real
    EntityNode.save() verified against graphiti-core==0.19.0 source: `attributes` keys flatten
    onto the node as top-level properties, and the whole dict is passed as a single
    `entity_data` kwarg (never a second positional arg -- see the execute_query calling
    convention regression covered below)."""

    def __init__(
        self,
        *,
        uuid,
        name,
        group_id,
        summary="",
        name_embedding=None,
        attributes=None,
        created_at=None,
    ):
        self.uuid = uuid
        self.name = name
        self.group_id = group_id
        self.summary = summary
        self.name_embedding = name_embedding
        self.attributes = attributes or {}
        self.created_at = created_at

    async def save(self, driver):
        entity_data = {
            "uuid": self.uuid,
            "name": self.name,
            "name_embedding": self.name_embedding,
            "group_id": self.group_id,
            "summary": self.summary,
            "created_at": self.created_at,
        }
        entity_data.update(self.attributes or {})
        return await driver.execute_query("FAKE_ENTITY_SAVE", entity_data=entity_data)


class _FakeEntityEdge:
    """Stands in for graphiti_core.edges.EntityEdge -- see _FakeEntityNode docstring. save()
    mirrors the FalkorDB branch of the real EntityEdge.save(): source_node_uuid/
    target_node_uuid become source_uuid/target_uuid in the persisted payload."""

    def __init__(
        self,
        *,
        uuid,
        group_id,
        source_node_uuid,
        target_node_uuid,
        name,
        fact,
        fact_embedding=None,
        episodes=None,
        created_at=None,
        attributes=None,
    ):
        self.uuid = uuid
        self.group_id = group_id
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.name = name
        self.fact = fact
        self.fact_embedding = fact_embedding
        self.episodes = episodes or []
        self.created_at = created_at
        self.attributes = attributes or {}

    async def save(self, driver):
        edge_data = {
            "source_uuid": self.source_node_uuid,
            "target_uuid": self.target_node_uuid,
            "uuid": self.uuid,
            "name": self.name,
            "group_id": self.group_id,
            "fact": self.fact,
            "fact_embedding": self.fact_embedding,
            "episodes": self.episodes,
            "created_at": self.created_at,
        }
        edge_data.update(self.attributes or {})
        return await driver.execute_query("FAKE_EDGE_SAVE", edge_data=edge_data)


def _patch_graphiti_node_edge_modules():
    """patch.dict target covering both `graphiti_core.nodes` and `graphiti_core.edges` so
    ingest_episode()'s function-local `from graphiti_core.nodes import EntityNode` /
    `from graphiti_core.edges import EntityEdge` resolve to the fakes above instead of
    requiring the real graphiti-core package in this venv."""
    nodes_mod = types.ModuleType("graphiti_core.nodes")
    nodes_mod.EntityNode = _FakeEntityNode
    edges_mod = types.ModuleType("graphiti_core.edges")
    edges_mod.EntityEdge = _FakeEntityEdge
    return patch.dict(
        "sys.modules",
        {"graphiti_core.nodes": nodes_mod, "graphiti_core.edges": edges_mod},
    )


def _entity_data_calls(mock_driver):
    return [c for c in mock_driver.execute_query.await_args_list if "entity_data" in c.kwargs]


def _edge_data_calls(mock_driver):
    return [c for c in mock_driver.execute_query.await_args_list if "edge_data" in c.kwargs]


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
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver), _patch_graphiti_node_edge_modules():
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
async def test_graphiti_core_ingest_flattens_attributes_onto_entity_node():
    """Root-cause regression: entities must be keyed on .uuid (not a custom .id property)
    and crystallization_id/sensitivity must land as top-level node properties (flattened
    from EntityNode.attributes) -- that's exactly what
    _filter_intimate_crystallization_ids()'s `n.crystallization_id` / `n.sensitivity` Cypher
    already expects."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    pool.fetchval = AsyncMock(return_value=None)
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver), _patch_graphiti_node_edge_modules():
        await core_backend.ingest_episode(
            pool,
            episode_id="gep_crys_b",
            crystallization_id="crys_b",
            kind="stance",
            subject="B subject",
            summary="B summary text",
            status="active",
            metadata={"sensitivity": "public"},
            links=[],
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    entity_calls = _entity_data_calls(mock_driver)
    assert entity_calls, "expected EntityNode.save() to write via an entity_data kwarg"
    entity_data = entity_calls[0].kwargs["entity_data"]
    assert entity_data["uuid"] == "gent_crys_b"
    assert entity_data["name"] == "B subject"
    assert entity_data["crystallization_id"] == "crys_b"
    assert entity_data["sensitivity"] == "public"


@pytest.mark.asyncio
async def test_graphiti_core_ingest_writes_self_referential_edge_with_no_links():
    """Graphiti.search() only ever returns edges, never bare nodes -- without a
    self-referential RELATES_TO edge, a crystallization with no CrystallizationLinkV1 links
    (the common case) would never be searchable at all."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    pool.fetchval = AsyncMock(return_value=None)
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver), _patch_graphiti_node_edge_modules():
        await core_backend.ingest_episode(
            pool,
            episode_id="gep_crys_b",
            crystallization_id="crys_b",
            kind="stance",
            subject="B subject",
            summary="B summary text",
            status="active",
            metadata={"sensitivity": "public"},
            links=[],
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    edge_calls = _edge_data_calls(mock_driver)
    self_edges = [
        c
        for c in edge_calls
        if c.kwargs["edge_data"]["source_uuid"] == c.kwargs["edge_data"]["target_uuid"]
    ]
    assert self_edges, "expected a self-referential RELATES_TO edge even with zero links"
    self_edge_data = self_edges[0].kwargs["edge_data"]
    assert self_edge_data["source_uuid"] == "gent_crys_b"
    assert "B subject" in self_edge_data["fact"]
    # fact is a deterministic string template, never LLM-generated (hard constraint).
    assert self_edge_data["fact"] == "B subject: B summary text"


@pytest.mark.asyncio
async def test_graphiti_core_ingest_writes_link_edge_and_target_stub_when_target_unseen():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    pool.fetchval = AsyncMock(return_value=None)  # target never ingested -> fallback subject
    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(return_value=([], None, None))  # no existing target node

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver), _patch_graphiti_node_edge_modules():
        result = await core_backend.ingest_episode(
            pool,
            episode_id="gep_crys_a",
            crystallization_id="crys_a",
            kind="stance",
            subject="A subject",
            summary="A summary",
            status="active",
            metadata={"sensitivity": "public"},
            links=[{"target_crystallization_id": "crys_c", "relation": "supports", "confidence": 0.9}],
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    edge_calls = _edge_data_calls(mock_driver)
    link_edges = [c for c in edge_calls if c.kwargs["edge_data"]["name"] == "supports"]
    assert link_edges, "expected a RELATES_TO edge for the crystallization link"
    link_edge_data = link_edges[0].kwargs["edge_data"]
    assert link_edge_data["source_uuid"] == "gent_crys_a"
    assert link_edge_data["target_uuid"] == "gent_crys_c"
    assert "supports" in link_edge_data["fact"]

    entity_calls = _entity_data_calls(mock_driver)
    target_stub_calls = [c for c in entity_calls if c.kwargs["entity_data"]["uuid"] == "gent_crys_c"]
    assert target_stub_calls, "expected a stub EntityNode for the not-yet-ingested link target"
    assert result["edge_ids"]


@pytest.mark.asyncio
async def test_graphiti_core_ingest_does_not_clobber_existing_target_entity():
    """EntityNode.save() is a full property replace (SET n = $entity_data). If the link
    target was already ingested for real, re-saving it as a stub would destroy its real
    name/summary/attributes -- the existence check must skip the stub write in that case."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    pool.fetchval = AsyncMock(return_value="Real target subject")

    async def fake_execute_query(query, **kwargs):
        if "uuid" in kwargs:
            return ([{"uuid": kwargs["uuid"]}], None, None)
        return ([], None, None)

    mock_driver = AsyncMock()
    mock_driver.execute_query = AsyncMock(side_effect=fake_execute_query)

    with patch.object(core_backend, "_falkor_driver", return_value=mock_driver), _patch_graphiti_node_edge_modules():
        await core_backend.ingest_episode(
            pool,
            episode_id="gep_crys_a",
            crystallization_id="crys_a",
            kind="stance",
            subject="A subject",
            summary="A summary",
            status="active",
            metadata={"sensitivity": "public"},
            links=[{"target_crystallization_id": "crys_c", "relation": "supports", "confidence": 0.9}],
            falkordb_uri="redis://localhost:6379/0",
            graph_name="orion_graphiti",
        )

    entity_calls = _entity_data_calls(mock_driver)
    target_stub_calls = [c for c in entity_calls if c.kwargs["entity_data"]["uuid"] == "gent_crys_c"]
    assert not target_stub_calls, "must not overwrite an already-real target entity with stub data"

    edge_calls = _edge_data_calls(mock_driver)
    link_edges = [c for c in edge_calls if c.kwargs["edge_data"]["name"] == "supports"]
    assert link_edges
    assert "Real target subject" in link_edges[0].kwargs["edge_data"]["fact"]


def test_extract_crystallization_ids_from_real_entity_edge_shaped_objects():
    """Graphiti.search() returns list[EntityEdge]; EntityEdge carries source_node_uuid/
    target_node_uuid as plain strings, not a nested source_node/target_node object with its
    own .crystallization_id -- the old fallback branch read a shape that never existed on
    the real object and silently extracted nothing."""

    class FakeEntityEdge:
        def __init__(self, source_node_uuid, target_node_uuid):
            self.source_node_uuid = source_node_uuid
            self.target_node_uuid = target_node_uuid

    edges = [
        FakeEntityEdge("gent_crys_a", "gent_crys_a"),  # self-referential edge
        FakeEntityEdge("gent_crys_a", "gent_crys_b"),  # link edge
    ]
    ids = core_backend._extract_crystallization_ids(edges)
    assert ids == ["crys_a", "crys_b"]


@pytest.mark.asyncio
async def test_ensure_graphiti_indices_swallows_already_indexed_error(monkeypatch):
    core_backend._indices_ready = False

    async def fake_build_indices_and_constraints(driver):
        raise Exception("Attribute 'name' is already indexed")

    maintenance_mod = types.ModuleType("graphiti_core.utils.maintenance.graph_data_operations")
    maintenance_mod.build_indices_and_constraints = fake_build_indices_and_constraints

    with patch.object(core_backend, "_falkor_driver", return_value=MagicMock()), patch.dict(
        "sys.modules",
        {"graphiti_core.utils.maintenance.graph_data_operations": maintenance_mod},
    ):
        await core_backend.ensure_graphiti_indices("redis://localhost:6379/0", "orion_graphiti")

    assert core_backend._indices_ready is True
    core_backend._indices_ready = False


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
