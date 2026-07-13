from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import app.main as main_mod


def _mock_pool():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool, conn


def test_ingest_with_supports_link_writes_cross_edge():
    # Exercises the generic orion_postgres ingest path (this test predates backend
    # selection); pin the backend so it doesn't depend on the graphiti_core package,
    # which is only installed in the service's Docker image, not this test venv.
    pool, conn = _mock_pool()
    with patch.object(main_mod, "pg_pool", pool), patch.object(
        main_mod.settings, "GRAPHITI_BACKEND", "orion_postgres"
    ), patch("app.main.sync_to_falkordb", return_value=None):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/episodes",
            json={
                "crystallization_id": "crys_a",
                "kind": "stance",
                "subject": "A",
                "summary": "A summary",
                "links": [
                    {
                        "target_crystallization_id": "crys_b",
                        "relation": "supports",
                        "confidence": 0.9,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        sql_calls = [str(c.args[0]) for c in conn.execute.await_args_list]
        assert any("graphiti_edges" in s for s in sql_calls)
        assert any("crys_b" in str(c) for c in conn.execute.await_args_list)


def test_neighborhood_depth_two_returns_linked_crystallization():
    pool, conn = _mock_pool()
    conn.fetch = AsyncMock(
        side_effect=[
            [
                {
                    "edge_id": "ged_a_b_supports",
                    "from_id": "gent_crys_a",
                    "to_id": "gent_crys_b",
                    "relation": "supports",
                },
                {
                    "edge_id": "ged_has",
                    "from_id": "gent_crys_b",
                    "to_id": "gep_crys_b",
                    "relation": "has_episode",
                },
            ],
            [],
            [],
            [{"episode_id": "gep_crys_a", "crystallization_id": "crys_a"}, {"episode_id": "gep_crys_b", "crystallization_id": "crys_b"}],
            [{"entity_id": "gent_crys_a", "crystallization_id": "crys_a"}, {"entity_id": "gent_crys_b", "crystallization_id": "crys_b"}],
        ]
    )
    with patch.object(main_mod, "pg_pool", pool):
        client = TestClient(main_mod.app)
        resp = client.get("/v1/neighborhood/crys_a?depth=2")
        assert resp.status_code == 200
        cids = {n.get("crystallization_id") for n in resp.json()["nodes"]}
        assert "crys_a" in cids
        assert "crys_b" in cids
