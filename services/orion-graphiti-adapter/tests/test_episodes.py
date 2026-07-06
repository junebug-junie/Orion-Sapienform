from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod


@pytest.fixture
def mock_pool():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="OK")

    @asynccontextmanager
    async def acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = acquire
    return pool, conn


def test_ingest_episode_returns_ids(mock_pool):
    pool, conn = mock_pool
    with patch.object(main_mod, "pg_pool", pool), patch(
        "app.main.sync_to_falkordb", return_value=None
    ):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/episodes",
            json={
                "crystallization_id": "crys_test001",
                "kind": "stance",
                "subject": "Test subject",
                "summary": "Test summary",
                "status": "active",
                "metadata": {"scope": ["project:orion"]},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == "gep_crys_test001"
        assert data["entity_id"] == "gent_crys_test001"
        assert data["edge_id"] == "ged_crys_test001"
        assert data["canonical_mutated"] is False
        assert conn.execute.await_count >= 3


def test_neighborhood_after_ingest(mock_pool):
    pool, conn = mock_pool
    conn.fetch = AsyncMock(
        side_effect=[
            [{"episode_id": "gep_crys_test001", "crystallization_id": "crys_test001"}],
            [{"entity_id": "gent_crys_test001", "crystallization_id": "crys_test001"}],
            [{"edge_id": "ged_crys_test001", "from_id": "gent_crys_test001", "to_id": "gep_crys_test001"}],
        ]
    )
    with patch.object(main_mod, "pg_pool", pool):
        client = TestClient(main_mod.app)
        resp = client.get("/v1/neighborhood/crys_test001")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) >= 1
        assert data["crystallization_id"] == "crys_test001"
