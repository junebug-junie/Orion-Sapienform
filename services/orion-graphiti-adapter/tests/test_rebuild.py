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


def test_rebuild_batch_ingests_items():
    pool, conn = _mock_pool()
    with patch.object(main_mod, "pg_pool", pool), patch(
        "app.main.sync_to_falkordb", return_value=None
    ):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/rebuild",
            json={
                "items": [
                    {
                        "crystallization_id": "crys_rebuild_a",
                        "kind": "stance",
                        "subject": "A",
                        "summary": "Summary A",
                        "status": "active",
                        "metadata": {"scope": ["project:orion"]},
                    },
                    {
                        "crystallization_id": "crys_rebuild_b",
                        "kind": "stance",
                        "subject": "B",
                        "summary": "Summary B",
                        "status": "active",
                        "metadata": {"scope": ["project:orion"]},
                    },
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 2
        assert data["skipped_intimate"] == 0
        assert data["canonical_mutated"] is False
        assert conn.execute.await_count >= 6


def test_rebuild_skips_intimate_items():
    pool, _conn = _mock_pool()
    with patch.object(main_mod, "pg_pool", pool):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/rebuild",
            json={
                "items": [
                    {
                        "crystallization_id": "crys_intimate",
                        "kind": "stance",
                        "subject": "Private",
                        "summary": "Private summary",
                        "status": "active",
                        "metadata": {"sensitivity": "intimate"},
                    },
                    {
                        "crystallization_id": "crys_public",
                        "kind": "stance",
                        "subject": "Public",
                        "summary": "Public summary",
                        "status": "active",
                        "metadata": {"sensitivity": "public"},
                    },
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 1
        assert data["skipped_intimate"] == 1
        assert data["canonical_mutated"] is False


def test_ingest_skips_intimate_sensitivity():
    with patch.object(main_mod, "pg_pool", None):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/episodes",
            json={
                "crystallization_id": "crys_intimate_only",
                "kind": "stance",
                "subject": "Private",
                "summary": "Private summary",
                "status": "active",
                "metadata": {"sensitivity": "intimate"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is True
        assert data["reason"] == "intimate_sensitivity"
        assert data["canonical_mutated"] is False
