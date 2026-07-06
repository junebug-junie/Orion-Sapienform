from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod


@pytest.mark.asyncio
async def test_search_returns_crystallization_ids(monkeypatch):
    async def fake_search(query, **kwargs):
        assert query == "stance on memory"
        return {
            "crystallization_ids": ["crys_a", "crys_b"],
            "trace": {"backend": "graphiti_core", "rails": ["vector", "graph"]},
        }

    monkeypatch.setattr("app.backends.graphiti_core.search", fake_search)
    with patch.object(main_mod.settings, "GRAPHITI_BACKEND", "graphiti_core"):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/search",
            json={
                "query": "stance on memory",
                "seed_crystallization_id": "crys_seed",
                "limit": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["crystallization_ids"] == ["crys_a", "crys_b"]
        assert data["trace"]["backend"] == "graphiti_core"
        assert data["trace"]["rails"] == ["vector", "graph"]


def test_search_returns_501_for_orion_postgres_backend():
    with patch.object(main_mod.settings, "GRAPHITI_BACKEND", "orion_postgres"):
        client = TestClient(main_mod.app)
        resp = client.post(
            "/v1/search",
            json={"query": "stance on memory", "limit": 5},
        )
        assert resp.status_code == 501
        assert resp.json()["detail"] == "search_requires_graphiti_core_backend"
