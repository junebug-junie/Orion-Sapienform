from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_mod


def test_health_without_postgres():
    with patch.object(main_mod, "pg_pool", None):
        client = TestClient(main_mod.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["postgres"] is False
        assert data["service"] == "orion-graphiti-adapter"
        assert data["backend"] == "orion_postgres"
