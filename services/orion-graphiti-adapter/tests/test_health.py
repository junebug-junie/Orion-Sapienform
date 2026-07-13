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
        # graphiti_core is the settings.py/.env_example default as of the 2026-07-13
        # hardening pass (was orion_postgres before /v1/search was proven live).
        assert data["backend"] == "graphiti_core"
