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
    pool, conn = _mock_pool()
    with patch.object(main_mod, "pg_pool", pool), patch("app.main.sync_to_falkordb", return_value=None):
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
