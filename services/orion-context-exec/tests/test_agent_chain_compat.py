from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from orion.schemas.agents.schemas import AgentChainRequest


@pytest.mark.asyncio
async def test_agent_chain_compat() -> None:
    transport = ASGITransport(app=app)
    body = AgentChainRequest(
        text="Where did the Denver claim come from?",
        mode="agent",
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/agent/chain/run", json=body.model_dump(mode="json"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"]
    assert "context_exec" in (data.get("structured") or {})
    assert "runtime_debug" in (data.get("planner_raw") or {})
