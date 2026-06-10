from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.runner import FAKE_ORGANS
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.mark.asyncio
async def test_belief_provenance_contract() -> None:
    FAKE_ORGANS.memory_hits = [
        {
            "claim": "User is from Denver",
            "source_ref": "memory:denver:1",
            "verified": True,
            "confidence": 0.9,
        }
    ]
    FAKE_ORGANS.trace_hits = [
        {"handle": "trace:1", "snippet": "Denver in corr abc", "corr_id": "abc"}
    ]
    transport = ASGITransport(app=app)
    req = ContextExecRequestV1(
        text="Where did Orion get the claim that I am from Denver?",
        mode="belief_provenance",
        expected_artifact_type="BeliefProvenanceReportV1",
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/context-exec/run", json=req.model_dump(mode="json"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifact_type"] == "BeliefProvenanceReportV1"
    assert data["artifact"]["status"] in {
        "supported",
        "unsupported",
        "contradicted",
        "stale",
        "inferred",
        "unknown",
    }
    assert data.get("findings_bundle") is not None
    FAKE_ORGANS.memory_hits = None
    FAKE_ORGANS.trace_hits = None
