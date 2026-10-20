"""A heat-shed refusal reaches the caller with the gateway's own retry-after hint, and a refusal the
shed backoff answered locally is not logged as a warning per call (orion/gpu_pool/shed_backoff.py)."""
from __future__ import annotations

import contextlib
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.gpu_pool.client import LeaseUnavailable

from app import main as gateway
from app import pool_placement
from app.settings import settings


def _refusing(reason: str, *, retry_after_sec=None, local=False):
    @contextlib.asynccontextmanager
    async def gpu_lease(bus, **kw):
        raise LeaseUnavailable(reason, retry_after_sec=retry_after_sec, local=local)
        yield  # pragma: no cover
    return gpu_lease


def _env() -> BaseEnvelope:
    return BaseEnvelope(
        kind="llm.chat.request", source=ServiceRef(name="vision-council", node="n", version="0"),
        correlation_id=str(uuid.uuid4()),
        payload=ChatRequestPayload(messages=[LLMMessage(role="user", content="ping")], route="quick").model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_bus_reply_carries_retry_after_on_a_shed_refusal(fake_pool, monkeypatch, caplog):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    monkeypatch.setattr(pool_placement, "gpu_lease", _refusing("shed:cabinet_hot", retry_after_sec=57.25, local=True))
    with caplog.at_level(logging.WARNING):
        out = await gateway.handle_chat(_env())
    details = out.payload.raw["details"]
    assert out.payload.raw["error"] == "gpu_pool_unavailable"
    assert details["reason"] == "shed:cabinet_hot" and details["retry_after_sec"] == 57.2
    assert not [r for r in caplog.records if "gpu_pool_unavailable" in r.getMessage()]


@pytest.mark.asyncio
async def test_non_shed_refusal_shape_and_warning_unchanged(fake_pool, monkeypatch, caplog):
    monkeypatch.setattr(gateway, "run_llm_chat", lambda *a: pytest.fail("must not run"))
    monkeypatch.setattr(pool_placement, "gpu_lease", _refusing("no_serviceable_role"))
    with caplog.at_level(logging.WARNING):
        out = await gateway.handle_chat(_env())
    assert "retry_after_sec" not in out.payload.raw["details"]
    assert [r for r in caplog.records if "gpu_pool_unavailable" in r.getMessage()]


def test_passthrough_shed_refusal_sets_retry_after(fake_pool, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    monkeypatch.setattr(pool_placement, "gpu_lease", _refusing("shed:cabinet_hot", retry_after_sec=29.6))
    from app.main import app

    resp = TestClient(app).post("/v1/chat/completions",
                                json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"
    assert resp.json()["error"]["retry_after_sec"] == 29.6


def test_passthrough_non_shed_refusal_has_no_retry_after(fake_pool, monkeypatch):
    monkeypatch.setattr(settings, "llm_gateway_openai_passthrough_enabled", True)
    monkeypatch.setattr(pool_placement, "gpu_lease", _refusing("no_serviceable_role"))
    from app.main import app

    resp = TestClient(app).post("/v1/chat/completions",
                                json={"model": "quick", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503 and "retry-after" not in resp.headers
    assert "retry_after_sec" not in resp.json()["error"]
