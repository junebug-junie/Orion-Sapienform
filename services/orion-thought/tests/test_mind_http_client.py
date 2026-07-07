from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app import mind_enrichment
from orion.mind.v1 import MindHandoffBriefV1, MindRunRequestV1, MindRunResultV1
from orion.mind.v1 import MindRunPolicyV1


def _settings(**over):
    base = dict(
        mind_base_url="http://orion-mind:6611",
        mind_timeout_sec=5.0,
        mind_max_response_bytes=2_000_000,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _req() -> MindRunRequestV1:
    return MindRunRequestV1(
        correlation_id="corr-1",
        snapshot_inputs={"user_text": "hi", "messages_tail": []},
        policy=MindRunPolicyV1(n_loops_max=1, wall_time_ms_max=12000),
    )


def _ok_result_json() -> dict:
    return MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(mind_quality="meaningful_synthesis"),
        mind_quality="meaningful_synthesis",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_ok_returns_result(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/mind/run"
        return httpx.Response(200, json=_ok_result_json())

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is not None
    assert result.ok is True
    assert result.brief.mind_quality == "meaningful_synthesis"


@pytest.mark.asyncio
async def test_timeout_fails_open(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is None


@pytest.mark.asyncio
async def test_http_500_fails_open(monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="nope"))
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(_req(), settings=_settings(), correlation_id="corr-1")
    assert result is None


@pytest.mark.asyncio
async def test_oversized_body_fails_open(monkeypatch):
    big = _ok_result_json()
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=big))
    monkeypatch.setattr(mind_enrichment, "_mind_transport", lambda: transport, raising=False)
    result = await mind_enrichment.run_mind_for_thought(
        _req(), settings=_settings(mind_max_response_bytes=1), correlation_id="corr-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_base_url_fails_open(monkeypatch):
    result = await mind_enrichment.run_mind_for_thought(
        _req(), settings=_settings(mind_base_url=""), correlation_id="corr-1"
    )
    assert result is None
