from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from app.substrate_reads import (
    SubstrateReadCache,
    fetch_execution_trajectory,
    fetch_grammar_truth,
    fetch_reasoning_activity,
)


@pytest.mark.asyncio
async def test_fetch_grammar_truth_degraded(monkeypatch) -> None:
    client = AsyncMock()
    client.get.return_value.json.return_value = {"ok": False, "degraded": True, "degraded_reasons": ["x"]}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_grammar_truth(client, "http://substrate/grammar/truth")
    assert out.degraded is True
    assert out.degraded_reasons == ["x"]


@pytest.mark.asyncio
async def test_fetch_grammar_truth_timeout_fail_closed(monkeypatch) -> None:
    client = AsyncMock()
    client.get.side_effect = TimeoutError("slow")
    out = await fetch_grammar_truth(client, "http://substrate/grammar/truth")
    assert out.degraded is True
    assert "http_error" in out.degraded_reasons[0]


def test_cache_reuses_within_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=2.0)
    cache.put_grammar({"degraded": False})
    assert cache.get_grammar() == {"degraded": False}


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_happy_path() -> None:
    client = AsyncMock()
    projection = {
        "call_count": 5,
        "reasoning_present_rate": 0.2,
        "completion_tokens_sum": 100,
        "thinking_tokens_sum": None,
    }
    client.get.return_value.json.return_value = {"ok": True, "projection": projection}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is True
    assert out.projection == projection


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_http_error_fails_closed() -> None:
    client = AsyncMock()
    client.get.side_effect = TimeoutError("slow")
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is False
    assert out.projection is None


@pytest.mark.asyncio
async def test_fetch_reasoning_activity_non_dict_projection_fails_closed() -> None:
    client = AsyncMock()
    client.get.return_value.json.return_value = {"ok": True, "projection": "not-a-dict"}
    client.get.return_value.raise_for_status = lambda: None
    out = await fetch_reasoning_activity(client, "http://thought/projections/reasoning_activity")
    assert out.ok is True
    assert out.projection is None


def test_cache_reuses_reasoning_activity_within_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=2.0)
    cache.put_reasoning_activity({"ok": True, "projection": {"call_count": 1}})
    assert cache.get_reasoning_activity() == {"ok": True, "projection": {"call_count": 1}}


def test_cache_reasoning_activity_expires_past_ttl() -> None:
    cache = SubstrateReadCache(ttl_sec=0.0)
    cache.put_reasoning_activity({"ok": True, "projection": {}})
    time.sleep(0.01)
    assert cache.get_reasoning_activity() is None
