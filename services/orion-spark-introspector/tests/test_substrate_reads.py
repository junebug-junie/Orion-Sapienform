from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from app.substrate_reads import SubstrateReadCache, fetch_grammar_truth, fetch_execution_trajectory


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
