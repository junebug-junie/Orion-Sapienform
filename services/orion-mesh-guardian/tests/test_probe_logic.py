from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.probe import run_probe
from app.roster import ProbeConfig, ProbeMode


@pytest.mark.asyncio
async def test_probe_bad_on_redis_ping_failure() -> None:
    redis = AsyncMock()
    redis.ping.side_effect = RuntimeError("down")
    result = await run_probe(
        redis=redis,
        entry_probe=ProbeConfig(mode=ProbeMode.redis, intake_channels=["orion:test"]),
    )
    assert result.status == "probe_bad"


@pytest.mark.asyncio
async def test_probe_bad_on_zero_subscribers() -> None:
    redis = AsyncMock()
    redis.ping.return_value = True
    with patch("app.probe.redis_pubsub_numsub", new=AsyncMock(return_value={"orion:test": 0})):
        result = await run_probe(
            redis=redis,
            entry_probe=ProbeConfig(mode=ProbeMode.redis, intake_channels=["orion:test"]),
        )
    assert result.status == "probe_bad"


@pytest.mark.asyncio
async def test_probe_ok_when_checks_pass() -> None:
    redis = AsyncMock()
    redis.ping.return_value = True
    with patch("app.probe.redis_pubsub_numsub", new=AsyncMock(return_value={"orion:test": 1})):
        result = await run_probe(
            redis=redis,
            entry_probe=ProbeConfig(mode=ProbeMode.redis, intake_channels=["orion:test"]),
        )
    assert result.status == "probe_ok"


@pytest.mark.asyncio
async def test_http_probe_bad_on_non_200() -> None:
    redis = MagicMock()
    mock_resp = MagicMock(status_code=503)
    mock_resp.json.return_value = {"ok": False}
    with patch("app.probe.httpx.AsyncClient") as mock_client:
        instance = mock_client.return_value.__aenter__.return_value
        instance.get = AsyncMock(return_value=mock_resp)
        result = await run_probe(
            redis=redis,
            entry_probe=ProbeConfig(mode=ProbeMode.http, ready_url="http://svc/ready"),
        )
    assert result.status == "probe_bad"
