"""Regression tests for cortex-gateway bus intake consumer resilience."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.bus_client import BusClient  # type: ignore  # noqa: E402


@pytest.mark.asyncio
async def test_connect_forks_separate_intake_and_rpc_buses(monkeypatch) -> None:
    """Orch RPC traffic uses a forked bus distinct from the intake subscriber."""
    client = BusClient()
    fork_calls = 0

    async def fake_connect() -> None:
        return None

    async def fake_fork(*, start_rpc_worker: bool = False):
        nonlocal fork_calls
        fork_calls += 1
        child = BusClient()
        child.connect = fake_connect  # type: ignore[method-assign]
        child.close = AsyncMock()
        return child

    client.bus = AsyncMock()
    client.bus.connect = fake_connect
    client.bus.fork = fake_fork

    await client.connect()

    assert fork_calls == 2
    assert client._intake_bus is not None
    assert client._rpc_bus is not None
    assert client._intake_bus is not client._rpc_bus
    assert client._intake_bus is not client.bus
    assert client._rpc_bus is not client.bus
