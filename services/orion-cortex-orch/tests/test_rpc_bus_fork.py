from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app import main as orch_main


def test_bus_for_rpc_prefers_fork() -> None:
    sentinel = object()
    orch_main._rpc_bus = sentinel  # type: ignore[assignment]
    try:
        assert orch_main._bus_for_rpc() is sentinel
    finally:
        orch_main._rpc_bus = None


def test_bus_for_rpc_falls_back_to_svc_bus() -> None:
    orch_main._rpc_bus = None
    assert orch_main._bus_for_rpc() is orch_main.svc.bus


@pytest.mark.asyncio
async def test_close_rpc_bus_closes_and_clears_global() -> None:
    fake = AsyncMock()
    orch_main._rpc_bus = fake  # type: ignore[assignment]
    await orch_main._close_rpc_bus()
    fake.close.assert_awaited_once()
    assert orch_main._rpc_bus is None
