from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.rpc_fork import fork_rpc_client


@pytest.mark.asyncio
async def test_fork_rpc_client_starts_worker() -> None:
    parent = OrionBusAsync("redis://127.0.0.1:6379/0", enabled=True)
    child = AsyncMock(spec=OrionBusAsync)
    child.enabled = True
    child.url = parent.url

    with patch.object(parent, "fork", new=AsyncMock(return_value=child)) as fork_mock:
        got = await fork_rpc_client(parent)

    fork_mock.assert_awaited_once_with(start_rpc_worker=True)
    assert got is child
