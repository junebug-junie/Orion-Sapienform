from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.zwave_client import ZWaveJSClient


@pytest.mark.asyncio
async def test_connect_closes_on_bootstrap_failure() -> None:
    mock_ws = AsyncMock()
    mock_ws.close = AsyncMock()

    with patch(
        "app.zwave_client.websockets.connect",
        new_callable=AsyncMock,
        return_value=mock_ws,
    ):
        client = ZWaveJSClient("ws://test", node_id=2)
        client._bootstrap = AsyncMock(side_effect=RuntimeError("bootstrap failed"))

        with pytest.raises(RuntimeError, match="bootstrap failed"):
            await client.connect()

    assert client._ws is None
    assert client._listener_task is None
    assert client._dispatch_task is None
    mock_ws.close.assert_awaited_once()
