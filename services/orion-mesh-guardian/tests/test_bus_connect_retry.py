from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.service import MeshGuardianService
from app.settings import Settings


@pytest.mark.asyncio
async def test_connect_bus_with_retry_recovers_after_transient_failure() -> None:
    service = MeshGuardianService(Settings())
    attempts = 0

    async def _flaky_connect() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("Network is unreachable")

    service.bus.connect = AsyncMock(side_effect=_flaky_connect)
    service.bus.close = AsyncMock()

    await service._connect_bus_with_retry(max_wait_sec=10.0, initial_backoff_sec=0.01)

    assert attempts == 3
    assert service.bus.close.await_count == 2
