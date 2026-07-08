import asyncio
from unittest.mock import MagicMock

import pytest

from app import main as gw_main
from app.settings import settings


@pytest.mark.asyncio
async def test_main_rabbit_enables_concurrent_handlers_when_setting_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_gateway_concurrent_handlers", True)

    captured: dict[str, bool] = {}

    class FakeRabbit:
        def __init__(
            self,
            cfg,
            *,
            request_channel,
            handler,
            concurrent_handlers: bool = False,
        ) -> None:
            captured["concurrent_handlers"] = concurrent_handlers
            self.enabled = False
            self.bus = MagicMock()

        async def start(self) -> None:
            return None

    async def noop_async() -> None:
        return None

    monkeypatch.setattr(gw_main, "Rabbit", FakeRabbit)
    monkeypatch.setattr(gw_main, "_probe_route_targets", noop_async)
    monkeypatch.setattr(gw_main, "_serve_health", noop_async)

    await gw_main.main()

    assert captured.get("concurrent_handlers") is True
