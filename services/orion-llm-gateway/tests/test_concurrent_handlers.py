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
    async def noop_pool_bus(parent) -> None:
        return None

    monkeypatch.setattr(gw_main, "_connect_pool_bus", noop_pool_bus)
    monkeypatch.setattr(gw_main, "_serve_health", noop_async)
    # The token-ledger publisher runs forever in production (like chat_svc.start); stub it so
    # main() returns, and pin that main() actually starts it on the chassis bus.
    started: dict = {}

    async def fake_publisher(bus, sink, **kw) -> None:
        started["bus"] = bus

    monkeypatch.setattr(gw_main, "run_usage_publisher", fake_publisher)
    monkeypatch.setattr(settings, "llm_gateway_call_usage_enabled", True)

    await gw_main.main()

    assert captured.get("concurrent_handlers") is True
    assert "bus" in started
