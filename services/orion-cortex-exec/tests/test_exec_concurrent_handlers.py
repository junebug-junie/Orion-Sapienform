from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app import main as exec_main
from app.settings import settings


def test_legacy_rabbit_enables_concurrent_handlers_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "exec_concurrent_handlers", True)
    assert exec_main.svc.concurrent_handlers is True


def test_legacy_rabbit_respects_concurrent_handlers_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "exec_concurrent_handlers", False)
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
            self.concurrent_handlers = concurrent_handlers
            self.bus = MagicMock()

    monkeypatch.setattr(exec_main, "Rabbit", FakeRabbit)
    exec_main.svc = FakeRabbit(
        MagicMock(),
        request_channel=settings.channel_exec_request,
        handler=exec_main.handle,
        concurrent_handlers=settings.exec_concurrent_handlers,
    )
    assert captured["concurrent_handlers"] is False
