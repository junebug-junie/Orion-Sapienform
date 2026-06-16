from __future__ import annotations

from unittest.mock import patch

from orion.core.bus.async_service import OrionBusAsync


def test_pubsub_redis_uses_no_socket_read_timeout() -> None:
    bus = OrionBusAsync("redis://127.0.0.1:6379/0", enabled=True)
    pubsub_kwargs = bus._pubsub_redis_kwargs()
    command_kwargs = bus._command_redis_kwargs()

    assert pubsub_kwargs["socket_timeout"] is None
    assert command_kwargs["socket_timeout"] == 60.0


def test_create_pubsub_redis_uses_pubsub_kwargs() -> None:
    bus = OrionBusAsync("redis://127.0.0.1:6379/0", enabled=True)
    with patch("orion.core.bus.async_service.aioredis.from_url") as from_url:
        from_url.return_value = object()
        bus._create_pubsub_redis()
        _args, kwargs = from_url.call_args
        assert kwargs["socket_timeout"] is None
