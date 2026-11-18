"""Regression: a normal bus message must not produce an INFO log line.

The Hunter chassis used to log every received envelope at INFO; for this
service that was ~200k lines/hour (2026-10-11), enough to evict every other
service's history from the shared journald cap.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter


def _env() -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.edge.frame.v1",
        source=ServiceRef(name="vision-edge", node="n1"),
        payload={},
    )


def _hunter(n_messages: int, handled: list) -> Hunter:
    async def handler(env: BaseEnvelope) -> None:
        handled.append(env.kind)

    hunter = Hunter(
        ChassisConfig(
            service_name="test-gateway",
            service_version="0",
            node_name="n",
            bus_url="redis://localhost:6379/0",
            bus_enabled=True,
        ),
        handler=handler,
        patterns=["orion:vision:*"],
    )
    hunter.bus = MagicMock()
    hunter.bus.enabled = True
    hunter.bus.redis = object()
    hunter.bus.codec.decode = MagicMock(return_value=MagicMock(ok=True, envelope=_env()))

    @asynccontextmanager
    async def subscribe_ctx(*_a, patterns: bool = False):
        yield MagicMock()

    async def iter_messages(_pubsub):
        for _ in range(n_messages):
            yield {"type": "pmessage", "channel": b"orion:vision:frames", "pattern": b"orion:vision:*", "data": b"x"}
        hunter._stop.set()

    hunter.bus.subscribe = subscribe_ctx
    hunter.bus.iter_messages = iter_messages
    hunter._publish_error = AsyncMock()
    return hunter


@pytest.fixture
def info_records():
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(m.record["message"]), level="INFO")
    yield records
    logger.remove(sink_id)


async def test_normal_messages_do_not_log_at_info(info_records) -> None:
    handled: list = []
    hunter = _hunter(50, handled)
    await hunter._run()

    assert len(handled) == 50
    intake = [m for m in info_records if m.startswith("Hunter intake channel=")]
    # Only the first message after subscribe is kept at INFO (reconnect diagnosis).
    assert len(intake) == 1
    assert "first_after_subscribe=true" in intake[0]


async def test_intake_summary_rolls_up_counts(info_records) -> None:
    handled: list = []
    hunter = _hunter(5, handled)
    hunter.INTAKE_SUMMARY_INTERVAL_SEC = 0.0  # every message closes a window
    await hunter._run()

    summaries = [m for m in info_records if m.startswith("Hunter intake summary")]
    assert summaries, info_records
    assert "orion:vision:frames:vision.edge.frame.v1=1" in summaries[0]


def test_bad_log_level_falls_back_instead_of_crashing() -> None:
    from app.main import configure_loguru

    try:
        configure_loguru("WARN")  # stdlib accepts, loguru does not
        handlers = logger._core.handlers  # type: ignore[attr-defined]
        assert [h.levelno for h in handlers.values()] == [20]
    finally:
        logger.remove()
        import sys

        logger.add(sys.stderr)


def test_log_level_applies_to_loguru() -> None:
    from app.main import configure_loguru

    try:
        configure_loguru("INFO")
        # The default DEBUG stderr sink is replaced by one at LOG_LEVEL.
        handlers = logger._core.handlers  # type: ignore[attr-defined]
        assert handlers and all(h.levelno >= 20 for h in handlers.values())
    finally:
        logger.remove()
        import sys

        logger.add(sys.stderr)
