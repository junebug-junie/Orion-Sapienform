"""settlement_loop must outlive a Redis drop (live 2026-10-20: one ConnectionError ended it)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

import pytest

from app import main as energy_main
from app.settings import Settings

REPO = Path(__file__).resolve().parents[3]
TARIFF = REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"


class _Session:
    """One pubsub session: hands out ``messages`` then sets ``stop``."""

    def __init__(self, messages: list[dict], stop: asyncio.Event) -> None:
        self._messages = list(messages)
        self._stop = stop

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if not self._messages:
            self._stop.set()
            return None
        return self._messages.pop(0)


class _DroppingBus:
    """First subscribe raises like a Redis drop; later ones hand out scripted sessions."""

    def __init__(self, sessions: list[object]) -> None:
        self.sessions = list(sessions)
        self.subscribe_calls = 0

    def subscribe(self, *channels: str, patterns: bool = False):
        bus = self

        @contextlib.asynccontextmanager
        async def _cm():
            bus.subscribe_calls += 1
            session = bus.sessions.pop(0)
            if isinstance(session, BaseException):
                raise session
            yield session

        return _cm()


def test_settlement_loop_resubscribes_after_redis_drop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    s = Settings(
        ORION_BUS_ENABLED=False,
        ENERGY_TARIFF_PATH=str(TARIFF),
        ENERGY_PROCESSED_DIR=str(tmp_path / "processed"),
        ENERGY_BILL_PROCESSED_DIR=str(tmp_path / "bills/processed"),
        ENERGY_PUBLISH_MAX_PER_SEC=0,
    )
    pipeline = energy_main.build_pipeline(s)

    async def _run() -> _DroppingBus:
        stop = asyncio.Event()
        bus = _DroppingBus(
            [ConnectionError("bus down"), _Session([{"type": "message", "data": b"not an envelope"}], stop)]
        )
        await asyncio.wait_for(
            energy_main.settlement_loop(bus, s, pipeline, asyncio.Lock(), stop=stop), timeout=10
        )
        return bus

    with caplog.at_level(logging.WARNING, logger="orion-energy"):
        bus = asyncio.run(_run())
    assert bus.subscribe_calls == 2
    # The message after the reconnect reached the settlement handler.
    assert "energy_settlement_decode_failed" in caplog.text
