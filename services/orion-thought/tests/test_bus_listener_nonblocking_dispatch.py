from __future__ import annotations

import asyncio

import pytest

from app import bus_listener


@pytest.fixture(autouse=True)
def _clear_pending_handler_tasks() -> None:
    bus_listener._pending_handler_tasks.clear()
    yield
    bus_listener._pending_handler_tasks.clear()


@pytest.mark.asyncio
async def test_dispatch_bus_message_runs_handler_in_background() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_handler(raw_msg: dict) -> None:
        started.set()
        await release.wait()

    original = bus_listener._run_bus_message_handler
    bus_listener._run_bus_message_handler = slow_handler  # type: ignore[assignment]
    try:
        bus_listener._dispatch_bus_message({"data": b"x"})
        await asyncio.sleep(0)
        assert started.is_set()
        assert bus_listener._pending_handler_tasks
        release.set()
        await bus_listener._drain_pending_handler_tasks(timeout_sec=1.0)
        assert not bus_listener._pending_handler_tasks
    finally:
        bus_listener._run_bus_message_handler = original


@pytest.mark.asyncio
async def test_drain_pending_handler_tasks_waits_for_in_flight_work() -> None:
    release = asyncio.Event()

    async def slow_handler(raw_msg: dict) -> None:
        await release.wait()

    original = bus_listener._run_bus_message_handler
    bus_listener._run_bus_message_handler = slow_handler  # type: ignore[assignment]
    try:
        bus_listener._dispatch_bus_message({"data": b"x"})
        await asyncio.sleep(0)
        assert bus_listener._pending_handler_tasks
        release.set()
        await bus_listener._drain_pending_handler_tasks(timeout_sec=1.0)
        assert not bus_listener._pending_handler_tasks
    finally:
        bus_listener._run_bus_message_handler = original
