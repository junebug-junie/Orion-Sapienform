"""Regression tests for cortex-gateway bus intake consumer resilience."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.bus_client import BusClient  # type: ignore  # noqa: E402


def _make_intake_bus(messages: list[dict[str, Any]], events: list[str]):
    """Fake intake bus that records subscribe/listen lifecycle per message."""

    pending = list(messages)

    @asynccontextmanager
    async def subscribe(*_channels: str):
        events.append("subscribe_enter")
        yield object()
        events.append("subscribe_exit")

    async def iter_messages(_pubsub: Any):
        events.append("listen_start")
        if not pending:
            return
        msg = pending.pop(0)
        yield msg
        events.append("listen_end")

    bus = AsyncMock()
    bus.subscribe = subscribe
    bus.iter_messages = iter_messages
    return bus


@pytest.mark.asyncio
async def test_gateway_consumer_exits_intake_subscribe_before_dispatch(monkeypatch) -> None:
    """Dispatch must run outside intake subscribe so orch RPC pubsub cannot overlap listen()."""
    events: list[str] = []
    client = BusClient()
    client._intake_bus = _make_intake_bus([{"data": b"one"}], events)
    holder: dict[str, asyncio.Task[None]] = {}

    async def fake_dispatch(_message: dict[str, Any]) -> None:
        events.append("dispatch_enter")
        await asyncio.sleep(0)
        events.append("dispatch_exit")
        holder["task"].cancel()

    monkeypatch.setattr(client, "_gateway_dispatch_one", fake_dispatch)

    holder["task"] = asyncio.create_task(client._consume_gateway_request())
    with pytest.raises(asyncio.CancelledError):
        await holder["task"]

    assert events.index("subscribe_exit") < events.index("dispatch_enter")
    assert events[:5] == [
        "subscribe_enter",
        "listen_start",
        "subscribe_exit",
        "dispatch_enter",
        "dispatch_exit",
    ]


@pytest.mark.asyncio
async def test_gateway_consumer_processes_multiple_messages_sequentially(monkeypatch) -> None:
    """Consumer must stay alive across back-to-back Hub RPCs (regression for dead subscriber)."""
    events: list[str] = []
    client = BusClient()
    client._intake_bus = _make_intake_bus(
        [{"data": b"one"}, {"data": b"two"}],
        events,
    )
    dispatch_count = 0
    holder: dict[str, asyncio.Task[None]] = {}

    async def fake_dispatch(_message: dict[str, Any]) -> None:
        nonlocal dispatch_count
        events.append(f"dispatch_{dispatch_count}")
        dispatch_count += 1
        await asyncio.sleep(0)
        if dispatch_count >= 2:
            holder["task"].cancel()

    monkeypatch.setattr(client, "_gateway_dispatch_one", fake_dispatch)

    holder["task"] = asyncio.create_task(client._consume_gateway_request())
    with pytest.raises(asyncio.CancelledError):
        await holder["task"]

    assert dispatch_count == 2
    subscribe_exits = [idx for idx, event in enumerate(events) if event == "subscribe_exit"]
    dispatch_starts = [idx for idx, event in enumerate(events) if event.startswith("dispatch_")]
    assert len(subscribe_exits) == 2
    assert len(dispatch_starts) == 2
    assert all(exit_idx < start_idx for exit_idx, start_idx in zip(subscribe_exits, dispatch_starts))


@pytest.mark.asyncio
async def test_connect_forks_separate_intake_and_rpc_buses(monkeypatch) -> None:
    """Orch RPC traffic uses a forked bus distinct from the intake subscriber."""
    client = BusClient()
    fork_calls = 0

    async def fake_connect() -> None:
        return None

    async def fake_fork(*, start_rpc_worker: bool = False):
        nonlocal fork_calls
        fork_calls += 1
        child = BusClient()
        child.connect = fake_connect  # type: ignore[method-assign]
        child.close = AsyncMock()
        return child

    client.bus = AsyncMock()
    client.bus.connect = fake_connect
    client.bus.fork = fake_fork

    await client.connect()

    assert fork_calls == 2
    assert client._intake_bus is not None
    assert client._rpc_bus is not None
    assert client._intake_bus is not client._rpc_bus
    assert client._intake_bus is not client.bus
    assert client._rpc_bus is not client.bus
