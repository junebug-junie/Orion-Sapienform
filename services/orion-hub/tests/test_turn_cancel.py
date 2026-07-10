from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState


@pytest.mark.asyncio
async def test_cancel_in_flight_turn_agent_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.turn_cancel as turn_cancel

    calls: list[str] = []

    async def fake_cancel(corr: str) -> bool:
        calls.append(corr)
        return True

    monkeypatch.setattr(turn_cancel, "cancel_agent_claude_turn", fake_cancel)
    await turn_cancel.cancel_in_flight_turn(
        bus=None,
        correlation_id="corr-ac",
        kind="agent-claude",
        reason="client_disconnect",
    )
    assert calls == ["corr-ac"]


@pytest.mark.asyncio
async def test_cancel_in_flight_turn_orion_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.turn_cancel as turn_cancel

    published: list[tuple[str, str]] = []

    async def fake_publish(bus: Any, *, correlation_id: str, reason: str = "client_disconnect") -> None:
        published.append((correlation_id, reason))

    monkeypatch.setattr(turn_cancel, "publish_harness_run_cancel", fake_publish)
    await turn_cancel.cancel_in_flight_turn(
        bus=object(),
        correlation_id="corr-orion",
        kind="orion",
        reason="client_disconnect",
    )
    assert published == [("corr-orion", "client_disconnect")]


@pytest.mark.asyncio
async def test_run_awaitable_cancel_on_ws_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.turn_cancel as turn_cancel

    cancelled: list[str] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append(correlation_id)

    monkeypatch.setattr(turn_cancel, "cancel_in_flight_turn", fake_cancel)

    class _WS:
        def __init__(self) -> None:
            self._state = WebSocketState.CONNECTED

        @property
        def client_state(self) -> WebSocketState:
            return self._state

    ws = _WS()

    async def _work() -> str:
        await asyncio.sleep(0.05)
        ws._state = WebSocketState.DISCONNECTED
        await asyncio.sleep(0.25)
        return "done"

    out = await turn_cancel.run_awaitable_cancel_on_ws_disconnect(
        ws,  # type: ignore[arg-type]
        _work(),
        bus=None,
        correlation_id="corr-watch",
        kind="orion",
        poll_sec=0.05,
    )
    assert out == "done"
    assert cancelled == ["corr-watch"]


@pytest.mark.asyncio
async def test_run_awaitable_cancel_on_websocket_disconnect_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-turn send raising WebSocketDisconnect must still cancel (not only poller)."""
    import scripts.turn_cancel as turn_cancel

    cancelled: list[str] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append(correlation_id)

    monkeypatch.setattr(turn_cancel, "cancel_in_flight_turn", fake_cancel)

    class _WS:
        client_state = WebSocketState.CONNECTED

    async def _work() -> str:
        await asyncio.sleep(0.01)
        raise WebSocketDisconnect()

    with pytest.raises(WebSocketDisconnect):
        await turn_cancel.run_awaitable_cancel_on_ws_disconnect(
            _WS(),  # type: ignore[arg-type]
            _work(),
            bus=None,
            correlation_id="corr-exc",
            kind="orion",
            poll_sec=5.0,
        )
    assert cancelled == ["corr-exc"]
