"""General 'stop chat' command: a per-connection active-turn registry in
websocket_handler.py, and the /api/chat/turn/cancel endpoint that uses it.

The WS receive loop blocks awaiting the in-flight turn between
websocket.receive_text() calls, so a same-connection "stop" message would sit
unread until the turn already finished. This registry is the side channel that
lets an HTTP request find and cancel a connection's in-flight turn instead.

Keyed by connection_id, not session_id: session_id is persisted client-side in
localStorage and shared across every browser tab on the same origin, so keying
on it would let a stop click in one tab cancel another tab's turn. Each
registry entry is the connection's own `active_turn` dict, registered once by
reference — mutating it in place (as the WS loop already does) is
automatically visible to a lookup, with no separate register/clear calls to
keep in sync.
"""
from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_cancel_active_turn_for_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.websocket_handler as wh

    cancelled: list[tuple[str, str, str]] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append((correlation_id, kind, reason))

    monkeypatch.setattr(wh, "cancel_in_flight_turn", fake_cancel)
    wh._ACTIVE_TURNS_BY_CONNECTION.clear()

    active_turn = {"correlation_id": "corr-1", "kind": "orion"}
    wh._ACTIVE_TURNS_BY_CONNECTION["conn-1"] = active_turn

    result = await wh.cancel_active_turn_for_connection("conn-1", bus=object(), reason="user_stop")

    assert result == "corr-1"
    assert cancelled == [("corr-1", "orion", "user_stop")]


@pytest.mark.asyncio
async def test_cancel_active_turn_for_unknown_connection_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.websocket_handler as wh

    cancelled: list[str] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append(correlation_id)

    monkeypatch.setattr(wh, "cancel_in_flight_turn", fake_cancel)
    wh._ACTIVE_TURNS_BY_CONNECTION.clear()

    result = await wh.cancel_active_turn_for_connection("no-such-connection", bus=object())

    assert result is None
    assert cancelled == []


@pytest.mark.asyncio
async def test_cancel_active_turn_for_connection_with_no_turn_in_flight_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection is registered at setup even before any turn starts (correlation_id
    is None until a turn actually begins) — cancel must no-op, not crash or cancel a
    turn that isn't real.
    """
    import scripts.websocket_handler as wh

    cancelled: list[str] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append(correlation_id)

    monkeypatch.setattr(wh, "cancel_in_flight_turn", fake_cancel)
    wh._ACTIVE_TURNS_BY_CONNECTION.clear()
    wh._ACTIVE_TURNS_BY_CONNECTION["conn-2"] = {"correlation_id": None, "kind": None}

    result = await wh.cancel_active_turn_for_connection("conn-2", bus=object())

    assert result is None
    assert cancelled == []


@pytest.mark.asyncio
async def test_cancel_active_turn_reflects_live_mutation_no_manual_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry stores the active_turn dict by reference: mutating it directly
    (as the WS loop's turn sites do) must be visible without any separate
    register/clear call — this is the whole point of not keeping two structures
    in sync by hand.
    """
    import scripts.websocket_handler as wh

    cancelled: list[str] = []

    async def fake_cancel(*, bus: Any, correlation_id: str, kind: str, reason: str = "client_disconnect") -> None:
        cancelled.append(correlation_id)

    monkeypatch.setattr(wh, "cancel_in_flight_turn", fake_cancel)
    wh._ACTIVE_TURNS_BY_CONNECTION.clear()

    active_turn = {"correlation_id": None, "kind": None}
    wh._ACTIVE_TURNS_BY_CONNECTION["conn-3"] = active_turn

    # Simulate a turn starting: mutate in place, exactly as the WS loop does.
    active_turn["correlation_id"] = "corr-3"
    active_turn["kind"] = "agent-claude"

    result = await wh.cancel_active_turn_for_connection("conn-3", bus=object())

    assert result == "corr-3"
    assert cancelled == ["corr-3"]


@pytest.mark.asyncio
async def test_api_chat_turn_cancel_requires_connection_id() -> None:
    import scripts.api_routes as api_routes
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await api_routes.api_chat_turn_cancel(api_routes.ChatTurnCancelRequest(connection_id="  "))
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_api_chat_turn_cancel_503_when_bus_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    from fastapi import HTTPException

    monkeypatch.setattr(hub_main, "bus", None)

    with pytest.raises(HTTPException) as exc_info:
        await api_routes.api_chat_turn_cancel(api_routes.ChatTurnCancelRequest(connection_id="conn-3"))
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_api_chat_turn_cancel_returns_cancelled_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.api_routes as api_routes
    import scripts.main as hub_main
    import scripts.websocket_handler as wh

    class _FakeBus:
        enabled = True

    monkeypatch.setattr(hub_main, "bus", _FakeBus())
    monkeypatch.setattr(hub_main, "rpc_bus", None)

    async def fake_cancel_active_turn_for_connection(connection_id: str, *, bus: Any, reason: str = "user_stop"):
        assert connection_id == "conn-4"
        return "corr-4"

    monkeypatch.setattr(wh, "cancel_active_turn_for_connection", fake_cancel_active_turn_for_connection)

    result = await api_routes.api_chat_turn_cancel(api_routes.ChatTurnCancelRequest(connection_id="conn-4"))

    assert result == {"cancelled": True, "correlation_id": "corr-4"}
