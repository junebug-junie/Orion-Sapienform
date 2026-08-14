"""Regression: the Hub must never publish a raw dict to the chat-history channel.

Background (2026-08-14). The Hub published each finished turn to
`chat_history_channel` three times: two `chat.history.message.v1` envelopes, one
`chat.turn` envelope, and a third *raw dict* carrying `prompt`/`response`. That
third publish had no `kind`, so `orion/core/bus/codec.py:72` stamped it
`legacy.message`, which matches no entry in the sql-writer route map. Every one
of them fell through to `_write_fallback` and landed in `bus_fallback_log` --
written nowhere else, one WARNING per turn, and (after the backlog watcher
shipped) an escalating email alert driven purely by normal chat volume.

There were two such publishes, one per transport:

* WS/unified turn: `orion/hub/turn_orchestrator.py`
* HTTP chat:       `services/orion-hub/scripts/api_routes.py`

Both are deleted. These tests fail if either comes back.

`test_static_no_raw_dict_publish_to_chat_history` is the durable gate: the
behavioral test below only covers the WS path, and a reintroduction on any other
Hub code path would slip past it.
"""
from __future__ import annotations

import ast
import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]

# Every Hub source file that may publish to a chat-history channel.
_SCANNED_SOURCES = (
    _REPO_ROOT / "orion" / "hub" / "turn_orchestrator.py",
    _HUB_ROOT / "scripts" / "api_routes.py",
    _HUB_ROOT / "scripts" / "chat_history.py",
    _HUB_ROOT / "scripts" / "websocket_handler.py",
)

# Attribute names on `settings` that resolve to a chat-history bus channel.
_CHAT_HISTORY_CHANNEL_ATTRS = {
    "chat_history_channel",
    "chat_history_turn_channel",
    "CHAT_HISTORY_LOG_CHANNEL",
    "CHANNEL_CHAT_HISTORY_LOG",
    "CHAT_HISTORY_TURN_CHANNEL",
    "CHANNEL_CHAT_HISTORY_TURN",
}


class _RecordingBus:
    """Minimal bus double that records every publish."""

    enabled = True

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, channel: str, payload: Any) -> None:
        self.published.append((channel, payload))


_CORR_ID = str(uuid.UUID("11111111-2222-3333-4444-555555555555"))


def _make_run(correlation_id: str = _CORR_ID):
    from orion.schemas.harness_finalize import HarnessRunV1

    return HarnessRunV1(
        correlation_id=correlation_id,
        final_text="an answer",
        finalize_ran=True,
        step_count=3,
        compliance_verdict="completed",
        grounding_status="grounded",
    )


@pytest.fixture()
def _publish_enabled(monkeypatch):
    """Turn on PUBLISH_CHAT_HISTORY_LOG, which gates the two *real* publishes."""
    from scripts.settings import settings as hub_settings

    monkeypatch.setattr(hub_settings, "PUBLISH_CHAT_HISTORY_LOG", True)
    return hub_settings


def _run_ws_publish(bus: _RecordingBus, response_text: str = "an answer") -> None:
    from orion.hub.turn_orchestrator import _publish_unified_turn_chat_history

    asyncio.run(
        _publish_unified_turn_chat_history(
            bus=bus,
            correlation_id=_CORR_ID,
            session_id="sess-1",
            user_message="a question",
            response_text=response_text,
            payload={"user_id": "juniper"},
            run=_make_run(),
        )
    )


def test_ws_turn_publishes_only_envelopes(_publish_enabled) -> None:
    """The unified-turn path emits envelopes only -- never a bare dict."""
    bus = _RecordingBus()
    _run_ws_publish(bus)

    assert bus.published, "expected the unified turn to publish something"

    raw_dict_publishes = [
        (channel, payload)
        for channel, payload in bus.published
        if isinstance(payload, dict)
    ]
    assert raw_dict_publishes == [], (
        "the Hub published a raw dict to the bus; the codec stamps these "
        "`legacy.message`, which no sql-writer route matches, so they are "
        f"dropped into bus_fallback_log: {raw_dict_publishes!r}"
    )

    for channel, payload in bus.published:
        kind = getattr(payload, "kind", None)
        assert kind, f"publish to {channel} carried no `kind`: {payload!r}"


def test_ws_turn_still_persists_the_turn(_publish_enabled) -> None:
    """Deleting the raw publish must not cost us the real persistence path.

    Guards the tempting-but-wrong fix: flipping PUBLISH_CHAT_HISTORY_LOG off
    would also silence these two, killing chat_history_log writes.
    """
    from scripts.settings import settings as hub_settings

    bus = _RecordingBus()
    _run_ws_publish(bus)

    observed = [(channel, getattr(payload, "kind", None)) for channel, payload in bus.published]

    # Exactly three publishes survive: the user message, the assistant message
    # (both `chat.history.message.v1` on the log channel), and the turn
    # envelope (`chat.history` on the *turn* channel). The deleted raw dict was
    # a fourth, on the log channel, with no kind at all.
    assert observed == [
        (hub_settings.chat_history_channel, "chat.history.message.v1"),
        (hub_settings.chat_history_channel, "chat.history.message.v1"),
        (hub_settings.chat_history_turn_channel, "chat.history"),
    ], observed

    rendered = " ".join(repr(getattr(p, "payload", p)) for _, p in bus.published)
    assert "a question" in rendered, "user message missing from published envelopes"
    assert "an answer" in rendered, "assistant response missing from published envelopes"


def test_ws_turn_publishes_nothing_when_response_is_empty(_publish_enabled) -> None:
    """No-empty-shell guard: a blank answer must not be persisted as a turn."""
    bus = _RecordingBus()
    _run_ws_publish(bus, response_text="   ")
    assert bus.published == []


def _dict_literal_publishes_to_chat_history(source: Path) -> list[int]:
    """Return line numbers of `*.publish(<chat-history channel>, {...})` calls."""
    tree = ast.parse(source.read_text(), filename=str(source))
    offenders: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "publish"):
            continue
        if len(node.args) < 2:
            continue

        channel_arg, payload_arg = node.args[0], node.args[1]

        # Channel must resolve to a chat-history channel (settings.<attr>).
        if not (
            isinstance(channel_arg, ast.Attribute)
            and channel_arg.attr in _CHAT_HISTORY_CHANNEL_ATTRS
        ):
            continue

        # A dict literal, or a Name bound to one, is an unenveloped payload.
        if isinstance(payload_arg, ast.Dict):
            offenders.append(node.lineno)
        elif isinstance(payload_arg, ast.Name):
            for other in ast.walk(tree):
                if (
                    isinstance(other, ast.Assign)
                    and isinstance(other.value, ast.Dict)
                    and any(
                        isinstance(t, ast.Name) and t.id == payload_arg.id
                        for t in other.targets
                    )
                ):
                    offenders.append(node.lineno)
                    break

    return offenders


def test_static_no_raw_dict_publish_to_chat_history() -> None:
    """No Hub source may publish a dict literal to a chat-history channel."""
    found: dict[str, list[int]] = {}
    for source in _SCANNED_SOURCES:
        if not source.exists():
            continue
        lines = _dict_literal_publishes_to_chat_history(source)
        if lines:
            found[str(source.relative_to(_REPO_ROOT))] = lines

    assert found == {}, (
        "raw dict published to a chat-history channel; the bus codec stamps a "
        "payload with no `kind` as `legacy.message`, which routes nowhere and "
        f"is dropped into bus_fallback_log: {found}"
    )


def test_static_scanner_detects_a_planted_offender(tmp_path: Path) -> None:
    """Mutation check: the static gate must actually fail on the deleted code."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        "async def go(bus, settings, corr, text):\n"
        "    chat_log_payload = {'correlation_id': corr, 'response': text}\n"
        "    await bus.publish(settings.chat_history_channel, chat_log_payload)\n"
        "    await bus.publish(settings.chat_history_channel, {'response': text})\n"
    )
    assert _dict_literal_publishes_to_chat_history(planted) == [3, 4]


def test_static_scanner_accepts_envelope_publishes(tmp_path: Path) -> None:
    """...and must not fire on the legitimate enveloped publishes."""
    clean = tmp_path / "clean.py"
    clean.write_text(
        "async def go(bus, settings, env):\n"
        "    await bus.publish(settings.chat_history_channel, env)\n"
        "    await bus.publish('orion:some:other:channel', {'a': 1})\n"
    )
    assert _dict_literal_publishes_to_chat_history(clean) == []
