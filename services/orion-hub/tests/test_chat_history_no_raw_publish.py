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
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

# Hub settings read `env_file=".env"`, which pydantic resolves *relative to the
# process cwd*. Run from the repo root it finds the 5.9K root `.env`, which has
# no CHANNEL_VOICE_* keys, and `scripts.settings` raises ValidationError at
# import. Run from `services/orion-hub` it finds the 20.6K hub `.env` and works.
# Without this block the behavioral tests below pass only when invoked from the
# hub directory, or when some *other* test module happens to be collected first
# and sets these as an import side effect. Same three lines as
# tests/test_social_room_turn_publish.py:6-8. Must precede any `scripts.*`
# import.
os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")

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

# The same channels written as bare string literals, which the repo does in 20+
# places (e.g. orion/signals/registry.py, concept_induction/settings.py).
_CHAT_HISTORY_CHANNEL_LITERALS = {
    "orion:chat:history:log",
    "orion:chat:history:turn",
}

# `OrionBusAsync.publish` is `publish(self, channel, msg)`; other bus wrappers in
# the tree name the second parameter differently, so accept any of these when
# the call is made with keyword arguments.
_PAYLOAD_KEYWORDS = ("msg", "payload", "message", "event", "envelope", "body")


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


def _unwrap(node: ast.AST) -> ast.AST:
    """Strip walrus wrappers so `(d := {...})` reads as `{...}`."""
    while isinstance(node, ast.NamedExpr):
        node = node.value
    return node


def _scope_bodies(tree: ast.Module):
    """Yield (scope_node, owned_nodes) for the module and every function.

    Bindings are resolved *per scope*. Resolving module-wide would bleed: in
    `chat_history.py` a local named `channel` is bound to the chat-history
    channel in `publish_chat_history`, while in `publish_response_feedback` the
    same name is a parameter defaulting to `orion:chat:response:feedback`. Only
    per-scope resolution tells those apart.
    """
    scopes: list[tuple[ast.AST, list[ast.AST]]] = []

    def owned(scope: ast.AST) -> list[ast.AST]:
        """Every node under `scope` except those inside a nested definition."""
        collected: list[ast.AST] = []

        def walk(node: ast.AST, is_root: bool) -> None:
            if not is_root and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                return
            collected.append(node)
            for child in ast.iter_child_nodes(node):
                walk(child, False)

        walk(scope, True)
        return collected

    scopes.append((tree, owned(tree)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append((node, owned(node)))
    return scopes


def _is_chat_history_channel(node: ast.AST, channel_names: set[str]) -> bool:
    node = _unwrap(node)
    if isinstance(node, ast.Attribute) and node.attr in _CHAT_HISTORY_CHANNEL_ATTRS:
        return True
    if isinstance(node, ast.Constant) and node.value in _CHAT_HISTORY_CHANNEL_LITERALS:
        return True
    return isinstance(node, ast.Name) and node.id in channel_names


def _is_raw_dict(node: ast.AST, dict_names: set[str]) -> bool:
    """True for a payload that is a bare mapping rather than an envelope."""
    node = _unwrap(node)
    if isinstance(node, ast.Dict):
        return True
    if isinstance(node, ast.Name):
        return node.id in dict_names
    if isinstance(node, ast.Call):
        func = node.func
        # `dict(a=1)`
        if isinstance(func, ast.Name) and func.id == "dict":
            return True
        # `payload.copy()` where payload is a known dict
        if isinstance(func, ast.Attribute) and func.attr in {"copy", "model_dump"}:
            return _is_raw_dict(func.value, dict_names)
    return False


def _bindings(nodes: list[ast.AST]) -> tuple[set[str], set[str]]:
    """Resolve, to a fixpoint, which local names hold a channel and which a dict."""
    assignments: list[tuple[str, ast.AST]] = []
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                assignments.append((node.target.id, node.value))
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                assignments.append((node.target.id, node.value))

    channel_names: set[str] = set()
    dict_names: set[str] = set()
    # Iterate so chains resolve: `a = {...}` then `b = a` then `c = b`.
    for _ in range(len(assignments) + 1):
        before = (len(channel_names), len(dict_names))
        for name, value in assignments:
            if _is_chat_history_channel(value, channel_names):
                channel_names.add(name)
            if _is_raw_dict(value, dict_names):
                dict_names.add(name)
        if (len(channel_names), len(dict_names)) == before:
            break
    return channel_names, dict_names


def _dict_literal_publishes_to_chat_history(source: Path) -> list[int]:
    """Return line numbers of `publish(<chat-history channel>, <raw dict>)` calls.

    Deliberately over-approximates rather than under-approximates: a false
    positive fails loudly and gets fixed, a false negative silently re-opens the
    hole this whole file exists to keep shut.
    """
    tree = ast.parse(source.read_text(), filename=str(source))
    offenders: list[int] = []

    for _scope, nodes in _scope_bodies(tree):
        channel_names, dict_names = _bindings(nodes)

        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name != "publish":
                continue

            keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}

            channel_arg = node.args[0] if node.args else keywords.get("channel")
            if channel_arg is None or not _is_chat_history_channel(
                channel_arg, channel_names
            ):
                continue

            payload_arg = None
            if len(node.args) >= 2:
                payload_arg = node.args[1]
            else:
                for key in _PAYLOAD_KEYWORDS:
                    if key in keywords:
                        payload_arg = keywords[key]
                        break
            if payload_arg is None:
                continue

            if _is_raw_dict(payload_arg, dict_names):
                offenders.append(node.lineno)

    return sorted(set(offenders))


def test_static_no_raw_dict_publish_to_chat_history() -> None:
    """No Hub source may publish a dict literal to a chat-history channel."""
    missing = [str(s) for s in _SCANNED_SOURCES if not s.exists()]
    # Skipping a renamed/moved file would turn this gate green by accident,
    # which is the exact failure mode it exists to prevent.
    assert missing == [], f"scanned source missing -- update _SCANNED_SOURCES: {missing}"

    found: dict[str, list[int]] = {}
    for source in _SCANNED_SOURCES:
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


# Every one of these was a confirmed false negative in the first version of this
# scanner. The bare-attribute form was the *only* shape it caught, which made a
# passing mutation test misleading: the gate looked durable and was not.
_EVASION_FORMS = {
    "keyword_args": "    await bus.publish(channel=settings.chat_history_channel, msg={'a': 1})\n",
    "dict_call": (
        "    d = dict(prompt='p', response='r')\n"
        "    await bus.publish(settings.chat_history_channel, d)\n"
    ),
    "variable_chain": (
        "    a = {'prompt': 'p'}\n"
        "    b = a\n"
        "    c = b\n"
        "    await bus.publish(settings.chat_history_channel, c)\n"
    ),
    "annotated_assign": (
        "    d: dict = {'prompt': 'p'}\n"
        "    await bus.publish(settings.chat_history_channel, d)\n"
    ),
    "walrus": "    await bus.publish(settings.chat_history_channel, (d := {'a': 1}))\n",
    "bare_publish_name": "    await publish(settings.chat_history_channel, {'a': 1})\n",
    "literal_channel_log": "    await bus.publish('orion:chat:history:log', {'a': 1})\n",
    "literal_channel_turn": "    await bus.publish('orion:chat:history:turn', {'a': 1})\n",
    "copy_of_dict": (
        "    base = {'prompt': 'p'}\n"
        "    await bus.publish(settings.chat_history_channel, base.copy())\n"
    ),
    "channel_via_local": (
        "    channel = settings.chat_history_channel\n"
        "    await bus.publish(channel, {'a': 1})\n"
    ),
    "channel_via_local_literal": (
        "    channel = 'orion:chat:history:log'\n"
        "    await bus.publish(channel, {'prompt': 'p'})\n"
    ),
}


@pytest.mark.parametrize("form", sorted(_EVASION_FORMS))
def test_static_scanner_catches_every_known_evasion(form: str, tmp_path: Path) -> None:
    planted = tmp_path / f"{form}.py"
    planted.write_text("async def go(bus, settings, publish):\n" + _EVASION_FORMS[form])
    assert _dict_literal_publishes_to_chat_history(planted), (
        f"scanner missed the {form!r} reintroduction form"
    )


def test_static_scanner_catches_a_plant_into_the_real_chat_history_module(
    tmp_path: Path,
) -> None:
    """The gate must fire on the file most likely to reintroduce this.

    `scripts/chat_history.py` owns every legitimate chat-history publish, so it
    is where a future raw publish would most plausibly be added. It binds the
    channel to a local (`channel = settings.chat_history_channel`) before
    publishing, which the first version of this scanner skipped entirely --
    the module was listed in `_SCANNED_SOURCES` and silently never checked.
    """
    real = _HUB_ROOT / "scripts" / "chat_history.py"
    source = real.read_text()
    assert "await bus.publish(channel, env)" in source, (
        "chat_history.py no longer has the publish this test plants into"
    )

    planted = tmp_path / "chat_history_planted.py"
    planted.write_text(
        source.replace(
            "await bus.publish(channel, env)",
            "await bus.publish(channel, {'prompt': 'p', 'response': 'r'})",
            1,
        )
    )
    assert _dict_literal_publishes_to_chat_history(planted), (
        "scanner did not catch a raw publish planted into the real chat_history.py"
    )


def test_static_scanner_does_not_confuse_scopes(tmp_path: Path) -> None:
    """A `channel` local in one function must not taint another function's.

    Real case: `chat_history.py` binds `channel` to the chat-history channel in
    `publish_chat_history`, while `publish_response_feedback` has a `channel`
    parameter defaulting to `orion:chat:response:feedback`. Module-wide binding
    resolution would flag the second as a chat-history publish.
    """
    mixed = tmp_path / "mixed.py"
    mixed.write_text(
        "async def a(bus, settings, env):\n"
        "    channel = settings.chat_history_channel\n"
        "    await bus.publish(channel, env)\n"
        "\n"
        "async def b(bus, channel='orion:chat:response:feedback'):\n"
        "    await bus.publish(channel, {'rating': 1})\n"
    )
    assert _dict_literal_publishes_to_chat_history(mixed) == []
