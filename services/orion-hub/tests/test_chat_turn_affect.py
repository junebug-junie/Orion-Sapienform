"""2026-08-25: the per-chat-turn affect bracket's policy and failure handling.

What is actually worth testing here is not "does it POST" -- it is the three
properties that make this safe to leave running against a real webcam:

1. The scope gate fails CLOSED. An unrecognized AFFECT_CHAT_TURN_SCOPE must
   never fall through to recording; a typo in a `.env` is the realistic way
   that happens.
2. The shared capture lock is always released. `_capture_blocking` runs in a
   detached thread, so a leaked lock would wedge every later capture --
   manual, ambient and chat-turn alike -- with no visible error anywhere.
3. Losing the slot is a drop, not a queue and not a retry, matching
   vision_affect_ambient's own stated no-retry policy for a live recording
   trigger.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import requests

from scripts import chat_turn_affect, vision_affect_ambient


def _settings(**overrides):
    base = {
        "AFFECT_CHAT_TURN_SCOPE": "voice",
        "JUNIPER_AFFECTIVE_STATE_BASE_URL": "http://affect.invalid:32799",
        "JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC": 5.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_capture_state():
    """The lock and state are module-level singletons shared with the real
    ambient loop, so leaving either dirty would leak across tests."""
    yield
    if vision_affect_ambient.state.tick_in_progress:
        vision_affect_ambient.end_capture(ok=False, error="test cleanup")


# --------------------------------------------------------------- scope gate

@pytest.mark.parametrize(
    "scope,is_voice,expected",
    [
        ("voice", True, True),
        ("voice", False, False),
        ("all", True, True),
        ("all", False, True),
        ("off", True, False),
        ("off", False, False),
    ],
)
def test_scope_matrix(scope, is_voice, expected):
    assert (
        chat_turn_affect.should_fire(_settings(AFFECT_CHAT_TURN_SCOPE=scope), is_voice_turn=is_voice)
        is expected
    )


@pytest.mark.parametrize("bad", ["Voice ", "typo", "true", "1", "", None])
def test_unrecognized_scope_fails_closed(bad):
    """Anything not exactly off/voice/all resolves to "off" -- NOT to the
    "voice" default. A misconfigured env var must not start a webcam."""
    if bad == "Voice ":
        # ...except for surrounding whitespace and case, which are normalized
        # rather than treated as a typo.
        assert chat_turn_affect.resolve_scope(_settings(AFFECT_CHAT_TURN_SCOPE=bad)) == "voice"
        return
    assert chat_turn_affect.resolve_scope(_settings(AFFECT_CHAT_TURN_SCOPE=bad)) == "off"
    assert chat_turn_affect.should_fire(
        _settings(AFFECT_CHAT_TURN_SCOPE=bad), is_voice_turn=True
    ) is False


def test_missing_scope_attribute_entirely_is_off():
    """A settings object predating this feature (an older container image
    still on the old build) must not start capturing on upgrade of the
    code alone."""
    assert chat_turn_affect.resolve_scope(SimpleNamespace()) == "off"


# ------------------------------------------------------------------- firing

def test_fire_returns_none_and_does_nothing_when_out_of_scope():
    assert (
        chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_PRE,
            correlation_id="corr-1",
            is_voice_turn=False,  # scope=voice, typed turn
        )
        is None
    )


def test_fire_returns_none_when_base_url_unconfigured():
    assert (
        chat_turn_affect.fire(
            settings=_settings(JUNIPER_AFFECTIVE_STATE_BASE_URL=""),
            trigger=chat_turn_affect.TRIGGER_PRE,
            correlation_id="corr-2",
            is_voice_turn=True,
        )
        is None
    )


def test_fire_sends_trigger_and_chat_correlation_id(monkeypatch):
    seen = {}

    def _fake_call(base_url, timeout_sec, trigger, *, chat_correlation_id=None):
        seen.update(
            base_url=base_url,
            timeout_sec=timeout_sec,
            trigger=trigger,
            chat_correlation_id=chat_correlation_id,
        )
        return {"result": {"ok": True, "raw_response": "calm"}, "capture": {"video_sha256": "abc"}}

    monkeypatch.setattr(vision_affect_ambient, "call_capture_and_assess", _fake_call)

    async def _run():
        task = chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_POST,
            correlation_id="corr-3",
            is_voice_turn=True,
        )
        assert task is not None
        await task

    asyncio.run(_run())
    assert seen["trigger"] == "chat_turn_post"
    assert seen["chat_correlation_id"] == "corr-3"
    assert seen["base_url"] == "http://affect.invalid:32799"
    # Released, not leaked.
    assert vision_affect_ambient.state.tick_in_progress is False


def test_transport_failure_still_releases_the_lock(monkeypatch):
    def _boom(*a, **k):
        raise requests.ConnectionError("affect service down")

    monkeypatch.setattr(vision_affect_ambient, "call_capture_and_assess", _boom)

    async def _run():
        task = chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_PRE,
            correlation_id="corr-4",
            is_voice_turn=True,
        )
        await task

    asyncio.run(_run())
    assert vision_affect_ambient.state.tick_in_progress is False
    assert vision_affect_ambient.state.last_result_ok is False


def test_unexpected_exception_still_releases_the_lock(monkeypatch):
    """Not just requests.RequestException -- a detached task that dies on a
    ValueError would strand the lock just as thoroughly and with even less
    warning."""
    def _boom(*a, **k):
        raise ValueError("malformed something")

    monkeypatch.setattr(vision_affect_ambient, "call_capture_and_assess", _boom)

    async def _run():
        await chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_PRE,
            correlation_id="corr-5",
            is_voice_turn=True,
        )

    asyncio.run(_run())
    assert vision_affect_ambient.state.tick_in_progress is False


def test_losing_the_shared_slot_drops_the_capture(monkeypatch):
    """Never queued, never retried -- and crucially, the caller that already
    holds the slot must still hold it afterwards."""
    called = []
    monkeypatch.setattr(
        vision_affect_ambient,
        "call_capture_and_assess",
        lambda *a, **k: called.append(1) or {"result": {"ok": True}},
    )
    assert vision_affect_ambient.try_begin_capture("ambient") is True
    try:

        async def _run():
            await chat_turn_affect.fire(
                settings=_settings(),
                trigger=chat_turn_affect.TRIGGER_PRE,
                correlation_id="corr-6",
                is_voice_turn=True,
            )

        asyncio.run(_run())
        assert called == [], "chat-turn capture ran while another held the slot"
        assert vision_affect_ambient.state.tick_in_progress is True
    finally:
        vision_affect_ambient.end_capture(ok=True, error=None)


def test_inflight_task_is_strongly_referenced_then_released(monkeypatch):
    """asyncio holds only a WEAK reference to a running task, and both real
    call sites discard fire()'s return value -- so without an explicit
    strong ref the capture can be garbage-collected mid-flight, stranding
    the shared lock forever. Assert the ref exists WHILE running and is
    dropped after, so the set cannot grow unbounded either."""
    started = asyncio.Event()
    release = asyncio.Event()
    loop_holder = {}

    def _slow(*a, **k):
        loop_holder["loop"].call_soon_threadsafe(started.set)
        # Block the worker thread until the test says go.
        import time

        while not loop_holder.get("go"):
            time.sleep(0.01)
        return {"result": {"ok": True}}

    monkeypatch.setattr(vision_affect_ambient, "call_capture_and_assess", _slow)

    async def _run():
        loop_holder["loop"] = asyncio.get_running_loop()
        task = chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_PRE,
            correlation_id="corr-inflight",
            is_voice_turn=True,
        )
        await started.wait()
        assert task in chat_turn_affect._INFLIGHT, "task not strongly referenced while running"
        loop_holder["go"] = True
        await task
        assert task not in chat_turn_affect._INFLIGHT, "done-callback did not release the ref"

    asyncio.run(_run())
    assert chat_turn_affect._INFLIGHT == set()


def test_fire_never_raises_when_no_loop_is_running():
    """The post-turn call site is inside a `finally`. If fire() raised there
    it would REPLACE the exception the turn was already unwinding, turning a
    real turn failure into a misleading affect error. Called with no running
    event loop, asyncio.create_task raises RuntimeError -- fire() must
    swallow it and return None instead."""
    assert (
        chat_turn_affect.fire(
            settings=_settings(),
            trigger=chat_turn_affect.TRIGGER_POST,
            correlation_id="corr-noloop",
            is_voice_turn=True,
        )
        is None
    )


def test_manual_and_ambient_request_bodies_are_unchanged(monkeypatch):
    """chat_correlation_id must be OMITTED, not sent as null, for the two
    pre-existing callers -- so their wire format is byte-identical to
    before this feature existed."""
    posted = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"ok": True}}

    def _fake_post(url, json=None, timeout=None):
        posted["json"] = json
        return _Resp()

    monkeypatch.setattr(vision_affect_ambient.requests, "post", _fake_post)
    vision_affect_ambient.call_capture_and_assess("http://x", 1.0, "ambient")
    assert posted["json"] == {"trigger": "ambient"}

    vision_affect_ambient.call_capture_and_assess(
        "http://x", 1.0, "chat_turn_pre", chat_correlation_id="corr-7"
    )
    assert posted["json"] == {"trigger": "chat_turn_pre", "chat_correlation_id": "corr-7"}
