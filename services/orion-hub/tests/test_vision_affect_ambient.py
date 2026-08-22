"""Covers services/orion-hub/scripts/vision_affect_ambient.py -- the real
recurring-capture toggle (design correction, 2026-08-22: replaces a one-shot
button that had silently shipped in place of the toggle Juniper actually
asked for and approved). No real HTTP, no real Hub app -- exercises
run_ambient_tick()/affect_ambient_loop() directly against a mocked
requests.post, matching the direct-call testing convention already used by
test_vision_affect_capture_api.py.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts import vision_affect_ambient as ambient  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    ambient.state = ambient.AffectAmbientState()
    # threading.Lock instances can't be "reset" -- a test that acquires
    # without releasing (intentionally, to test the collision path) would
    # otherwise leave every later test unable to ever acquire it again.
    ambient._capture_lock = ambient.threading.Lock()
    yield
    ambient.state = ambient.AffectAmbientState()
    ambient._capture_lock = ambient.threading.Lock()


def test_call_capture_and_assess_sends_trigger_and_url():
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"result": {"ok": True}}
    with patch.object(ambient.requests, "post", return_value=fake_resp) as mock_post:
        body = ambient.call_capture_and_assess("http://circe:32799/", 240.0, "ambient")

    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "http://circe:32799/v1/juniper/affect/capture_and_assess"
    assert mock_post.call_args.kwargs["json"] == {"trigger": "ambient"}
    assert mock_post.call_args.kwargs["timeout"] == 240.0
    assert body["result"]["ok"] is True


@pytest.mark.asyncio
async def test_run_ambient_tick_success_updates_state():
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"result": {"ok": True, "raw_response": "calm"}}
    with patch.object(ambient.requests, "post", return_value=fake_resp):
        await ambient.run_ambient_tick("http://circe:32799", 240.0)

    assert ambient.state.tick_count == 1
    assert ambient.state.last_result_ok is True
    assert ambient.state.last_error is None
    assert ambient.state.tick_in_progress is False
    assert ambient.state.last_attempt_at is not None


@pytest.mark.asyncio
async def test_run_ambient_tick_business_failure_updates_state_without_raising():
    """The orchestrator's own endpoint replies 200 with ok=False inside the
    body on internal failure -- this must be read as a real failed tick,
    not silently treated as success."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"result": {"ok": False, "error": "capture failed: busy"}}
    with patch.object(ambient.requests, "post", return_value=fake_resp):
        await ambient.run_ambient_tick("http://circe:32799", 240.0)

    assert ambient.state.last_result_ok is False
    assert ambient.state.last_error == "capture failed: busy"
    assert ambient.state.tick_in_progress is False


@pytest.mark.asyncio
async def test_run_ambient_tick_transport_failure_never_raises():
    """No retries, per Juniper's explicit instruction -- but ALSO the tick
    coroutine itself must never raise, or one bad tick would kill the whole
    background loop task."""
    import requests as requests_module

    with patch.object(
        ambient.requests, "post", side_effect=requests_module.ConnectionError("refused")
    ):
        await ambient.run_ambient_tick("http://circe:32799", 240.0)

    assert ambient.state.last_result_ok is False
    assert "refused" in ambient.state.last_error
    assert ambient.state.tick_in_progress is False


@pytest.mark.asyncio
async def test_loop_does_not_tick_while_disabled():
    tick_calls = []

    async def _fake_tick(base_url, timeout_sec):
        tick_calls.append((base_url, timeout_sec))

    with patch.object(ambient, "run_ambient_tick", _fake_tick):
        task = asyncio.create_task(
            ambient.affect_ambient_loop(
                base_url="http://circe:32799", interval_sec=0.01, timeout_sec=1.0, poll_sec=0.01
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert tick_calls == [], "loop ticked while state.enabled was False"


@pytest.mark.asyncio
async def test_loop_ticks_when_enabled_and_due():
    tick_calls = []

    async def _fake_tick(base_url, timeout_sec):
        tick_calls.append((base_url, timeout_sec))
        ambient.state.last_attempt_at = ambient.time.time()

    ambient.state.enabled = True
    with patch.object(ambient, "run_ambient_tick", _fake_tick):
        task = asyncio.create_task(
            ambient.affect_ambient_loop(
                base_url="http://circe:32799", interval_sec=0.02, timeout_sec=1.0, poll_sec=0.01
            )
        )
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(tick_calls) >= 2, f"expected multiple ticks over 0.15s at a 0.02s interval, got {tick_calls}"
    assert all(call == ("http://circe:32799", 1.0) for call in tick_calls)


@pytest.mark.asyncio
async def test_loop_never_ticks_concurrently():
    """tick_in_progress must exclude the loop from starting a second tick
    while one is still running, even if it's overdue by the time the first
    finishes."""
    concurrent_count = {"value": 0, "max_seen": 0}

    async def _slow_tick(base_url, timeout_sec):
        ambient.state.tick_in_progress = True
        concurrent_count["value"] += 1
        concurrent_count["max_seen"] = max(concurrent_count["max_seen"], concurrent_count["value"])
        await asyncio.sleep(0.05)
        concurrent_count["value"] -= 1
        ambient.state.last_attempt_at = ambient.time.time()
        ambient.state.tick_in_progress = False

    ambient.state.enabled = True
    with patch.object(ambient, "run_ambient_tick", _slow_tick):
        task = asyncio.create_task(
            ambient.affect_ambient_loop(
                base_url="http://circe:32799", interval_sec=0.01, timeout_sec=1.0, poll_sec=0.01
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert concurrent_count["max_seen"] == 1, (
        f"loop allowed {concurrent_count['max_seen']} concurrent ticks -- "
        "tick_in_progress must serialize them"
    )


def test_status_payload_shape():
    ambient.state.enabled = True
    ambient.state.tick_count = 3
    ambient.state.last_result_ok = False
    ambient.state.last_error = "timeout"
    ambient.state.last_trigger = "ambient"
    ambient.state.last_raw_response = "calm"
    ambient.state.last_video_sha256 = "a" * 64
    payload = ambient.state.status_payload()
    assert payload == {
        "enabled": True,
        "tick_in_progress": False,
        "tick_count": 3,
        "last_attempt_at": None,
        "last_trigger": "ambient",
        "last_result_ok": False,
        "last_error": "timeout",
        "last_raw_response": "calm",
        "last_video_sha256": "a" * 64,
    }


# --- Shared capture slot (review finding, 2026-08-22) -----------------------
# Originally the manual "Check now" route bypassed this module's state
# entirely -- a collision with an in-flight ambient tick was only ever
# caught incidentally by retina's own device lock (a confusing generic
# "busy", and the ambient loop losing its cycle with no record of why).


def test_try_begin_capture_excludes_a_concurrent_caller():
    assert ambient.try_begin_capture("manual") is True
    assert ambient.state.tick_in_progress is True
    assert ambient.state.last_trigger == "manual"

    # A second caller (either trigger) must NOT win the slot while the
    # first is still in flight.
    assert ambient.try_begin_capture("ambient") is False
    assert ambient.state.last_trigger == "manual", "a losing caller must not overwrite state"


def test_end_capture_releases_the_slot_for_the_next_caller():
    assert ambient.try_begin_capture("manual") is True
    ambient.end_capture(ok=True, error=None)

    assert ambient.state.tick_in_progress is False
    assert ambient.state.last_result_ok is True
    assert ambient.try_begin_capture("ambient") is True, "the slot must be free again after end_capture"


def test_end_capture_stores_raw_response_only_on_success():
    ambient.state.last_raw_response = "sad, contemplative"
    ambient.state.last_video_sha256 = "a" * 64

    ambient.try_begin_capture("manual")
    ambient.end_capture(ok=False, error="timeout")

    # A failure must NOT erase the last real reading -- "Carbon (affect
    # snapshot)" should keep showing the last successful result, not go
    # blank the moment one attempt fails.
    assert ambient.state.last_raw_response == "sad, contemplative"
    assert ambient.state.last_video_sha256 == "a" * 64

    ambient.try_begin_capture("ambient")
    ambient.end_capture(ok=True, error=None, raw_response="calm", video_sha256="b" * 64)

    assert ambient.state.last_raw_response == "calm"
    assert ambient.state.last_video_sha256 == "b" * 64


def test_result_content_extracts_raw_response_and_video_sha256():
    body = {
        "capture": {"ok": True, "video_sha256": "a" * 64},
        "result": {"ok": True, "raw_response": "focused"},
    }
    raw_response, video_sha256 = ambient.result_content(body)
    assert raw_response == "focused"
    assert video_sha256 == "a" * 64


def test_result_content_handles_a_missing_result_or_capture():
    assert ambient.result_content({}) == (None, None)
    assert ambient.result_content({"result": {"ok": False}}) == (None, None)


@pytest.mark.asyncio
async def test_ambient_tick_skips_without_retrying_when_manual_holds_the_slot():
    """No retries, per Juniper's explicit instruction -- a collision with a
    manual capture is handled exactly like any other failed tick: skip,
    wait for the next scheduled attempt."""
    assert ambient.try_begin_capture("manual") is True  # simulate an in-flight manual call

    called = False

    async def _should_not_run(base_url, timeout_sec):
        nonlocal called
        called = True

    with patch.object(ambient, "call_capture_and_assess", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called"))):
        await ambient.run_ambient_tick("http://circe:32799", 1.0)

    assert not called
    # tick_in_progress must still be True -- it's the manual caller's to
    # release, not something the failed ambient attempt should touch.
    assert ambient.state.tick_in_progress is True
    assert ambient.state.last_trigger == "manual"


def test_call_capture_and_assess_returns_error_shape_on_non_dict_response():
    """Review finding, 2026-08-22: this used to degrade to a bare {} on a
    malformed (non-dict) JSON response, silently dropping the error signal
    for both callers (the manual route saw neither ok nor error; the
    ambient loop's status line showed a generic "failed (unknown)")."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = ["not", "a", "dict"]
    with patch.object(ambient.requests, "post", return_value=fake_resp):
        body = ambient.call_capture_and_assess("http://circe:32799", 240.0, "manual")

    assert body == {"result": {"ok": False, "error": "invalid_response"}}
    ok, error = ambient.result_ok_and_error(body)
    assert ok is False
    assert error == "invalid_response"
