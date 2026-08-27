"""2026-08-26/27: the Orion (unified-turn) chat lane speaks its replies.

Real incident (corr=7dc1bab2-97a4-4390-89a2-cdd1fa4f0092): a spoken turn got
a real, good reply, and no voice came back. Root cause was structural, not a
regression -- the orion-mode branch's own `continue` (immediately after
`run_unified_turn`'s `finally:`) always exited the message loop before ever
reaching the classic lane's "4. TTS" block, which lives ~40 lines further
down in the same function and is the ONLY place `run_tts_remote` was ever
called from. Voice INPUT (STT) always worked for this lane; voice OUTPUT
never did, regardless of `disable_tts` or anything else -- there was no code
path to it at all.

`extract_unified_turn_final_text` is the one genuinely non-trivial piece of
the fix (which real frame carries the assistant's actual words) and is
tested directly, against real frame shapes `_success_frames`/
`_harness_error_frame`/`_thought_deferred_frame` (turn_orchestrator.py)
produce. The rest of the wiring (the `will_tts` gate, the `client_state`
skip, the log line) mirrors the classic lane's own established, unit-tested-
only-by-shape convention -- this repo has no full WebSocket TestClient
harness for `websocket_handler.py` (see
test_websocket_agent_claude_routing.py's own docstring for why), so those
pieces are covered the same way that file already covers the import guard:
asserting the real source's control-flow shape, not a reimplementation of
it.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.websocket_handler as ws_handler
from scripts.websocket_handler import extract_unified_turn_final_text

HUB_ROOT = Path(__file__).resolve().parents[1]
WS_PATH = HUB_ROOT / "scripts" / "websocket_handler.py"


# ------------------------------------------------- extract_unified_turn_final_text

def test_extracts_llm_response_from_the_final_frame():
    frames = [
        {"type": "substrate_appraisal", "correlation_id": "c1", "appraisal": {}},
        {"type": "final", "correlation_id": "c1", "llm_response": "hello there"},
    ]
    assert extract_unified_turn_final_text(frames) == "hello there"


def test_scans_for_final_rather_than_trusting_frame_order():
    """_success_frames can prepend substrate_appraisal/reflection frames
    before the final one -- frames[-1] would be wrong if that ordering ever
    changes, or if a caller appends anything after the final frame."""
    frames = [
        {"type": "final", "llm_response": "the real answer"},
        {"type": "some_future_trailer_frame", "note": "not the answer"},
    ]
    assert extract_unified_turn_final_text(frames) == "the real answer"


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [{"type": "turn_deferred", "correlation_id": "c1", "reason": "defer"}],
        [{"type": "turn_error", "correlation_id": "c1", "error": "boom"}],
        [{"type": "turn_degraded", "correlation_id": "c1", "reason": "substrate_unavailable"}],
    ],
)
def test_no_final_frame_yields_none(frames):
    """turn_deferred/turn_error/turn_degraded-only, or an empty list (e.g. a
    cancelled turn) -- nothing to speak, and callers must not crash on it."""
    assert extract_unified_turn_final_text(frames) is None


def test_a_turn_error_frames_partial_draft_is_not_spoken():
    """Deliberate: a partial_draft on a turn_error frame is real
    assistant-authored text (the browser does render it as a bubble), but
    speaking an error-path partial aloud is a different, untested product
    decision -- not folded into this fix."""
    frames = [
        {
            "type": "turn_error",
            "correlation_id": "c1",
            "error": "harness_rpc_timeout",
            "partial_draft": "I was in the middle of saying",
        }
    ]
    assert extract_unified_turn_final_text(frames) is None


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_present_but_empty_final_text_yields_none(blank):
    """A final frame with no real text (context-overflow edge case, or a
    genuinely empty finalize output) must not be treated as speakable --
    matches the classic lane's own `orion_response_text and ...` truthiness
    gate rather than a bare `is not None` check."""
    frames = [{"type": "final", "llm_response": blank}]
    assert extract_unified_turn_final_text(frames) is None


def test_a_non_string_llm_response_does_not_crash():
    """Defensive: llm_response should always be a string in practice, but a
    malformed/None value must degrade to "nothing to speak", not raise."""
    frames = [{"type": "final", "llm_response": None}]
    assert extract_unified_turn_final_text(frames) is None


# ------------------------------------------------------- dispatch_tts_reply

# Real behavioral tests, not string-matching -- review finding, 2026-08-27:
# an earlier version of this test file asserted
# `source.count("run_tts_remote(") >= 2`, which was ALREADY true on
# `origin/main` before this fix (2: the `async def run_tts_remote(`
# definition plus the pre-existing classic-lane call), confirmed live via
# `git show origin/main:... | grep -c run_tts_remote`. A future revert of
# the orion-lane call would leave the count at 2, still satisfying `>= 2`,
# and this test would keep passing while the exact bug it exists to catch
# (corr=7dc1bab2-97a4-4390-89a2-cdd1fa4f0092) came back. Testing the real,
# shared `dispatch_tts_reply` function directly -- which BOTH lanes now
# call -- closes that gap for real: it exercises the actual gate and
# dispatch logic, not a proxy count of a substring.

def _fake_tts_client(ok: bool = True):
    client = MagicMock()
    if ok:
        client.speak = AsyncMock()
    return client


def test_dispatch_fires_when_all_conditions_are_met():
    async def _run():
        client = _fake_tts_client()
        q = asyncio.Queue()
        dispatched = ws_handler.dispatch_tts_reply(
            text="hello",
            disable_tts=False,
            tts_client=client,
            tts_q=q,
            correlation_id="c1",
            session_id="s1",
            lane="orion",
        )
        assert dispatched is True
        await asyncio.sleep(0)  # let the fire-and-forget task actually run
        client.speak.assert_awaited_once()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "text,disable_tts,has_client",
    [
        (None, False, True),
        ("", False, True),
        ("   ", False, True),
        ("hello", True, True),
        ("hello", False, False),
    ],
)
def test_dispatch_does_not_fire_when_any_condition_fails(text, disable_tts, has_client):
    client = _fake_tts_client() if has_client else None
    q = asyncio.Queue()
    dispatched = ws_handler.dispatch_tts_reply(
        text=text,
        disable_tts=disable_tts,
        tts_client=client,
        tts_q=q,
        correlation_id="c1",
        session_id="s1",
        lane="orion",
    )
    assert dispatched is False


def test_extra_gate_can_suppress_dispatch_even_with_everything_else_ok():
    """This is how the classic lane's own `workflow_metadata_only` exclusion
    folds in without dispatch_tts_reply needing to know what that concept
    is -- the shared function only needs a bool."""
    client = _fake_tts_client()
    q = asyncio.Queue()
    dispatched = ws_handler.dispatch_tts_reply(
        text="hello",
        disable_tts=False,
        tts_client=client,
        tts_q=q,
        correlation_id="c1",
        session_id="s1",
        lane="classic",
        extra_gate=False,
    )
    assert dispatched is False


def test_the_orion_lane_actually_calls_the_shared_dispatch_function():
    """The real regression-proof, replacing the substring-count check: the
    orion branch must call the SAME `dispatch_tts_reply` the classic lane
    calls, via source verification of an unambiguous call site, not a raw
    count of an unrelated substring that was already >= 2 before the fix."""
    source = WS_PATH.read_text(encoding="utf-8")
    orion_call = "extract_unified_turn_final_text(orion_turn_frames)"
    assert orion_call in source
    idx = source.index(orion_call)
    following = source[idx : idx + 500]
    assert "dispatch_tts_reply(" in following
    assert 'lane="orion"' in following


def test_dispatched_task_is_strongly_referenced_then_released():
    """Review finding, 2026-08-27: asyncio holds only a WEAK reference to a
    running task, and both dispatch_tts_reply call sites discard
    create_task's return value -- without an explicit strong ref, a
    synthesis task can be garbage-collected mid-flight and simply vanish
    with nothing surfaced. Same class of bug, same fix shape, as
    services/orion-whisper-tts/app/cuda_watchdog.py's own _INFLIGHT set
    (PR #1901, same day)."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_speak(*a, **k):
        started.set()
        await release.wait()

    async def _run():
        client = MagicMock()
        client.speak = _slow_speak
        q = asyncio.Queue()
        dispatched = ws_handler.dispatch_tts_reply(
            text="hello",
            disable_tts=False,
            tts_client=client,
            tts_q=q,
            correlation_id="c1",
            session_id="s1",
            lane="orion",
        )
        assert dispatched is True
        await started.wait()
        assert len(ws_handler._TTS_DISPATCH_INFLIGHT) == 1, (
            "task not strongly referenced while its synthesis is in flight"
        )
        release.set()
        # Let the task actually finish and its done-callback fire.
        for _ in range(50):
            if not ws_handler._TTS_DISPATCH_INFLIGHT:
                break
            await asyncio.sleep(0.01)
        assert ws_handler._TTS_DISPATCH_INFLIGHT == set(), (
            "done-callback did not release the ref after completion"
        )

    asyncio.run(_run())


# --------------------------------------------- real source, control-flow shape

def _source() -> str:
    return WS_PATH.read_text(encoding="utf-8")


def test_orion_tts_trigger_sits_between_the_affect_post_leg_and_continue():
    """Placement matters: this must run AFTER run_unified_turn has actually
    returned (so real frames exist to read from) and BEFORE the branch's
    `continue`, which is the only way the orion branch's own iteration of
    the message loop ends."""
    source = _source()
    affect_post_marker = 'trigger=chat_turn_affect.TRIGGER_POST,'
    assert affect_post_marker in source
    affect_idx = source.index(affect_post_marker)

    tts_marker = "extract_unified_turn_final_text(orion_turn_frames)"
    assert tts_marker in source
    tts_idx = source.index(tts_marker)
    assert tts_idx > affect_idx, "TTS trigger must come after the affect post-fire"

    # The next `continue` after the TTS trigger is the branch's own exit --
    # confirms the trigger is inside the orion branch, not accidentally
    # placed somewhere it would never run (e.g. after the classic lane's
    # own logic starts).
    following = source[tts_idx : tts_idx + 800]
    assert "\n                continue" in following


def test_orion_tts_trigger_skips_a_disconnected_client():
    """Synthesizing speech for a socket that already closed is pure waste
    -- must reuse the same client_state check the affect post-leg already
    established, not fire unconditionally."""
    source = _source()
    tts_marker = "extract_unified_turn_final_text(orion_turn_frames)"
    idx = source.index(tts_marker)
    preceding = source[max(0, idx - 400) : idx]
    assert "websocket.client_state == WebSocketState.CONNECTED" in preceding


def test_orion_lane_reuses_the_single_shared_tts_queue():
    """tts_q is defined once per connection (above the message loop) and
    the classic lane's own drain_task already relays whatever lands in it
    back over the websocket -- the orion lane must reuse that SAME queue,
    not create a second one, or audio would have nothing draining it."""
    source = _source()
    idx = source.index("extract_unified_turn_final_text(orion_turn_frames)")
    following = source[idx : idx + 400]
    assert "tts_q=tts_q," in following


def test_classic_lane_also_routes_through_the_shared_dispatch_function():
    """The whole point of extracting dispatch_tts_reply: both lanes must
    call it, not just the orion one -- otherwise the duplication (and the
    risk of the two silently diverging again) is not actually fixed."""
    source = _source()
    idx = source.index("# 4. TTS")
    following = source[idx : idx + 400]
    assert "dispatch_tts_reply(" in following
    assert 'lane="classic"' in following
