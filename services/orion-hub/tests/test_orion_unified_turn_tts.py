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

from pathlib import Path

import pytest

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


# --------------------------------------------- real source, control-flow shape

def _source() -> str:
    return WS_PATH.read_text(encoding="utf-8")


def test_orion_lane_calls_run_tts_remote():
    """The actual regression: before this fix, `run_tts_remote` appeared
    exactly once in this file (the classic lane's own "4. TTS" block) and
    the orion branch's own `continue` could never reach it. Now it must be
    reachable from the orion branch too."""
    source = _source()
    assert source.count("run_tts_remote(") >= 2, (
        "run_tts_remote should now be called from both the classic lane "
        "and the orion lane -- found only the pre-existing classic-lane call"
    )


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
    following = source[tts_idx : tts_idx + 2000]
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


def test_orion_tts_gate_mirrors_the_classic_lanes_conditions():
    """disable_tts and tts_client must gate the orion lane exactly as they
    already gate the classic lane -- same user-facing toggle, same
    dependency, so the two lanes behave identically from the browser's
    point of view."""
    source = _source()
    idx = source.index("orion_will_tts = bool(")
    following = source[idx : idx + 200]
    assert "disable_tts" in following
    assert "tts_client" in following


def test_orion_lane_reuses_the_single_shared_tts_queue():
    """tts_q is defined once per connection (above the message loop) and
    the classic lane's own drain_task already relays whatever lands in it
    back over the websocket -- the orion lane must reuse that SAME queue,
    not create a second one, or audio would have nothing draining it."""
    source = _source()
    idx = source.index("extract_unified_turn_final_text(orion_turn_frames)")
    following = source[idx : idx + 1500]
    assert "run_tts_remote(orion_final_text, tts_client, tts_q)" in following
