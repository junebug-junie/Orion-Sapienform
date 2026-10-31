"""An answer the CLI's output cap split across two responses reaches the draft whole.

Live 2026-10-31 (reading stage-2 turn c5df42da): the model's JSON answer ran past
CLAUDE_CODE_MAX_OUTPUT_TOKENS (16384, mostly thinking). The CLI sent its fixed
"Output token limit hit. Resume directly ..." user message and the model wrote
the rest in a new response. The CLI's ``result`` held only that last piece, so the
harness draft started mid-string and ``parse_json_object`` failed.

The stream shape below is the one the real CLI (2.1.316) printed against a fake
Messages server that cut the first response with ``stop_reason: max_tokens``:
``stop_reason`` is null on both assistant events; the resume message is a plain
``user`` event; ``result`` is the tail alone.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from orion.core.llm_json import parse_json_object
from orion.fcc.output_cap_continuation import OutputCapStitcher, is_output_cap_resume_event
from orion.harness import fcc_motor as motor
from orion.harness.tests.test_fcc_motor_write_stamp import _Proc, _Stream, _patch_motor

RESUME = ("Output token limit hit. Resume directly — no apology, no recap of what you were doing. "
          "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")
HEAD = '```json\n{\n  "summary": "the page states the counts, not the ratio", "weight": 0.'
TAIL = '047,\n  "need_stage1_urls": ["https://example.org/a b"]\n}\n```'


def _assistant(msg_id: str, *blocks: dict) -> dict:
    return {"type": "assistant", "message": {"id": msg_id, "role": "assistant", "stop_reason": None,
                                             "content": list(blocks)}}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _user_text(t: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": t}]}}


def _tool_result(tool_id: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]}}


CUT_STREAM = [
    {"type": "system", "subtype": "init", "session_id": "s1"},
    _assistant("m1", {"type": "thinking", "thinking": "long reasoning"}),
    _assistant("m1", _text(HEAD)),
    _user_text(RESUME),
    _assistant("m2", {"type": "thinking", "thinking": "more"}),
    _assistant("m2", _text(TAIL)),
    {"type": "result", "subtype": "success", "result": TAIL, "session_id": "s1"},
]


def _replay(events: list[dict]) -> tuple[str, OutputCapStitcher]:
    """The motor's own per-event draft update, without the subprocess."""
    st = OutputCapStitcher()
    acc = ""
    for ev in events:
        st.observe(ev)
        text, _sid, _dur = motor.extract_final_from_stream_event(ev, accumulated=acc)
        if text:
            acc = st.draft_for(ev, text)
    return acc, st


def test_resume_marker_detected_only_on_user_events() -> None:
    assert is_output_cap_resume_event(_user_text(RESUME))
    assert not is_output_cap_resume_event(_assistant("x", _text(RESUME)))
    assert not is_output_cap_resume_event(_tool_result("t1"))


def test_cut_answer_is_stitched_and_parses() -> None:
    draft, st = _replay(CUT_STREAM)
    assert st.continuations == 1
    assert draft.startswith("```json")
    parsed = parse_json_object(draft)
    assert parsed["weight"] == 0.047
    assert parsed["need_stage1_urls"] == ["https://example.org/a b"]


def test_cut_mid_word_keeps_the_space_the_tail_starts_with() -> None:
    events = [_assistant("m1", _text("The answer is")), _user_text(RESUME),
              _assistant("m2", _text(" forty-two.")),
              {"type": "result", "result": "forty-two.", "session_id": "s"}]
    draft, _ = _replay(events)
    assert draft == "The answer is forty-two."


def test_two_cuts_stitch_three_pieces() -> None:
    events = [_assistant("m1", _text('{"a": "x')), _user_text(RESUME),
              _assistant("m2", {"type": "thinking", "thinking": "t"}), _user_text(RESUME),
              _assistant("m3", _text('y')), _user_text(RESUME),
              _assistant("m4", _text('z"}')),
              {"type": "result", "result": 'z"}', "session_id": "s"}]
    draft, st = _replay(events)
    assert st.continuations == 3
    assert parse_json_object(draft) == {"a": "xyz"}


def test_tool_call_between_ends_the_continuation() -> None:
    """A cut piece followed by tool work was mid-work prose, not the start of the answer."""
    events = [_assistant("m1", _text("Let me check one more")), _user_text(RESUME),
              _assistant("m2", {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}),
              _tool_result("t1"),
              _assistant("m3", _text('{"ok": true}')),
              {"type": "result", "result": '{"ok": true}', "session_id": "s"}]
    draft, _ = _replay(events)
    assert draft == '{"ok": true}'


def test_uncut_turn_is_unchanged() -> None:
    events = [_assistant("m1", _text('{"ok": true}')),
              {"type": "result", "result": '{"ok": true}', "session_id": "s"}]
    draft, st = _replay(events)
    assert draft == '{"ok": true}'
    assert st.continuations == 0


async def _drain(**kwargs: Any) -> list:
    events = []
    async for ev in motor.run_fcc_turn(
        prompt="hi", correlation_id="corr-cap", workspace="/tmp",
        fcc_server_url="http://127.0.0.1:8082", auth_token="tok", claude_bin="claude",
        timeout_sec=5.0, **kwargs,
    ):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_motor_final_frame_carries_the_whole_answer(monkeypatch) -> None:
    proc = _Proc(_Stream([json.dumps(e) for e in CUT_STREAM]))
    _patch_motor(monkeypatch, proc, None)
    events = await _drain()
    final = events[-1]
    assert final["type"] == "final"
    # Before the fix this was TAIL alone ("047, ..."), which no JSON parser can read.
    assert final["llm_response"].startswith("```json")
    assert parse_json_object(final["llm_response"])["weight"] == 0.047
    assert final["metadata"]["fcc_output_cap_continuations"] == 1


def test_turn_ending_on_a_thinking_only_response_keeps_one_copy_of_the_head() -> None:
    """Killed after the resume, mid-thinking: the draft is the head once, never doubled."""
    events = [_assistant("m1", _text('{"a": "x')), _user_text(RESUME),
              _assistant("m2", {"type": "thinking", "thinking": "t"})]
    draft, _ = _replay(events)
    assert draft == '{"a": "x'


def test_rewrite_from_the_top_replaces_the_head() -> None:
    """Live: in 9 of 23 cut JSON answers the model ignored "resume directly" and began again.
    Stitching the old head onto a fresh full answer would break an answer that already parses."""
    full = '```json\n{\n  "summary": "Stage 2 read of the story", "n": 2\n}\n```'
    events = [_assistant("m1", _text('```json\n{\n  "summary": "Stage 2 read of the sto')), _user_text(RESUME),
              _assistant("m2", _text(full)),
              {"type": "result", "result": full, "session_id": "s"}]
    draft, _ = _replay(events)
    assert draft == full
    assert parse_json_object(draft)["n"] == 2
