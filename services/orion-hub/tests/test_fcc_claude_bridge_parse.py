from __future__ import annotations

from pathlib import Path

from scripts.fcc_claude_bridge import (
    build_step_frame,
    extract_final_from_stream_event,
    parse_stream_json_line,
    parse_stream_json_lines,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fcc_claude_stream.jsonl"


def test_parse_stream_json_line_skips_blank_and_raw_fallback() -> None:
    assert parse_stream_json_line("") is None
    assert parse_stream_json_line("{not json") == {"type": "raw", "content": "{not json"}


def test_parse_stream_json_lines_and_final_text() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    events = parse_stream_json_lines(lines)
    assert len(events) == 4
    assert events[0]["type"] == "system"
    final_text, session_id, duration_ms = extract_final_from_stream_event(events[-1], accumulated="ignored")
    assert final_text == "Here are the hub scripts."
    assert session_id == "sess-abc"
    assert duration_ms == 1200


def test_build_step_frame_shape() -> None:
    raw = {"type": "assistant", "message": {"content": []}}
    step = build_step_frame(raw)
    assert step == {"type": "assistant", "raw": raw}
