"""Unit tests for the pure helpers in scripts/context_mode_hooks_smoke.py.

These never spawn claude — they only exercise stream-json parsing, dir
snapshot diffing, and status/exit-code aggregation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.context_mode_hooks_smoke import (
    FAIL,
    PASS,
    PLUGIN_TOOL_PREFIX,
    UNVERIFIED,
    CheckResult,
    dir_diff,
    exit_code_for,
    extract_final_text,
    extract_session_id,
    extract_tool_use_names,
    has_compact_event,
    is_truthy,
    parse_stream_lines,
    snapshot_dir,
    worst_status,
)

_PLUGIN_TOOL = "mcp__plugin_context-mode_context-mode__ctx_stats"

_ASSISTANT_EVENT = {
    "type": "assistant",
    "message": {
        "content": [
            {"type": "text", "text": "Calling the stats tool now."},
            {"type": "tool_use", "name": _PLUGIN_TOOL, "input": {}},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/tmp/x"}},
        ]
    },
    "session_id": "sess-assistant",
}

_RESULT_EVENT = {
    "type": "result",
    "result": "DONE",
    "session_id": "sess-final-123",
    "duration_ms": 4200,
}


def _stream_lines() -> list[str]:
    return [
        json.dumps({"type": "system", "subtype": "init"}),
        "",
        "not-json garbage line",
        json.dumps(_ASSISTANT_EVENT),
        json.dumps(["a", "bare", "list"]),
        json.dumps(_RESULT_EVENT),
    ]


def test_parse_stream_lines_skips_blank_nonjson_and_nondict() -> None:
    events = parse_stream_lines(_stream_lines())
    assert [e["type"] for e in events] == ["system", "assistant", "result"]


def test_extract_tool_use_names_includes_plugin_tool() -> None:
    events = parse_stream_lines(_stream_lines())
    names = extract_tool_use_names(events)
    assert names == [_PLUGIN_TOOL, "Read"]
    assert names[0].startswith(PLUGIN_TOOL_PREFIX)


def test_extract_session_id_prefers_result_event() -> None:
    events = parse_stream_lines(_stream_lines())
    assert extract_session_id(events) == "sess-final-123"


def test_extract_session_id_falls_back_to_assistant_event() -> None:
    assert extract_session_id([_ASSISTANT_EVENT]) == "sess-assistant"
    assert extract_session_id([{"type": "system"}]) is None


def test_extract_final_text_prefers_result_string() -> None:
    events = parse_stream_lines(_stream_lines())
    assert extract_final_text(events) == "DONE"


def test_extract_final_text_falls_back_to_assistant_text() -> None:
    assert extract_final_text([_ASSISTANT_EVENT]) == "Calling the stats tool now."
    assert extract_final_text([]) == ""


def test_has_compact_event_matches_type_and_subtype() -> None:
    assert has_compact_event([{"type": "system", "subtype": "compact_boundary"}])
    assert has_compact_event([{"type": "compact", "trigger": "auto"}])
    assert not has_compact_event(parse_stream_lines(_stream_lines()))


def test_snapshot_dir_missing_root_is_empty(tmp_path: Path) -> None:
    assert snapshot_dir(tmp_path / "does-not-exist") == {}


def test_dir_diff_detects_added_file(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "old.json").write_text("{}", encoding="utf-8")
    before = snapshot_dir(sessions)
    (sessions / "snap-1.json").write_text("{}", encoding="utf-8")
    diff = dir_diff(before, snapshot_dir(sessions))
    assert diff["added"] == ["snap-1.json"]
    assert diff["modified"] == []


def test_dir_diff_detects_modified_file(tmp_path: Path) -> None:
    target = tmp_path / "sessions" / "snap.json"
    target.parent.mkdir()
    target.write_text("{}", encoding="utf-8")
    before = snapshot_dir(target.parent)
    mtime = target.stat().st_mtime
    os.utime(target, (mtime + 5, mtime + 5))
    diff = dir_diff(before, snapshot_dir(target.parent))
    assert diff["added"] == []
    assert diff["modified"] == ["snap.json"]


def test_worst_status_fail_beats_unverified_beats_pass() -> None:
    assert worst_status([]) == PASS
    assert worst_status([PASS, PASS]) == PASS
    assert worst_status([PASS, UNVERIFIED]) == UNVERIFIED
    assert worst_status([UNVERIFIED, FAIL, PASS]) == FAIL


def test_exit_code_zero_only_without_fail() -> None:
    assert exit_code_for([PASS, UNVERIFIED, PASS]) == 0
    assert exit_code_for([PASS, FAIL, UNVERIFIED]) == 1
    assert exit_code_for([]) == 0


def test_check_result_json_line_round_trips() -> None:
    line = CheckResult("plugin_installed", PASS, {"matches": ["context-mode"]}).to_json_line()
    assert json.loads(line) == {
        "check": "plugin_installed",
        "status": "PASS",
        "evidence": {"matches": ["context-mode"]},
    }


def test_is_truthy_matches_harness_convention() -> None:
    assert is_truthy("1") and is_truthy("true") and is_truthy("YES") and is_truthy(" on ")
    assert not is_truthy("") and not is_truthy(None) and not is_truthy("false")
