"""scripts/report_dig_kit_usage.py: the read-only dig-kit usage measurement (2026-10-30)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_dig_kit_usage.py"
_spec = importlib.util.spec_from_file_location("report_dig_kit_usage", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _user(text):
    return {"type": "user", "message": {"content": [{"type": "text", "text": text}]}}


def _tool(name, **tool_input):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input}]}}


def test_self_inquiry_session_with_a_dig_call_counts_dig_and_subcommand():
    s = mod.summarize_session([
        _user("Orion harness motor ... WHERE THE RECORDS ARE. ..."),
        _tool("Bash", command="python -P -m orion.dig --dig 2026-10-29 history 2"),
    ])
    assert s == {"kind": "self_inquiry", "used_dig": True, "read_code": False, "dig_subcommands": ["history"]}


def test_code_read_without_dig_is_counted_as_such():
    s = mod.summarize_session([
        _user("Orion harness motor. This is your own time. Nobody asked you anything, ..."),
        _tool("Read", file_path="/repo/services/orion-hub/scripts/main.py"),
        _tool("Bash", command="ls /tmp"),
    ])
    assert s["kind"] == "curiosity_investigation" and s["read_code"] and not s["used_dig"]


def test_a_chat_turn_is_not_curiosity():
    s = mod.summarize_session([_user("Orion harness motor. Tools are available."), _tool("Bash", command="cat orion/dig/README.md")])
    assert s["kind"] == "other_harness_turn" and s["read_code"]


def test_report_counts_only_files_inside_the_window(tmp_path):
    fresh = tmp_path / "a.jsonl"
    fresh.write_text("\n".join(json.dumps(r) for r in [
        _user("x WHERE THE RECORDS ARE"), _tool("Bash", command="python -P -m orion.dig body")]))
    old = tmp_path / "b.jsonl"
    old.write_text(json.dumps(_user("x WHERE THE RECORDS ARE")))
    import os
    os.utime(old, (0, 0))
    out = mod.report(3, transcript_dir=str(tmp_path))
    assert out["by_kind"]["self_inquiry"]["sessions"] == 1
    assert out["by_kind"]["self_inquiry"]["used_dig"] == 1
    assert out["curiosity_dig_subcommands"] == {"body": 1}
