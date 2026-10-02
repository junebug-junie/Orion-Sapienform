"""Repeat-failing-call breaker: replay of urgent run a153451fe423 + unit rules."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from orion.fcc import repeat_failure_breaker as rfb

FIXTURE = Path(__file__).parent / "fixtures" / "fcc_repeat_failure_a153451fe423.jsonl"
SENSORS = {"url": "http://host.docker.internal:8080/api/cabinet/sensors/latest"}
SCRAPE = "mcp__firecrawl__firecrawl_scrape"


def _entries():
    return list(rfb.read_transcript(FIXTURE))


def _replay(entries, threshold=3):
    """Re-run the turn: before each tool_use, decide from the transcript prefix.

    Returns [(tool_name, input, blocked)] in call order. Counterfactual past the
    first block (the real model never saw it), which is fine for counting.
    """
    calls = []
    for i, entry in enumerate(entries):
        for block in (entry.get("message") or {}).get("content") or []:
            if block.get("type") == "tool_use":
                history = rfb.CallHistory().feed_all(entries[:i])
                verdict = rfb.decide(history, block["name"], block["input"], threshold=threshold)
                calls.append((block["name"], block["input"], verdict is not None))
    return calls


def test_fixture_is_the_real_incident():
    sensors = [c for c in _replay(_entries()) if c[0] == SCRAPE and c[1] == SENSORS]
    assert len(sensors) == 19  # the brief's "~20": 19 identical scrape calls in one turn


def test_replay_cuts_nineteen_identical_failures_to_three():
    sensors = [c for c in _replay(_entries()) if c[0] == SCRAPE and c[1] == SENSORS]
    allowed = [c for c in sensors if not c[2]]
    assert len(allowed) == 3
    assert all(c[2] for c in sensors[3:])  # every call after the third failure is blocked
    assert sum(c[2] for c in sensors) == 16


def test_replay_does_not_block_calls_below_threshold_or_succeeded():
    calls = _replay(_entries())
    blocked_other = {json.dumps(c[1], sort_keys=True) for c in calls if c[2] and c[1] != SENSORS}
    # biometrics?node=circe failed only twice; psql/Bash calls succeeded -> never blocked.
    assert blocked_other == set()


def test_block_message_names_count_and_real_error():
    entries = _entries()
    history = rfb.CallHistory().feed_all(entries)
    count, err = rfb.decide(history, SCRAPE, SENSORS, threshold=3)
    assert count >= 3
    assert "DNS resolution failed" in err
    msg = rfb.block_message(SCRAPE, count, err)
    assert rfb.BLOCK_MARKER in msg and f"failed {count} times" in msg
    assert "DNS resolution failed" in msg and "Change approach" in msg


def _use(i, name, inp):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}]}}


def _res(i, err, text="x"):
    block = {"type": "tool_result", "tool_use_id": f"t{i}", "content": text}
    if err:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}}


def _hist(*pairs):
    entries = []
    for i, (name, inp, err) in enumerate(pairs):
        entries += [_use(i, name, inp), _res(i, err, "boom" if err else "ok")]
    return rfb.CallHistory().feed_all(entries)


def test_previously_succeeded_call_is_never_blocked():
    cmd = {"command": "curl -s http://x"}
    h = _hist(("Bash", cmd, False), ("Bash", cmd, True), ("Bash", cmd, True), ("Bash", cmd, True))
    assert rfb.decide(h, "Bash", cmd, threshold=3) is None


def test_bash_blocked_after_three_identical_failures():
    cmd = {"command": "false"}
    h = _hist(("Bash", cmd, True), ("Bash", cmd, True))
    assert rfb.decide(h, "Bash", cmd, threshold=3) is None
    h = _hist(("Bash", cmd, True), ("Bash", cmd, True), ("Bash", cmd, True))
    assert rfb.decide(h, "Bash", cmd, threshold=3) == (3, "boom")


def test_different_input_or_tool_not_blocked():
    cmd = {"command": "false"}
    h = _hist(("Bash", cmd, True), ("Bash", cmd, True), ("Bash", cmd, True))
    assert rfb.decide(h, "Bash", {"command": "true"}, threshold=3) is None
    assert rfb.decide(h, "mcp__x__y", cmd, threshold=3) is None


def test_normalization_key_order_and_whitespace():
    a = {"url": "http://a", "formats": ["markdown"]}
    h = _hist(("T", a, True), ("T", a, True), ("T", a, True))
    assert rfb.decide(h, "T", {"formats": ["markdown"], "url": "  http://a "}, threshold=3) is not None


def test_bash_description_label_does_not_split_identical_calls():
    """Live smoke 2026-10-02: the model numbered each attempt in `description`."""
    h = _hist(*[("Bash", {"command": "cat /nope", "description": f"Read file (attempt {i})"}, True) for i in (1, 2, 3)])
    assert rfb.decide(h, "Bash", {"command": "cat /nope", "description": "attempt 4"}, threshold=3) == (3, "boom")
    # ...but only for Bash: an MCP tool's `description` can be real content.
    h = _hist(*[("mcp__x__y", {"description": f"d{i}"}, True) for i in (1, 2, 3)])
    assert rfb.decide(h, "mcp__x__y", {"description": "d4"}, threshold=3) is None


def test_successful_edit_resets_failure_counts():
    """Edit -> re-run failing tests is a fair retry, not a stuck loop."""
    cmd = {"command": "pytest tests/x.py"}
    pairs = []
    for i in range(3):
        pairs += [("Bash", cmd, True), ("Edit", {"file_path": "/a.py", "old_string": str(i), "new_string": "y"}, False)]
    assert rfb.decide(_hist(*pairs), "Bash", cmd, threshold=3) is None
    # A failed edit does not reset.
    pairs = []
    for i in range(3):
        pairs += [("Bash", cmd, True), ("Edit", {"file_path": "/a.py", "old_string": str(i)}, True)]
    assert rfb.decide(_hist(*pairs), "Bash", cmd, threshold=3) is not None
    # Nor does an unrelated successful read.
    pairs = []
    for i in range(3):
        pairs += [("Bash", cmd, True), ("Read", {"file_path": f"/{i}"}, False)]
    assert rfb.decide(_hist(*pairs), "Bash", cmd, threshold=3) is not None


def test_threshold_zero_disables():
    cmd = {"command": "false"}
    h = _hist(*[("Bash", cmd, True)] * 5)
    assert rfb.decide(h, "Bash", cmd, threshold=0) is None


def test_sidechain_entries_ignored():
    cmd = {"command": "false"}
    entries = []
    for i in range(3):
        u, r = _use(i, "Bash", cmd), _res(i, True)
        u["isSidechain"] = r["isSidechain"] = True
        entries += [u, r]
    assert rfb.decide(rfb.CallHistory().feed_all(entries), "Bash", cmd, threshold=3) is None


def _run_hook(tmp_path, transcript_entries, tool_name, tool_input, threshold=3):
    t = tmp_path / "session.jsonl"
    t.write_text("".join(json.dumps(e) + "\n" for e in transcript_entries))
    payload = {"hook_event_name": "PreToolUse", "transcript_path": str(t), "tool_name": tool_name, "tool_input": tool_input}
    return subprocess.run(
        [sys.executable, rfb.__file__, "--threshold", str(threshold)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30,
    )


def test_per_session_isolation_via_transcript(tmp_path):
    """Two turns = two transcripts; failures in one never block the other."""
    cmd = {"command": "false"}
    failing = []
    for i in range(3):
        failing += [_use(i, "Bash", cmd), _res(i, True)]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert _run_hook(tmp_path / "a", failing, "Bash", cmd).returncode == 2
    assert _run_hook(tmp_path / "b", [], "Bash", cmd).returncode == 0


def test_hook_process_blocks_with_exit_2_and_stderr(tmp_path):
    proc = _run_hook(tmp_path, _entries(), SCRAPE, SENSORS)
    assert proc.returncode == 2
    assert rfb.BLOCK_MARKER in proc.stderr and "DNS resolution failed" in proc.stderr


def test_hook_fails_open_on_missing_transcript_or_garbage(tmp_path):
    payload = {"transcript_path": str(tmp_path / "nope.jsonl"), "tool_name": "Bash", "tool_input": {}}
    proc = subprocess.run([sys.executable, rfb.__file__], input=json.dumps(payload), capture_output=True, text=True)
    assert proc.returncode == 0
    proc = subprocess.run([sys.executable, rfb.__file__], input="not json", capture_output=True, text=True)
    assert proc.returncode == 0
    # Bad argv must not turn into argparse's exit 2 (= block every call).
    proc = subprocess.run([sys.executable, rfb.__file__, "--threshold", "x"], input=json.dumps(payload), capture_output=True, text=True)
    assert proc.returncode == 0


def test_settings_json_installs_catch_all_pretooluse_hook():
    data = json.loads(rfb.hook_settings_json(threshold=4, python_bin="/usr/local/bin/python3"))
    (entry,) = data["hooks"]["PreToolUse"]
    assert entry["matcher"] == "*"
    cmd = entry["hooks"][0]["command"]
    assert cmd.startswith("/usr/local/bin/python3 ") and "repeat_failure_breaker.py" in cmd
    assert cmd.endswith("--threshold 4")
