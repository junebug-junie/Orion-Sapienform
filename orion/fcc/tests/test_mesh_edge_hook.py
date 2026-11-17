"""Issue #3409: the FCC PreToolUse hook refuses a GitHub write quoting a guarded sentence."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from orion.fcc import mesh_edge_hook as hook
from orion.memory.mesh_edge import MeshEdgeGuard, MeshEdgeUnavailable

SENTENCE = "Robin told me the blue kite got stuck in the old oak tree."
GUARD = MeshEdgeGuard.from_texts([("page:juniper", SENTENCE)])


def _bash(command: str, cwd: str | None = None) -> dict:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}}
    if cwd:
        payload["cwd"] = cwd
    return payload


@pytest.mark.parametrize("command", [
    "gh issue create --title t --body 'x'",
    "gh pr comment 12 --body-file notes.md",
    "gh pr edit 3 -b hi",
    "cd repo && gh pr create --fill",
    "gh api -X PATCH repos/o/r/pulls/1 -F body=@pr.md",
    "gh api repos/o/r/issues/1/comments -f body=hello",
    "gh release create v1 --notes x",
    "curl -s -X POST https://api.github.com/repos/o/r/issues -d '{}'",
    # review regressions: line continuations, absolute path, curl --json, unknown write verbs
    "gh api repos/o/r/issues \\\n  -f title=x -f body=y",
    "gh api \\\n  --method POST repos/o/r/issues --input f.json",
    "/usr/bin/gh issue comment 1 --body hi",
    "curl https://api.github.com/repos/o/r/issues --json @f.json",
    "curl -d@/tmp/x.json https://api.github.com/repos/o/r/issues",
    "gh label create bug",
    "gh repo edit --description x",
    "gh project item-create 1 --title x",
    "gh release upload v1 notes.md",
])
def test_github_writes_are_recognized(command):
    assert hook.is_github_write(command)


@pytest.mark.parametrize("command", [
    "gh pr view 12 --json body",
    "gh api repos/o/r/pulls/1",
    "gh issue list --search kite",
    "git log --oneline",
    "curl -s https://api.github.com/repos/o/r",
])
def test_reads_are_not_github_writes(command):
    assert not hook.is_github_write(command)


@pytest.mark.parametrize("command,expected", [
    ("curl -d@/tmp/x.json https://api.github.com/x", ["/tmp/x.json"]),
    ("curl --json @b.json https://api.github.com/x", ["b.json"]),
    ("gh pr comment 12 --body-file - < /tmp/b.md", ["/tmp/b.md"]),
    ("gh api -X PATCH r -F body=@pr.md", ["pr.md"]),
])
def test_referenced_files(command, expected):
    assert hook.referenced_files(command) == expected


@pytest.mark.parametrize("command", [
    "cat /tmp/b.md | gh pr comment 12 --body-file -",
    'gh pr comment 12 --body "$(cat notes.md)"',
    'gh pr comment 12 --body "$BODY"',
    "gh issue create --title t --body `cat x`",
])
def test_bodies_built_at_run_time_are_held(command):
    msg = hook.decide(_bash(command), guard_loader=lambda: GUARD)
    assert msg and "--body-file" in msg


def test_shell_quoted_apostrophes_still_match():
    guard = MeshEdgeGuard.from_texts([("episode_memory:x", "Robin doesn't like loud noises in the morning at all.")])
    cmd = "gh pr comment 3 --body 'Robin doesn'\\''t like loud noises in the morning at all.'"
    assert hook.decide(_bash(cmd), guard_loader=lambda: guard)


def test_redirected_stdin_body_is_read(tmp_path: Path):
    (tmp_path / "b.md").write_text(SENTENCE, encoding="utf-8")
    cmd = f"gh pr comment 12 --body-file - < {tmp_path / 'b.md'}"
    assert hook.decide(_bash(cmd), guard_loader=lambda: GUARD)


def test_a_hook_error_holds_a_write_but_not_other_calls(monkeypatch, tmp_path: Path):
    def boom(*_a, **_k):
        raise RuntimeError("bug")

    monkeypatch.setattr(hook, "decide", boom)
    import io
    for command, code in (("gh issue create --title t --body hi", 2), ("ls -la", 0)):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_bash(command))))
        assert hook.main() == code


def test_inline_body_quoting_a_page_sentence_is_blocked():
    msg = hook.decide(_bash(f"gh issue create --title t --body \"Context: {SENTENCE}\""), guard_loader=lambda: GUARD)
    assert msg and msg.startswith(hook.BLOCK_MARKER) and "(page)" in msg
    assert "kite" not in msg  # the refusal never echoes the private sentence


def test_body_file_quoting_a_page_sentence_is_blocked(tmp_path: Path):
    (tmp_path / "pr.md").write_text(f"## Summary\n\n{SENTENCE.upper()}\n", encoding="utf-8")
    cmd = "gh api -X PATCH repos/o/r/pulls/7 -F body=@pr.md"
    assert hook.decide(_bash(cmd, cwd=str(tmp_path)), guard_loader=lambda: GUARD)
    assert hook.decide(_bash("gh pr create --body-file pr.md", cwd=str(tmp_path)), guard_loader=lambda: GUARD)


def test_clean_write_and_any_read_pass():
    assert hook.decide(_bash("gh pr comment 3 --body 'CI is green now.'"), guard_loader=lambda: GUARD) is None
    assert hook.decide(_bash(f"echo '{SENTENCE}'"), guard_loader=lambda: GUARD) is None
    assert hook.decide({"tool_name": "Read", "tool_input": {"file_path": "/x"}}, guard_loader=lambda: GUARD) is None


def test_write_is_blocked_when_the_guard_cannot_load():
    def down():
        raise MeshEdgeUnavailable("no database DSN for the mesh-edge check")

    msg = hook.decide(_bash("gh pr comment 3 --body 'CI is green now.'"), guard_loader=down)
    assert msg and "could not be checked" in msg


def test_script_exits_2_for_a_write_without_a_dsn(tmp_path: Path):
    """End to end, the way claude runs it: by file path, payload on stdin, empty env (no DSN)."""
    payload = json.dumps(_bash("gh issue create --title t --body hi"))
    proc = subprocess.run([sys.executable, str(Path(hook.__file__))], input=payload, text=True,
                          capture_output=True, env={"PATH": "/usr/bin:/bin"}, cwd=str(tmp_path))
    assert proc.returncode == 2 and hook.BLOCK_MARKER in proc.stderr
    read = subprocess.run([sys.executable, str(Path(hook.__file__))], input=json.dumps(_bash("gh pr view 1")),
                          text=True, capture_output=True, env={"PATH": "/usr/bin:/bin"}, cwd=str(tmp_path))
    assert read.returncode == 0


def test_fcc_turn_settings_always_install_the_hook(monkeypatch):
    from orion.harness import fcc_motor

    monkeypatch.setattr(fcc_motor, "repeat_failure_threshold", lambda: 0)
    argv = fcc_motor.repeat_failure_breaker_argv(correlation_id="c")
    hooks = json.loads(argv[1])["hooks"]["PreToolUse"]
    assert argv[0] == "--settings" and any("mesh_edge_hook.py" in h["hooks"][0]["command"] for h in hooks)
    monkeypatch.setattr(fcc_motor, "repeat_failure_threshold", lambda: 3)
    hooks = json.loads(fcc_motor.repeat_failure_breaker_argv(correlation_id="c")[1])["hooks"]["PreToolUse"]
    assert len(hooks) == 2
