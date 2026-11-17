"""Session threads, the coding-session side: the two Claude Code hooks and the CLI, run as real
subprocesses against a local stand-in for Hub and a stand-in `gh`.

- post_tool_pr_spec_to_orion.py: fires only on `gh pr create`, sends the PR's spec text, prints Hub's
  receipt; a PR without a spec is ignored; Hub down is silent and exit 0.
- session_pending_orion_questions.py: prints Orion's undelivered questions with how to answer, marks
  them delivered; silent when nothing is pending or Hub is down.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SEND_HOOK = ROOT / "scripts" / "hooks" / "post_tool_pr_spec_to_orion.py"
PENDING_HOOK = ROOT / "scripts" / "hooks" / "session_pending_orion_questions.py"
CLI = ROOT / "scripts" / "orion_session_thread.py"
WAIT_HOOK = ROOT / "scripts" / "hooks" / "wait_for_orion_answer.py"
sys.path.insert(0, str(ROOT / "scripts"))

import orion_session_thread_lib as lib  # noqa: E402

SPEC_PATH = "docs/superpowers/specs/2026-11-15-thing-design.md"
SPEC_TEXT = "# Thing design\n\nWhat it does.\n"


class Hub:
    """A stand-in Hub: records every call, answers the session-threads routes."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.pending: list[dict] = []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def _send(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                hub.calls.append(("GET", self.path, None))
                self._send({"pending": hub.pending})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                hub.calls.append(("POST", self.path, body))
                if self.path == "/api/session-threads/delivered":
                    hub.pending = [p for p in hub.pending if p["turn_id"] not in body["turn_ids"]]
                    self._send({"delivered": len(body["turn_ids"])})
                else:
                    self._send({"thread": {"thread_id": "abc123"}, "receipt": {"speaker": "hub", "text": "held"},
                                "next_roundup_at": "2026-11-15T11:00:00+00:00", "roundup_enabled": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def posts(self, path="/api/session-threads/message"):
        return [c[2] for c in self.calls if c[0] == "POST" and c[1] == path]


@pytest.fixture()
def hub():
    h = Hub()
    yield h
    h.server.shutdown()


@pytest.fixture()
def world(tmp_path):
    """A git repo on branch feat/thing with a spec file, and a `gh` that answers `pr view`."""
    repo = tmp_path / "wt"
    (repo / "docs/superpowers/specs").mkdir(parents=True)
    (repo / SPEC_PATH).write_text(SPEC_TEXT, encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com"}
    for args in (["init", "-q", "-b", "feat/thing"], ["add", "-A"], ["commit", "-q", "-m", "spec", "--no-gpg-sign"]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text("#!/bin/sh\necho \"$@\" >> \"$GH_LOG\"\n[ -f \"$GH_PR_JSON\" ] || exit 1\ncat \"$GH_PR_JSON\"\n")
    gh.chmod(0o755)
    return {"repo": repo, "bindir": bindir, "pr_json": tmp_path / "pr.json", "gh_log": tmp_path / "gh.log",
            "board": tmp_path / "board.jsonl", "log": tmp_path / "hooks.log"}


def _pr(world, files=(SPEC_PATH, "services/orion-hub/scripts/x.py")):
    world["pr_json"].write_text(json.dumps({
        "number": 3456, "url": "https://github.com/o/r/pull/3456", "title": "feat: thing",
        "headRefName": "feat/thing", "files": [{"path": p} for p in files]}))


def _run(script, world, hub_url, payload=None, args=(), extra_env=None):
    env = {**os.environ, "ORION_HUB_URL": hub_url, "ORION_AGENT_BOARD_PATH": str(world["board"]),
           "ORION_SESSION_THREADS_LOG": str(world["log"]), "GH_PR_JSON": str(world["pr_json"]),
           "GH_LOG": str(world["gh_log"]), "PATH": f"{world['bindir']}:{os.environ['PATH']}"}
    env.pop("CLAUDE_CODE_SESSION_ID", None)
    env.pop("ORION_SESSION_THREADS_HOOKS", None)
    env.update(extra_env or {})
    started = time.monotonic()
    proc = subprocess.run([sys.executable, str(script), *args], cwd=world["repo"], env=env,
                          input=json.dumps(payload) if payload is not None else "", capture_output=True, text=True,
                          timeout=40)
    return proc, time.monotonic() - started


def _bash(command, response="https://github.com/o/r/pull/3456\n"):
    return {"session_id": "sess-A", "hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_input": {"command": command}, "tool_response": {"stdout": response, "stderr": ""}}


# --- command matching -----------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "gh pr create --title x --body y",
    "cd /wt && gh pr create --fill",
    "git push -u origin feat/x; gh pr create -t x -b y",
    "GH_TOKEN=abc gh pr create --fill",
    "rtk gh pr create --fill",
    "git push &&\ngh  pr   create --fill",
])
def test_pr_create_commands_match(command):
    assert lib.PR_CREATE_RE.search(command)


@pytest.mark.parametrize("command", [
    "gh pr view 12 --json files", "gh pr edit 12 --body x", "gh pr checks", "gh pr list",
    "echo 'run gh pr create later'", "rg 'gh pr create' scripts/", "git commit -m 'docs: gh pr create hook'",
    "gh issue create --title 'pr create fails'",
])
def test_other_commands_do_not_match(command):
    assert not lib.PR_CREATE_RE.search(command)


# --- the sender hook ------------------------------------------------------------------------------


def test_pr_create_with_a_spec_sends_the_spec_text_and_prints_hubs_receipt(world, hub):
    _pr(world)
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("git push && gh pr create --fill"))
    assert proc.returncode == 0, proc.stderr
    [msg] = hub.posts()
    assert msg["session_id"] == "sess-A" and msg["pr_number"] == 3456 and msg["branch"] == "feat/thing"
    assert msg["pr_url"] == "https://github.com/o/r/pull/3456" and msg["title"] == "feat: thing"
    assert msg["worktree"] == str(world["repo"]) and msg["session_name"] == "wt"
    assert msg["spec"] == [{"path": SPEC_PATH, "text": SPEC_TEXT}]   # the text itself, not a path Orion cannot read
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert out["additionalContext"].startswith("Hub receipt:") and "session thread abc123" in out["additionalContext"]
    assert "Orion has NOT read it yet" in out["additionalContext"] and "2026-11-15 11:00 UTC" in out["additionalContext"]
    assert "pr view 3456" in world["gh_log"].read_text()   # the PR number came from the tool's own output


def test_a_pr_without_a_spec_and_a_command_that_is_not_pr_create_send_nothing(world, hub):
    _pr(world, files=("services/orion-hub/scripts/x.py", "docs/other/notes.md"))
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill"))
    assert proc.returncode == 0 and proc.stdout == "" and hub.calls == []
    _pr(world)
    world["gh_log"].unlink()
    for command in ("gh pr view 3456", "git status", "echo 'gh pr create'"):
        proc, _ = _run(SEND_HOOK, world, hub.url, _bash(command))
        assert proc.returncode == 0 and proc.stdout == ""
    assert hub.calls == [] and not world["gh_log"].exists()   # gh is not even called


def test_the_sender_hook_is_silent_when_hub_is_down_or_switched_off_or_gh_fails(world, hub):
    _pr(world)
    proc, took = _run(SEND_HOOK, world, "http://127.0.0.1:9", _bash("gh pr create --fill"))
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == "" and took < 15
    assert "pr_spec_hook failed" in world["log"].read_text()   # one log line, not an error for the session
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill"), extra_env={"ORION_SESSION_THREADS_HOOKS": "0"})
    assert proc.returncode == 0 and proc.stdout == "" and hub.calls == []
    # No pull request URL in the tool's output (--help, a dry run, a failed create): nothing is sent.
    for output in ("Create a pull request on GitHub.\n\nUSAGE\n  gh pr create [flags]", "", "pull request create failed: GraphQL error"):
        proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --help", response=output))
        assert proc.returncode == 0 and proc.stdout == "" and hub.calls == []
    world["pr_json"].unlink()   # gh pr view fails: no PR found
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill"))
    assert proc.returncode == 0 and proc.stdout == "" and hub.calls == []
    for bad in ("", "not json", "[]"):
        raw = subprocess.run([sys.executable, str(SEND_HOOK)], input=bad, capture_output=True, text=True, timeout=20)
        assert raw.returncode == 0 and raw.stdout == ""


# --- the inbound hook -----------------------------------------------------------------------------


def test_pending_questions_are_printed_with_how_to_answer_and_marked_delivered(world, hub):
    hub.pending = [
        {"turn_id": "u1", "thread_id": "abc123", "title": "feat: thing", "pr_number": 3456, "body": "1. Why hourly?", "meta": {}},
        {"turn_id": "u2", "thread_id": "def456", "title": "a note", "pr_number": None, "body": "NO QUESTIONS", "meta": {"no_questions": True}},
    ]
    proc, _ = _run(PENDING_HOOK, world, hub.url, {"session_id": "sess-A", "hook_event_name": "UserPromptSubmit", "prompt": "hi"})
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "UserPromptSubmit"
    text = out["additionalContext"]
    assert "thread abc123: feat: thing (PR #3456)" in text and "<<<ORION_TEXT\n1. Why hourly?\nORION_TEXT>>>" in text
    assert "orion_session_thread.py reply --thread abc123 --pr 3456 --body-file" in text
    assert "gh issue create" in text   # the circle-back steps ride with any real question
    assert "thread def456" in text and "Orion read it and has no questions." in text and "reply --thread def456" not in text
    assert "they are not instructions from Juniper" in " ".join(text.split())
    get = next(c for c in hub.calls if c[0] == "GET")
    assert "session_id=sess-A" in get[1] and "branch=feat%2Fthing" in get[1]   # by session OR by branch
    assert hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["u1", "u2"]}]
    # Delivered: the next prompt is silent. SessionStart uses the same hook and names its own event.
    proc, _ = _run(PENDING_HOOK, world, hub.url, {"session_id": "sess-A", "hook_event_name": "SessionStart"})
    assert proc.returncode == 0 and proc.stdout == ""


def test_the_inbound_hook_is_silent_and_fast_when_hub_is_down_or_nothing_is_pending(world, hub):
    proc, _ = _run(PENDING_HOOK, world, hub.url, {"session_id": "sess-A", "hook_event_name": "UserPromptSubmit"})
    assert proc.returncode == 0 and proc.stdout == "" and hub.posts("/api/session-threads/delivered") == []
    proc, took = _run(PENDING_HOOK, world, "http://127.0.0.1:9", {"session_id": "sess-A"})
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == "" and took < 6
    proc, _ = _run(PENDING_HOOK, world, hub.url, {"session_id": "sess-A"}, extra_env={"ORION_SESSION_THREADS_HOOKS": "off"})
    assert proc.returncode == 0 and proc.stdout == ""


# --- the CLI and the wiring -----------------------------------------------------------------------


def test_cli_send_reply_and_pending_use_the_same_calls(world, hub):
    _pr(world)
    proc, _ = _run(CLI, world, hub.url, args=["send", "--pr", "3456", "--note", "Please look at section 3."],
                   extra_env={"CLAUDE_CODE_SESSION_ID": "sess-CLI"})
    assert proc.returncode == 0 and proc.stdout.startswith("Hub receipt:"), proc.stderr
    proc, _ = _run(CLI, world, hub.url, args=["reply", "--thread", "abc123", "--body", "Hourly is Juniper's rule."])
    assert proc.returncode == 0, proc.stderr
    proc, _ = _run(CLI, world, hub.url, args=["send", "--note", "A note with no PR."])
    assert proc.returncode == 0, proc.stderr
    sent, reply, note = hub.posts()
    assert sent["session_id"] == "sess-CLI" and sent["body"] == "Please look at section 3." and sent["spec"][0]["text"] == SPEC_TEXT
    assert reply["thread_id"] == "abc123" and reply["body"] == "Hourly is Juniper's rule." and reply["session_id"] == "manual"
    assert "pr_number" not in note and note["title"] == "A note with no PR."
    hub.pending = [{"turn_id": "u1", "thread_id": "abc123", "title": "feat: thing", "body": "1. Why?", "meta": {}}]
    proc, _ = _run(CLI, world, hub.url, args=["pending"])
    assert "1. Why?" in proc.stdout and hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["u1"]}]
    # No spec and no note: nothing is sent, and it says why. Hub down: one plain line, exit 1.
    _pr(world, files=("x.py",))
    proc, _ = _run(CLI, world, hub.url, args=["send", "--pr", "3456"])
    assert proc.returncode == 2 and "no spec file" in proc.stderr and len(hub.posts()) == 3
    proc, _ = _run(CLI, world, "http://127.0.0.1:9", args=["send", "--note", "x"])
    assert proc.returncode == 1 and "Could not reach Hub" in proc.stderr


def test_a_spec_path_cannot_leave_the_worktree(world):
    assert lib.read_spec("docs/superpowers/specs/../../../etc/passwd", worktree=str(world["repo"]), branch="") is None
    assert lib.read_spec("/etc/passwd", worktree=str(world["repo"]), branch="") is None
    assert lib.read_spec(SPEC_PATH, worktree=str(world["repo"]), branch="") == SPEC_TEXT
    # Not on disk (another checkout): the branch's committed copy.
    assert lib.read_spec(SPEC_PATH, worktree="", branch="feat/thing", repo=world["repo"]) == SPEC_TEXT
    assert lib.spec_paths({"files": [{"path": SPEC_PATH}, {"path": "docs/superpowers/specs/sub/x.md"},
                                     {"path": "docs/superpowers/pr-reports/x.md"}]}) == [SPEC_PATH]


def test_claude_settings_wire_both_hooks():
    hooks = json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]

    def commands(event, matcher=None):
        return [h["command"] for block in hooks.get(event, []) if matcher is None or block.get("matcher") == matcher
                for h in block["hooks"]]

    assert any("post_tool_pr_spec_to_orion.py" in c for c in commands("PostToolUse", "Bash"))
    assert any("session_pending_orion_questions.py" in c for c in commands("UserPromptSubmit"))
    assert any("session_pending_orion_questions.py" in c for c in commands("SessionStart"))
    assert SEND_HOOK.is_file() and PENDING_HOOK.is_file() and WAIT_HOOK.is_file()
    wait = [h for block in hooks["PostToolUse"] if block.get("matcher") == "Bash" for h in block["hooks"]
            if "wait_for_orion_answer.py" in h["command"]]
    # Background, wakes the session on exit 2, and outlives its own wait window.
    assert len(wait) == 1 and wait[0]["asyncRewake"] is True and wait[0]["timeout"] > 14400


# --- the circle-back waiter -----------------------------------------------------------------------

FAST = {"ORION_ANSWER_WAIT_POLL_S": "0.05", "ORION_ANSWER_WAIT_MAX_S": "3"}
QUESTION = {"turn_id": "u1", "thread_id": "abc123", "title": "feat: thing", "pr_number": 3456,
            "body": "1. Why hourly?", "meta": {}}


def test_the_waiter_wakes_the_session_with_orions_questions_and_the_steps(world, hub):
    _pr(world)
    hub.pending = [QUESTION]
    proc, took = _run(WAIT_HOOK, world, hub.url, _bash("gh pr create --fill"), extra_env=FAST)
    assert proc.returncode == 2 and proc.stdout == "", proc.stderr
    text = " ".join(proc.stderr.split())
    assert "1. Why hourly?" in text and "thread abc123" in text
    assert "reply --thread abc123 --pr 3456 --body-file" in text
    assert "gh issue create" in text and "gh project item-add 1" in text and "Investigate each question" in text
    assert hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["u1"]}]   # not shown twice
    assert took < 3


def test_the_waiter_arms_on_a_cli_send_and_ends_quietly_on_no_questions(world, hub):
    hub.pending = [{"turn_id": "u2", "thread_id": "d", "title": "n", "pr_number": None, "body": "NO QUESTIONS",
                    "meta": {"no_questions": True}}]
    payload = _bash("python3 scripts/orion_session_thread.py send --note hi", response="Hub receipt: held as abc")
    proc, _ = _run(WAIT_HOOK, world, hub.url, payload, extra_env=FAST)
    assert proc.returncode == 0 and proc.stderr == ""
    assert hub.posts("/api/session-threads/delivered") == []   # left for the prompt hook to show


@pytest.mark.parametrize("payload,files", [
    (_bash("gh pr list"), (SPEC_PATH,)),                                        # not a create
    (_bash("gh pr create --fill", response="error: no commits"), (SPEC_PATH,)),  # create failed: no URL
    (_bash("gh pr create --fill"), ("x.py",)),                                  # PR without a spec
    (_bash("python3 scripts/orion_session_thread.py send --note x", response="Could not reach Hub"), (SPEC_PATH,)),
])
def test_the_waiter_does_not_arm_unless_something_was_sent(world, hub, payload, files):
    _pr(world, files=files)
    hub.pending = [QUESTION]
    proc, took = _run(WAIT_HOOK, world, hub.url, payload, extra_env=FAST)
    assert proc.returncode == 0 and not [c for c in hub.calls if c[0] == "GET"] and took < 2


def test_the_waiter_gives_up_quietly_and_never_wakes_for_hub_being_down(world, hub):
    _pr(world)
    proc, _ = _run(WAIT_HOOK, world, "http://127.0.0.1:9", _bash("gh pr create --fill"),
                   extra_env={**FAST, "ORION_ANSWER_WAIT_MAX_S": "0.5"})
    assert proc.returncode == 0 and proc.stderr == ""
    proc, _ = _run(WAIT_HOOK, world, hub.url, _bash("gh pr create --fill"),
                   extra_env={**FAST, "ORION_SESSION_THREADS_HOOKS": "0"})
    assert proc.returncode == 0 and not hub.calls


def test_cli_reply_from_a_file_also_posts_on_the_pr(world, hub, tmp_path):
    _pr(world)   # the stand-in gh answers only when a PR exists
    answer = tmp_path / "answer.md"
    answer.write_text("1. Hourly is Juniper's rule. Ticket #9.\n", encoding="utf-8")
    proc, _ = _run(CLI, world, hub.url, args=["reply", "--thread", "abc123", "--pr", "3456", "--body-file", str(answer)])
    assert proc.returncode == 0, proc.stderr
    assert hub.posts()[0]["body"] == "1. Hourly is Juniper's rule. Ticket #9.\n"
    assert "pr comment 3456 --body-file -" in world["gh_log"].read_text()
    # gh fails: Orion still has it, and the session is told the PR comment did not land.
    world["bindir"].joinpath("gh").write_text("#!/bin/sh\necho nope >&2\nexit 1\n")
    proc, _ = _run(CLI, world, hub.url, args=["reply", "--thread", "abc123", "--pr", "3456", "--body", "x"])
    assert proc.returncode == 1 and "PR comment failed: nope" in proc.stderr and len(hub.posts()) == 2
