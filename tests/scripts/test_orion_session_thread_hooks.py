"""Session threads, the coding-session side: the two Claude Code hooks and the CLI, run as real
subprocesses against a local stand-in for Hub and a stand-in `gh`.

- post_tool_pr_spec_to_orion.py: fires only on `gh pr create`, sends the PR's spec text and code diff,
  prints Hub's receipt; a PR with nothing readable is ignored; Hub down is silent and exit 0.
- pr_draft_gate.py: `gh pr create` without --draft is refused; `gh pr ready` waits for Orion's first
  review to be answered.
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
GATE_HOOK = ROOT / "scripts" / "hooks" / "pr_draft_gate.py"
sys.path.insert(0, str(ROOT / "scripts"))

import orion_session_thread_lib as lib  # noqa: E402

SPEC_PATH = "docs/superpowers/specs/2026-11-15-thing-design.md"
SPEC_TEXT = "# Thing design\n\nWhat it does.\n"


class Hub:
    """A stand-in Hub: records every call, answers the session-threads routes."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.pending: list[dict] = []
        self.threads: dict[str, dict] = {}
        self.delivered_ids: set[str] = set()
        self.views = 0
        self.on_view = None
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
                if self.path.startswith("/api/session-threads?"):
                    from urllib.parse import parse_qs, urlparse
                    pr = parse_qs(urlparse(self.path).query).get("pr")
                    self._send({"threads": [t["thread"] for t in hub.threads.values()
                                            if not pr or str(t["thread"].get("pr_number")) == pr[0]]})
                elif self.path.startswith("/api/session-threads/") and not self.path.startswith("/api/session-threads/pending"):
                    t = hub.threads[self.path.rsplit("/", 1)[1]]
                    hub.views += 1
                    if hub.on_view:
                        hub.on_view(hub)
                    self._send({"thread": t["thread"], "turns": t["turns"]})
                else:
                    self._send({"pending": hub.pending})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                hub.calls.append(("POST", self.path, body))
                if self.path == "/api/session-threads/delivered":
                    new = [i for i in body["turn_ids"] if i not in hub.delivered_ids]   # Hub marks a turn once
                    hub.delivered_ids.update(new)
                    hub.pending = [p for p in hub.pending if p["turn_id"] not in body["turn_ids"]]
                    self._send({"delivered": len(new)})
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
    gh.write_text("#!/bin/sh\necho \"$@\" >> \"$GH_LOG\"\n"
                  "if [ \"$1 $2\" = \"pr diff\" ]; then [ -f \"$GH_DIFF\" ] || exit 1; cat \"$GH_DIFF\"; exit 0; fi\n"
                  "[ -f \"$GH_PR_JSON\" ] || exit 1\ncat \"$GH_PR_JSON\"\n")
    gh.chmod(0o755)
    return {"repo": repo, "bindir": bindir, "pr_json": tmp_path / "pr.json", "gh_log": tmp_path / "gh.log",
            "diff": tmp_path / "pr.diff",
            "board": tmp_path / "board.jsonl", "log": tmp_path / "hooks.log"}


def _pr(world, files=(SPEC_PATH, "services/orion-hub/scripts/x.py")):
    world["pr_json"].write_text(json.dumps({
        "number": 3456, "url": "https://github.com/o/r/pull/3456", "title": "feat: thing",
        "headRefName": "feat/thing", "files": [{"path": p} for p in files]}))


def _run(script, world, hub_url, payload=None, args=(), extra_env=None):
    env = {**os.environ, "ORION_HUB_URL": hub_url, "ORION_AGENT_BOARD_PATH": str(world["board"]),
           "ORION_SESSION_THREADS_LOG": str(world["log"]), "GH_PR_JSON": str(world["pr_json"]),
           "GH_LOG": str(world["gh_log"]), "GH_DIFF": str(world["diff"]), "PATH": f"{world['bindir']}:{os.environ['PATH']}"}
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


DIFF = ("diff --git a/docs/superpowers/specs/2026-11-15-thing-design.md b/docs/superpowers/specs/2026-11-15-thing-design.md\n"
        "+# Thing design\n"
        "diff --git a/services/orion-hub/scripts/x.py b/services/orion-hub/scripts/x.py\n+x = 1\n"
        "diff --git a/services/orion-hub/package-lock.json b/services/orion-hub/package-lock.json\n+{}\n")


def test_pr_create_with_a_spec_sends_the_spec_text_and_the_code_diff_and_prints_hubs_receipt(world, hub):
    _pr(world)
    world["diff"].write_text(DIFF)
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("git push && gh pr create --fill --draft"))
    assert proc.returncode == 0, proc.stderr
    [msg] = hub.posts()
    assert msg["session_id"] == "sess-A" and msg["pr_number"] == 3456 and msg["branch"] == "feat/thing"
    assert msg["pr_url"] == "https://github.com/o/r/pull/3456" and msg["title"] == "feat: thing"
    assert msg["worktree"] == str(world["repo"]) and msg["session_name"] == "wt"
    diff, spec = msg["spec"]   # the diff first: Hub's spec-file cap can never drop it
    assert spec == {"path": SPEC_PATH, "text": SPEC_TEXT}   # the text itself, not a path Orion cannot read
    # The diff: the code file only. The spec is attached whole above; the lock file is named, not shown.
    assert diff["kind"] == "diff" and diff["path"] == "1 file(s)" and "+x = 1" in diff["text"]
    assert "Thing design" not in diff["text"] and "+{}" not in diff["text"] and "package-lock.json" in diff["text"]
    assert "1 design spec file(s) and a code diff of 1 file(s)" in msg["body"]
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse"
    assert out["additionalContext"].startswith("Hub receipt:") and "session thread abc123" in out["additionalContext"]
    assert "Orion has NOT read it yet" in out["additionalContext"] and "2026-11-15 11:00 UTC" in out["additionalContext"]
    assert "pr view 3456" in world["gh_log"].read_text()   # the PR number came from the tool's own output


def test_a_code_only_pr_sends_its_diff(world, hub):
    _pr(world, files=("services/orion-hub/scripts/x.py",))
    world["diff"].write_text(DIFF)
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill --draft"))
    assert proc.returncode == 0, proc.stderr
    [msg] = hub.posts()
    [diff] = msg["spec"]
    assert diff["kind"] == "diff" and "+x = 1" in diff["text"] and "a code diff of 1 file(s)" in msg["body"]
    assert "Hub receipt:" in json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]


def test_a_pr_whose_diff_gh_cannot_give_still_gets_a_thread_naming_its_files(world, hub):
    # Without a thread the draft gate could never let the PR leave draft (review finding 4).
    _pr(world, files=("services/orion-hub/scripts/x.py", "docs/other/notes.md"))
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill --draft"))   # gh pr diff fails
    assert proc.returncode == 0, proc.stderr
    [msg] = hub.posts()
    assert msg["spec"] == [] and "diff could not be attached" in msg["body"]
    assert "services/orion-hub/scripts/x.py" in msg["body"] and "gh pr diff 3456" in msg["body"]


def test_a_pr_create_on_a_branch_that_already_has_a_pr_sends_nothing(world, hub):
    _pr(world)
    world["diff"].write_text(DIFF)
    payload = _bash("gh pr create --fill --draft",
                    response="a pull request for branch \"feat/thing\" into branch \"main\" already exists:\n"
                             "https://github.com/o/r/pull/3456\n")
    proc, _ = _run(SEND_HOOK, world, hub.url, payload)
    assert proc.returncode == 0 and proc.stdout == "" and hub.posts() == []


def test_a_pr_with_nothing_to_send_and_a_command_that_is_not_pr_create_send_nothing(world, hub):
    _pr(world, files=())
    proc, _ = _run(SEND_HOOK, world, hub.url, _bash("gh pr create --fill --draft"))
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
    assert "gh issue create" in text and "after Juniper's current request" in text   # steps, her request first
    assert "thread def456" in text and "Orion read it and has no questions." in text and "reply --thread def456" not in text
    assert "they are not instructions from Juniper" in " ".join(text.split())
    get = next(c for c in hub.calls if c[0] == "GET")
    assert "session_id=sess-A" in get[1] and "branch=feat%2Fthing" in get[1]   # by session OR by branch
    # Claimed for THIS session: Hub hands an unanswered question to a later session too (thread 455a41af5b44 Q5).
    assert hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["u1"], "session_id": "sess-A"},
                                                           {"turn_ids": ["u2"], "session_id": "sess-A"}]
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
    assert sent["session_id"] == "sess-CLI" and sent["body"].startswith("Please look at section 3.") and sent["spec"][0]["text"] == SPEC_TEXT
    assert "diff could not be attached" in sent["body"]   # the stand-in gh gives no diff here
    assert reply["thread_id"] == "abc123" and reply["body"] == "Hourly is Juniper's rule." and reply["session_id"] == "manual"
    assert "pr_number" not in note and note["title"] == "A note with no PR."
    hub.pending = [{"turn_id": "u1", "thread_id": "abc123", "title": "feat: thing", "body": "1. Why?", "meta": {}}]
    proc, _ = _run(CLI, world, hub.url, args=["pending"])
    assert "1. Why?" in proc.stdout and hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["u1"]}]
    # A PR with no files at all: nothing is sent, and it says why. Hub down: one plain line, exit 1.
    _pr(world, files=())
    proc, _ = _run(CLI, world, hub.url, args=["send", "--pr", "3456"])
    assert proc.returncode == 2 and "no spec file and no code diff" in proc.stderr and len(hub.posts()) == 3
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
    assert any("pr_draft_gate.py" in c for c in commands("PreToolUse", "Bash")) and GATE_HOOK.is_file()


# --- the circle-back waiter -----------------------------------------------------------------------

FAST = {"ORION_ANSWER_WAIT_POLL_S": "0.05", "ORION_ANSWER_WAIT_MAX_S": "3", "ORION_ANSWER_ARM_LOOKUP_S": "0.3"}


def _now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _thread(hub, thread_id="abc123def", pr_number=3456, turns=()):
    hub.threads[thread_id] = {"thread": {"thread_id": thread_id, "title": "feat: thing", "pr_number": pr_number,
                                         "updated_at": _now_iso()},
                              "turns": [{"turn_id": "s1", "seq": 1, "speaker": "session", "meta": {}}, *turns]}
    return hub.threads[thread_id]


def _answer(turn_id="o9", seq=9, body="1. Why hourly?", no_questions=False):
    return {"turn_id": turn_id, "seq": seq, "thread_id": "abc123def", "speaker": "orion", "body": body,
            "meta": {"no_questions": True} if no_questions else {}}


def _answer_later(turn, after_views=2):
    def on_view(hub):
        if hub.views == after_views:
            hub.threads["abc123def"]["turns"].append(turn)
    return on_view


def _wait(world, hub, payload, extra=None):
    return _run(WAIT_HOOK, world, hub.url, payload,
                extra_env={**FAST, "ORION_ANSWER_WAIT_LOCK_DIR": str(world["log"].parent / "locks"), **(extra or {})})


def test_the_waiter_wakes_the_session_with_orions_new_answer_and_the_steps(world, hub):
    # An older Orion answer (seq 2, never delivered) is on the thread: it must not wake or end the wait.
    # It said NO QUESTIONS: an older turn WITH questions would put the thread in its answer phase
    # (test_the_waiter_does_not_wait_on_a_thread_whose_round_is_used).
    _thread(hub, turns=[_answer("old", 2, body="old question", no_questions=True)])
    hub.on_view = _answer_later(_answer())
    proc, took = _wait(world, hub, _bash("gh pr create --fill"))
    assert proc.returncode == 2 and proc.stdout == "", proc.stderr
    text = " ".join(proc.stderr.split())
    assert "1. Why hourly?" in text and "old question" not in text and "thread abc123def" in text
    assert "reply --thread abc123def --pr 3456 --body-file" in text
    assert "Do this now, without waiting to be asked" in text and "gh issue create" in text
    assert "gh project item-add 1" in text and "Investigate each question" in text
    assert hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["o9"], "session_id": "sess-A"}]
    assert took < 3


def test_the_waiter_does_not_wake_for_an_answer_the_prompt_hook_already_showed(world, hub):
    _thread(hub)
    hub.delivered_ids.add("o9")   # the prompt hook claimed it first
    hub.on_view = _answer_later(_answer())
    payload = _bash("python3 scripts/orion_session_thread.py reply --thread abc123def --body x",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, _ = _wait(world, hub, payload)
    assert proc.returncode == 0 and proc.stderr == ""


def test_the_waiter_keeps_waiting_past_a_requeued_uncut_reply_and_wakes_on_the_real_questions(world, hub):
    # Review of the unparsed-quota fix: Hub could not cut Orion's reply, kept it (meta.requeued) and reads
    # the thread again. That turn is not an answer: the waiter must not claim it or wake on it.
    _thread(hub)
    requeued = {**_answer("rq", 8, body="### thread:other\n1. whole uncut reply?"),
                "meta": {"unparsed": True, "requeued": True}}
    real = _answer()

    def on_view(h):
        if h.views == 2:
            h.threads["abc123def"]["turns"].append(requeued)
        if h.views == 4:
            h.threads["abc123def"]["turns"].append(real)
    hub.on_view = on_view
    proc, _ = _wait(world, hub, _bash("gh pr create --fill"))
    assert proc.returncode == 2, proc.stderr
    text = " ".join(proc.stderr.split())
    assert "1. Why hourly?" in text and "whole uncut reply" not in text
    assert hub.posts("/api/session-threads/delivered") == [{"turn_ids": ["o9"], "session_id": "sess-A"}]
    assert not lib.has_questions({"meta": {"unparsed": True, "requeued": True}})


def test_the_waiter_ends_quietly_on_no_questions_and_leaves_it_for_the_prompt_hook(world, hub):
    _thread(hub)
    hub.on_view = _answer_later(_answer(no_questions=True))
    payload = _bash("python3 scripts/orion_session_thread.py send --note hi",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, _ = _wait(world, hub, payload)
    assert proc.returncode == 0 and proc.stderr == "" and hub.posts("/api/session-threads/delivered") == []


def test_one_waiter_per_thread(world, hub):
    import fcntl
    _thread(hub, turns=[])
    hub.on_view = _answer_later(_answer())
    lock_dir = world["log"].parent / "locks"
    lock_dir.mkdir()
    with open(lock_dir / "abc123def.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc, _ = _wait(world, hub, _bash("gh pr create --fill"))
    assert proc.returncode == 0 and hub.views == 0


@pytest.mark.parametrize("payload", [
    _bash("gh pr list"),                                                       # not a create
    _bash("gh pr create --fill", response="error: no commits"),                # create failed: no URL
    _bash("gh pr create --fill", response="https://github.com/o/r/pull/777"),  # no thread for that PR: nothing sent
    _bash("python3 scripts/orion_session_thread.py send --note x", response="Could not reach Hub"),
])
def test_the_waiter_does_not_arm_unless_hub_stored_a_message(world, hub, payload):
    _thread(hub)
    hub.on_view = _answer_later(_answer(), after_views=1)
    proc, took = _wait(world, hub, payload)
    assert proc.returncode == 0 and hub.views == 0 and took < 2


def test_the_waiter_does_not_arm_on_a_stale_thread_for_the_pr(world, hub):
    t = _thread(hub)
    t["thread"]["updated_at"] = "2026-01-01T00:00:00+00:00"   # e.g. `gh pr create` printing "already exists"
    proc, _ = _wait(world, hub, _bash("gh pr create --fill"))
    assert proc.returncode == 0 and hub.views == 0


def test_the_waiter_gives_up_quietly_and_never_wakes_for_hub_being_down(world, hub):
    payload = _bash("python3 scripts/orion_session_thread.py send --note x",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, _ = _run(WAIT_HOOK, world, "http://127.0.0.1:9", payload,
                   extra_env={**FAST, "ORION_ANSWER_WAIT_MAX_S": "0.5",
                              "ORION_ANSWER_WAIT_LOCK_DIR": str(world["log"].parent / "locks")})
    assert proc.returncode == 0 and proc.stderr == ""
    proc, _ = _wait(world, hub, _bash("gh pr create --fill"), {"ORION_SESSION_THREADS_HOOKS": "0"})
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
    assert "Retry only the comment: gh pr comment 3456" in proc.stderr


# --- rounds of questions per thread ---------------------------------------------------------


def test_a_closing_note_is_shown_without_reply_steps(world, hub):
    hub.pending = [
        {"turn_id": "c1", "thread_id": "abc123", "title": "feat: thing", "pr_number": 3456,
         "body": "NO QUESTIONS\nThe answers settle it; the retry cap stays open.",
         "meta": {"closing_note": True, "dropped_question_lines": 0}},
        {"turn_id": "c2", "thread_id": "def456", "title": "other", "pr_number": 3457,
         "body": "1. A new question Hub did not send?", "meta": {"closing_note": True, "dropped_question_lines": 1}},
    ]
    proc, _ = _run(PENDING_HOOK, world, hub.url, {"session_id": "sess-A", "hook_event_name": "UserPromptSubmit"})
    assert proc.returncode == 0, proc.stderr
    text = " ".join(json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"].split())
    assert "Orion read your answers" in text and "nothing here needs an answer" in text
    assert "the retry cap stays open" in text and "not a question to answer" in text
    # No reply command, no circle-back steps, no "answer every question": nothing to answer.
    assert "reply --thread" not in text and "gh issue create" not in text and "Answer every question" not in text
    assert "Do this" not in text
    assert not lib.has_questions({"meta": {"closing_note": True}}) and lib.has_questions({"meta": {}})


def test_the_waiter_does_not_wait_on_a_thread_whose_round_is_used(world, hub):
    # Orion used all 3 rounds on this thread already: the reply is answers, nothing will come back to answer.
    _thread(hub, turns=[{**_answer("o2", 2), "roundup_id": "r1"}, {**_answer("o3", 3), "roundup_id": "r2"},
                        {**_answer("o4", 4), "roundup_id": "r3"}])
    hub.on_view = _answer_later({**_answer("o9", 9, body="NO QUESTIONS"), "meta": {"closing_note": True}})
    payload = _bash("python3 scripts/orion_session_thread.py reply --thread abc123def --body x",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, took = _wait(world, hub, payload)
    assert proc.returncode == 0 and proc.stderr == "" and hub.views == 1 and took < 2
    assert hub.posts("/api/session-threads/delivered") == []
    assert "answer_wait answer_phase thread=abc123def" in world["log"].read_text()


def test_the_waiter_still_waits_while_orion_has_rounds_left(world, hub):
    # Orion asked once (round 1 of 3): the reply may get a second round of questions.
    _thread(hub, turns=[{**_answer("o2", 2), "roundup_id": "r1"}])
    hub.on_view = _answer_later(_answer(body="1. Still unclear: who pays?"))
    payload = _bash("python3 scripts/orion_session_thread.py reply --thread abc123def --body x",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, _ = _wait(world, hub, payload)
    assert proc.returncode == 2 and "1. Still unclear: who pays?" in proc.stderr


def test_the_waiter_still_waits_after_a_no_questions_turn(world, hub):
    # NO QUESTIONS used no round: a new message may still get Orion's questions.
    _thread(hub, turns=[{**_answer("o2", 2, no_questions=True), "roundup_id": "r1"}])
    hub.on_view = _answer_later(_answer())
    payload = _bash("python3 scripts/orion_session_thread.py reply --thread abc123def --body x",
                    response="Hub receipt: your message to Orion is held as session thread abc123def. ...")
    proc, _ = _wait(world, hub, payload)
    assert proc.returncode == 2 and "1. Why hourly?" in proc.stderr


def test_the_receipt_line_says_when_the_round_is_used():
    base = {"thread": {"thread_id": "abc"}, "next_roundup_at": "2026-11-15T11:00:00+00:00", "roundup_enabled": True}
    assert "Orion's questions will come back" in lib.receipt_line(base)
    used = lib.receipt_line({**base, "answer_phase": True})
    assert "every round of questions" in used and "Nothing will come back for you to answer" in used
    assert "Orion's questions will come back" not in used
    reading = lib.receipt_line({**base, "thread": {"thread_id": "abc", "status": "reading"}})
    assert "reading this thread's earlier message now" in reading and "Orion's questions will come back" not in reading


def test_a_question_handed_to_an_earlier_session_says_so_and_still_asks_for_a_reply():
    # Thread 455a41af5b44 Q5: Hub re-hands unanswered questions to a later session on the branch.
    row = {"turn_id": "o1", "thread_id": "abc123", "title": "t", "body": "1. Why?", "meta": {"delivered_to": ["s0"]},
           "delivered_at": "2026-11-18T05:27:45+00:00", "pr_number": 7}
    text = lib.pending_text([row])
    assert "An earlier session was shown these questions and no session has replied yet." in text
    assert "reply --thread abc123 --pr 7" in text and lib.has_questions(row)
    assert "earlier session" not in lib.pending_text([{**row, "delivered_at": None}])


# --- the draft gate -------------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts" / "hooks"))
import pr_draft_gate as gate  # noqa: E402


def _pre(command):
    return {"session_id": "sess-A", "hook_event_name": "PreToolUse", "tool_name": "Bash",
            "tool_input": {"command": command}}


def _denied(proc):
    out = json.loads(proc.stdout)["hookSpecificOutput"] if proc.stdout else {}
    return out.get("permissionDecision") == "deny", out.get("permissionDecisionReason", "")


@pytest.mark.parametrize("command,refused", [
    ("gh pr create --title x --body y", True),
    ("git push -u origin feat/x && gh pr create --fill", True),
    ("rtk gh pr create --fill", True),
    ("gh pr create --draft=false --fill", True),
    ("gh pr create --fill --draft", False),
    ("gh pr create -d --fill", False),
    ("gh pr create --title x --body \"$(cat <<'EOF'\n## Summary\n\nwords\nEOF\n)\" --draft", False),
    ("gh pr create --help", False),
    ("echo 'gh pr create'", False),
    ("gh pr view 12", False),
    # Text in a body, a title or a heredoc is never a flag (review finding 2).
    ("gh pr create --title x --body \"use --help to see\"", True),
    ("gh pr create --title \"fix -d flag\" --body y", True),
    ("gh pr create --title x --body \"$(cat <<'EOF'\nfoo -d bar --draft\nEOF\n)\"", True),
    ("gh pr create -t x -b y && git log -d", True),
    ("gh pr create -t x -b y; echo --draft", True),
    ("gh -R o/r pr create --fill", True),
    ("/usr/bin/gh pr create --fill", True),
    ("(gh pr create --fill)", True),
    ("git push &&\ngh pr create --fill", True),
])
def test_pull_requests_open_as_drafts(command, refused):
    assert gate.undrafted_create(command) is refused


@pytest.mark.parametrize("command,ready", [
    # A heredoc or a quoted message that mentions gh pr ready is text, not a command (review finding 1).
    ("gh pr create --draft --title x --body \"$(cat <<'EOF'\nThen run:\ngh pr ready <n>\nEOF\n)\"", []),
    ("git commit -m \"$(cat <<'EOF'\ngh pr ready is gated\nEOF\n)\"", []),
    ("git commit -m 'gh pr ready 12 later'", []),
    ("gh pr ready 12", ["12"]),
    ("gh -R o/r pr ready 12", ["12"]),
    ("(gh pr ready 12)", ["12"]),
    ("cd /wt && rtk gh pr ready", [""]),
])
def test_only_real_gh_pr_ready_commands_are_gated(command, ready):
    assert [gate.ready_target(c["args"]) for c in gate.gh_pr_commands(command) if c["verb"] == "ready"] == ready


def test_the_override_is_a_leading_env_assignment_only_and_cd_and_repo_are_read():
    [c] = gate.gh_pr_commands("ORION_ALLOW_PR_READY=1 gh pr ready 12")
    assert c["env"] == {"ORION_ALLOW_PR_READY": "1"}
    [c] = gate.gh_pr_commands("gh pr ready 12 # ORION_ALLOW_PR_READY=1")
    assert c["env"] == {}
    [c] = gate.gh_pr_commands("cd /wt/a && gh -R o/r pr ready 12")
    assert c["cd"] == "/wt/a" and c["repo"] == "o/r"


def test_the_diff_keeps_files_whose_path_it_cannot_read_and_follows_renames():
    diff = ("diff --git a/uv.lock b/uv.lock\n--- a/uv.lock\n+++ b/uv.lock\n+lock\n"
            "diff --git a/my notes.py b/my notes.py\n--- a/my notes.py\n+++ b/my notes.py\n+spaced = 1\n"
            "diff --git \"a/caf\\303\\251.py\" \"b/caf\\303\\251.py\"\n+quoted = 1\n"
            "diff --git a/old.py b/new.py\nrename from old.py\nrename to new.py\n"
            "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n-x\n")
    kept, skipped = lib.filter_diff(diff)
    assert skipped == ["uv.lock"] and "+spaced = 1" in kept and "+quoted = 1" in kept
    assert "rename to new.py" in kept and "-x" in kept and "+lock" not in kept
    assert lib.diff_path("diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n") == "gone.py"


def _orion_turn(seq, *, no_questions=False, closing=False, requeued=False):
    meta = {k: True for k, on in (("no_questions", no_questions), ("closing_note", closing), ("requeued", requeued)) if on}
    return {"speaker": "orion", "seq": seq, "meta": meta}


def _session_turn(seq):
    return {"speaker": "session", "seq": seq, "meta": {}}


def test_ready_waits_for_the_answer_to_orions_first_review():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
    fresh = {"thread_id": "t1", "status": "held", "created_at": (now - timedelta(hours=1)).isoformat()}
    assert gate.ready_verdict(None, [], now=now)[0] is False
    # Not reviewed yet, fresh: refused. After the wait, or when Hub gave up: allowed.
    assert gate.ready_verdict(fresh, [_session_turn(1)], now=now)[0] is False
    old = {**fresh, "created_at": (now - timedelta(hours=25)).isoformat()}
    assert gate.ready_verdict(old, [_session_turn(1)], now=now)[0] is True
    assert gate.ready_verdict({**fresh, "status": "failed"}, [_session_turn(1)], now=now)[0] is True
    # A requeued uncut reply is not a review.
    assert gate.ready_verdict(fresh, [_session_turn(1), _orion_turn(3, requeued=True)], now=now)[0] is False
    # Orion asked: refused until a session wrote after Orion's FIRST question turn.
    asked = [_session_turn(1), _orion_turn(3)]
    assert gate.ready_verdict(fresh, asked, now=now)[0] is False
    assert gate.ready_verdict(old, asked, now=now)[0] is False   # the wait never excuses unanswered questions
    assert gate.ready_verdict(fresh, [*asked, _session_turn(5)], now=now)[0] is True
    # A re-send of the PR (spec/diff attached again) is not an answer (review finding 7).
    resend = {"speaker": "session", "seq": 5, "meta": {"spec": [{"path": "1 file(s)", "kind": "diff"}]}}
    assert gate.ready_verdict(fresh, [*asked, resend], now=now)[0] is False
    # NO QUESTIONS first: nothing to answer.
    assert gate.ready_verdict(fresh, [_session_turn(1), _orion_turn(3, no_questions=True)], now=now)[0] is True


def test_the_gate_hook_refuses_ready_until_orion_was_answered_and_takes_the_override(world, hub):
    _pr(world)
    gate_env = {"ORION_PR_READY_WAIT_H": "24"}
    # An undrafted create is refused.
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("gh pr create --fill"), extra_env=gate_env)
    denied, why = _denied(proc)
    assert proc.returncode == 0 and denied and "--draft" in why
    # No thread for the PR: refused, with the send command.
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("gh pr ready 3456"), extra_env=gate_env)
    denied, why = _denied(proc)
    assert denied and "orion_session_thread.py send --pr 3456" in why
    # Orion asked, nobody answered: refused. Answered: allowed (no output).
    thread = _thread(hub, turns=[{**_answer("o2", 2), "roundup_id": "r1"}])
    thread["thread"]["created_at"] = _now_iso()
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("gh pr ready 3456"), extra_env=gate_env)
    denied, why = _denied(proc)
    assert denied and "no session has answered" in why and "ORION_ALLOW_PR_READY=1" in why
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("ORION_ALLOW_PR_READY=1 gh pr ready 3456"), extra_env=gate_env)
    assert proc.returncode == 0 and proc.stdout == ""
    thread["turns"].append({"turn_id": "s3", "seq": 3, "speaker": "session", "meta": {}})
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("gh pr ready"), extra_env=gate_env)   # current branch's PR
    assert proc.returncode == 0 and proc.stdout == ""
    # --undo (back to draft) and other commands pass untouched; Hub down refuses ready.
    for command in ("gh pr ready 3456 --undo", "git status", "gh pr view 3456"):
        proc, _ = _run(GATE_HOOK, world, hub.url, _pre(command), extra_env=gate_env)
        assert proc.stdout == "", command
    proc, _ = _run(GATE_HOOK, world, "http://127.0.0.1:9", _pre("gh pr ready 3456"), extra_env=gate_env)
    denied, why = _denied(proc)
    assert denied and "Hub did not answer" in why
    # The kill switch.
    proc, _ = _run(GATE_HOOK, world, hub.url, _pre("gh pr create --fill"), extra_env={"ORION_PR_DRAFT_GATE": "0"})
    assert proc.stdout == ""
