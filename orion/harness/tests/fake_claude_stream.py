#!/usr/bin/env python3
"""Stand-in ``claude`` for the warm-pool tests. No real model, no network beyond the test's stub.

Speaks the stream-json shapes the real CLI 2.1.291 produced against a local
stub on 2026-10-06 (see the PR report): in ``--input-format stream-json`` mode
nothing is printed until a stdin message arrives; ``/clear`` answers
``conversation_reset`` -> ``system/init`` (new session) -> ``result`` (empty);
every prompt answers ``system/init`` -> ``assistant`` -> ``result``.

Unlike the real CLI it keeps a plain message history and sends ALL of it to
the model on every turn, so a turn that was not preceded by ``/clear`` leaks
the previous prompt into the upstream request -- which is what the isolation
test checks for.

Prompt verbs: ``CRASH`` exits 3 after init; ``CRASH_SILENT`` exits 3 before
saying anything; ``HANG`` sleeps; ``ENVFILE`` reports the turn clock a Bash
call would see (process env overlaid by ``$CLAUDE_ENV_FILE`` exports).
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
import uuid


def emit(ev: dict) -> None:
    sys.stdout.write(json.dumps(ev) + "\n")
    sys.stdout.flush()


def _argv_value(flag: str) -> str:
    args = sys.argv[1:]
    return args[args.index(flag) + 1] if flag in args else ""


STREAMING = "--input-format" in sys.argv
MODEL = _argv_value("--model")
SESSION = [str(uuid.uuid4())]
HISTORY: list[dict] = []

log_path = os.environ.get("FAKE_CLAUDE_SPAWN_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "streaming": STREAMING, "model": MODEL}) + "\n")

# Stand-in for MCP server start-up, the cost the pool exists to avoid.
time.sleep(float(os.environ.get("FAKE_CLAUDE_BOOT_SEC", "0") or 0))


def bash_view_of_turn_clock() -> dict:
    seen = {k: os.environ.get(k) for k in ("ORION_TURN_BUDGET_SEC", "ORION_TURN_DEADLINE_EPOCH", "ORION_TURN_STEP_STALL_SEC")}
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if env_file and os.path.exists(env_file):
        for line in open(env_file, encoding="utf-8"):
            parts = shlex.split(line)
            if len(parts) == 2 and parts[0] == "export" and "=" in parts[1]:
                key, _, value = parts[1].partition("=")
                seen[key] = value
    return seen


def call_model() -> str:
    base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
        # The test wrapper execs Python, so this is the pid the pool holds.
        "x-fake-pid": str(os.getpid()),
    }
    for line in os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            headers[key.strip()] = value.strip()
    body = json.dumps({"model": MODEL, "messages": HISTORY, "stream": False}).encode()
    req = urllib.request.Request(base + "/v1/messages", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict))


def run_turn(text: str) -> bool:
    emit({"type": "system", "subtype": "init", "session_id": SESSION[0], "model": MODEL})
    if text.startswith("CRASH"):
        sys.exit(3)
    if text.startswith("HANG"):
        time.sleep(3600)
    if text.startswith("BGSHELL"):
        emit({"type": "assistant", "session_id": SESSION[0], "message": {"model": "stub-model", "content": [
            {"type": "tool_use", "id": "toolu_bg", "name": "Bash",
             "input": {"command": "sleep 60", "run_in_background": True}}]}})
        reply = "started"
    elif text.startswith("ENVFILE"):
        reply = "CLOCK " + json.dumps(bash_view_of_turn_clock(), sort_keys=True)
    else:
        HISTORY.append({"role": "user", "content": text})
        try:
            reply = call_model()
        except urllib.error.HTTPError as exc:
            emit({"type": "result", "subtype": "error", "is_error": True,
                  "result": f"API Error: {exc.code}", "session_id": SESSION[0]})
            return False
        HISTORY.append({"role": "assistant", "content": reply})
    emit({"type": "assistant", "session_id": SESSION[0],
          "message": {"model": "stub-model", "content": [{"type": "text", "text": reply}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": reply,
          "session_id": SESSION[0], "num_turns": 1})
    return True


def main() -> int:
    if not STREAMING:
        prompt = sys.argv[sys.argv.index("-p") + 1]
        if prompt.startswith("CRASH_SILENT"):
            return 3
        return 0 if run_turn(prompt) else 1
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        text = msg["message"]["content"]
        if text == "/clear":
            HISTORY.clear()
            emit({"type": "conversation_reset", "trigger": "clear", "session_id": SESSION[0]})
            SESSION[0] = str(uuid.uuid4())
            emit({"type": "system", "subtype": "init", "session_id": SESSION[0], "model": MODEL})
            emit({"type": "result", "subtype": "success", "is_error": False, "result": "",
                  "session_id": SESSION[0], "num_turns": 0})
            continue
        if text.startswith("CRASH_SILENT"):
            return 3
        run_turn(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
