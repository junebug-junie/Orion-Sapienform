#!/usr/bin/env python3
"""Bridge Claude-Code hook scripts into Cursor's hooks.json I/O dialect.

The safety/nudge scripts under scripts/hooks/ speak Claude Code's contract:

  stdin:  tool_input.command / tool_input.file_path / session_id / cwd
  stdout: {"hookSpecificOutput": {"permissionDecision": "deny", ...}}
       or {"hookSpecificOutput": {"additionalContext": "..."}}

Cursor speaks a flatter dialect (see cursor.com/docs/hooks):

  beforeShellExecution / preToolUse deny -> {"permission": "deny", ...}
  sessionStart context                   -> {"additional_context": "..."}
  postToolUse nudge                      -> {"additional_context": "..."}
  stop reminder                          -> {"followup_message": "..."}

This bridge is the only Cursor-specific layer. It does not reimplement the
guards; it normalizes stdin, runs the existing script(s), and translates
stdout. Keep Claude's .claude/settings.json and Cursor's .cursor/hooks.json
pointing at the same scripts/hooks/* bodies.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


_MODES = ("context", "deny", "nudge", "stop")
_PATH_ALIASES = ("path", "target_file", "target_notebook", "filePath")
# Claude metric_lineage_nudge looks for content/new_string/new_source; Cursor Write uses contents.
_BODY_ALIASES = (("contents", "content"), ("new_contents", "content"))


def normalize_cursor_to_claude(payload: dict[str, Any]) -> dict[str, Any]:
    """Make a Cursor hook stdin payload readable by Claude-shaped scripts."""
    out = dict(payload)
    tool_input = out.get("tool_input")
    if isinstance(tool_input, dict):
        tool_input = dict(tool_input)
    else:
        tool_input = {}

    # Cursor common fields use conversation_id; Claude board hooks only read session_id.
    session_id = out.get("session_id")
    if not (isinstance(session_id, str) and session_id.strip()):
        conversation_id = out.get("conversation_id")
        if isinstance(conversation_id, str) and conversation_id.strip():
            out["session_id"] = conversation_id

    # beforeShellExecution: command sits at the top level, not under tool_input.
    top_command = out.get("command")
    if isinstance(top_command, str) and top_command.strip() and "command" not in tool_input:
        tool_input["command"] = top_command

    # Write/edit tools vary: Claude uses file_path; Cursor often uses path.
    if "file_path" not in tool_input:
        for alias in _PATH_ALIASES:
            val = tool_input.get(alias)
            if isinstance(val, str) and val.strip():
                tool_input["file_path"] = val
                break
    if "file_path" not in tool_input:
        top_path = out.get("file_path")
        if isinstance(top_path, str) and top_path.strip():
            tool_input["file_path"] = top_path

    for src, dest in _BODY_ALIASES:
        if dest not in tool_input:
            val = tool_input.get(src)
            if isinstance(val, str):
                tool_input[dest] = val

    if tool_input:
        out["tool_input"] = tool_input
    return out

def translate_claude_to_cursor(raw: str, mode: str) -> dict[str, Any] | None:
    """Map Claude hookSpecificOutput JSON (or empty) to a Cursor response."""
    text = (raw or "").strip()
    if not text:
        if mode == "deny":
            return {"permission": "allow"}
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # graphify's test double echoes plain text; treat as a nudge body.
        if mode in ("nudge", "context") and text:
            return {"additional_context": text}
        if mode == "stop" and text:
            return {"followup_message": text}
        if mode == "deny":
            return {"permission": "allow"}
        return None

    if not isinstance(data, dict):
        if mode == "deny":
            return {"permission": "allow"}
        return None

    # Already Cursor-shaped (passthrough / idempotent).
    if "permission" in data or "additional_context" in data or "followup_message" in data:
        return data

    hso = data.get("hookSpecificOutput")
    if not isinstance(hso, dict):
        if mode == "deny":
            return {"permission": "allow"}
        return None

    decision = hso.get("permissionDecision")
    reason = str(hso.get("permissionDecisionReason") or "").strip()
    context = str(hso.get("additionalContext") or "").strip()

    if mode == "deny":
        if decision == "deny":
            return {
                "permission": "deny",
                "user_message": reason or "Blocked by Orion hook.",
                "agent_message": reason or "Blocked by Orion hook.",
            }
        return {"permission": "allow"}

    if mode in ("context", "nudge"):
        if context:
            return {"additional_context": context}
        return None

    if mode == "stop":
        if context:
            return {"followup_message": context}
        return None

    return None


def _resolve_command(hook_path: str, hook_args: list[str]) -> list[str]:
    path = Path(hook_path)
    suffix = path.suffix.lower()
    if suffix == ".py":
        return [sys.executable, hook_path, *hook_args]
    if suffix == ".sh":
        return ["bash", hook_path, *hook_args]
    return [hook_path, *hook_args]


def _run_one(cmd: list[str], stdin_bytes: bytes, timeout: float) -> str:
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_bytes,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""
    except OSError:
        return ""
    if proc.stderr:
        sys.stderr.buffer.write(proc.stderr)
    # Fail-open on non-zero: same stance as the underlying Claude hooks.
    # Still return stdout so a deny/nudge payload is not dropped on a weird exit.
    return proc.stdout.decode("utf-8", "replace")


def run_bridge(mode: str, hook_cmds: list[list[str]], stdin_bytes: bytes, timeout: float) -> dict[str, Any] | None:
    """Run one or more Claude hooks and produce a single Cursor response.

    For context mode with N scripts, `timeout` is the *shared* wall budget
    split across children so N * per-script cannot exceed the Cursor outer
    hook timeout when hooks.json sets both.
    """
    if mode == "context":
        parts: list[str] = []
        n = max(len(hook_cmds), 1)
        per = max(timeout / n, 1.0)
        for cmd in hook_cmds:
            translated = translate_claude_to_cursor(_run_one(cmd, stdin_bytes, per), "context")
            if translated and translated.get("additional_context"):
                parts.append(str(translated["additional_context"]))
        if not parts:
            return None
        return {"additional_context": "\n\n".join(parts)}

    # deny / nudge / stop: single hook command
    cmd = hook_cmds[0]
    return translate_claude_to_cursor(_run_one(cmd, stdin_bytes, timeout), mode)

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt Claude Code hook scripts for Cursor hooks.json",
    )
    parser.add_argument(
        "mode",
        choices=_MODES,
        help="context=sessionStart, deny=preToolUse/beforeShell, nudge=postToolUse, stop=followup",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-script subprocess timeout in seconds (default 20)",
    )
    parser.add_argument(
        "hooks",
        nargs="+",
        help=(
            "For context: one or more script paths (no script args). "
            "For deny/nudge/stop: script path followed by optional script args."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    raw_in = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw_in.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    # Cursor stop followup_message auto-submits a user turn. Never do that on
    # aborted/error stops — and don't run the board nag script either, or its
    # per-session digest would burn without the agent seeing the reminder.
    if args.mode == "stop" and payload.get("status") in ("aborted", "error"):
        return 0

    normalized = normalize_cursor_to_claude(payload)
    stdin_bytes = json.dumps(normalized, ensure_ascii=False).encode("utf-8")

    if args.mode == "context":
        hook_cmds = [_resolve_command(path, []) for path in args.hooks]
    else:
        hook_path, *hook_args = args.hooks
        hook_cmds = [_resolve_command(hook_path, hook_args)]

    result = run_bridge(args.mode, hook_cmds, stdin_bytes, args.timeout)
    if result is not None:
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
