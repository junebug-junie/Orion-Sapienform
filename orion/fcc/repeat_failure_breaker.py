"""Repeat-failing-call breaker for FCC's ``claude -p`` turns (a PreToolUse hook).

Incident (urgent run a153451fe423, 2026-10-01): Orion called
``mcp__firecrawl__firecrawl_scrape`` with the identical input ~20 times in one
turn, every call failing with the same DNS error, burning ~10 of its 15
minutes -- and resumed the same call even after auto-compaction summarized it
as unsolvable. Nothing in the harness said "stop".

This module is the deterministic stop. The harness (orion/harness/fcc_motor.py)
installs it as a PreToolUse hook on every FCC turn via ``--settings``. Before
each tool call it re-reads the turn's own Claude Code session transcript
(``transcript_path`` from the hook's stdin payload) and counts how often this
exact call -- same tool name, same normalized input -- has already FAILED.
At the threshold it blocks the call (exit code 2; stderr is shown to the model
as the tool's error) with a message naming the failure count and last error.

State is the transcript itself: one file per ``claude -p`` session, i.e. per
FCC turn, append-only across auto-compaction. So isolation between turns is
structural, there is no shared store, and nothing outlives the turn.

Rules:
- a call key that has EVER succeeded in this turn is never blocked;
- different input (after normalization) is a different key;
- applies to every tool (MCP, Bash, built-ins) -- the hook matcher is ``*``;
- a successful file edit (Edit/Write/MultiEdit/NotebookEdit) resets all failure
  counts: after a code change, re-running the same failing command is a fair retry;
- fail-open: any error reading/parsing exits 0 (Claude Code proceeds).

Out of scope: subagent (Task) calls, which Claude Code 2.x writes to separate
transcripts, and MCP tools that report failure only in text with is_error=false.

Stdlib-only on purpose: it runs as a separate Python process per tool call,
by file path, so it must not import the orion package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

DEFAULT_THRESHOLD = 3
# A successful call to one of these changes the world the failing calls ran in,
# so failure counts start over (edit -> re-run tests is not a stuck loop).
_STATE_CHANGING_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
# Stable marker the harness motor greps for in tool_result errors to log the trace.
BLOCK_MARKER = "[orion-repeat-failure-breaker]"
_ERROR_SNIPPET_CHARS = 300


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


# Input fields that label a call without changing what it does. Confirmed live
# 2026-10-02 (container smoke, Claude Code 2.1.286): the model wrote Bash
# ``description`` as "... (attempt 1)", "(attempt 2)", ... on byte-identical
# commands, so keying on it let four identical failures through unblocked.
_COSMETIC_INPUT_KEYS: Dict[str, frozenset] = {"Bash": frozenset({"description"})}


def call_key(tool_name: str, tool_input: Any) -> str:
    """Identity of a call: tool name + key-order/whitespace-normalized input,
    minus fields that are only a human-readable label for the call."""
    cosmetic = _COSMETIC_INPUT_KEYS.get(tool_name)
    if cosmetic and isinstance(tool_input, dict):
        tool_input = {k: v for k, v in tool_input.items() if k not in cosmetic}
    return f"{tool_name}\x00" + json.dumps(
        _normalize(tool_input), sort_keys=True, separators=(",", ":"), default=str
    )


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return "" if content is None else str(content)


class CallHistory:
    """Per-key failure count, last error, and ever-succeeded flag for one session."""

    def __init__(self) -> None:
        self._pending: Dict[str, Tuple[str, str]] = {}  # tool_use_id -> (tool_name, key)
        self.failures: Dict[str, int] = {}
        self.last_error: Dict[str, str] = {}
        self.succeeded: set[str] = set()

    def feed(self, entry: Dict[str, Any]) -> None:
        if not isinstance(entry, dict) or entry.get("isSidechain"):
            return
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use" and block.get("id") and isinstance(block.get("name"), str):
                self._pending[str(block["id"])] = (
                    block["name"], call_key(block["name"], block.get("input"))
                )
            elif btype == "tool_result":
                pending = self._pending.pop(str(block.get("tool_use_id") or ""), None)
                if pending is None:
                    continue
                tool_name, key = pending
                if block.get("is_error"):
                    self.failures[key] = self.failures.get(key, 0) + 1
                    text = _result_text(block.get("content")).strip()
                    if BLOCK_MARKER not in text:  # keep quoting the real error
                        self.last_error[key] = text
                else:
                    self.succeeded.add(key)
                    if tool_name in _STATE_CHANGING_TOOLS:
                        self.failures.clear()

    def feed_all(self, entries: Iterable[Dict[str, Any]]) -> "CallHistory":
        for entry in entries:
            self.feed(entry)
        return self


def read_transcript(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Only tool_use/tool_result lines matter; skip parsing the rest.
            if '"tool_use' not in line and '"tool_result"' not in line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                yield entry


def decide(
    history: CallHistory, tool_name: str, tool_input: Any, *, threshold: int
) -> Optional[Tuple[int, str]]:
    """``(failure_count, last_error)`` if this call must be blocked, else None."""
    if threshold <= 0:
        return None
    key = call_key(tool_name, tool_input)
    if key in history.succeeded:
        return None
    count = history.failures.get(key, 0)
    if count < threshold:
        return None
    return count, history.last_error.get(key, "")


def block_message(tool_name: str, count: int, last_error: str) -> str:
    err = " ".join(last_error.split())
    if len(err) > _ERROR_SNIPPET_CHARS:
        err = err[:_ERROR_SNIPPET_CHARS] + "..."
    return (
        f"{BLOCK_MARKER} Blocked: this exact {tool_name} call (same input) has already "
        f"failed {count} times in this turn, last error: {err or '(no error text)'}. Repeating it will not "
        "succeed and it will stay blocked for the rest of this turn. Change approach "
        "(a different tool, a different input, or a different source), or answer now "
        "with what you already have and say what you could not reach."
    )


def run_hook(payload: Dict[str, Any], *, threshold: int) -> Optional[str]:
    """Block message for this PreToolUse payload, or None to allow."""
    tool_name = payload.get("tool_name")
    transcript = payload.get("transcript_path")
    if not isinstance(tool_name, str) or not tool_name or not transcript:
        return None
    history = CallHistory().feed_all(read_transcript(transcript))
    verdict = decide(history, tool_name, payload.get("tool_input"), threshold=threshold)
    if verdict is None:
        return None
    return block_message(tool_name, *verdict)


def hook_settings_json(*, threshold: int, python_bin: str) -> str:
    """The ``--settings`` JSON that installs this module as a PreToolUse hook on all tools."""
    import shlex

    cmd = " ".join(
        shlex.quote(part)
        for part in (python_bin, str(Path(__file__).resolve()), "--threshold", str(int(threshold)))
    )
    return json.dumps(
        {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": cmd}]}]}}
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    try:
        # argparse exits 2 on bad args -- the same code that blocks. Stay fail-open.
        args = parser.parse_args(argv)
    except SystemExit:
        return 0
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return 0
        message = run_hook(payload, threshold=args.threshold)
    except Exception:  # fail-open: never break a tool call because the breaker broke
        return 0
    if message is None:
        return 0
    sys.stderr.write(message + "\n")
    return 2  # PreToolUse exit 2 = block; stderr goes back to the model


if __name__ == "__main__":
    sys.exit(main())
