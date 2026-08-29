#!/usr/bin/env python3
"""Block bare `graphify update` shell commands at tool-call time.

Companion to:
  - scripts/safe_graphify_update.sh (wraps the command, auto-restores on shrink)
  - scripts/check_graph_node_loss.py (pre-commit gate on staged graph.json)
  - scripts/check_graph_worktree_integrity.py (session-start restore)

Those only help when someone uses the wrapper, commits, or starts a new
session. A bare `graphify update .` still guts graphify-out/ immediately and
poisons every `graphify query` until something else restores it. This hook
blocks that path before the shell runs.

Use scripts/safe_graphify_update.sh instead. It runs the same underlying
command but refuses and auto-restores when the known ~91% node-loss bug fires.

Escape hatch: ORION_ALLOW_GRAPH_SHRINK=1 on the statement or in the hook
process environment -- same knob as scripts/check_graph_node_loss.py, for a
deliberate full re-extraction a human has consciously opted into.

Known gaps (disclosed, not silent):
  - Statement splitting uses the same quote-stripping heuristic as
    destructive_git_guard.py -- not a real shell tokenizer.
  - Variable expansion ($GRAPHIFY, $(which graphify)) is not performed.
  - `sh -c 'graphify update .'` may hide the invocation from outer-string
    matching if the agent wraps it opaquely.
  - This hook only fires for sessions that load this repo's hook config
    (.cursor/hooks.json / .claude/settings.json). A plain terminal is
    unprotected -- use the wrapper there too.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from destructive_git_guard import _split_statements  # noqa: E402

# Reuse the leading env-var assignment shape from destructive_git_guard, but
# for the graph shrink escape hatch name used by the commit gate.
_ESCAPE_HATCH_PREFIX = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*ORION_ALLOW_GRAPH_SHRINK=1(?:\s|$)"
)
_ENV_ESCAPE = "ORION_ALLOW_GRAPH_SHRINK"
_SAFE_WRAPPER = re.compile(r"\bsafe_graphify_update\.sh\b")
_GRAPHIFY_BIN = re.compile(r"(?:\bgraphify\b|/graphify\b)(?:\s+-[^\s]+)*(\s+\S+)")


def _first_graphify_subcommand(statement: str) -> str | None:
    """Return graphify's first subcommand token, or None if not invoked."""
    m = _GRAPHIFY_BIN.search(statement)
    if not m:
        return None
    tail = m.group(1).strip()
    return tail.split()[0] if tail else None


def _escape_hatch_set(statement: str) -> bool:
    if os.environ.get(_ENV_ESCAPE) == "1":
        return True
    return bool(_ESCAPE_HATCH_PREFIX.match(statement))


def _statement_is_bare_graphify_update(statement: str) -> bool:
    """`statement` must already be quote-stripped."""
    if _SAFE_WRAPPER.search(statement):
        return False
    return _first_graphify_subcommand(statement) == "update"


def _evaluate(command: str) -> str | None:
    """Return a short reason for the first bare graphify update statement,
    or None if the command is allowed."""
    for _original, statement in _split_statements(command):
        if not _statement_is_bare_graphify_update(statement):
            continue
        if _escape_hatch_set(statement):
            continue
        return "bare `graphify update` (use scripts/safe_graphify_update.sh)"
    return None


def main() -> None:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return
    command = str(tool_input.get("command") or "")
    if not command.strip():
        return

    reason = _evaluate(command)
    if reason is None:
        return

    decision = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"[graphify-safety] Blocked: {reason}. "
                f"`graphify update .` can silently discard ~95% of graphify-out/graph.json "
                f"(known unfixed bug since 2026-07-14). Run "
                f"scripts/safe_graphify_update.sh instead -- it wraps the same command, "
                f"refuses catastrophic shrink, and auto-restores. "
                f"If this is a deliberate full re-extraction, prefix with "
                f"ORION_ALLOW_GRAPH_SHRINK=1."
            ),
        }
    }
    sys.stdout.write(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
