#!/usr/bin/env python3
"""SessionStart hook: detects a destructively-modified (uncommitted)
graphify-out/graph.json -- or a stale sibling (manifest.json/GRAPH_REPORT.md)
left over from a partial restore -- and auto-restores from HEAD before the
session does anything else.

Delegates all detection/restore logic to
scripts/check_graph_worktree_integrity.py so the two entry points (this hook,
and a human/agent running the script by hand) can never disagree. See that
module's docstring for the two real incidents this guards against.

Root resolution: Claude Code's SessionStart/Stop hooks run with a process cwd
fixed to wherever the session originally started -- confirmed live in this
repo (see agent_board_lib.resolve_current_identity's docstring) that a hook's
own `git rev-parse --show-toplevel` from that fixed cwd resolves to the wrong
checkout when the session is actually operating in a linked worktree. Reuses
the exact fix session_start_agent_board.py already applies for the same bug:
look up this session's own session_id (from the hook's stdin payload) against
the agent board's presence rows, which git-hook-driven heartbeats
(scripts/git_hooks/post-commit, scripts/safe_docker_build.sh) tag with the
correct worktree path. Falls back to plain git-rev-parse (same limitation as
ever) only when no matching heartbeat exists yet -- e.g. the very first hook
fire in a brand new worktree session before any commit has landed.

Fails silently (no output) on any error or import problem -- never blocks
session start. This is a convenience auto-fix, not a gate; the commit-time
gate (scripts/check_graph_node_loss.py, wired via the pre-commit hook) is
what actually blocks bad data from reaching history.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_root(session_id: str | None) -> str | None:
    """Best-effort correct worktree root for THIS session -- see module
    docstring for why plain git-rev-parse alone is not reliable here."""
    try:
        from agent_board_lib import board_config_from_env, resolve_current_identity

        identity = resolve_current_identity(board_config_from_env(), session_id=session_id)
        path = identity.get("worktree_path")
        if path:
            return path
    except Exception:
        pass
    from check_graph_worktree_integrity import _repo_root

    return _repo_root()


def main() -> None:
    try:
        from check_graph_worktree_integrity import check, resolve_threshold
    except Exception:
        return

    try:
        from agent_board_lib import read_session_id_from_stdin_hook_payload

        session_id = read_session_id_from_stdin_hook_payload()
    except Exception:
        session_id = None

    root = _resolve_root(session_id)
    if root is None:
        return

    try:
        result = check(root, resolve_threshold(None), check_only=False)
    except Exception:
        return

    if not result["dirty"] or (not result["destructive"] and not result["desync"]):
        return  # clean, or dirty-but-fine -- no news is no news

    if result["escaped"]:
        summary = (
            "graphify-worktree-guard: destructive graphify-out/ drift detected but "
            "ORION_ALLOW_GRAPH_SHRINK=1 is set -- left as-is. "
            f"({result['detail']})"
        )
    elif result["restored"]:
        summary = (
            "graphify-worktree-guard: graphify-out/ was destructively modified "
            f"({result['detail']}) -- auto-restored from HEAD. Discarded content "
            f"backed up to {result['backup_dir']}."
        )
    else:
        summary = (
            f"graphify-worktree-guard: destructive graphify-out/ drift detected "
            f"({result['detail']}) but the auto-restore FAILED -- fix manually, "
            f"see scripts/check_graph_worktree_integrity.py --json for detail."
        )

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": summary,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
