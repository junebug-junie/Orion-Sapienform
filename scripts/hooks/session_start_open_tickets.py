#!/usr/bin/env python3
"""SessionStart hook: a short summary of open GitHub issues (the ticket board)
as additionalContext, so open follow-ups are seen every session instead of
living in one agent's memory file.

Lists every `needs-juniper` issue by number and title (those are blocked on
her), and counts the rest by kind label. Fails silently on any error (no gh,
no network, no auth) -- never blocks session start.
"""
from __future__ import annotations

import json
import subprocess
import sys

REPO = "junebug-junie/Orion-Sapienform"  # explicit: hooks may start outside the repo dir
KIND_LABELS =("follow-up", "verify-live", "design", "bug")
MAX_LISTED = 8


def summarize(issues: list[dict]) -> str:
    if not issues:
        return ""
    def names(i: dict) -> set[str]:
        return {l.get("name", "") for l in i.get("labels") or []}

    blocked = [i for i in issues if "needs-juniper" in names(i)]
    counts = {k: sum(1 for i in issues if k in names(i)) for k in KIND_LABELS}
    kinds = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
    lines = [
        f"Open tickets: {len(issues)}" + (f" ({kinds})" if kinds else "")
        + ". File deferred findings with `gh issue create` (CLAUDE.md 'Ticket board')."
    ]
    if blocked:
        lines.append(f"Waiting on Juniper ({len(blocked)}):")
        for i in blocked[:MAX_LISTED]:
            lines.append(f"  #{i['number']} {i['title']}")
        if len(blocked) > MAX_LISTED:
            lines.append(f"  ...and {len(blocked) - MAX_LISTED} more (gh issue list -l needs-juniper)")
    return "\n".join(lines)


def main() -> None:
    try:
        out = subprocess.run(
            ["gh", "issue", "list", "-R", REPO, "--state", "open", "--limit", "300",
             "--json", "number,title,labels"],
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout
        summary = summarize(json.loads(out))
    except Exception:
        return
    if not summary:
        return
    sys.stdout.write(json.dumps({
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": summary}
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
