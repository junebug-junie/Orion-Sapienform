#!/usr/bin/env python3
"""How often do curiosity and self-inquiry runs read Orion's own code, and how often through the dig kit?

Read-only measurement for the 2026-10-30 fold of self-study into curiosity
(docs/superpowers/specs/2026-10-29-self-study-into-curiosity-proposal.md, question 4). Juniper's
decision: do not build a `self_repo` curiosity line; measure first. This is that measurement.

Source of truth: the `claude -p` session transcripts the harness governor keeps for every FCC
turn (curiosity runs go through the same harness as chat). Each transcript is one JSONL file. The
first user message carries the run's prompt, so it says which line the turn belongs to:

* ``self_inquiry``            -- the self-inquiry prompt ("WHERE THE RECORDS ARE")
* ``curiosity_investigation`` -- the investigate prompt ("This is your own time. ...")
* ``other_harness_turn``      -- any other harness turn (chat, reading, ...)

Per session it records whether any Bash call ran ``orion.dig`` and whether any tool call read repo
code (Read/Grep/Glob on a repo path, or a read-only shell command on one). Run it inside the
governor container, which owns the transcripts (read-only):

    docker exec -i orion-athena-harness-governor python3 - 3 < scripts/report_dig_kit_usage.py

The argument is the window in days (by file mtime; default 14). Production reads only.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
import time
from typing import Iterable

TRANSCRIPT_DIR = "/root/.claude/projects/-mnt-orion-fcc-repo"
SELF_INQUIRY_MARK = "WHERE THE RECORDS ARE"
INVESTIGATE_MARK = "This is your own time. Nobody asked you anything"
HARNESS_MARK = "Orion harness motor"
CURIOSITY_KINDS = ("self_inquiry", "curiosity_investigation")

_CODE_PATH_RE = re.compile(r"(services/|orion/|docs/|\.py\b|\.ya?ml\b)")
_READ_SHELL_RE = re.compile(r"\b(cat|sed|head|tail|grep|rg|find|ls|wc|awk|git)\b")
_DIG_SUBCOMMAND_RE = re.compile(r"orion\.dig\s+(?:--dig\s+\S+\s+)?([a-z_]+)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def classify_prompt(text: str) -> str:
    if SELF_INQUIRY_MARK in text:
        return "self_inquiry"
    if INVESTIGATE_MARK in text:
        return "curiosity_investigation"
    if text.startswith(HARNESS_MARK):
        return "other_harness_turn"
    return "non_turn"


def summarize_session(records: Iterable[dict]) -> dict:
    """One transcript -> {kind, used_dig, read_code, dig_subcommands}."""
    kind = None
    used_dig = read_code = False
    subcommands: list[str] = []
    for record in records:
        message = record.get("message") or {}
        content = message.get("content")
        if record.get("type") == "user" and kind is None:
            kind = classify_prompt(_text(content))
        if record.get("type") != "assistant" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "tool_use":
                continue
            name = part.get("name")
            tool_input = part.get("input") or {}
            command = str(tool_input.get("command") or "")
            if name == "Bash" and "orion.dig" in command:
                used_dig = True
                match = _DIG_SUBCOMMAND_RE.search(command)
                subcommands.append(match.group(1) if match else "?")
            elif name in ("Read", "Grep", "Glob") and _CODE_PATH_RE.search(json.dumps(tool_input)):
                read_code = True
            elif name == "Bash" and _READ_SHELL_RE.search(command) and _CODE_PATH_RE.search(command):
                read_code = True
    return {"kind": kind or "empty", "used_dig": used_dig, "read_code": read_code, "dig_subcommands": subcommands}


def _records(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def report(days: int, transcript_dir: str = TRANSCRIPT_DIR, now: float | None = None) -> dict:
    cutoff = (now or time.time()) - days * 86400
    by_kind: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    subcommands: collections.Counter = collections.Counter()
    for path in glob.glob(os.path.join(transcript_dir, "*.jsonl")):
        if os.path.getmtime(path) < cutoff:
            continue
        session = summarize_session(_records(path))
        counter = by_kind[session["kind"]]
        counter["sessions"] += 1
        counter["used_dig"] += session["used_dig"]
        counter["read_code"] += session["read_code"]
        counter["read_code_without_dig"] += session["read_code"] and not session["used_dig"]
        if session["kind"] in CURIOSITY_KINDS:
            subcommands.update(session["dig_subcommands"])
    return {"window_days": days, "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
            "curiosity_dig_subcommands": dict(subcommands.most_common())}


def main(argv: list[str]) -> int:
    days = int(argv[1]) if len(argv) > 1 else 14
    print(json.dumps(report(days), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
