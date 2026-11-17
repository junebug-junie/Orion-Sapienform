"""`dig take-ons [<days>]`: each take-on (intention.carry run) and how many sub-agents its carry turn started.

Why the transcript and not a table (#3639): a carry turn runs on the durable path, which passes no step queue, so it
writes no ``motor_hop`` rows to ``cockpit_turn_sighting`` (``dig turn-steps`` sees nothing), and
``harness_turn_trace.run_artifact`` holds no tool names. The one record that has every tool call is the `claude`
transcript on the governor's volume, and its file name is derivable: Hub resumes a durable step with
``resume_key=f"{run_id}:{source_tag}"`` and the session id is ``session_resume.session_id_for(resume_key, gen)``.
A take-on whose transcript is not found reads UNREAD, never 0.

Read-only: one SELECT through ReadOnlyPsql, then plain file reads.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from orion.schemas.intention import INTENTION_CARRY_TAG, INTENTION_CARRY_WORKFLOW

MAX_DAYS = 90
MAX_RUNS = 50
# Generations per take-on: a lease/ceiling/resume-limit bump moves the session to ``<key>#<n>``, with no upper
# bound. The resume registry names the last one; without it, probing stops after MISS_RUN misses in a row (a bump
# for a missing transcript leaves a gap) or at MAX_GENERATIONS, and a cap hit is said, never hidden.
MAX_GENERATIONS = 64
MISS_RUN = 4
# Claude Code's sub-agent tool: "Agent" today (2.1.x transcripts), "Task" in older ones.
SUBAGENT_TOOLS = frozenset({"Agent", "Task"})
# An Agent call without subagent_type runs Claude Code's default agent.
DEFAULT_SUBAGENT = "general-purpose"
RUN_ID_RE = re.compile(r"intention-([0-9a-f]{16})")

SQL = f"""
SELECT run_id, to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') AS created_at,
       coalesce(terminal, 'running') AS terminal
  FROM durable_run_workflow
 WHERE workflow = '{INTENTION_CARRY_WORKFLOW}'
   AND created_at >= now() - interval '{{days}} days'
 ORDER BY created_at DESC
 LIMIT {MAX_RUNS}
""".strip()


def build_sql(days: int) -> str:
    return SQL.format(days=max(1, min(int(days), MAX_DAYS)))


@dataclass
class TakeOn:
    run_id: str
    created_at: str
    terminal: str
    concern_id: str = ""
    transcripts: list[str] = field(default_factory=list)
    tool_calls: int = 0
    subagents: dict[str, int] = field(default_factory=dict)
    capped: bool = False

    @property
    def read(self) -> bool:
        return bool(self.transcripts)

    @property
    def dispatches(self) -> int:
        return sum(self.subagents.values())


def resume_key(run_id: str, concern_id: str) -> str:
    """Hub's resume key for the carry step (curiosity_investigation ``resume_key=f"{run_id}:{source_tag}"``,
    source tag from durable-runs ``_intention_step``)."""
    return f"{run_id}:{INTENTION_CARRY_TAG}:{concern_id}"


def transcripts_for(run_id: str, concern_id: str, find: Callable[[str], Optional[Path]],
                    last_generation: Optional[int] = None) -> tuple[list[Path], bool]:
    """(transcripts in generation order, capped). ``last_generation`` from the resume registry when known."""
    from orion.harness.session_resume import session_id_for

    key = resume_key(run_id, concern_id)
    if last_generation is not None and 0 <= last_generation < MAX_GENERATIONS:
        return [p for g in range(last_generation + 1) if (p := find(session_id_for(key, g))) is not None], False
    found: list[Path] = []
    misses = 0
    for g in range(MAX_GENERATIONS):
        p = find(session_id_for(key, g))
        if p is None:
            misses += 1
            if misses >= MISS_RUN:
                return found, False
            continue
        misses = 0
        found.append(p)
    return found, True


def count_tools(lines: Iterable[str], seen: set[str]) -> tuple[int, dict[str, int]]:
    """(tool calls, sub-agent calls by type) from one transcript's lines. ``seen`` dedupes tool_use ids across a
    take-on's transcripts (a resumed session may repeat history)."""
    tools = 0
    subagents: dict[str, int] = {}
    for line in lines:
        if '"tool_use"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        # Older CLIs log a sub-agent's own events into the parent file as a sidechain: not the turn's calls.
        if not isinstance(event, dict) or event.get("type") != "assistant" or event.get("isSidechain"):
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tid = str(block.get("id") or "")
            if tid:
                if tid in seen:
                    continue
                seen.add(tid)
            tools += 1
            if block.get("name") in SUBAGENT_TOOLS:
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                kind = str(inp.get("subagent_type") or DEFAULT_SUBAGENT)[:60]
                subagents[kind] = subagents.get(kind, 0) + 1
    return tools, subagents


def read_take_ons(rows: Iterable[dict[str, str]], find: Callable[[str], Optional[Path]],
                  last_generation: Callable[[str], Optional[int]] = lambda key: None) -> list[TakeOn]:
    out: list[TakeOn] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        m = RUN_ID_RE.fullmatch(run_id)
        t = TakeOn(run_id, str(row.get("created_at") or ""), str(row.get("terminal") or ""))
        out.append(t)
        if m is None:
            continue
        t.concern_id = m.group(1)
        seen: set[str] = set()
        paths, t.capped = transcripts_for(run_id, t.concern_id, find,
                                          last_generation(resume_key(run_id, t.concern_id)))
        for path in paths:
            try:
                with path.open(encoding="utf-8", errors="replace") as fh:
                    tools, subs = count_tools(fh, seen)
            except OSError:
                continue
            t.transcripts.append(str(path))
            t.tool_calls += tools
            for kind, n in subs.items():
                t.subagents[kind] = t.subagents.get(kind, 0) + n
    return out


def render(take_ons: list[TakeOn], days: int) -> list[str]:
    if not take_ons:
        return [f"no take-ons in the last {days} day(s)"]
    read = [t for t in take_ons if t.read]
    used = [t for t in read if t.dispatches]
    lines = [f"{len(take_ons)} take-on(s) in the last {days} day(s); transcripts read for {len(read)}; "
             f"{len(used)} started a sub-agent ({sum(t.dispatches for t in read)} sub-agent call(s))"]
    for t in take_ons:
        head = f"{t.created_at}Z {t.run_id} {t.terminal}:"
        if not t.read:
            lines.append(f"{head} UNREAD (no carry transcript found here; run this in a turn's Bash)")
            continue
        kinds = ", ".join(f"{k} {n}" for k, n in sorted(t.subagents.items())) or "none"
        lines.append(f"{head} {len(t.transcripts)} attempt(s), {t.tool_calls} tool call(s), "
                     f"sub-agents: {kinds}" + (f" (+ generations past {MAX_GENERATIONS} not read)" if t.capped else ""))
    return lines
