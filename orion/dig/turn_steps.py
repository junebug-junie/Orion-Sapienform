"""`dig turn-steps`: Orion's own earlier tool steps in one Hub chat, from cockpit_turn_sighting.

Spec: docs/superpowers/specs/2026-11-05-turn-working-notes-design.md (Patch 1). Read-only:
every query runs through ReadOnlyPsql (orion_readonly role, READ ONLY transaction).
This is the exact record; working notes (orion/harness/working_notes.py) are the short version.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
MAX_TURNS = 5
RESULT_CHARS = 300
ACTION_CHARS = 240

# One motor_hop's tool blocks, trimmed in SQL: big inputs (a Write's content) and tool results
# never leave the DB whole, and jsonb text escapes newlines, so each row is one short CSV line.
CONTENT_SQL = """
(SELECT jsonb_agg(jsonb_build_object(
          'type', b->>'type', 'id', b->>'id', 'name', b->>'name', 'tool_use_id', b->>'tool_use_id',
          'is_error', b->'is_error',
          'input', CASE WHEN length(((b->'input') - 'content' - 'new_string' - 'old_string')::text) > 2000
                        THEN jsonb_build_object('truncated',
                               left(((b->'input') - 'content' - 'new_string' - 'old_string')::text, 600))
                        ELSE (b->'input') - 'content' - 'new_string' - 'old_string' END,
          'content', left(CASE jsonb_typeof(b->'content')
                            WHEN 'string' THEN b->>'content'
                            WHEN 'array' THEN (SELECT string_agg(x->>'text', ' ')
                                                 FROM jsonb_array_elements(b->'content') x)
                          END, 400)))
   FROM jsonb_array_elements(CASE jsonb_typeof(s.raw->'step'->'raw'->'message'->'content')
                               WHEN 'array' THEN s.raw->'step'->'raw'->'message'->'content'
                               ELSE '[]'::jsonb END) b
  WHERE b->>'type' IN ('tool_use', 'tool_result'))::text
""".strip()

# Last <n> chat turns of this Hub session that ran a tool, then their content-bearing steps.
SQL = """
WITH turns AS (
  SELECT c.correlation_id, c.created_at
    FROM chat_history_log c
   WHERE c.session_id = '{sid}'
     AND c.correlation_id IS NOT NULL
     AND EXISTS (SELECT 1 FROM cockpit_turn_sighting s
                  WHERE s.correlation_id = c.correlation_id AND s.stage = 'motor_hop'
                    AND s.raw->'step'->'raw'->'message'->'content' @> '[{{"type": "tool_use"}}]')
   ORDER BY c.created_at DESC
   LIMIT {n}
)
SELECT s.correlation_id, s.seq, t.created_at AS turn_at, s.ts,
       {content} AS content
  FROM cockpit_turn_sighting s JOIN turns t USING (correlation_id)
 WHERE s.stage = 'motor_hop'
   AND s.raw->'step'->'raw'->>'type' IN ('assistant', 'user')
 ORDER BY t.created_at, s.seq
"""


def build_sql(session_id: str, turns: int) -> str:
    if not SESSION_RE.fullmatch(session_id or ""):
        raise ValueError(f"chat id must be a plain id (letters, digits, - _), got {session_id!r}")
    return SQL.format(sid=session_id, n=max(1, min(int(turns), MAX_TURNS)), content=CONTENT_SQL)


@dataclass
class Step:
    correlation_id: str
    turn_at: str
    ts: str
    tool: str
    action: str
    status: str = "no result seen"
    result: str = ""


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def action_of(name: str, inp: dict[str, Any]) -> str:
    for key in ("command", "file_path", "path", "pattern", "url", "query"):
        if inp.get(key):
            return _one_line(str(inp[key]), ACTION_CHARS)
    return _one_line(json.dumps(inp, sort_keys=True, default=str), ACTION_CHARS) if inp else ""


def result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def parse_rows(rows: Iterable[dict[str, str]]) -> list[Step]:
    """tool_use blocks in order, each joined to its tool_result by id."""
    steps: list[Step] = []
    by_id: dict[str, Step] = {}
    for row in rows:
        try:
            content = json.loads(row.get("content") or "null")
        except ValueError:
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                step = Step(row.get("correlation_id", ""), row.get("turn_at", ""), row.get("ts", ""),
                            str(block.get("name") or "?"), action_of(str(block.get("name") or ""), inp))
                steps.append(step)
                if block.get("id"):
                    by_id[str(block["id"])] = step
            elif block.get("type") == "tool_result":
                step = by_id.get(str(block.get("tool_use_id") or ""))
                if step is None:
                    continue
                step.status = "failed" if block.get("is_error") else "ok"
                step.result = result_text(block.get("content"))
    return steps


def render(steps: list[Step], session_id: str) -> list[str]:
    if not steps:
        return [f"no recorded tool steps in chat {session_id} (a turn may have only replied)"]
    lines: list[str] = []
    current = None
    for i, s in enumerate(steps, 1):
        if s.correlation_id != current:
            current = s.correlation_id
            lines.append(f"turn {s.turn_at[:16]}Z (corr {s.correlation_id[:8]}):")
        lines.append(f"  {i}. {s.ts[11:16]}Z {s.tool}: {s.action}")
        lines.append(f"     -> {s.status}: {_one_line(s.result, RESULT_CHARS)}")
    return lines
