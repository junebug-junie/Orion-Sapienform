"""`dig digs <date>`: every dig command Orion actually ran on one Denver day, read-only.

Spec: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 1. The nightly day-dream reads the
day's digs from here. No model call.

One line per executed ``python -m orion.dig ...`` tool call: the time (Denver), where it ran (a Hub chat
session, or a background workflow), the command, the first line of its result, and a ref. Only tool_use
blocks count. ``orion.dig`` text inside a prompt (the dig help in every motor prompt: ~200 hits a day) is
never a command, so a plain text search is wrong here.

Two sources, joined by tool_use id (the same id is in both for a chat turn):

* ``cockpit_turn_sighting`` ``motor_hop`` rows: chat turns' tool blocks (same shape ``dig turn-steps`` reads),
  session through ``chat_history_log.correlation_id``. One read-only SELECT through ReadOnlyPsql.
* The governor's ``claude`` transcripts (``session_resume.claude_home()/projects/*/<session>.jsonl``): the only
  record of a background run's tool calls. A durable run passes no step queue, so it writes no ``motor_hop``
  rows, and ``harness_turn_trace.run_artifact`` holds no tool steps (checked live 2026-11-17: no key of it
  holds a tool_use; same finding as ``dig take-ons``, #3639). The run's correlation id is the
  ``Strain refs: hub:turn:<id>`` line its own motor prompt carries; its workflow is then
  ``harness_turn_trace.workflow`` (a second read-only SELECT). A durable run's step prompt carries no such
  line; its transcript name is ``session_resume.session_id_for(f"{run_id}:{source_tag}", generation)`` (Hub's
  resume key, as ``dig take-ons`` derives it), so it is matched exactly through Hub's resume registry
  (``<claude_home>/orion-resume/*.json``: every key and its last generation); the key's source tag is the
  workflow. Plain file reads.

The line rule: every printed line is a time, a dig command and a ref (``check_lines``; ``render`` runs it on
its own output). A source that cannot be read is said in the CLI tail as UNREAD, never shown as 0.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from orion.dig.organs import ZONE, day_bounds

# Text that looks like a dig invocation (fast prefilter, and what a printed line must show).
DIG_RE = re.compile(r"(?<![\w.])-m\s+orion\.dig(?![\w.])")
# is_dig_command then asks the shell question: is `python ... -m orion.dig` a command the line runs, or only
# words inside a quoted string or a heredoc (a note written with printf/cat, live 2026-11-17)?
_PYTHON_RE = re.compile(r"(?:.*/)?python(?:\d+(?:\.\d+)*)?")
# A heredoc body, only when its terminator line exists (so `1 << shift` in Python code is not one).
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n.*?\n\s*\2[ \t]*(?:\n|$)", re.S)
_SHELLS = frozenset({"bash", "sh", "zsh", "/bin/bash", "/bin/sh", "/usr/bin/bash", "/usr/bin/zsh"})
# What may stand before a command word: an operator, a keyword, or a wrapper (with its options and values).
_OPS = frozenset({";", "&", "&&", "|", "||", "|&", "(", ")", ";;", "!", "{", "do", "then", "else", "elif", "if",
                  "while", "until", "time"})
_WRAPPERS = frozenset({"env", "timeout", "nice", "nohup", "exec", "sudo", "command", "stdbuf", "ionice"})
_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.S)
_WRAPPER_ARG_RE = re.compile(r"-.*|\d+(?:\.\d+)?[smhd]?|[A-Z]+", re.S)  # `timeout -s KILL 120`, `nice -n 5`
# The run's own correlation id in its motor prompt (orion/harness/prefix.py: "Strain refs: ...";
# orion/thought/coalition.py anchors a stance to `hub:turn:<correlation_id>`).
STRAIN_REF_RE = re.compile(r"Strain refs:[^\n]*?hub:turn:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
COMMAND_CHARS = 100
RESULT_CHARS = 80
RESULT_GRACE_HOURS = 2  # a call at 23:59 may get its result after midnight
REF_KINDS = ("chat", "harness_turn_trace", "durable_run", "transcript", "cockpit_turn_sighting")

LINE_RE = re.compile(
    r"^\d\d:\d\d:\d\d \S.*? \$ .*(?<![\w.])-m\s+orion\.dig(?![\w.]).* "
    r"\[(?:" + "|".join(REF_KINDS) + r"):[^\]\s]+\]$"
)

# Chat turns' dig calls, each with its result (a later motor_hop of the same turn) and its chat row.
# ts is a timestamp without time zone holding UTC (dig turn-steps prints it as Z).
COCKPIT_SQL = """
WITH hop AS (
  SELECT s.correlation_id, s.ts, b
    FROM cockpit_turn_sighting s,
         jsonb_array_elements(CASE jsonb_typeof(s.raw->'step'->'raw'->'message'->'content')
                                WHEN 'array' THEN s.raw->'step'->'raw'->'message'->'content'
                                ELSE '[]'::jsonb END) b
   WHERE s.stage = 'motor_hop'
     AND s.ts >= ('{lo}'::timestamptz AT TIME ZONE 'UTC')
     AND s.ts < ('{hi}'::timestamptz AT TIME ZONE 'UTC') + interval '{grace} hours'
     AND b->>'type' IN ('tool_use', 'tool_result')
), used AS (
  SELECT correlation_id, ts, b->>'id' AS id, b->'input'->>'command' AS command
    FROM hop
   WHERE b->>'type' = 'tool_use' AND b->'input'->>'command' LIKE '%orion.dig%'
     AND ts < ('{hi}'::timestamptz AT TIME ZONE 'UTC')
)
SELECT u.ts, u.correlation_id, coalesce(u.id, '') AS id, left(u.command, 2000) AS command,
       (r.b IS NOT NULL) AS has_result, coalesce(r.b->>'is_error', 'false') AS is_error,
       left(CASE jsonb_typeof(r.b->'content')
              WHEN 'string' THEN r.b->>'content'
              WHEN 'array' THEN (SELECT string_agg(x->>'text', ' ') FROM jsonb_array_elements(r.b->'content') x)
            END, 600) AS result,
       coalesce(c.id, '') AS chat_id, coalesce(c.session_id, '') AS session_id
  FROM used u
  LEFT JOIN LATERAL (SELECT h.b FROM hop h
                      WHERE h.correlation_id = u.correlation_id AND h.b->>'type' = 'tool_result'
                        AND h.b->>'tool_use_id' = u.id
                      LIMIT 1) r ON true
  LEFT JOIN LATERAL (SELECT id, session_id FROM chat_history_log c
                      WHERE c.correlation_id = u.correlation_id ORDER BY created_at LIMIT 1) c ON true
 ORDER BY u.ts
"""

# Where a transcript's run ran: a Hub chat turn, or a harness run's workflow.
LOOKUP_SQL = """
SELECT t.corr AS correlation_id, coalesce(h.workflow, '') AS workflow, (h.correlation_id IS NOT NULL) AS traced,
       coalesce(c.id, '') AS chat_id, coalesce(c.session_id, '') AS session_id
  FROM unnest(ARRAY[{ids}]::text[]) t(corr)
  LEFT JOIN harness_turn_trace h ON h.correlation_id = t.corr
  LEFT JOIN LATERAL (SELECT id, session_id FROM chat_history_log c
                      WHERE c.correlation_id = t.corr ORDER BY created_at LIMIT 1) c ON true
"""


def _command_position(tokens: list[str], i: int) -> bool:
    """Is tokens[i] a command word: first, after an operator/keyword, past assignments and wrappers?"""
    j = i - 1
    while j >= 0:
        tok = tokens[j]
        if _ASSIGN_RE.fullmatch(tok) or tok in _WRAPPERS:
            j -= 1
        elif _WRAPPER_ARG_RE.fullmatch(tok) and any(t in _WRAPPERS for t in tokens[max(0, j - 4):j]):
            j -= 1
        else:
            break
    return j < 0 or tokens[j] in _OPS or set(tokens[j]) <= set(";&|()")


def _invocation(tokens: list[str], depth: int) -> Optional[str]:
    for i, tok in enumerate(tokens):
        if not _command_position(tokens, i):
            continue
        if _PYTHON_RE.fullmatch(tok):
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-") and tokens[j] != "-m":
                j += 2 if tokens[j] in ("-X", "-W") else 1  # options that take a value
            if j + 1 < len(tokens) and tokens[j] == "-m" and tokens[j + 1] == "orion.dig":
                end = j + 2  # up to the next operator or redirection (`2>&1` lexes as 2 >& 1: drop the 2)
                while end < len(tokens) and not set(tokens[end]) <= set(";&|()<>"):
                    end += 1
                if end < len(tokens) and set(tokens[end]) & set("<>") and tokens[end - 1].isdigit() and end - 1 > j + 1:
                    end -= 1
                return shlex.join(tokens[i:end])
        # `bash -c "python -P -m orion.dig ..."` (or -lc) runs it too
        if depth < 2 and tok in _SHELLS:
            for k in range(i + 1, min(len(tokens) - 1, i + 4)):
                flag = tokens[k]
                if not flag.startswith("-") or flag.startswith("--"):
                    break
                if "c" in flag[1:]:
                    inner = dig_invocation(tokens[k + 1], depth + 1)
                    if inner:
                        return inner
                    break
    return None


def dig_invocation(command: Any, depth: int = 0) -> Optional[str]:
    """The ``python [opts] -m orion.dig ...`` command a shell line runs (re-quoted, up to the next operator), or
    None when the dig words only sit inside a quoted string (``printf '... python -P -m orion.dig help ...'``),
    a heredoc body, a comment, or after a non-command word (``echo python -P -m orion.dig ...``).

    Known misses (documented in tests): combined flags ``python -Pm orion.dig`` and a quoted module name."""
    if not isinstance(command, str) or DIG_RE.search(command) is None:
        return None
    text = _HEREDOC_RE.sub("\n", command).replace("\n", "\n;")  # a newline ends a command (and a comment)
    if DIG_RE.search(text) is None:
        return None
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = "#"
        tokens = list(lex)
    except ValueError:  # unbalanced quotes: the shell would refuse it too
        return None
    return _invocation(tokens, depth)


def is_dig_command(command: Any) -> bool:
    return dig_invocation(command) is not None


def window(day: date) -> tuple[datetime, datetime]:
    """The Denver day as a half-open UTC window (organs.day_bounds: one definition of the day)."""
    lo, hi = day_bounds(day)
    return datetime.fromisoformat(lo), datetime.fromisoformat(hi)


def cockpit_sql(day: date) -> str:
    lo, hi = day_bounds(day)
    return COCKPIT_SQL.format(lo=lo, hi=hi, grace=RESULT_GRACE_HOURS)


def lookup_sql(correlation_ids: Iterable[str]) -> Optional[str]:
    ids = sorted({c for c in correlation_ids if UUID_RE.fullmatch(c or "")})
    if not ids:
        return None
    return LOOKUP_SQL.format(ids=", ".join(f"'{c}'" for c in ids))


# --------------------------------------------------------------------------- one call


@dataclass
class Call:
    at: str  # ISO UTC
    tool_use_id: str
    command: str
    result: Optional[str] = None  # None: no tool_result recorded
    is_error: bool = False
    source: str = ""  # cockpit | transcript
    correlation_id: str = ""
    transcript: str = ""  # claude session id (transcript file stem)
    chat_id: str = ""
    session_id: str = ""
    workflow: str = ""
    traced: bool = False
    run_id: str = ""  # a durable run matched by its resume key
    chat_pool: bool = False  # its transcript is a warm-pool chat session (reset by /clear): no turn id in it

    @property
    def needs_link(self) -> bool:
        """A transcript call whose chat or workflow is still unknown (a traced row may lack a workflow)."""
        return self.source == "transcript" and not self.chat_id and not self.workflow

    @property
    def where(self) -> str:
        if self.chat_id:
            return f"chat {self.session_id[:8] or '?'}"
        if self.workflow:
            return self.workflow
        if self.traced:
            return "run (no workflow)"
        if self.source == "cockpit":
            return "turn (no chat row)"
        if self.chat_pool:
            return "chat (no turn id)"
        return "unlinked run"

    @property
    def ref(self) -> str:
        if self.chat_id:
            return f"[chat:{self.chat_id}]"
        if self.traced and self.correlation_id:
            return f"[harness_turn_trace:{self.correlation_id}]"
        if self.run_id:
            return f"[durable_run:{self.run_id}]"
        if self.transcript:
            return f"[transcript:{self.transcript}]"
        return f"[cockpit_turn_sighting:{self.correlation_id or 'unknown'}]"


def _utc(text: str) -> Optional[datetime]:
    t = (text or "").strip().replace(" ", "T", 1)
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    if re.search(r"[+-]\d\d$", t):
        t += ":00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def result_text(content: Any) -> str:
    """A tool_result's content: a string, or an array of {text} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " \n".join(str(x.get("text") or "") for x in content if isinstance(x, dict) and x.get("text"))
    return ""


def first_line(text: Optional[str]) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


# --------------------------------------------------------------------------- source 1: cockpit rows


def parse_cockpit_rows(rows: Iterable[Mapping[str, Any]]) -> list[Call]:
    out: list[Call] = []
    for r in rows:
        command = str(r.get("command") or "")
        at = _utc(str(r.get("ts") or ""))
        if not is_dig_command(command) or at is None:
            continue
        has = str(r.get("has_result") or "").lower() in ("t", "true", "1")
        out.append(Call(
            at=at.isoformat(), tool_use_id=str(r.get("id") or ""), command=command,
            result=str(r.get("result") or "") if has else None,
            is_error=str(r.get("is_error") or "").lower() in ("t", "true", "1"),
            source="cockpit", correlation_id=str(r.get("correlation_id") or ""),
            chat_id=str(r.get("chat_id") or ""), session_id=str(r.get("session_id") or "")))
    return out


# --------------------------------------------------------------------------- source 2: transcripts


def _blocks(event: Mapping[str, Any]) -> list[Any]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return content if isinstance(content, list) else []


def _prompt_text(event: Mapping[str, Any]) -> str:
    message = event.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text") or "") for b in content if isinstance(b, Mapping) and b.get("type") == "text")
    return ""


_TS_RE = re.compile(r'"timestamp":\s*"([^"]+)"')
_ASSISTANT_RE = re.compile(r'"type":\s*"assistant"')


def _iso_z(dt: datetime) -> str:
    """The transcript's own timestamp form (``2026-11-16T09:08:55.118Z``): string order = time order."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def scan_transcript(lines: Iterable[str], lo: datetime, hi: datetime, transcript: str) -> list[Call]:
    """Dig tool calls made in [lo, hi) in one claude transcript, each with its result when recorded.

    Only assistant tool_use blocks count; a user prompt that quotes a dig command (the help text) does not.
    A sub-agent's sidechain events are not the run's own calls."""
    calls: dict[str, Call] = {}
    order: list[Call] = []
    corr = ""
    pool = False
    lo_s, stop_s = _iso_z(lo), _iso_z(hi + timedelta(hours=RESULT_GRACE_HOURS))
    for line in lines:
        if "tool_" not in line and "Strain refs:" not in line and "<command-name>/clear<" not in line:
            continue
        m_ts = _TS_RE.search(line)
        if m_ts:
            if m_ts.group(1) >= stop_s:
                break  # transcripts are append-ordered: nothing later can be in the day or answer a call in it
            if m_ts.group(1) < lo_s and _ASSISTANT_RE.search(line):
                continue  # a call before the day (user lines still read: turn ids, /clear)
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("isSidechain"):
            continue
        kind = event.get("type")
        if kind == "user":
            prompt = _prompt_text(event)
            m = STRAIN_REF_RE.search(prompt)
            if m:
                corr = m.group(1)
            if "<command-name>/clear</command-name>" in prompt:
                # the warm chat pool's reset handshake (orion/harness/fcc_motor.py): this is a chat session,
                # and its turn prompts carry no turn id; a later turn must not inherit an earlier one's
                pool, corr = True, ""
            for b in _blocks(event):
                if isinstance(b, Mapping) and b.get("type") == "tool_result":
                    call = calls.get(str(b.get("tool_use_id") or ""))
                    if call is not None and call.result is None:
                        call.result = result_text(b.get("content"))
                        call.is_error = bool(b.get("is_error"))
        elif kind == "assistant":
            at = _utc(str(event.get("timestamp") or ""))
            if at is None or not (lo <= at < hi):
                continue
            for b in _blocks(event):
                if not isinstance(b, Mapping) or b.get("type") != "tool_use":
                    continue
                inp = b.get("input") if isinstance(b.get("input"), Mapping) else {}
                if not is_dig_command(inp.get("command")):
                    continue
                tid = str(b.get("id") or "")
                if tid and tid in calls:
                    continue  # a resumed session may repeat its history
                call = Call(at=at.isoformat(), tool_use_id=tid, command=str(inp["command"]), source="transcript",
                            correlation_id=corr, transcript=transcript, chat_pool=pool)
                order.append(call)
                if tid:
                    calls[tid] = call
    return order


def candidate_transcripts(home: Path, lo: datetime, hi: datetime) -> list[Path]:
    """Transcripts that may hold events in [lo, hi): written to at or after lo (mtime) and begun before hi (first
    line's timestamp); the rest are never opened."""
    out: list[Path] = []
    projects = home / "projects"
    lo_epoch = lo.timestamp()
    for project in sorted(projects.iterdir()) if projects.is_dir() else []:
        if not project.is_dir():
            continue
        for f in os.scandir(project):
            if not f.name.endswith(".jsonl") or not f.is_file():
                continue
            if f.stat().st_mtime < lo_epoch:
                continue
            try:
                with open(f.path, encoding="utf-8", errors="replace") as fh:
                    head = fh.readline(65536)
            except OSError:
                continue
            m = _TS_RE.search(head)
            began = _utc(m.group(1)) if m else None
            if began is not None and began >= hi:
                continue
            out.append(Path(f.path))
    return out


def read_transcripts(paths: Iterable[Path], lo: datetime, hi: datetime) -> list[Call]:
    """Streamed line by line; a file stops being read past the day (plus the result grace)."""
    out: list[Call] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                out += scan_transcript(fh, lo, hi, p.stem)
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- join


def merge(cockpit: list[Call], transcript: list[Call]) -> list[Call]:
    """One call per tool_use id; a chat turn's cockpit row wins (it knows the chat session)."""
    seen: set[str] = set()
    out: list[Call] = []
    for c in [*cockpit, *transcript]:  # cockpit first: a duplicated hop row or a transcript copy is dropped
        if c.tool_use_id and c.tool_use_id in seen:
            continue
        if c.tool_use_id:
            seen.add(c.tool_use_id)
        out.append(c)
    return sorted(out, key=lambda c: (c.at, c.tool_use_id))


def attach(calls: list[Call], rows: Iterable[Mapping[str, Any]]) -> None:
    """Fill where a transcript call ran from LOOKUP_SQL rows (chat row first, then the harness trace)."""
    by = {str(r.get("correlation_id") or ""): r for r in rows}
    for c in calls:
        r = by.get(c.correlation_id)
        if c.source != "transcript" or r is None:
            continue
        c.chat_id = str(r.get("chat_id") or "")
        c.session_id = str(r.get("session_id") or "")
        c.workflow = str(r.get("workflow") or "")
        c.traced = str(r.get("traced") or "").lower() in ("t", "true", "1")


MAX_GENERATIONS = 64  # resume generations per key (session_resume bumps `<key>#<n>`; the registry names the last)


def read_resume_registry(home: Path) -> list[dict[str, Any]]:
    """Hub's resume registry (``orion/harness/session_resume.py``: one ``{"key", "generation"}`` file per durable
    step under ``<claude_home>/orion-resume/``). Plain reads; an unreadable file is skipped."""
    from orion.harness.session_resume import _REGISTRY_DIRNAME

    out: list[dict[str, Any]] = []
    folder = home / _REGISTRY_DIRNAME
    for p in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            got = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(got, dict) and isinstance(got.get("key"), str):
            out.append(got)
    return out


def link_by_resume_key(calls: list[Call], records: Iterable[Mapping[str, Any]],
                       session_id_for: Optional[Callable[[str, int], str]] = None) -> int:
    """Name the durable run of each transcript call still unlinked. A durable step's transcript is
    ``session_id_for(resume_key, generation)`` and Hub's key is ``<run_id>:<source_tag>[:<more>]``
    (``<run_id>:intention_carry:<concern>``, ``<run_id>:<self_fit_tag>:<field>``); the registry holds every key
    and its last generation, so the match is exact. The source tag is the workflow. Returns calls linked."""
    if session_id_for is None:
        from orion.harness.session_resume import session_id_for
    todo = {c.transcript for c in calls if c.needs_link and c.transcript}
    if not todo:
        return 0
    found: dict[str, str] = {}
    for r in records:
        key = str(r.get("key") or "")
        gen = r.get("generation")
        last = gen if isinstance(gen, int) and 0 <= gen < MAX_GENERATIONS else 0
        for g in range(last + 1):
            sid = session_id_for(key, g)
            if sid in todo:
                found[sid] = key
    linked = 0
    for c in calls:
        key = found.get(c.transcript) if c.needs_link else None
        if key and ":" in key:
            run_id, rest = key.split(":", 1)
            c.run_id, c.workflow = run_id, rest.split(":", 1)[0]
            linked += 1
    return linked


# --------------------------------------------------------------------------- the line rule and the page


def check_lines(lines: Iterable[str]) -> None:
    """Raise ValueError on the first line that is not a time, a dig command and a ref."""
    for i, line in enumerate(lines):
        if not LINE_RE.match(line):
            raise ValueError(f"digs line {i} is not a time + dig command + ref: {line[:120]!r}")


def _command(cmd: str) -> str:
    """The dig command the line ran (not a quoted note before it), re-quoted, up to the next shell operator."""
    shown = " ".join((dig_invocation(cmd) or cmd).split())
    return shown if len(shown) <= COMMAND_CHARS else shown[: COMMAND_CHARS - 1] + "…"


def _result(c: Call) -> str:
    if c.result is None:
        return "(no result recorded)"
    text = first_line(c.result).replace('"', "'")
    if len(text) > RESULT_CHARS:
        text = text[: RESULT_CHARS - 1] + "…"
    return ("failed: " if c.is_error else "") + f'"{text}"'


def line_of(c: Call) -> str:
    at = _utc(c.at)
    hhmmss = at.astimezone(ZONE).strftime("%H:%M:%S") if at else "??:??:??"
    return f"{hhmmss} {c.where} $ {_command(c.command)} -> {_result(c)} {c.ref}"


def render(calls: list[Call]) -> list[str]:
    lines = [line_of(c) for c in calls]
    check_lines(lines)
    return lines


def to_file(day: date, calls: list[Call], unread: Mapping[str, str]) -> dict[str, Any]:
    return {"day": day.isoformat(), "zone": "America/Denver", "count": len(calls), "unread": dict(unread),
            "calls": [{**c.__dict__, "where": c.where, "ref": c.ref, "first_line": first_line(c.result)}
                      for c in calls]}
