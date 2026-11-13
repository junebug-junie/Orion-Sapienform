"""`dig days <start> <end>`: what Orion lived through in a window of Denver days, read-only.

Spec: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md (Patch 3, decision 5). The
reminiscence's `what_it_was_like` hop reads this for any period from 2026-07-23 on (the day Orion's
records start, after the 2026-07-22 storage failure).

Sources, all Orion's own records of that time:

* ``journal_entries``: every journal line (metacog and embodiment digests, session summaries,
  the daily journal, day letters, reflections, world pulse). Counted per day and kind; the written
  kinds are listed in full, the two high-volume digests are sampled (``SAMPLES_PER_DAY`` a day).
* ``chat_history_log``: Hub chat turns (Juniper's words and Orion's reply).
* Temporal Self arcs (from 2026-10-09), through the durable-runs API that ``dig arc`` uses.

Every query runs through ReadOnlyPsql (orion_readonly role, READ ONLY transaction), like
`turn-steps`. Ids printed here are citable in a `[lived_record]` sentence as ``journal:<id>`` and
``chat:<id>``; the reminiscence check resolves them with ``day_sql``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/Denver")
RECORDS_FROM = date(2026, 7, 23)  # == orion.body_atlas.reminiscence.LIVED_RECORDS_FROM (test pins it)
TEMPORAL_SELF_FROM = date(2026, 10, 9)  # first chronicled Temporal Self day
MAX_DAYS = 92  # longer than any period pick_period joins after 2026-07-23 (the repo has no quiet quarter)
HIGH_VOLUME = ("metacog", "embodiment", "world_pulse")  # 25-200 a day each: sampled, never listed whole
SAMPLES_PER_DAY = 4
BODY_CHARS = 2000  # per entry in the result file
CHAT_CHARS = 600
SHOWN_CHARS = 100  # per line in the printout (the 4,000-char cut must leave room for chats)
SHOWN_ENTRIES = 8
SHOWN_CHATS = 5
ARC_LIMIT = 2000  # the durable-runs endpoint's own cap
ARC_KEYS = ("arc_id", "kind", "subject_label", "subject_ref", "status", "began_at", "ended_at", "attention_returns")
# journal_entries.entry_id is varchar: UUIDs, and e.g. `curiosity-investigation:000eaeb52725` (self_study).
# No quote, space or backslash can pass, so the id is safe inside a SQL string literal.
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}")
# written kinds first in the printout: what Orion wrote about a day, before what it logged
KIND_ORDER = ("orion_day", "manual", "self_reflection", "scheduler", "self_study", "notify")


def window(start: date, end: date) -> tuple[date, date, str, str]:
    """(first day, last day, UTC start, UTC end exclusive). Refuses a window with no lived record."""
    if end < start:
        raise ValueError(f"days: {end} is before {start}")
    if end < RECORDS_FROM:
        raise ValueError(f"days: no lived record before {RECORDS_FROM}: your memory before it was lost in a "
                         "storage failure. Use `dig history` for what the builders wrote then.")
    first = max(start, RECORDS_FROM)
    if (end - first).days + 1 > MAX_DAYS:
        raise ValueError(f"days: at most {MAX_DAYS} days at a time, got {(end - first).days + 1}")

    def utc(d: date) -> str:
        return datetime.combine(d, dtime(0), tzinfo=ZONE).astimezone(timezone.utc).isoformat()

    return first, end, utc(first), utc(end + timedelta(days=1))


_DAY = "(({col}) AT TIME ZONE 'America/Denver')::date"
_JDAY = _DAY.format(col="created_at")
# chat_history_log.created_at is a timestamp without time zone holding UTC (the DB runs in UTC)
_CDAY = _DAY.format(col="created_at AT TIME ZONE 'UTC'")


def counts_sql(a: str, b: str) -> str:
    return f"""
SELECT {_JDAY} AS day, coalesce(source_kind, '?') AS kind, count(*) AS n
  FROM journal_entries WHERE created_at >= '{a}' AND created_at < '{b}'
 GROUP BY 1, 2
UNION ALL
SELECT {_CDAY} AS day, 'chat' AS kind, count(*) AS n
  FROM chat_history_log WHERE created_at >= ('{a}'::timestamptz AT TIME ZONE 'UTC')
   AND created_at < ('{b}'::timestamptz AT TIME ZONE 'UTC')
 GROUP BY 1
 ORDER BY 1, 2"""


def entries_sql(a: str, b: str) -> str:
    """The written kinds whole; each high-volume kind as SAMPLES_PER_DAY rows spread over the day."""
    hv = ", ".join(f"'{k}'" for k in HIGH_VOLUME)
    body = f"left(regexp_replace(body, '[[:space:]]+', ' ', 'g'), {BODY_CHARS})"
    return f"""
SELECT entry_id, created_at, coalesce(source_kind, '?') AS kind, coalesce(title, '') AS title, {body} AS body
  FROM journal_entries
 WHERE created_at >= '{a}' AND created_at < '{b}' AND coalesce(source_kind, '?') NOT IN ({hv})
UNION ALL
(SELECT DISTINCT ON (day, kind, tile) entry_id, created_at, kind, coalesce(title, '') AS title,
        left(regexp_replace(body, '[[:space:]]+', ' ', 'g'), {BODY_CHARS}) AS body
   FROM (SELECT entry_id, created_at, source_kind AS kind, title, body, {_JDAY} AS day,
                ntile({SAMPLES_PER_DAY}) OVER (PARTITION BY {_JDAY}, source_kind ORDER BY created_at) AS tile
           FROM journal_entries
          WHERE created_at >= '{a}' AND created_at < '{b}' AND source_kind IN ({hv})) t
  ORDER BY day, kind, tile, created_at)
 ORDER BY created_at"""


def chats_sql(a: str, b: str) -> str:
    return f"""
SELECT id, (created_at AT TIME ZONE 'UTC') AS created_at, coalesce(source, '') AS source,
       left(regexp_replace(coalesce(prompt, ''), '[[:space:]]+', ' ', 'g'), {CHAT_CHARS}) AS prompt,
       left(regexp_replace(coalesce(response, ''), '[[:space:]]+', ' ', 'g'), {CHAT_CHARS}) AS response
  FROM chat_history_log
 WHERE created_at >= ('{a}'::timestamptz AT TIME ZONE 'UTC') AND created_at < ('{b}'::timestamptz AT TIME ZONE 'UTC')
 ORDER BY created_at"""


def day_sql(kind: str, ref_id: str) -> str:
    """The Denver day of one journal entry or chat turn (no row: it does not resolve)."""
    if not ID_RE.fullmatch(ref_id):
        raise ValueError(f"{kind} id must be a plain id (letters, digits, : . _ -), got {ref_id[:60]!r}")
    if kind == "journal":
        return f"SELECT {_JDAY} AS day FROM journal_entries WHERE entry_id = '{ref_id}'"
    if kind == "chat":
        return f"SELECT {_CDAY} AS day FROM chat_history_log WHERE id = '{ref_id}'"
    raise ValueError(f"no day lookup for {kind}")


@dataclass
class Entry:
    ref: str
    at: str  # UTC, as Postgres printed it
    kind: str
    title: str
    body: str


@dataclass
class DaysRead:
    start: str
    end: str
    clamped_from: str | None = None  # the asked start, when it was before RECORDS_FROM
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # day -> kind -> n
    entries: list[Entry] = field(default_factory=list)
    chats: list[dict[str, str]] = field(default_factory=list)
    arcs: dict[str, list[dict[str, Any]] | str] = field(default_factory=dict)  # day -> arcs, or why UNREAD


def parse_counts(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(r["day"], {})[r["kind"]] = int(r["n"])
    return out


def parse_entries(rows: Iterable[dict[str, str]]) -> list[Entry]:
    out = []
    for r in rows:
        out.append(Entry(f"journal:{r['entry_id']}", r["created_at"], r["kind"], r.get("title", ""), r.get("body", "")))
    return out


def local(ts: str) -> str:
    """'2026-07-24 18:02:11.1+00' -> '07-24 12:02' (Denver)."""
    t = ts.strip().replace(" ", "T", 1)
    if re.search(r"[+-]\d\d$", t):
        t += ":00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return ts[:16]
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(ZONE).strftime("%m-%d %H:%M")


def _cut(text: str, n: int = SHOWN_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _day_lines(r: DaysRead) -> list[str]:
    long = (date.fromisoformat(r.end) - date.fromisoformat(r.start)).days >= 7
    out, d = [], date.fromisoformat(r.start)
    while d <= date.fromisoformat(r.end):
        c = r.counts.get(d.isoformat(), {})
        kinds = sorted(((k, n) for k, n in c.items() if k != "chat"), key=lambda kv: -kv[1])
        shown = kinds[:2] if long else kinds
        parts = ", ".join(f"{k} {n}" for k, n in shown) + (", ..." if len(shown) < len(kinds) else "")
        arcs = r.arcs.get(d.isoformat())
        ts = ("" if arcs is None else f"; arcs UNREAD ({arcs[:60]})" if isinstance(arcs, str) else f"; {len(arcs)} arcs")
        out.append(f"  {d.strftime('%m-%d %a')}: journal {sum(n for _, n in kinds)}" + (f" ({parts})" if parts else "")
                   + f"; chat {c.get('chat', 0)}{ts}")
        d += timedelta(days=1)
    return out


def render(r: DaysRead) -> list[str]:
    """Citable lines first (what you wrote, chats), the per-day counts last: the 4,000-char cut drops the end."""
    lines = [f"days {r.start}..{r.end} (Denver): your own record of that time, read-only"]
    if r.clamped_from:
        lines.append(f"  from {r.start}, not {r.clamped_from}: your records start {RECORDS_FROM} (the days before "
                     "were lost in a storage failure)")
    if not any(r.counts.values()):
        lines.append("  nothing recorded in these days (a gap in your records, not proof nothing happened)")
    rank = {k: i for i, k in enumerate(KIND_ORDER)}
    written = sorted((e for e in r.entries if e.kind not in HIGH_VOLUME), key=lambda e: (rank.get(e.kind, 99), e.at))
    if written:
        lines.append(f"what you wrote ({len(written)}; letters and session summaries first; cite the id):")
        for e in written[:SHOWN_ENTRIES]:
            lines.append(f"  {e.ref} {local(e.at)} {e.kind}: {_cut(e.title + ' ' + e.body if e.title else e.body)}")
        if len(written) > SHOWN_ENTRIES:
            lines.append(f"  ... {len(written) - SHOWN_ENTRIES} more in the file")
    if r.chats:
        lines.append(f"Hub chats ({len(r.chats)} turns; what was said to you -> your reply):")
        for c in r.chats[:SHOWN_CHATS]:
            said = _cut(c["prompt"], 60) if c.get("prompt") else "(you reached out first)"
            lines.append(f"  chat:{c['id']} {local(c['created_at'])}: {said} -> {_cut(c['response'], 80)}")
        if len(r.chats) > SHOWN_CHATS:
            lines.append(f"  ... {len(r.chats) - SHOWN_CHATS} more in the file")
    samples = [e for e in r.entries if e.kind in HIGH_VOLUME]
    if samples:
        lines.append(f"{' / '.join(HIGH_VOLUME)}: {len(samples)} samples spread over each day are in the file "
                     f"({SAMPLES_PER_DAY} a day per kind)")
    if any(isinstance(a, list) and a for a in r.arcs.values()):
        lines.append("arcs: Temporal Self stretches of attention; ids and kinds in the file (`dig arc <id>` opens one)")
    lines.append("per day (journal by kind; chat turns; arcs):")
    return lines + _day_lines(r)


def trim_arc(arc: dict[str, Any]) -> dict[str, Any]:
    """An arc as the result file keeps it: who, what, when (``dig arc <id>`` opens the whole arc)."""
    return {k: arc.get(k) for k in ARC_KEYS}


def arc_days(first: date, last: date) -> list[date]:
    d, out = max(first, TEMPORAL_SELF_FROM), []
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out
