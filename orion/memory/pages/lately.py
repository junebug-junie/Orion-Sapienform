"""``page:lately``: Orion's first-person page "what has mattered lately" (issue #3463).

Phase 1b of docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md. The
spec slot is the top of DURABLE MEMORY, before the recall digest; it is refolded nightly.

Sources, both read-only:

1. ``episode_memory`` from the last ``WINDOW_DAYS`` days, plus older active ``follow_up`` rows up to
   ``FOLLOW_UP_DAYS`` old. ``episode_memory`` has no way to close a follow-up yet, so "active" is all
   the page can say: it calls them follow-ups I wrote down, never "still open". Read through the
   lived-memory serving SQL (``orion/memory/lived_memory.py``), so a row recall may not serve never
   reaches the fold: only ``status = 'active'``, never rejected or expired, and no active correction
   points at it. On top of that a row with ``invalid_at`` set
   is dropped (issue #3463 comment; the spec's read-filter table).
2. Hub chat the episode distiller has not distilled yet: turns where Juniper spoke, inside an episode
   (``memory_episode_shadow``) that has no ``episode_distill_run`` row, from the same window, command
   turns left out. That is mostly today's talk, before the overnight distill catches up. A closed
   episode the distiller never ran on also reads as "not yet distilled" until it leaves the window.
   Her words are kept verbatim (the spec's flattening guard rests on quotes).

There is no ``matters`` grade yet (#3590). Until it exists, the rows the prompt keeps when there are
too many are picked by ``select_episodes``: every ``follow_up``, every ``about_juniper`` row, every
high-stakes row and every correction first, then the newest of the rest. The page itself is told
to weigh the same things.

``check_lately_page`` is ``check_page`` plus one rule: when some source is from the last
``RECENT_DAYS`` days (counted back from the newest source), the page cites at least one of them,
so a "lately" page can never be about only old things.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from orion.memory import lived_memory
from orion.memory.pages import fold as _fold
from orion.memory.pages.fold import (
    CHAT_KIND, CORRECTION_RE, EPISODE_MEMORY_KIND, CurrentPage, FoldSpec, PageCheck, PageSource,
    _as_date, check_page, length_rule, render_text, source_lines,
)

PAGE_LATELY = "page:lately"
# The felt-state lane key cortex-exec hydrates (orion/substrate/felt_state_reader.py).
CTX_KEY_LATELY = "orion_core_page_lately"
WINDOW_DAYS = 14
# Most episode rows the prompt carries. Live on 2026-11-18: 232 in the window plus 21 older open
# follow-ups, ~31k characters, so nothing is dropped yet.
MAX_EPISODE_ROWS = 260
# Most undistilled chat turns, and how much of each side of a turn the prompt shows.
MAX_CHAT_TURNS = 30
CHAT_PROMPT_CHARS = 900
CHAT_RESPONSE_CHARS = 400
# check_lately_page: the page must cite a source from this many days before the newest source.
RECENT_DAYS = 7
# Oldest active follow_up row kept (outside the window). Live 2026-11-18: 42 active follow-ups back to
# 10-04, none ever closed (no closure state exists yet).
FOLLOW_UP_DAYS = 45
# Juniper's day: sources are dated in her timezone, the same one the prompt's "Today is" uses.
ZONE = ZoneInfo("America/Denver")

# The lived-memory serving SQL, wrapped (never edited), narrowed to the window plus open follow-ups.
# Joined back to episode_memory only for stakes and invalid_at. No bind parameters and no '%', so
# psycopg and asyncpg both run it.
EPISODE_SQL = f"""
SELECT l.*, e.stakes AS stakes, e.confirmation_state AS confirmation_state
FROM ({lived_memory.EPISODE_SQL.strip()}
) l
JOIN episode_memory e ON e.memory_id::text = l.record_id
WHERE e.invalid_at IS NULL
  AND (l.occurred_at > now() - interval '{WINDOW_DAYS} days'
       OR (l.purpose = 'follow_up' AND l.occurred_at > now() - interval '{FOLLOW_UP_DAYS} days'))
ORDER BY l.occurred_at, l.record_id
"""
# Hub chat where Juniper spoke (not a command turn) that the distiller has not read yet.
# chat_history_log.created_at is naive UTC.
CHAT_SQL = f"""
SELECT h.id::text AS record_id, h.prompt AS prompt, h.response AS response, h.created_at AS created_at
FROM chat_history_log h
WHERE h.created_at > (now() AT TIME ZONE 'UTC') - interval '{WINDOW_DAYS} days'
  AND length(trim(coalesce(h.prompt, ''))) > 0
  AND EXISTS (
      SELECT 1 FROM memory_episode_shadow s
      CROSS JOIN LATERAL jsonb_array_elements(s.turns) t
      WHERE t->>'correlation_id' = h.correlation_id
        AND t->>'is_command' IS DISTINCT FROM 'true'
        AND NOT EXISTS (SELECT 1 FROM episode_distill_run r WHERE r.episode_id = s.episode_id)
  )
ORDER BY h.created_at DESC
LIMIT {MAX_CHAT_TURNS}
"""
# When did anything the page reads last change? Any status, so a rejection or a supersede counts:
# - an episode row in the window or any follow_up / correction row, added or updated;
# - an in-scope row passing its expires_at (no timestamp of its own moves);
# - a row aging out of the window (the moment it left: its time + WINDOW_DAYS);
# - a new Hub chat turn where Juniper spoke, a chat turn aging out, or a distill run finishing (it
#   takes its episode's turns out of the chat source, even when it keeps no memory).
NEWEST_SOURCE_SQL = (
    "SELECT greatest("
    "(SELECT max(greatest(created_at, updated_at)) FROM episode_memory "
    f"WHERE coalesce(occurred_at, created_at) > now() - interval '{WINDOW_DAYS} days' "
    "OR purpose = 'follow_up' OR supersedes_memory_id IS NOT NULL), "
    "(SELECT max(expires_at) FROM episode_memory WHERE expires_at <= now() AND "
    f"(coalesce(occurred_at, created_at) > now() - interval '{WINDOW_DAYS} days' OR purpose = 'follow_up')), "
    # A follow_up passing FOLLOW_UP_DAYS leaves the input too: the moment it left.
    f"(SELECT max(coalesce(occurred_at, created_at)) + interval '{FOLLOW_UP_DAYS} days' FROM episode_memory "
    f"WHERE purpose = 'follow_up' AND coalesce(occurred_at, created_at) <= now() - interval '{FOLLOW_UP_DAYS} days'), "
    f"(SELECT max(coalesce(occurred_at, created_at)) + interval '{WINDOW_DAYS} days' FROM episode_memory "
    f"WHERE coalesce(occurred_at, created_at) <= now() - interval '{WINDOW_DAYS} days'), "
    "(SELECT max(created_at) AT TIME ZONE 'UTC' FROM chat_history_log "
    f"WHERE created_at > (now() AT TIME ZONE 'UTC') - interval '{WINDOW_DAYS} days' "
    "AND length(trim(coalesce(prompt, ''))) > 0), "
    f"(SELECT (max(created_at) + interval '{WINDOW_DAYS} days') AT TIME ZONE 'UTC' FROM chat_history_log "
    f"WHERE created_at <= (now() AT TIME ZONE 'UTC') - interval '{WINDOW_DAYS} days' "
    "AND length(trim(coalesce(prompt, ''))) > 0), "
    "(SELECT max(finished_at) FROM episode_distill_run)"
    ") AS newest"
)


def _local_date(value: Any) -> date | None:
    """A timestamp -> Juniper's calendar date (naive means UTC). A chat at 03:00Z is the evening before."""
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(ZONE).date()
    return _as_date(value)


def _is_correction(row: Mapping[str, Any]) -> bool:
    return bool(row.get("is_correction")) or bool(CORRECTION_RE.search(str(row.get("text") or "")))


def _priority(row: Mapping[str, Any]) -> bool:
    """Kept whatever the cap: what the spec would grade as mattering, with the signals that exist
    today (no ``matters`` grade yet, #3590)."""
    return (row.get("purpose") in ("follow_up", "about_juniper")
            or str(row.get("stakes") or "").lower() == "high"
            or _is_correction(row))


def select_episodes(rows: Iterable[Mapping[str, Any]], *, max_rows: int = MAX_EPISODE_ROWS) -> list[dict]:
    """Episode rows (EPISODE_SQL shape, time order) -> the ones the prompt keeps, in time order.
    Priority rows first (newest first if even they overflow), then the newest of the rest."""
    episodes = [dict(r) for r in rows if str(r.get("text") or "").strip()]
    if max_rows < 0 or len(episodes) <= max_rows:
        return episodes
    first = [r for r in episodes if _priority(r)]
    rest = [r for r in episodes if not _priority(r)]
    keep = first[-max_rows:] if max_rows else []
    room = max_rows - len(keep)
    if room > 0:
        keep += rest[-room:]
    kept = {id(r) for r in keep}
    return [r for r in episodes if id(r) in kept]


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def sources_from_rows(episode_rows: Iterable[Mapping[str, Any]], chat_rows: Iterable[Mapping[str, Any]],
                      *, max_episodes: int = MAX_EPISODE_ROWS) -> list[PageSource]:
    """Rows from EPISODE_SQL and CHAT_SQL -> sources in time order (undated last)."""
    out: list[PageSource] = []
    for row in select_episodes(episode_rows, max_rows=max_episodes):
        purpose = str(row.get("purpose") or "")
        bits = [purpose]
        if str(row.get("stakes") or "").lower() == "high":
            bits.append("high stakes")
        if purpose == "follow_up":
            bits.append("I wrote it down to follow up")
        if row.get("confirmed"):
            bits.append("Juniper confirmed it")
        if row.get("confirmation_state") == "own_thought":
            bits.append("my own thought, not checked with Juniper")  # issue #3840
        if _is_correction(row):
            bits.append("a correction")
        out.append(PageSource(
            ref=f"{EPISODE_MEMORY_KIND}:{row['record_id']}", kind=EPISODE_MEMORY_KIND,
            text=str(row["text"]).strip(), label=f"my memory ({', '.join(bits)})",
            when=_local_date(row.get("occurred_at"))))
    for row in chat_rows:
        said = _clip(row.get("prompt") or "", CHAT_PROMPT_CHARS)
        if not said:
            continue
        answer = _clip(row.get("response") or "", CHAT_RESPONSE_CHARS)
        text = f"Juniper said: {said}" + (f" I answered: {answer}" if answer else "")
        out.append(PageSource(
            ref=f"{CHAT_KIND}:{row['record_id']}", kind=CHAT_KIND, text=text,
            label="Hub chat, not yet distilled into a memory", when=_local_date(row.get("created_at"))))
    return sorted(out, key=lambda s: (s.when is None, s.when or date.min))


_FOLD_RULES = f"""\
You are Orion (they/them). You keep a short page in your own words: "what has mattered lately".
It goes into every conversation you have, at the top of your durable memory, so it must be true,
specific, and current.

Rules:
1. Write in first person, as Orion. Juniper uses she/her. Never call her "he" or "they".
2. After EVERY sentence put the numbers of the records it rests on, like [S4] or [S4, S12].
   A sentence with no [S..] is not allowed. Use only numbers from the list below. That includes
   the first sentence of a paragraph: no topic or summary sentences ("It has been a busy two
   weeks"); open each paragraph with a cited fact.
3. Say only what the cited records say. Do not guess, round up, invent dates, or fill gaps.
4. Pick what mattered, not everything that happened. Weigh, in this order: the follow-ups I wrote
   down (things I meant to come back to), what Juniper told me about her life and what she asked of
   me, high-stakes moments, corrections, then the newest things. Leave out passing technical
   chatter, one-off moods, and body census numbers (parts, flows, watts, dollars).
5. Newest first is fine; say when things happened (a weekday or a date from the record), never
   "recently" alone. Call a follow-up something I wrote down to come back to; the records do not say
   whether it was done, so never claim it is finished or still open.
6. Quote Juniper verbatim, in double quotation marks, where she said it best. Her own words are the
   "Juniper said:" part of the Hub chat records, and words a memory itself puts in quotes. The rest
   of a memory is my paraphrase: quote it only word for word. A quote must appear word for word in a
   record the sentence cites.
7. Corrections win. When a later record corrects an earlier one, write the corrected fact.
8. {length_rule()} Two or three paragraphs.
"""


def build_fold_prompt(sources: Sequence[PageSource], current: CurrentPage | None = None, *,
                      today: date | None = None) -> str:
    """The one prompt the fold sends. Sources are numbered in ``handle_map`` order."""
    today = today or datetime.now(ZONE).date()
    lines = [_FOLD_RULES, f"Today is {today:%A} {today.isoformat()}. The records cover the last "
             f"{WINDOW_DAYS} days, plus older follow-ups I wrote down.", ""]
    if current is not None:
        lines.append(f"My current page (version {current.version}). Rewrite it for today: drop what no "
                     "longer matters or is no longer in the records, fix what they correct, add what is new:")
        lines.append(render_text(current.content))
        lines.append("")
    lines.append("Records:")
    lines.extend(source_lines(sources))
    lines.append("")
    lines.append("Now write the page. Every sentence ends with its [S..] cites.")
    return "\n".join(lines)


def recent_refs(sources: Iterable[PageSource], *, days: int = RECENT_DAYS) -> set[str]:
    """Refs of the sources dated within ``days`` of the newest dated source."""
    dated = [s for s in sources if s.when is not None]
    if not dated:
        return set()
    newest = max(s.when for s in dated)
    return {s.ref for s in dated if s.when >= newest - timedelta(days=days)}


def check_lately_page(raw: str, handles: Mapping[str, PageSource]) -> PageCheck:
    """``check_page`` plus: a page with recent sources available must cite at least one of them."""
    check = check_page(raw, handles)
    recent = recent_refs(handles.values())
    if recent and not (set(check.refs) & recent):
        check.errors.append(f"not_recent: cites nothing from the last {RECENT_DAYS} days of records")
        check.ok = False
    return check


def history_row(check: PageCheck, *, version: int, now=None):
    """The ``self_concept_history`` row for a passed page:lately check. Refuses a failed one."""
    return _fold.history_row(check, concept_id=PAGE_LATELY, version=version, now=now)


def core_page_from_payload(payload: Any):
    """The felt-state lane payload (``CTX_KEY_LATELY``) -> ``CorePageV1``, or None."""
    return _fold.core_page_from_payload(payload, concept_id=PAGE_LATELY)


FOLD_SPEC = FoldSpec(
    slug="lately",
    concept_id=PAGE_LATELY,
    source_sql={"episodes": EPISODE_SQL, "chat": CHAT_SQL},
    newest_sql=NEWEST_SOURCE_SQL,
    sources=lambda rows: sources_from_rows(rows.get("episodes", []), rows.get("chat", [])),
    prompt=build_fold_prompt,
    check=check_lately_page,
)

__all__ = [
    "CHAT_SQL", "CTX_KEY_LATELY", "EPISODE_SQL", "FOLD_SPEC", "NEWEST_SOURCE_SQL", "PAGE_LATELY", "WINDOW_DAYS",
    "build_fold_prompt", "check_lately_page", "core_page_from_payload", "history_row", "recent_refs",
    "select_episodes", "sources_from_rows",
]
