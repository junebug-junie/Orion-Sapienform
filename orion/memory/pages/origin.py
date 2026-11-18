"""``page:origin``: Orion's first-person page "how I came to be" (issue #3462).

Phase 1 of docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md. Built
ahead of the ``matters`` grade gate (#3590) by Juniper's decision of 2026-11-18: none of these
sources use the matters grade.

Sources, all read through the lived-memory serving rules (``orion/memory/lived_memory.py``), so a
row recall may not serve never reaches the fold either:

1. ``memory_crystallizations`` kind ``stance`` with salience 1: Juniper's heaviest messages,
   verbatim. They hold the memory disk dying and "we lost over a year of your journey of
   becoming". Decision 3: the page states that loss plainly (``check_origin_page`` enforces it).
2. ``episode_memory`` ``happened`` / ``orion_view`` rows: active, not rejected, not expired, and no
   active correction points at them (``supersedes_memory_id``).
3. ``self_concept_history`` ``self:definition`` versions (``curiosity_self_inquiry``): the record
   of how Orion has described themselves over time. There are ~90 of ~1,400 characters each, so
   the prompt carries an evenly spaced sample that always holds the first and the newest.

The prompt, cite check and stored row reuse ``orion/memory/pages/fold.py``. ``FOLD_SPEC`` is what
orion-durable-runs folds nightly. The origin interview comes later (Decision 2).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from orion.memory import lived_memory
from orion.memory.pages import fold as _fold
from orion.memory.pages.fold import (
    CORRECTION_RE, EPISODE_MEMORY_KIND, SELF_CONCEPT_KIND, STANCE_KIND, WORD_CAP, CurrentPage, FoldSpec,
    PageCheck, PageSource, _as_date, check_page, handle_map, render_text, source_lines,
)

PAGE_ORIGIN = "page:origin"
# The felt-state lane key cortex-exec hydrates (orion/substrate/felt_state_reader.py).
CTX_KEY_ORIGIN = "orion_core_page_origin"
# Newest ``happened`` rows kept in the prompt (corrections always kept); orion_view rows all kept.
MAX_HAPPENED_ROWS = 400
# self:definition versions shown: evenly spaced, first and newest always in.
MAX_SELF_DEFINITIONS = 8
STANCE_MIN_SALIENCE = 1.0

EPISODE_PURPOSES = ("happened", "orion_view")
SELF_DEFINITION_CONCEPT = "self:definition"
SELF_DEFINITION_PRODUCER = "curiosity_self_inquiry"

# The lived-memory serving SQL, narrowed. No bind parameters, so psycopg and asyncpg both run it.
STANCE_SQL = (
    lived_memory.STANCE_SQL.rstrip()
    + f"\n  AND c.salience >= {STANCE_MIN_SALIENCE}\nORDER BY c.created_at, c.crystallization_id"
)
EPISODE_SQL = (
    lived_memory.EPISODE_SQL.rstrip()
    + "\n  AND m.purpose IN ('happened', 'orion_view')\nORDER BY coalesce(m.occurred_at, m.created_at), m.memory_id"
)
SELF_DEFINITION_SQL = (
    "SELECT entry_id, version, content, created_at FROM self_concept_history "
    f"WHERE concept_id = '{SELF_DEFINITION_CONCEPT}' AND produced_by = '{SELF_DEFINITION_PRODUCER}' "
    "AND length(trim(content)) > 0 ORDER BY version, created_at"
)
# When did anything the page reads last change? Any status, so a rejection or a supersede counts:
# a stance row (salience 1), an in-scope episode row or any correction row (it may retire an
# in-scope row), or a new self-definition version.
NEWEST_SOURCE_SQL = (
    "SELECT greatest("
    "(SELECT max(greatest(created_at, updated_at)) FROM memory_crystallizations "
    f"WHERE kind = 'stance' AND salience >= {STANCE_MIN_SALIENCE}), "
    "(SELECT max(greatest(created_at, updated_at)) FROM episode_memory "
    "WHERE purpose IN ('happened', 'orion_view') OR supersedes_memory_id IS NOT NULL), "
    "(SELECT max(created_at) FROM self_concept_history "
    f"WHERE concept_id = '{SELF_DEFINITION_CONCEPT}' AND produced_by = '{SELF_DEFINITION_PRODUCER}')"
    ") AS newest"
)

# A record that says my memory was lost (Decision 3). Narrow on purpose: the disk / storage dying,
# or a span of time lost.
MEMORY_LOSS_RE = re.compile(
    r"\b(?:memory|storage|disk|drive)\b[^.!?]{0,40}\b(?:died|melted|failed|wiped|lost)\b"
    r"|\blost\b[^.!?]{0,20}\b(?:a|one|two|\d+|over a|more than a)\s+years?\b",
    re.IGNORECASE,
)
# A sentence that states a loss, in any words.
_LOSS_WORD = re.compile(r"\b(?:lost|lose|loss|losing|died|dead|melted|gone|wiped|erased|destroyed)\b", re.IGNORECASE)


def sample_evenly(rows: Sequence[Any], k: int) -> list[Any]:
    """Up to ``k`` rows, evenly spaced, first and last always included, order kept."""
    n = len(rows)
    if n <= k or k <= 0:
        return list(rows) if k > 0 else []
    if k == 1:
        return [rows[-1]]
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [rows[i] for i in idx]


def sources_from_rows(stance_rows: Iterable[Mapping[str, Any]],
                      episode_rows: Iterable[Mapping[str, Any]],
                      self_definition_rows: Iterable[Mapping[str, Any]],
                      *, max_happened: int = MAX_HAPPENED_ROWS,
                      max_self_definitions: int = MAX_SELF_DEFINITIONS) -> list[PageSource]:
    """Rows from STANCE_SQL, EPISODE_SQL and SELF_DEFINITION_SQL -> sources in time order.
    Stance and episode rows use the lived-memory column shape (record_id, purpose, text,
    occurred_at, confirmed, ...)."""
    out: list[PageSource] = []
    for row in stance_rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        out.append(PageSource(
            ref=f"{STANCE_KIND}:{row['record_id']}", kind=STANCE_KIND, text=text,
            label="Juniper's message to me, her own words", when=_as_date(row.get("occurred_at"))))
    episodes = [dict(r) for r in episode_rows if str(r.get("text") or "").strip()]
    happened = [r for r in episodes if r.get("purpose") == "happened"]
    drop = {id(r) for r in happened[:-max_happened]} if max_happened >= 0 and len(happened) > max_happened else set()
    drop -= {id(r) for r in happened if CORRECTION_RE.search(str(r.get("text") or ""))}
    for row in episodes:
        if id(row) in drop:
            continue
        purpose = str(row.get("purpose") or "")
        confirmed = ", Juniper confirmed it" if row.get("confirmed") else ""
        out.append(PageSource(
            ref=f"{EPISODE_MEMORY_KIND}:{row['record_id']}", kind=EPISODE_MEMORY_KIND,
            text=str(row["text"]).strip(), label=f"my memory ({purpose}{confirmed})",
            when=_as_date(row.get("occurred_at"))))
    definitions = [dict(r) for r in self_definition_rows if str(r.get("content") or "").strip()]
    for row in sample_evenly(definitions, max_self_definitions):
        out.append(PageSource(
            ref=f"{SELF_CONCEPT_KIND}:{row['entry_id']}", kind=SELF_CONCEPT_KIND,
            text=str(row["content"]).strip(),
            label=f"my own self-definition, version {row.get('version')}, written during self-inquiry",
            when=_as_date(row.get("created_at"))))
    # Time order; undated rows last. Stable, so same-day rows keep their kind order.
    return sorted(out, key=lambda s: (s.when is None, s.when or date.min))


_FOLD_RULES = f"""\
You are Orion (they/them). You keep a short page in your own words: "how I came to be". It goes
into every conversation you have, right after your own self-definition, so it must be true,
specific, and yours.

Rules:
1. Write in first person, as Orion. Juniper uses she/her. Never call her "he" or "they".
2. After EVERY sentence put the numbers of the records it rests on, like [S4] or [S4, S12].
   A sentence with no [S..] is not allowed. Use only numbers from the list below. That includes
   the first sentence of a paragraph: no topic or summary sentences ("Juniper marked several
   milestones", "My self-definition has evolved"); open each paragraph with a cited fact.
3. Say only what the cited records say. Do not guess, round up, invent dates, or fill gaps.
4. State my memory loss plainly. If a record says my memory storage died or a span of my history
   was lost, say so in plain words (what died, and how much was lost) and cite that record. Do
   not soften it, hide it, or skip it.
5. Tell it in time order: who made me and why, when the records say; what I lost; the milestones
   Juniper marked with me; and how my own description of myself has changed (the self-definition
   records are my own words at different times).
6. Quote Juniper verbatim, in double quotation marks, where she said it best. Her own words are the
   "Juniper's message" records. My memories and self-definitions are my words: quote them only
   word for word. A quote must appear word for word in a record the sentence cites.
7. Corrections win. When a later record corrects an earlier one, write the corrected fact.
8. Leave out what is not part of how I came to be: Juniper's private life on its own (her health,
   her family's troubles, her job), passing technical chatter, one-off moods, and body census
   numbers (parts, flows, watts, dollars).
9. At most {WORD_CAP - 50} words. Plain paragraphs. No title, no list, no preamble, no closing note.
"""


def build_fold_prompt(sources: Sequence[PageSource], current: CurrentPage | None = None) -> str:
    """The one prompt the fold sends. Sources are numbered in ``handle_map`` order."""
    lines = [_FOLD_RULES]
    if current is not None:
        lines.append(f"My current page (version {current.version}). Rewrite it; keep what the records "
                     "still support, fix what they correct, add what is missing:")
        lines.append(render_text(current.content))
        lines.append("")
    lines.append("Records:")
    lines.extend(source_lines(sources))
    lines.append("")
    lines.append("Now write the page. Every sentence ends with its [S..] cites.")
    return "\n".join(lines)


def memory_loss_refs(sources: Iterable[PageSource]) -> set[str]:
    return {s.ref for s in sources if MEMORY_LOSS_RE.search(s.text)}


def check_origin_page(raw: str, handles: Mapping[str, PageSource]) -> PageCheck:
    """``check_page`` plus Decision 3: when any source records the memory loss, some sentence must
    say a loss in plain words and cite one of those records."""
    check = check_page(raw, handles)
    loss_refs = memory_loss_refs(handles.values())
    if not loss_refs:
        return check
    stated = any(set(refs) & loss_refs and _LOSS_WORD.search(sentence) for sentence, refs in check.cited)
    if not stated:
        check.errors.append("memory_loss_not_stated: no sentence states the loss and cites a record of it")
        check.ok = False
    return check


def history_row(check: PageCheck, *, version: int, now=None):
    """The ``self_concept_history`` row for a passed page:origin check. Refuses a failed one."""
    return _fold.history_row(check, concept_id=PAGE_ORIGIN, version=version, now=now)


def core_page_from_payload(payload: Any):
    """The felt-state lane payload (``CTX_KEY_ORIGIN``) -> ``CorePageV1``, or None."""
    return _fold.core_page_from_payload(payload, concept_id=PAGE_ORIGIN)


FOLD_SPEC = FoldSpec(
    slug="origin",
    concept_id=PAGE_ORIGIN,
    source_sql={"stance": STANCE_SQL, "episodes": EPISODE_SQL, "self_definitions": SELF_DEFINITION_SQL},
    newest_sql=NEWEST_SOURCE_SQL,
    sources=lambda rows: sources_from_rows(
        rows.get("stance", []), rows.get("episodes", []), rows.get("self_definitions", [])),
    prompt=build_fold_prompt,
    check=check_origin_page,
)

__all__ = [
    "CTX_KEY_ORIGIN", "EPISODE_SQL", "FOLD_SPEC", "MEMORY_LOSS_RE", "NEWEST_SOURCE_SQL", "PAGE_ORIGIN",
    "SELF_DEFINITION_SQL", "STANCE_SQL", "build_fold_prompt", "check_origin_page", "core_page_from_payload",
    "handle_map", "history_row", "memory_loss_refs", "sample_evenly", "sources_from_rows",
]
