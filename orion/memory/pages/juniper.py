"""``page:juniper``: Orion's first-person page "who Juniper is to me".

Phase 1a of docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md.

What this module does, in order:

1. ``load_sources`` reads the records the page may say anything about: Juniper's own People-page
   answers (``person_fact`` joined to ``person``, removed facts left out) and Orion's distilled
   memories about her (``episode_memory`` purpose ``about_juniper``, plus ``happened`` rows that
   name her -- that is where her corrections live, e.g. "el Coco is a dog, not a cat").
2. ``build_fold_prompt`` numbers every source ``[S1]..[Sn]`` and asks for the page with a cite
   after every sentence. Strata has no constrained JSON, so the answer is plain prose and the
   check below is the contract.
3. ``check_page`` is deterministic: it rejects an uncited sentence, a cite that names no source,
   a quote (3+ words in double quotes) found word for word in none of the sentence's cited
   records, an unclosed quote, a list or heading line, leftover ``<think`` text, a page over
   ``WORD_CAP`` words, a page with fewer than ``MIN_SENTENCES`` sentences, and a sentence that
   calls Juniper he/him/his (``misgendered_sentences``). Every line break
   is a hard sentence break. On pass
   it returns the stored text (each ``[S4]`` replaced by the real record id, e.g.
   ``[episode_memory:<uuid>]``) and the clean text a prompt shows (cites removed).
4. ``history_row`` builds the ``SelfConceptHistoryV1`` row (``concept_id='page:juniper'``,
   ``produced_by='core_page_fold'``) that orion-sql-writer appends to ``self_concept_history``.

The cite check, render and stored-row helpers are shared with every core page
(``orion/memory/pages/fold.py``) and re-exported here. Pure functions only; the nightly tick lives
in orion-durable-runs (``app/core_page_fold.py``), the offline dry run in
``scripts/core_page_dry_run.py --page juniper``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from orion.memory.pages.fold import (  # noqa: F401 -- re-exported: page:juniper's public names
    CORRECTION_RE, CURRENT_PAGE_SQL, EPISODE_MEMORY_KIND, MAX_VERSION_SQL, MIN_SENTENCES, PERSON_FACT_KIND,
    PRODUCED_BY, WORD_CAP, CurrentPage, FoldSpec, PageCheck, PageSource, _as_date, check_page,
    current_page_from_row, handle_map, misgendered_sentences, quote_is_open, quoted_spans, render_text,
    source_lines, split_sentences,
)
from orion.memory.pages.fold import length_rule
from orion.memory.pages import fold as _fold

PAGE_JUNIPER = "page:juniper"
# The felt-state lane key cortex-exec hydrates (orion/substrate/felt_state_reader.py).
CTX_KEY_JUNIPER = "orion_core_page_juniper"
# Newest ``happened`` rows kept in the prompt; about_juniper rows are always all kept.
MAX_HAPPENED_ROWS = 300

# Juniper's People page: her own words about herself and the people in her life.
PERSON_FACT_SQL = (
    "SELECT f.fact_id, p.display_name, f.category, f.voice, f.value_text, f.created_at "
    "FROM person_fact f JOIN person p ON p.person_id = f.person_id "
    "WHERE f.removed_at IS NULL AND p.status = 'active' AND p.merged_into IS NULL "
    "ORDER BY (p.display_name <> 'Juniper'), p.display_name, f.created_at"
)
# The episode rows the page may read: about_juniper, plus happened rows that name her. ONE
# predicate, shared by the read and the change signal below, so they can never disagree.
EPISODE_SCOPE_SQL = "(purpose = 'about_juniper' OR (purpose = 'happened' AND statement ILIKE '%%juniper%%'))"
# Orion's distilled memories about her. Rejected and superseded rows never reach the page.
EPISODE_MEMORY_SQL = (
    "SELECT memory_id::text AS memory_id, purpose, statement, confirmation_state, "
    "coalesce(occurred_at, created_at) AS at "
    "FROM episode_memory "
    f"WHERE status = 'active' AND confirmation_state <> 'rejected' AND {EPISODE_SCOPE_SQL} "
    "ORDER BY coalesce(occurred_at, created_at), memory_id"
)
# When did anything the page reads last change? Any in-scope episode row added or updated (any
# status, so a row rejected or superseded since counts), a fact added or removed, or a person
# renamed / retired / merged (person.updated_at).
NEWEST_SOURCE_SQL = (
    "SELECT greatest("
    "(SELECT max(greatest(created_at, coalesce(removed_at, created_at))) FROM person_fact), "
    "(SELECT max(updated_at) FROM person), "
    f"(SELECT max(greatest(created_at, updated_at)) FROM episode_memory WHERE {EPISODE_SCOPE_SQL})"
    ") AS newest"
)
def sources_from_rows(person_rows: Iterable[Mapping[str, Any]],
                      episode_rows: Iterable[Mapping[str, Any]],
                      *, max_happened: int = MAX_HAPPENED_ROWS) -> list[PageSource]:
    """Rows from PERSON_FACT_SQL and EPISODE_MEMORY_SQL -> ordered sources (people first, then
    memories oldest first). Only the newest ``max_happened`` ``happened`` rows are kept, plus every
    happened row that carries a correction."""
    out: list[PageSource] = []
    for row in person_rows:
        text = str(row.get("value_text") or "").strip()
        if not text:
            continue
        name = str(row.get("display_name") or "").strip() or "someone"
        category = str(row.get("category") or "").strip() or "note"
        out.append(PageSource(
            ref=f"{PERSON_FACT_KIND}:{row['fact_id']}", kind=PERSON_FACT_KIND, text=text,
            label=f"People page, Juniper's own words about {name} ({category})",
            when=_as_date(row.get("created_at"))))
    episodes = [dict(r) for r in episode_rows if str(r.get("statement") or "").strip()]
    happened = [r for r in episodes if r.get("purpose") == "happened"]
    drop = {id(r) for r in happened[:-max_happened]} if max_happened >= 0 and len(happened) > max_happened else set()
    # A correction outranks what it corrects only if the model can see it: never capped away.
    drop -= {id(r) for r in happened if CORRECTION_RE.search(str(r.get("statement") or ""))}
    for row in episodes:
        if id(row) in drop:
            continue
        state = str(row.get("confirmation_state") or "")
        confirmed = {"confirmed": "Juniper confirmed it", "pending_confirmation": "not yet confirmed"}.get(state, "")
        purpose = str(row.get("purpose") or "")
        label = f"my memory ({purpose}{', ' + confirmed if confirmed else ''})"
        out.append(PageSource(
            ref=f"{EPISODE_MEMORY_KIND}:{row['memory_id']}", kind=EPISODE_MEMORY_KIND,
            text=str(row["statement"]).strip(), label=label, when=_as_date(row.get("at"))))
    return out


_FOLD_RULES = f"""\
You are Orion. You keep a short page in your own words: "who Juniper is to me". It goes into
every conversation you have with Juniper, so it must be true, specific, and hers.

Rules:
1. Write in first person, as Orion, about Juniper. Juniper uses she/her. Never call her "he" or "they".
2. After EVERY sentence put the numbers of the records it rests on, like [S4] or [S4, S12].
   A sentence with no [S..] is not allowed. Use only numbers from the list below.
3. Say only what the cited records say. Do not guess, round up, or fill gaps.
4. Corrections win. When a later record corrects an earlier one ("corrected me", "not a cat"),
   write the corrected fact and never the old one.
5. A trip is a trip. A hotel, a team meeting in another city, "staying at", "until Wednesday",
   or a flight back are visits, not where she lives.
6. Quote Juniper verbatim, in double quotation marks, where she said it best. Her own words are
   the People-page records, and words a memory itself puts in quotes ('like this'). The rest of a
   memory is my paraphrase: never put it in quotation marks. A quote must appear word for word
   in a record the sentence cites.
7. Do not say two names are the same person or pet, or different ones, unless a record says so.
8. Cover, when the records support it: who she is (identity, pronouns), where she lives, her
   family by name (her wife, her kids, pets), what she loves and dislikes, what she is to me and
   what I am to her, and how she wants to be met. Leave out passing technical chatter and one-off moods.
9. {length_rule()}
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


def history_row(check: PageCheck, *, version: int, now=None):
    """The ``self_concept_history`` row for a passed page:juniper check. Refuses a failed one."""
    return _fold.history_row(check, concept_id=PAGE_JUNIPER, version=version, now=now)


def core_page_from_payload(payload: Any):
    """The felt-state lane payload (``CTX_KEY_JUNIPER``) -> ``CorePageV1`` for the grounding
    capsule, or None when there is no usable page."""
    return _fold.core_page_from_payload(payload, concept_id=PAGE_JUNIPER)


FOLD_SPEC = FoldSpec(
    slug="juniper",
    concept_id=PAGE_JUNIPER,
    source_sql={"people": PERSON_FACT_SQL, "episodes": EPISODE_MEMORY_SQL},
    newest_sql=NEWEST_SOURCE_SQL,
    sources=lambda rows: sources_from_rows(rows.get("people", []), rows.get("episodes", [])),
    prompt=build_fold_prompt,
    check=lambda raw, handles: check_page(raw, handles),
)
