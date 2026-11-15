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
   a quote (3+ words in double quotes) found in none of the sentence's cited records, a page over
   ``WORD_CAP`` words, and a page with fewer than ``MIN_SENTENCES`` sentences. On pass
   it returns the stored text (each ``[S4]`` replaced by the real record id, e.g.
   ``[episode_memory:<uuid>]``) and the clean text a prompt shows (cites removed).
4. ``history_row`` builds the ``SelfConceptHistoryV1`` row (``concept_id='page:juniper'``,
   ``produced_by='core_page_fold'``) that orion-sql-writer appends to ``self_concept_history``.

Pure functions only; the nightly tick lives in orion-durable-runs (``app/core_page_fold.py``), the
offline dry run in ``scripts/core_page_juniper_dry_run.py``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

PAGE_JUNIPER = "page:juniper"
PRODUCED_BY = "core_page_fold"
# The felt-state lane key cortex-exec hydrates (orion/substrate/felt_state_reader.py).
CTX_KEY_JUNIPER = "orion_core_page_juniper"
WORD_CAP = 350
MIN_SENTENCES = 3
# Newest ``happened`` rows kept in the prompt; about_juniper rows are always all kept.
MAX_HAPPENED_ROWS = 300

PERSON_FACT_KIND = "person_fact"
EPISODE_MEMORY_KIND = "episode_memory"

# Juniper's People page: her own words about herself and the people in her life.
PERSON_FACT_SQL = (
    "SELECT f.fact_id, p.display_name, f.category, f.voice, f.value_text, f.created_at "
    "FROM person_fact f JOIN person p ON p.person_id = f.person_id "
    "WHERE f.removed_at IS NULL AND p.status = 'active' AND p.merged_into IS NULL "
    "ORDER BY (p.display_name <> 'Juniper'), p.display_name, f.created_at"
)
# Orion's distilled memories about her. Rejected and superseded rows never reach the page.
EPISODE_MEMORY_SQL = (
    "SELECT memory_id::text AS memory_id, purpose, statement, confirmation_state, "
    "coalesce(occurred_at, created_at) AS at "
    "FROM episode_memory "
    "WHERE status = 'active' AND confirmation_state <> 'rejected' "
    "AND (purpose = 'about_juniper' OR (purpose = 'happened' AND statement ILIKE '%%juniper%%')) "
    "ORDER BY coalesce(occurred_at, created_at), memory_id"
)
# When did anything the page reads last change? A new fact or memory, a removed fact, or a
# memory rejected/superseded (its status moved). Reinforcement of an active row does not count.
NEWEST_SOURCE_SQL = (
    "SELECT greatest("
    "(SELECT max(greatest(created_at, coalesce(removed_at, created_at))) FROM person_fact), "
    "(SELECT max(created_at) FROM episode_memory WHERE purpose IN ('about_juniper', 'happened')), "
    "(SELECT max(updated_at) FROM episode_memory "
    " WHERE purpose IN ('about_juniper', 'happened') AND status <> 'active')"
    ") AS newest"
)
CURRENT_PAGE_SQL = (
    "SELECT entry_id, version, content, evidence_refs, created_at FROM self_concept_history "
    "WHERE concept_id = %(cid)s AND produced_by = %(by)s ORDER BY created_at DESC LIMIT 1"
)
MAX_VERSION_SQL = "SELECT coalesce(max(version), 0) AS v FROM self_concept_history WHERE concept_id = %(cid)s"

_HANDLE_GROUP = re.compile(r"\s*\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_HANDLE = re.compile(r"S\d+")
# A stored cite: one or more real refs inside brackets.
_STORED_CITE = re.compile(
    r"\s*\[\s*(?:person_fact|episode_memory):[^\]\s,;]+(?:\s*[,;]\s*(?:person_fact|episode_memory):[^\]\s,;]+)*\s*\]"
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*$", re.MULTILINE)
_SENTENCE_START = re.compile(r"\s+(?=[\"“(']?[A-Z0-9])")
_LEADING_PUNCT = re.compile(r"^[\s.,;:!?)]+")


@dataclass(frozen=True)
class PageSource:
    ref: str                 # "person_fact:<fact_id>" | "episode_memory:<uuid>"
    kind: str                # PERSON_FACT_KIND | EPISODE_MEMORY_KIND
    text: str
    label: str               # one-line provenance shown to the model
    when: date | None = None


@dataclass
class CurrentPage:
    version: int
    content: str
    created_at: datetime | None = None


@dataclass
class PageCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    clean_text: str = ""
    stored_text: str = ""
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "word_count": self.word_count,
                "sentence_count": self.sentence_count, "refs": len(self.refs)}


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return None


def sources_from_rows(person_rows: Iterable[Mapping[str, Any]],
                      episode_rows: Iterable[Mapping[str, Any]],
                      *, max_happened: int = MAX_HAPPENED_ROWS) -> list[PageSource]:
    """Rows from PERSON_FACT_SQL and EPISODE_MEMORY_SQL -> ordered sources (people first, then
    memories oldest first). Only the newest ``max_happened`` ``happened`` rows are kept."""
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


def current_page_from_row(row: Mapping[str, Any] | None) -> CurrentPage | None:
    if not row:
        return None
    content = str(row.get("content") or "").strip()
    if not content:
        return None
    try:
        version = int(row.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    created = row.get("created_at")
    return CurrentPage(version=version, content=content, created_at=created if isinstance(created, datetime) else None)


def handle_map(sources: Sequence[PageSource]) -> dict[str, PageSource]:
    return {f"S{i}": src for i, src in enumerate(sources, start=1)}


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
    for handle, src in handle_map(sources).items():
        stamp = f" {src.when.isoformat()}" if src.when else ""
        lines.append(f"[{handle}]{stamp} {src.label}: {src.text}")
    lines.append("")
    lines.append("Now write the page. Every sentence ends with its [S..] cites.")
    return "\n".join(lines)


def _strip_wrapping(raw: str) -> str:
    text = _THINK.sub("", raw or "")
    text = _FENCE.sub("", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Sentences of one cited span. Quote-aware: a period inside "..." does not end a sentence
    (Juniper's quoted words often hold several). A double quote right after a digit is an inch
    mark (6'5"), not a quote."""
    sentences: list[str] = []
    start = 0
    in_quote = False
    for i, ch in enumerate(text):
        if ch == '"' and not (i > 0 and text[i - 1].isdigit()):
            in_quote = not in_quote
        elif ch == "“":
            in_quote = True
        elif ch == "”":
            in_quote = False
        if in_quote:
            continue
        ends = ch in ".!?" or (ch in '"”' and i > 0 and text[i - 1] in ".!?")
        if not ends:
            continue
        if _SENTENCE_START.match(text, i + 1):
            piece = text[start:i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


_QUOTE_MIN_WORDS = 3


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


def quoted_spans(sentence: str) -> list[str]:
    """Double-quoted spans of a sentence ("..." or “...”). A double quote right after a digit is an
    inch mark (6'5"), not a quote."""
    spans: list[str] = []
    start: int | None = None
    for i, ch in enumerate(sentence):
        if ch == '"' and i > 0 and sentence[i - 1].isdigit() and start is None:
            continue
        if ch in '"“' and start is None:
            start = i + 1
        elif ch in '"”' and start is not None:
            spans.append(sentence[start:i])
            start = None
    return spans


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def check_page(raw: str, handles: Mapping[str, PageSource], *, word_cap: int = WORD_CAP,
               min_sentences: int = MIN_SENTENCES) -> PageCheck:
    """Deterministic truth check on the model's page. Never repairs, only accepts or rejects."""
    text = _strip_wrapping(raw)
    errors: list[str] = []
    if not text:
        return PageCheck(ok=False, errors=["empty"])
    sentences: list[tuple[str, list[str]]] = []
    pos = 0
    for match in _HANDLE_GROUP.finditer(text):
        chunk = _LEADING_PUNCT.sub("", text[pos:match.start()])
        cites = _HANDLE.findall(match.group(1))
        pos = match.end()
        if not chunk.strip():
            if sentences:      # "[S4] [S9]" or "[S4]. [S9]": more cites for the same sentence
                sentences[-1][1].extend(cites)
            else:
                errors.append("cite_without_sentence")
            continue
        parts = split_sentences(chunk)
        for uncited in parts[:-1]:
            errors.append(f"uncited: {uncited[:120]}")
            sentences.append((uncited, []))
        sentences.append((parts[-1], list(cites)))
    tail = _LEADING_PUNCT.sub("", text[pos:]).strip()
    if tail:
        for uncited in split_sentences(tail):
            errors.append(f"uncited: {uncited[:120]}")
            sentences.append((uncited, []))

    refs: list[str] = []
    for sentence, cites in sentences:
        cited_texts: list[str] = []
        for handle in cites:
            src = handles.get(handle)
            if src is None:
                errors.append(f"unresolvable_cite: {handle}")
                continue
            cited_texts.append(_norm(src.text))
            if src.ref not in refs:
                refs.append(src.ref)
        if not cites:
            continue
        # A quote is a claim about someone's exact words: it must be in a record this sentence cites.
        for span in quoted_spans(sentence):
            quote = _norm(span)
            if len(quote.split()) >= _QUOTE_MIN_WORDS and not any(quote in t for t in cited_texts):
                errors.append(f"unsupported_quote: {span[:80]}")

    def _stored(match: re.Match[str]) -> str:
        real = [handles[h].ref for h in _HANDLE.findall(match.group(1)) if h in handles]
        return f" [{', '.join(dict.fromkeys(real))}]" if real else ""

    clean = _clean(_HANDLE_GROUP.sub("", text))
    stored = _clean(_HANDLE_GROUP.sub(_stored, text))
    words = len(clean.split())
    cited_sentences = sum(1 for _, cites in sentences if cites)
    if words > word_cap:
        errors.append(f"over_cap: {words} words > {word_cap}")
    if cited_sentences < min_sentences:
        errors.append(f"too_thin: {cited_sentences} cited sentences < {min_sentences}")
    return PageCheck(ok=not errors, errors=errors, word_count=words, sentence_count=len(sentences),
                     clean_text=clean, stored_text=stored, refs=refs)


def render_text(content: str) -> str:
    """Stored page text -> what a prompt shows: the cites removed, the words untouched."""
    return _clean(_STORED_CITE.sub("", content or ""))


def history_row(check: PageCheck, *, version: int, now: datetime | None = None):
    """The ``self_concept_history`` row for a passed check. Refuses a failed one."""
    from orion.schemas.self_concept_history import SelfConceptHistoryV1

    if not check.ok:
        raise ValueError(f"page check failed: {check.errors[:3]}")
    return SelfConceptHistoryV1(
        concept_id=PAGE_JUNIPER, version=int(version), content=check.stored_text,
        evidence_refs=list(check.refs), produced_by=PRODUCED_BY,
        **({"created_at": now} if now is not None else {}))


def core_page_from_payload(payload: Any):
    """The felt-state lane payload (``CTX_KEY_JUNIPER``) -> ``CorePageV1`` for the grounding
    capsule, or None when there is no usable page."""
    from orion.schemas.thought import CorePageV1

    if not isinstance(payload, Mapping):
        return None
    text = render_text(str(payload.get("content") or ""))
    if not text:
        return None
    try:
        version = int(payload.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    refs = payload.get("evidence_refs")
    return CorePageV1(
        concept_id=str(payload.get("concept_id") or PAGE_JUNIPER), version=version, text=text,
        written_at=str(payload.get("created_at") or "")[:10] or None,
        evidence_count=len(refs) if isinstance(refs, list) else 0)
