"""Orion's Day letters as orion-introspect items. Pure: a stored letter in, items out.

Numbering, citation lookup and the claim check are `orion/orion_day/letter_parts.py`'s,
the same functions the email numbers its parts with, so "2026-10-09 ¶3" in Juniper's
email and in this tool always name the same words.

Labels (design 2026-10-11, section 2): the letter's own words are `unsettled` -- what
Orion wrote then, not settled fact. A note paragraph carries a claim check (string
evidence only: which concrete tokens appear verbatim in which of that day's records);
a carry item carries its citations resolved against that day's material. A day-section
record is the stored record the letter was written from.

Budgets are in SERIALIZED JSON characters (the 12,000-character MCP result budget);
a test pins the worst case.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from orion.orion_day.letter_parts import (
    SECTION_PREFIXES,
    LetterPart,
    claim_check,
    record_excerpt,
    record_text,
    resolve_citations,
    section_records,
    split_carry,
    split_note,
)
from orion.schemas.introspect import IntrospectItemV1, clip_json_text
from orion.schemas.orion_day import OrionDayLetterV1

PART_KIND = "orion_day_letter_part"
RECORD_KIND = "orion_day_record"

# One part by index.
ONE_TEXT_JSON_BUDGET = 6000
ONE_EXTRA_JSON_BUDGET = 3000
ONE_EXCERPT_CHARS = 300
# Several parts or records share these totals evenly.
MANY_TEXT_JSON_BUDGET = 5000
MANY_EXTRA_JSON_BUDGET = 2500
MANY_EXCERPT_CHARS = 120
# part=list: three outline items.
OUTLINE_JSON_BUDGET = 3000
OUTLINE_LINE_CHARS = 120
FOUND_IN_SHOWN = 5
UNRESOLVED_SHOWN = 10
INDEX_TEXT_CHARS = 1800

# Records that are bare facts rather than something Orion wrote or concluded.
_RECORD_STATUS_PREFIXES = ("curiosity_failed:", "world_pulse_digest:")
_RECORD_CLOCKS = ("completed_at", "failed_at", "created_at", "occurred_at", "offered_at")


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def letter_clock(letter: OrionDayLetterV1) -> datetime:
    return _aware(letter.created_at or letter.window_end)


def _json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def fit_list(entries: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], bool]:
    """Leading entries whose serialized list fits ``budget``; True when any were left out."""
    kept: list[dict[str, Any]] = []
    used = 2
    for entry in entries:
        size = _json_len(entry) + 2
        if used + size > budget:
            return kept, True
        kept.append(entry)
        used += size
    return kept, False


def _first_line(text: str, cap: int = OUTLINE_LINE_CHARS) -> str:
    squashed = " ".join((text or "").split())
    return squashed if len(squashed) <= cap else squashed[: cap - 1].rstrip() + "…"


def part_name(part: LetterPart) -> str:
    return "note" if part.kind == "paragraph" else "carry_forward"


def claim_check_extra(text: str, letter: OrionDayLetterV1, budget: int) -> dict[str, Any]:
    tokens = claim_check(text, letter.material)
    entries = []
    for t in tokens:
        entry: dict[str, Any] = {"token": t.token, "kind": t.kind, "found_in": list(t.found_in[:FOUND_IN_SHOWN])}
        if len(t.found_in) > FOUND_IN_SHOWN:
            entry["found_in_more"] = len(t.found_in) - FOUND_IN_SHOWN
        entries.append(entry)
    kept, cut = fit_list(entries, budget)
    return {
        "claims_found": sum(1 for t in tokens if t.found),
        "claims_not_found": sum(1 for t in tokens if not t.found),
        "claim_check": kept,
        "claim_check_truncated": cut,
    }


def citations_extra(text: str, letter: OrionDayLetterV1, budget: int, excerpt_chars: int) -> dict[str, Any]:
    cites = resolve_citations(text, letter.material)
    entries = [
        {"ref": c.ref, "resolved": c.resolved, "excerpt": record_excerpt(c.record, excerpt_chars) if c.resolved else None}
        for c in cites
    ]
    kept, cut = fit_list(entries, budget)
    return {
        "citations_total": len(cites),
        "citations_unresolved": [c.ref for c in cites if not c.resolved],
        "citations": kept,
        "citations_truncated": cut,
    }


def part_item(
    letter: OrionDayLetterV1, part: LetterPart, *, full: bool, n_items: int = 1,
    extra: Optional[dict[str, Any]] = None,
) -> IntrospectItemV1:
    """One numbered note paragraph or carry item, its exact words first.

    ``full`` (one part by index) gets the large budget; otherwise ``n_items`` parts share
    the MANY_* budgets evenly.
    """
    date = letter.letter_date.isoformat()
    n = max(n_items, 1)
    text_budget = ONE_TEXT_JSON_BUDGET if full else MANY_TEXT_JSON_BUDGET // n
    extra_budget = ONE_EXTRA_JSON_BUDGET if full else MANY_EXTRA_JSON_BUDGET // n
    text, truncated = clip_json_text(part.text, text_budget)
    fields: dict[str, Any] = {"letter_date": date, "part": part_name(part), "index": part.index}
    if part.kind == "paragraph":
        fields.update(claim_check_extra(part.text, letter, extra_budget))
    else:
        fields.update(citations_extra(
            part.text, letter, extra_budget, ONE_EXCERPT_CHARS if full else MANY_EXCERPT_CHARS,
        ))
    fields.update(extra or {})
    return IntrospectItemV1(
        id=part.ref(date) or f"{date} {part_name(part)}", occurred_at=letter_clock(letter), kind=PART_KIND,
        epistemic_status="unsettled", text=text, truncated=truncated, extra=fields,
    )


def section_counts(letter: OrionDayLetterV1) -> dict[str, int]:
    return {name: len(section_records(letter.material, name)) for name in SECTION_PREFIXES}


def outline_items(
    letter: OrionDayLetterV1, note: list[LetterPart], carry: list[LetterPart],
) -> list[IntrospectItemV1]:
    """part=list: the note's paragraphs and the carry items by number with their opening
    words, and how many records each day section holds (counts only)."""
    date = letter.letter_date.isoformat()
    when = letter_clock(letter)
    paragraphs = [p for p in note if p.kind == "paragraph"]
    items_ = [p for p in carry if p.kind == "carry"]
    note_text, note_cut = clip_json_text(
        "\n".join(f"¶{p.index} {_first_line(p.text)}" for p in paragraphs), OUTLINE_JSON_BUDGET,
    )
    carry_text, carry_cut = clip_json_text(
        "\n".join(f"carry {p.index}: {_first_line(p.text.splitlines()[0] if p.text else '')}" for p in items_),
        OUTLINE_JSON_BUDGET,
    )
    cites = [c for p in items_ for c in resolve_citations(p.text, letter.material)]
    unresolved = [c.ref for c in cites if not c.resolved]
    counts = section_counts(letter)
    m = letter.material
    themes = Counter(c.theme_key for c in m.reverie_chains if c.theme_key)
    counts_text, counts_cut = clip_json_text(
        "\n".join(f"{name}: {n} record(s)" for name, n in counts.items())
        + f"\nreverie chains: {len(m.reverie_chains)} ({len(themes)} themes)",
        OUTLINE_JSON_BUDGET,
    )
    return [
        IntrospectItemV1(
            id=f"{date} note", occurred_at=when, kind=PART_KIND, epistemic_status="unsettled",
            text=note_text, truncated=note_cut,
            extra={"letter_date": date, "part": "note", "paragraphs": len(paragraphs),
                   "unnumbered": sum(1 for p in note if p.kind != "paragraph")},
        ),
        IntrospectItemV1(
            id=f"{date} carry", occurred_at=when, kind=PART_KIND, epistemic_status="unsettled",
            text=carry_text, truncated=carry_cut,
            extra={"letter_date": date, "part": "carry_forward", "items": len(items_),
                   "unnumbered": sum(1 for p in carry if p.kind != "carry"),
                   "citations": len(cites), "citations_unresolved": len(unresolved),
                   "unresolved_refs": unresolved[:UNRESOLVED_SHOWN]},
        ),
        IntrospectItemV1(
            id=f"{date} sections", occurred_at=when, kind=RECORD_KIND, epistemic_status="record",
            text=counts_text, truncated=counts_cut,
            extra={"letter_date": date, "part": "section", "section_counts": counts,
                   "reverie_chains": len(m.reverie_chains), "reverie_themes": len(themes),
                   "emailed": letter.emailed_at is not None},
        ),
    ]


def _record_clock(record: dict[str, Any], fallback: datetime) -> datetime:
    for key in _RECORD_CLOCKS:
        value = record.get(key)
        if isinstance(value, datetime):
            return _aware(value)
        if isinstance(value, str) and value:
            try:
                return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
            except ValueError:
                continue
    return fallback


def section_item(
    letter: OrionDayLetterV1, section: str, ref: str, record: dict[str, Any], *, n_items: int,
) -> IntrospectItemV1:
    text, truncated = clip_json_text(record_text(record), MANY_TEXT_JSON_BUDGET // max(n_items, 1))
    status = "record" if ref.startswith(_RECORD_STATUS_PREFIXES) else "unsettled"
    return IntrospectItemV1(
        id=ref, occurred_at=_record_clock(record, _aware(letter.window_end)), kind=RECORD_KIND,
        epistemic_status=status, text=text, truncated=truncated,
        extra={"letter_date": letter.letter_date.isoformat(), "section": section, "ref": ref},
    )


def index_docs_from_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    """(ref, text, meta) per note paragraph and carry item, for the search index."""
    docs = []
    for row in rows:
        day = row.get("letter_date")
        created = row.get("created_at")
        if day is None or not isinstance(created, datetime):
            continue
        date = day.isoformat()
        created = _aware(created)
        for part in split_note(str(row.get("note_md") or "")) + split_carry(str(row.get("carry_forward_md") or "")):
            if part.index is None or not part.text.strip():
                continue
            text = part.text.strip()[:INDEX_TEXT_CHARS]
            docs.append((part.ref(date), text, {
                "letter_date": date, "part": part_name(part),
                "occurred_at": created.isoformat(), "occurred_ts": created.timestamp(),
            }))
    return docs
