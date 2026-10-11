"""Addressable parts of an Orion's Day letter, and what the day's records say about each.

Juniper points at a letter by part ("2026-10-09 ¶3", "2026-10-09 carry 5"). The email numbers
the parts with these splitters, and Orion's reread tool looks them up with the same ones, so a
number in the email always names the text the tool returns. Nothing here is stored: numbering
is recomputed from ``note_md`` / ``carry_forward_md``.

Grounding:

* ``resolve_citations``: the carry-forward cites records (``[curiosity:<run_id>]``); each is
  looked up in that day's stored material. A ref with no record is ``unresolved``.
* ``claim_check``: the note cites nothing (0 refs in each of the 5 letters before 2026-10-10),
  so concrete tokens are pulled out of a paragraph -- decimals, long integers, timestamps,
  PR numbers, prior ids, code spans, long quotes -- and each is searched in the raw records.
  Found means "appears in that record"; not found means "not in the day's records verbatim"
  (derived, from another day, or wrong). String evidence only, never a verdict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from orion.orion_day.budget import extract_refs, material_ref_records
from orion.schemas.orion_day import OrionDayMaterialV1

PartKind = Literal["paragraph", "carry", "other"]


@dataclass(frozen=True)
class LetterPart:
    kind: PartKind
    text: str
    # 1-based for paragraphs and carry items; None for headings, rules and loose text.
    index: int | None = None

    def ref(self, letter_date: str) -> str | None:
        if self.index is None:
            return None
        return f"{letter_date} ¶{self.index}" if self.kind == "paragraph" else f"{letter_date} carry {self.index}"


_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")


def _blocks(text: str) -> list[str]:
    """Blank-line separated blocks; a fenced code block never splits."""
    blocks: list[str] = []
    current: list[str] = []
    fence: str | None = None  # the marker that opened the current fence; only it closes it
    for line in (text or "").splitlines():
        m = _FENCE_RE.match(line)
        if m:
            fence = m.group(1) if fence is None else (None if m.group(1) == fence else fence)
        if not line.strip() and fence is None:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_note(note_md: str) -> list[LetterPart]:
    """Orion's note in order. Each prose block is a numbered paragraph; a block that is only a
    heading or a horizontal rule is kept, unnumbered, so the email can render it in place."""
    parts: list[LetterPart] = []
    n = 0
    for block in _blocks(note_md):
        lines = block.splitlines()
        if len(lines) == 1 and (_RULE_RE.match(lines[0]) or _HEADING_RE.match(lines[0])):
            parts.append(LetterPart("other", block))
            continue
        n += 1
        parts.append(LetterPart("paragraph", block, n))
    return parts


_TOP_BULLET_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


def split_carry(carry_md: str) -> list[LetterPart]:
    """Carry-forward in order. Each top-level list item (with its continuation lines) is a
    numbered carry item; text outside items is kept unnumbered."""
    parts: list[LetterPart] = []
    n = 0
    item: list[str] | None = None
    loose: list[str] = []
    prev_blank = False

    def flush_loose() -> None:
        while loose and not loose[-1].strip():
            loose.pop()
        if loose:
            parts.append(LetterPart("other", "\n".join(loose).strip("\n")))
        loose.clear()

    def flush_item() -> None:
        nonlocal n, item
        if item is not None:
            n += 1
            parts.append(LetterPart("carry", "\n".join(item).rstrip(), n))
            item = None

    for line in (carry_md or "").splitlines():
        if _RULE_RE.match(line) and (item is None or not line[:1].isspace()):
            # A thematic break ends a list (markdown), even right under an item.
            flush_item()
            flush_loose()
            parts.append(LetterPart("other", line))
        elif _TOP_BULLET_RE.match(line):
            flush_item()
            flush_loose()
            item = [line]
        elif item is not None and (not line.strip() or line[:1].isspace() or not prev_blank):
            item.append(line)
        else:
            flush_item()
            loose.append(line)
        prev_blank = not line.strip()
    flush_item()
    flush_loose()
    return parts


def find_part(parts: list[LetterPart], kind: PartKind, index: int) -> LetterPart | None:
    return next((p for p in parts if p.kind == kind and p.index == index), None)


@dataclass(frozen=True)
class Citation:
    ref: str
    record: dict[str, Any] | None  # None: the letter cites a record that day's material does not hold

    @property
    def resolved(self) -> bool:
        return self.record is not None


def resolve_citations(text: str, material: OrionDayMaterialV1) -> list[Citation]:
    records = material_ref_records(material)
    return [Citation(ref, records.get(ref)) for ref in extract_refs(text)]


TokenKind = Literal["timestamp", "date", "decimal", "integer", "pr", "prior_id", "code", "quote"]

# Order matters: earlier patterns claim their span first, so "2026-10-09T06:35:27Z" is one
# timestamp, not a date plus a time plus three integers.
_TOKEN_PATTERNS: list[tuple[TokenKind, re.Pattern[str]]] = [
    ("code", re.compile(r"`([^`\n]{4,})`")),
    ("quote", re.compile(r"[\"“]([^\"”\n]{12,200})[\"”]")),
    ("timestamp", re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?\b|\b\d{2}:\d{2}:\d{2}Z?\b")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),  # claimed so its digits aren't integers, then skipped
    ("prior_id", re.compile(r"\b(?:self|world|lived|prior):[a-z0-9_.\-]{6,}\b")),
    ("pr", re.compile(r"(?<![\w#])#(\d{2,6})\b")),
    ("decimal", re.compile(r"(?<![\w.])\d+\.\d+(?![\d.]*\d)")),
    ("integer", re.compile(r"(?<![\w.,\-])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\w.,\-])\d{3,}(?![\w.,]*\d)")),
]


@dataclass(frozen=True)
class ClaimToken:
    token: str
    kind: TokenKind
    found_in: tuple[str, ...] = field(default_factory=tuple)  # refs of the records that contain it

    @property
    def found(self) -> bool:
        return bool(self.found_in)


def extract_claim_tokens(text: str) -> list[tuple[str, TokenKind]]:
    """Concrete, checkable tokens in reading order, de-duplicated. Small integers, bare words and
    dates without a time are skipped: they match almost any day's records and would read as support."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str, TokenKind]] = []
    for kind, pattern in _TOKEN_PATTERNS:
        for m in pattern.finditer(text or ""):
            span = m.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            if kind == "date":
                continue
            token = m.group(1) if kind in ("code", "quote", "pr") else m.group(0)
            found.append((span[0], token.strip(), kind))
    seen: dict[tuple[str, TokenKind], None] = {}
    for _, token, kind in sorted(found):
        seen.setdefault((token, kind), None)
    return list(seen)


def _needles(token: str, kind: TokenKind) -> list[re.Pattern[str]]:
    if kind == "timestamp":
        # A record stores "2026-10-09T06:35:27.123+00:00"; the letter says "06:35:27Z".
        core = token.rstrip("Z")
        return [re.compile(re.escape(core))]
    if kind in ("decimal", "integer"):
        forms = {token, token.replace(",", "")}
        return [re.compile(rf"(?<![\d.]){re.escape(f)}(?!\d|\.\d)") for f in forms]
    if kind == "pr":
        return [re.compile(rf"(?:#|pull/|PR ?){re.escape(token)}\b")]
    return [re.compile(re.escape(token), re.IGNORECASE)]


def _leaf_strings(value: Any) -> list[str]:
    """Every scalar in a record as plain text. Searching json.dumps output instead would miss a
    quote or code span containing '"' or a backslash, which JSON escapes."""
    if isinstance(value, dict):
        return [s for v in value.values() for s in _leaf_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _leaf_strings(v)]
    return [] if value is None else [str(value)]


def claim_check(text: str, material: OrionDayMaterialV1) -> list[ClaimToken]:
    haystacks = {ref: "\n".join(_leaf_strings(record)) for ref, record in material_ref_records(material).items()}
    results: list[ClaimToken] = []
    for token, kind in extract_claim_tokens(text):
        needles = _needles(token, kind)
        hits = tuple(ref for ref, hay in haystacks.items() if any(n.search(hay) for n in needles))
        results.append(ClaimToken(token, kind, hits))
    return results


# --- reading a letter back (orion-introspect `orion_day`) ------------------------------------

_REF_PART_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (?:¶(\d{1,4})|carry (\d{1,4}))$")


def parse_ref(ref: str) -> tuple[str, PartKind, int] | None:
    """``"2026-10-09 ¶3"`` -> ("2026-10-09", "paragraph", 3); ``"... carry 5"`` -> carry.
    The inverse of ``LetterPart.ref``; None for anything else."""
    m = _REF_PART_RE.match((ref or "").strip())
    if m is None:
        return None
    if m.group(2) is not None:
        return m.group(1), "paragraph", int(m.group(2))
    return m.group(1), "carry", int(m.group(3))


# Each email day section, as the material refs it is built from (``material_ref_records``
# prefixes). Reverie themes (``reverie_theme:``) are chain aggregates, not records, and are
# counted in the outline instead.
SECTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "curiosity": ("curiosity:", "curiosity_failed:"),
    "self_sense": ("self_sense:",),
    "readings": ("reading:", "reading_journal:"),
    "dreams": ("dream:", "dream_offered:"),
    "code_changes": ("github_compactor:",),
    "conversations": ("chat_compactor:",),
    "world_news": ("world_pulse_digest:",),
    "reveries": ("visual_reverie:", "reverie:"),
}


def section_records(material: OrionDayMaterialV1, section: str) -> list[tuple[str, dict[str, Any]]]:
    """(ref, record) for one day section, in the order the material holds them."""
    prefixes = SECTION_PREFIXES[section]
    return [(ref, rec) for ref, rec in material_ref_records(material).items() if ref.startswith(prefixes)]


# The first non-empty field names what a record is about, then what it says. Ordered by
# how each material model spells them (orion/schemas/orion_day.py).
_TITLE_FIELDS = ("journal_title", "title", "tldr", "question")
_BODY_FIELDS = (
    "journal_body", "finding_text", "answer_text", "learned", "narrative", "body", "claim",
    "interpretation", "description", "executive_summary", "lived_answer_text", "self_definition_text",
    "error",
)


def _squash(value: Any) -> str:
    return " ".join(str(value or "").split())


def record_text(record: dict[str, Any] | None) -> str:
    """A record as plain words: its title, then its main text (a world-pulse digest adds its
    item titles). Empty for a missing record."""
    if not record:
        return ""
    title = next((_squash(record.get(k)) for k in _TITLE_FIELDS if _squash(record.get(k))), "")
    body = next((_squash(record.get(k)) for k in _BODY_FIELDS if _squash(record.get(k))), "")
    items = record.get("items")
    if isinstance(items, list):
        titles = [_squash(i.get("title")) for i in items if isinstance(i, dict) and _squash(i.get("title"))]
        if titles:
            body = (body + " Items: " if body else "Items: ") + "; ".join(titles)
    if title and body and not body.startswith(title):
        return f"{title}: {body}"
    return body or title


def record_excerpt(record: dict[str, Any] | None, cap: int = 300) -> str:
    """``record_text`` cut to ``cap`` characters (an ellipsis marks the cut)."""
    text = record_text(record)
    return text if len(text) <= cap else text[: max(cap - 1, 0)].rstrip() + "…"
