"""Deterministic, structural tone dials for one Orion reply.

P0 of the dynamic personality graph design (PR #2726, section 5 "Tone"):
measure what Orion's chat replies actually look like before anything tries
to move them. The current voice reads as generic assistant prose (headings,
bullet lists, bold spans, a preamble paragraph); these dials put numbers on
that shape so a baseline exists and a later change can be checked against it.

Every dial is structural -- punctuation, layout, sentence shape. There is
deliberately no vocabulary list of "LLM words": per repo rule, a filter that
grows by adding words is a keyword cathedral. The one word set here
(first-person singular pronouns) is a closed grammatical class, fixed, and
not meant to be extended.

Pure and I/O-free. ``measure_tone_dials`` never raises on any ``str``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Code is not voice, so it is removed before measuring. Fences are tracked
# line by line (see _strip_code): a regex over the whole text mistook the
# closing line of a fence opened mid-line ("Here: ```python") for a new
# opener and deleted every sentence after it (review, 2026-10-16).
_FENCE_LINE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_MIDLINE_OPEN_RE = re.compile(r"(`{3,}|~{3,})[\w+.-]*[ \t]*$")
_INLINE_FENCE_RE = re.compile(r"(`{3,})[^\n]*?\1")
_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"\w+(?:['\-]\w+)*")
_LIST_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
# Sentence end: terminal punctuation (optionally followed by closing quotes /
# brackets) then whitespace. Each non-empty line also ends a sentence, so a
# bullet without a period still counts as one unit.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]”’*_]*\s+")
# Closers that can trail a question mark: quotes, brackets, and markdown
# emphasis ("**Want me to continue?**").
_QUESTION_TRAIL = "\"')]”’*_"
# Dash used as punctuation: an em dash anywhere, or a hyphen / double hyphen /
# en dash with whitespace on both sides. A hyphen inside a word
# ("well-known") has no surrounding space and is not counted.
_DASH_RE = re.compile(r"—|(?<=\S)[ \t]+(?:--?|–)[ \t]+(?=\S)")
# Closed grammatical class: first-person singular. Fixed; do not extend.
_FIRST_PERSON_CASED = frozenset({"I", "I'm", "I've", "I'd", "I'll"})
_FIRST_PERSON_FOLDED = frozenset({"me", "my"})


@dataclass(frozen=True)
class ToneDials:
    sentence_count: int  # sentences, splitting on .!? and on line breaks
    mean_sentence_words: float  # words per sentence
    list_line_ratio: float  # non-empty lines starting with -, *, bullet, or 1. / 1) over all non-empty lines
    heading_count: int  # markdown '#' heading lines
    bold_span_count: int  # **bold** spans
    question_count: int  # sentences ending in '?'
    em_dash_per_100_words: float  # em dashes plus spaced ' - ' / ' -- ' / ' en-dash ' per 100 words
    first_person_singular_per_100_words: float  # I, I'm, I've, I'd, I'll, me, my per 100 words
    opener_words: int  # words in the first paragraph if the reply has >1 paragraph, else 0 (preamble proxy)
    word_count: int  # words outside code fences

    def as_dict(self) -> dict:
        return asdict(self)


DIAL_NAMES: tuple[str, ...] = tuple(ToneDials.__dataclass_fields__)


def _strip_code(text: str) -> str:
    """Drop fenced code (a bare fence line, or a fence opened at the end of a
    prose line) and inline ```span``` code. An unterminated fence runs to the
    end of the text. A fence closes only on a bare line of at least as many
    of the same character."""
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        if fence is not None:
            stripped = line.strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                fence = None
            continue
        line = _INLINE_FENCE_RE.sub("", line)
        m = _FENCE_LINE_RE.match(line)
        if m:
            fence = m.group(1)
            continue
        m = _MIDLINE_OPEN_RE.search(line)
        if m:
            fence = m.group(1)
            line = line[: m.start()]
        out.append(line)
    return "\n".join(out)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _per_100(count: int, words: int) -> float:
    return round(100.0 * count / words, 4) if words else 0.0


def measure_tone_dials(text: str) -> ToneDials | None:
    """Return the tone dials for ``text``, or None if it is empty/whitespace
    (or only a code block). Never raises on any ``str``."""
    if not isinstance(text, str):
        return None
    try:
        return _measure(text)
    except Exception:  # pragma: no cover -- defensive: a measurement must never break a caller
        return None


def _measure(text: str) -> ToneDials | None:
    body = _strip_code(text.replace("\r\n", "\n").replace("\r", "\n")).replace("’", "'")
    # A URL is one word, not ten.
    body = _URL_RE.sub("link", body)
    if not body.strip():
        return None

    lines = [ln for ln in body.split("\n") if ln.strip()]
    list_lines = sum(1 for ln in lines if _LIST_RE.match(ln))
    heading_count = sum(1 for ln in lines if _HEADING_RE.match(ln))

    sentences: list[str] = []
    words: list[str] = []
    dash_count = 0
    for ln in lines:
        # Drop list / heading markers so they are neither words nor dashes.
        # Every word-based dial counts from this same stripped text.
        content = _HEADING_RE.sub("", _LIST_RE.sub("", ln)).strip()
        words.extend(_words(content))
        dash_count += len(_DASH_RE.findall(content))
        for piece in _SENT_SPLIT_RE.split(content):
            piece = piece.strip()
            if _words(piece):
                sentences.append(piece)

    word_count = len(words)
    if word_count == 0:
        return None
    sentence_count = len(sentences)
    sentence_words = sum(len(_words(s)) for s in sentences)

    first_person = sum(
        1 for w in words if w in _FIRST_PERSON_CASED or w.lower() in _FIRST_PERSON_FOLDED
    )

    # A leading "# Title" is not a preamble: headings are left out of paragraphs.
    no_headings = "\n".join(ln for ln in body.split("\n") if not _HEADING_RE.match(ln))
    paragraphs = [p for p in re.split(r"\n\s*\n", no_headings) if _words(p)]
    opener_words = len(_words(paragraphs[0])) if len(paragraphs) > 1 else 0

    return ToneDials(
        sentence_count=sentence_count,
        mean_sentence_words=round(sentence_words / sentence_count, 4) if sentence_count else 0.0,
        list_line_ratio=round(list_lines / len(lines), 4) if lines else 0.0,
        heading_count=heading_count,
        bold_span_count=len(_BOLD_RE.findall(body)),
        question_count=sum(1 for s in sentences if s.rstrip(_QUESTION_TRAIL).endswith("?")),
        em_dash_per_100_words=_per_100(dash_count, word_count),
        first_person_singular_per_100_words=_per_100(first_person, word_count),
        opener_words=opener_words,
        word_count=word_count,
    )
