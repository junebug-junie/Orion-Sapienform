"""The citation gate for anything Orion writes about its own day. Pure.

A reader of the day (stance_react, Orion's Day letter) is shown arcs with ``[arc:<16 hex>]``
on every clause. What it writes back is checked here against the ids it was actually shown:

* An ``arc:`` id that was not shown is removed from the text and counted (always).
* A sentence that names something from the shown record -- one of its subject labels or one of
  its clock times -- without an ``[arc:]`` in the same sentence (or, for stance, an arc of that
  label cited in ``evidence_refs``) is NOT blocked or retried: it is counted, and the check is
  labelled ``UNVERIFIED`` (Juniper, 2026-10-11).

The match set is the shown record's own labels, never a word list: Orion may name its plumbing
freely, the gate only asks that a claim about the recorded day point at the row it came from.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from orion.schemas.temporal_self import UNCITED_SENTENCES_SHOWN, DayCitationCheckV1

ARC_REF_RE = re.compile(r"\[arc:([0-9a-f]{16})\]")
# An evidence_refs / strain_refs entry ("arc:<id>"), as stance_react cites it.
ARC_EVIDENCE_RE = re.compile(r"^arc:([0-9a-f]{16})$")
# Any bracketed arc ref, well-formed or not, so a garbled one is still removed and counted.
_ANY_ARC_BRACKET_RE = re.compile(r"\s?\[arc:([^\]\s]*)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLOCK_LABEL_RE = re.compile(r"^\d{2}:\d{2}$")
LABEL_MIN_CHARS = 6
EXAMPLE_MAX_CHARS = 200


def arc_ref(arc_id: str) -> str:
    return f"arc:{arc_id}"


def _label_patterns(labels: Mapping[str, Sequence[str]]) -> list[tuple[re.Pattern[str], set[str]]]:
    out: list[tuple[re.Pattern[str], set[str]]] = []
    for label, arc_ids in sorted(labels.items()):
        text = (label or "").strip()
        if not text or not arc_ids:
            continue
        if _CLOCK_LABEL_RE.match(text):
            # A local clock time from the block; "15:05 UTC" elsewhere in a letter is another clock.
            pattern = re.compile(rf"(?<![\d:]){re.escape(text)}(?![\d:])(?!\s*UTC)")
        elif len(text) >= LABEL_MIN_CHARS:
            pattern = re.compile(re.escape(text), re.IGNORECASE)
        else:
            continue
        out.append((pattern, set(arc_ids)))
    return out


def check_arc_citations(
    text: str,
    shown_arc_ids: Iterable[str],
    *,
    labels: Mapping[str, Sequence[str]] | None = None,
    cited_elsewhere: Iterable[str] = (),
) -> tuple[str, DayCitationCheckV1]:
    """Gate ``text``. Returns (text with unknown arc refs removed, the check).

    ``cited_elsewhere`` are arc ids the writer cited outside ``text`` (stance's evidence_refs);
    they are already gated by the caller and only count toward a label sentence being cited."""
    shown = set(shown_arc_ids)
    elsewhere = {a for a in cited_elsewhere if a in shown}
    rejected: list[str] = []
    valid: list[str] = []

    def _strip(match: re.Match[str]) -> str:
        arc_id = match.group(1)
        if arc_id in shown:
            if arc_id not in valid:
                valid.append(arc_id)
            return match.group(0)
        rejected.append(arc_id)
        return ""

    cleaned = _ANY_ARC_BRACKET_RE.sub(_strip, text or "")
    for arc_id in sorted(elsewhere):
        if arc_id not in valid:
            valid.append(arc_id)

    uncited: list[str] = []
    patterns = _label_patterns(labels or {})
    if patterns:
        for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
            if not sentence.strip():
                continue
            named: set[str] = set()
            for pattern, arc_ids in patterns:
                if pattern.search(sentence):
                    named |= arc_ids
            if not named:
                continue
            in_sentence = {m.group(1) for m in ARC_REF_RE.finditer(sentence)} & shown
            if in_sentence or (named & elsewhere):
                continue
            uncited.append(" ".join(sentence.split())[:EXAMPLE_MAX_CHARS])

    if uncited:
        label = "UNVERIFIED"
    elif valid:
        label = "verified"
    else:
        label = "no_day_claims"
    check = DayCitationCheckV1(
        shown=len(shown),
        cited_valid=[arc_ref(a) for a in valid],
        rejected_unknown=[arc_ref(a) for a in rejected],
        uncited_day_sentences=len(uncited),
        uncited_examples=uncited[:UNCITED_SENTENCES_SHOWN],
        label=label,
        action="stripped" if rejected else "kept",
    )
    return cleaned, check


def gate_arc_refs(refs: Iterable[str], shown_arc_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split a ref list (evidence_refs / strain_refs): keep every non-arc ref and every shown arc
    ref; return (kept, rejected arc refs). An unknown arc ref must never widen an allowed set."""
    shown = set(shown_arc_ids)
    kept: list[str] = []
    rejected: list[str] = []
    for ref in refs:
        text = str(ref).strip()
        if text.startswith("[arc:") and text.endswith("]"):
            # Live 2026-10-12: the stance model copied the cue's bracketed form into evidence_refs.
            # Same citation, so normalise it rather than let the coalition filter drop it unseen.
            text = text[1:-1]
        if text.startswith("arc:"):
            if text in kept:
                continue
            m = ARC_EVIDENCE_RE.match(text)
            if m is None or m.group(1) not in shown:
                rejected.append(text)
                continue
        kept.append(text)
    return kept, rejected
