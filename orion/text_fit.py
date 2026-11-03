"""Fit text into a character budget without cutting a sentence in half.

A blind ``text[:n]`` cut lands mid-word and drops the end of the text, which is
often where the qualification lives. Live 2026-11-03: 77 of 141 lived-answer
rows in ``self_concept_history`` ended inside a word at exactly 1200 chars.

``fit_to_budget`` keeps the longest run of whole sentences that fits, and marks
any cut with an ellipsis. It never makes text longer than ``budget``.
"""

from __future__ import annotations

import re

ELLIPSIS = "…"

# A sentence ends at . ! ? (optionally followed by closing quotes or brackets)
# when whitespace or the end of the text follows.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s|$)")

# A sentence boundary this early would throw away most of the text; below it,
# a word boundary is the better cut.
_MIN_SENTENCE_KEEP = 0.5


def fit_to_budget(text: str, budget: int) -> str:
    """``text`` unchanged when it fits; else whole sentences plus an ellipsis.

    Falls back to the last word boundary when no sentence ends in the second
    half of the window, and to a hard cut only when the window holds no word
    boundary at all (one unbroken token)."""
    s = str(text or "").strip()
    if budget <= 0:
        return ""
    if len(s) <= budget:
        return s
    if budget == 1:
        return ELLIPSIS
    # Room for " …" after the kept text.
    window = s[: max(1, budget - 2)]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends and ends[-1] >= len(window) * _MIN_SENTENCE_KEEP:
        return window[: ends[-1]].rstrip() + " " + ELLIPSIS
    space = window.rfind(" ")
    if space > 0:
        return window[:space].rstrip(" ,;:") + ELLIPSIS
    return s[: budget - 1] + ELLIPSIS
