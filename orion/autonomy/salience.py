"""Deterministic term-overlap salience scorer for readonly-fetch articles.

Narrow deterministic sensor (per repo no-regex-swamp note): a single scoring
function over gap terms vs article text. Not a cognition architecture.
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1

_GAP_SIGNAL = "world_coverage_gap"
_SECTION_PREFIX = "section:"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def iter_gap_section_labels(
    signals: Sequence[FrontierInvocationSignalV1],
) -> Iterator[str]:
    """Yield normalized section labels from `section:` focal refs across
    world_coverage_gap signals, in signal/ref order.

    Single source of truth for gap-section parsing (prefix stripped,
    underscores -> spaces). Consumed by `gap_terms_from_signals` here and by the
    query/seed builders in `policy_act.py` so the traversal does not drift.
    """
    for sig in signals:
        if getattr(sig, "signal_type", None) != _GAP_SIGNAL:
            continue
        for ref in getattr(sig, "focal_node_refs", None) or []:
            section = str(ref or "").strip()
            if section.startswith(_SECTION_PREFIX):
                yield section[len(_SECTION_PREFIX):].replace("_", " ")


def gap_terms_from_signals(
    signals: Sequence[FrontierInvocationSignalV1],
    *,
    fallback_query: str = "",
) -> set[str]:
    """Union of tokens from `section:` focal refs across world_coverage_gap signals.

    Falls back to query tokens only when no section-derived terms are found.
    """
    terms: set[str] = set()
    for label in iter_gap_section_labels(signals):
        terms |= _tokenize(label)
    if not terms and fallback_query:
        terms |= _tokenize(fallback_query)
    return terms


def score_article_salience(article_text: str, gap_terms: set[str]) -> float:
    """Fraction of gap terms present in the article text, clamped to [0, 1]."""
    if not gap_terms:
        return 0.0
    tokens = _tokenize(article_text)
    if not tokens:
        return 0.0
    overlap = gap_terms & tokens
    score = len(overlap) / len(gap_terms)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
