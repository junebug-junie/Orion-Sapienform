"""Keyword/price interest scoring.

Deliberately NOT an NER pass or an LLM call. Category pages already narrow
the domain (every candidate here already came from an Electronics/Computers/
FREE KSL search), titles are short and keyword-dense (confirmed live
2026-09-04: real titles like "Asus Rog Nuc 2025 - RTX 5060 - 32 GB DDR5" and
"LOADED HP ELITEBOOK 13TH GEN I7 32GB 512GB WIN 11 W/FACTORY WARRANTY" carry
the whole signal in the title alone), and a fixed keyword list keeps every
score inspectable -- `interest_reasons` is always the literal rule text that
fired, never a bare number with no explanation (AGENTS.md "no empty-shell
cognition").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# A listing in the typical resale range for used tech reads as a more
# credible tech find than a $0.01 placeholder or a $50,000 outlier -- neither
# of which this crawl is built to evaluate. This is a confirmation bonus on
# top of keyword hits, never a standalone trigger: a listing with zero
# keyword hits gets zero reasons and zero score regardless of price.
_PRICE_BAND_MIN = 20.0
_PRICE_BAND_MAX = 10000.0
_PRICE_BAND_WEIGHT = 0.5


@dataclass(frozen=True)
class InterestRule:
    keyword: Optional[str]
    weight: float


def rules_from_rows(rows: Iterable[dict[str, Any]]) -> list[InterestRule]:
    """Build scoring rules from `exo_exploration_interest_rules` rows.

    Category-scope rows (keyword IS NULL) are informational only -- they
    document which category URLs this crawl covers -- and contribute no
    score. Only keyword rows score a candidate.
    """
    out: list[InterestRule] = []
    for row in rows:
        keyword = row.get("keyword")
        if not keyword:
            continue
        out.append(InterestRule(keyword=keyword, weight=float(row.get("weight") or 0.0)))
    return out


def _word_boundary_pattern(keyword: str) -> re.Pattern[str]:
    # Multi-word keywords ("network switch") match as a literal phrase;
    # single-word keywords match on a word boundary so "ram" does not fire
    # on "diagram" or "ramp".
    escaped = re.escape(keyword.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def score_candidate(
    *,
    title: str,
    description: Optional[str],
    price: Optional[float],
    rules: list[InterestRule],
) -> tuple[float, list[str]]:
    """Returns (score, reasons). `reasons` are literal, human-readable strings
    naming exactly what fired -- never a bare number.
    """
    haystack = f"{title} {description or ''}".lower()
    score = 0.0
    reasons: list[str] = []
    seen_keywords: set[str] = set()
    for rule in rules:
        if not rule.keyword:
            continue
        kw = rule.keyword.lower()
        if kw in seen_keywords:
            continue
        if _word_boundary_pattern(kw).search(haystack):
            seen_keywords.add(kw)
            score += rule.weight
            reasons.append(f"keyword:'{rule.keyword}' matched title/description (+{rule.weight:g})")

    if score > 0 and price is not None and _PRICE_BAND_MIN <= price <= _PRICE_BAND_MAX:
        score += _PRICE_BAND_WEIGHT
        reasons.append(
            f"price ${price:,.2f} is in the typical used-tech range "
            f"${_PRICE_BAND_MIN:g}-${_PRICE_BAND_MAX:g} (+{_PRICE_BAND_WEIGHT:g})"
        )

    return score, reasons
