"""Parse relative journal timeframes from natural language (V1: explicit durations only)."""

from __future__ import annotations

import re
from typing import Optional


_JOURNAL_PREFIX = re.compile(r"^\s*(please\s+)?(journal|log)\s+", re.IGNORECASE)


def parse_journal_discussion_lookback_seconds(user_text: str) -> Optional[int]:
    """
    Return lookback in seconds when the prompt is clearly a journal-discussion request
    with an explicit minute/hour/day window. None if no explicit window matched.
    """
    t = (user_text or "").strip()
    if not t:
        return None
    lowered = t.lower()
    if "journal" not in lowered and "log our chat" not in lowered:
        return None

    # Minutes: "last 34 minutes", "34 minute", "for the last 34 mins"
    m = re.search(
        r"(?:last|past|for\s+the\s+last)\s+(\d{1,4})\s*(?:minute|minutes|min|mins)\b",
        lowered,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)) * 60

    # Hours: "last hour", "last 2 hours", "an hour"
    if re.search(r"\b(last|past)\s+(?:an?\s+)?hour\b", lowered) or re.search(
        r"\bfor\s+the\s+last\s+hour\b", lowered
    ):
        return 3600
    m = re.search(r"(?:last|past|for\s+the\s+last)\s+(\d{1,3})\s*(?:hour|hours)\b", lowered, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 3600

    # Day(s)
    if re.search(r"\b(last|past)\s+day\b", lowered) or re.search(
        r"\b(last|past)\s+24\s*hours?\b", lowered
    ):
        return 86400
    m = re.search(r"(?:last|past)\s+(\d{1,3})\s*(?:day|days)\b", lowered, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 86400

    # "journal our chat discussion" without explicit window → default 24h
    if re.search(r"\b(journal|log)\b.*\b(chat|discussion)\b", lowered) and not re.search(
        r"\b(last|past)\s+\d", lowered
    ):
        if re.search(r"\b(last|past)\s+(?:an?\s+)?(?:hour|day)\b", lowered):
            return None
        return 86400

    return None


def strip_journal_command_prefix(user_text: str) -> str:
    """Best-effort strip of leading 'journal …' for secondary parsing."""
    return _JOURNAL_PREFIX.sub("", (user_text or "").strip()).strip()
