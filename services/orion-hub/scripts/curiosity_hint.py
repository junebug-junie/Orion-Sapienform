"""Curiosity focus hint for the Hub agent lane (self-observability v2).

When the agent REPL lane runs and fresh endogenous curiosity candidates exist,
prepend one bounded advisory line so introspective sessions can start from
Orion's own detected gaps. Gates are structural — flag + lane + data freshness —
never keyword classification of the prompt. Advisory only: fetch or parse
failures leave the prompt untouched.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("orion-hub.curiosity_hint")

_MAX_SUMMARIES = 2
_MAX_SUMMARY_CHARS = 120
_MAX_AGE_SEC = 120.0


def _fetch_fresh_candidates() -> list[dict[str, Any]]:
    """Latest curiosity candidate set newer than _MAX_AGE_SEC, else []."""
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return []
    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT candidates_json FROM substrate_endogenous_curiosity_candidates
                        WHERE generated_at >= now() - make_interval(secs => :max_age)
                        ORDER BY generated_at DESC LIMIT 1
                        """
                    ),
                    {"max_age": _MAX_AGE_SEC},
                )
                .mappings()
                .first()
            )
    finally:
        engine.dispose()
    if not row:
        return []
    candidates = row["candidates_json"]
    if isinstance(candidates, str):
        candidates = json.loads(candidates)
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, dict)]


def format_curiosity_hint(candidates: list[dict[str, Any]]) -> str | None:
    """One bounded hint line from the strongest candidate summaries, or None."""
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("signal_strength") or 0.0),
        reverse=True,
    )
    summaries: list[str] = []
    for candidate in ranked:
        summary = str(candidate.get("evidence_summary") or "").strip()
        if not summary:
            continue
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[: _MAX_SUMMARY_CHARS - 1] + "…"
        summaries.append(summary)
        if len(summaries) >= _MAX_SUMMARIES:
            break
    if not summaries:
        return None
    return "[curiosity focus] Self-observed gaps: " + "; ".join(summaries)


def apply_curiosity_hint(prompt: str) -> str:
    """Prepend the focus hint to ``prompt`` when available; never raises."""
    try:
        candidates = _fetch_fresh_candidates()
        hint = format_curiosity_hint(candidates) if candidates else None
        if not hint:
            return prompt
        return f"{hint}\n\n{prompt}"
    except Exception as exc:
        logger.warning("curiosity_hint_failed error=%s", exc)
        return prompt
