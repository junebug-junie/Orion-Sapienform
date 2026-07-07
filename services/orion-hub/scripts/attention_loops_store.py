"""Pending Attention cognitive-loop cards + closure persistence (orion-hub).

Builds operator-legible PendingAttentionCardV1 (never id-only — hard UX rule),
reads recent loops from the salience trace table, and writes human Resolve/Dismiss
outcomes. Privacy: cards carry only plain summaries; no raw private trace/journal
material. Direct SQL (conjourney), matching the reverie persistence precedent.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from orion.schemas.attention_frame import OpenLoopV1
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    PendingAttentionCardV1,
)

logger = logging.getLogger("orion-hub.attention_loops")

_FEATURE_LABELS = {
    "evidence_strength": "strong evidence",
    "evidence_breadth": "corroborated across detectors",
    "recurrence": "keeps recurring",
    "recency": "recently observed",
    "novelty_vs_known": "novel vs known",
    "dwell": "held attention",
    "habituation": "over-attended (habituating)",
}


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _engine():
    from sqlalchemy import create_engine

    return create_engine(_database_url(), pool_pre_ping=True)


def _top_features(features: dict[str, Any], *, limit: int = 3) -> list[str]:
    scored = []
    for name, label in _FEATURE_LABELS.items():
        try:
            val = float(features.get(name, 0.0))
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.0:
            scored.append((val, label))
    scored.sort(reverse=True)
    return [label for _, label in scored[:limit]]


def build_pending_card(
    loop: OpenLoopV1,
    *,
    first_seen: datetime,
    recurrence_count: int,
    narrative: str,
    now: datetime | None = None,
) -> PendingAttentionCardV1:
    now = now or datetime.now(timezone.utc)
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    age = max(0.0, (now - first_seen).total_seconds())

    title = (loop.description or "").strip() or f"An unresolved {loop.target_type} loop"
    why = (loop.why_it_matters or "").strip() or (
        f"This {loop.target_type} has stayed active without resolution."
    )
    source = str((loop.provenance or {}).get("signal_source") or "the substrate")
    what_triggered = f"Raised by {source}; still open."

    return PendingAttentionCardV1(
        loop_id=loop.id,
        theme_key=loop.id,
        title=title,
        why_it_matters=why,
        what_triggered=what_triggered,
        narrative=(narrative or "").strip(),
        age_seconds=age,
        recurrence_count=int(recurrence_count),
        salience=float(loop.salience),
        weights_version=str((loop.salience_features or {}).get("weights_version") or "seed-v1"),
        top_contributing_features=_top_features(loop.salience_features or {}),
        status="pending",
    )
