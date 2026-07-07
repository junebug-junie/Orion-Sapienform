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


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine

        _ENGINE = create_engine(_database_url(), pool_pre_ping=True)
    return _ENGINE


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
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
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


_VALID_VERDICTS = {"resolved", "dismissed", "decayed_unattended"}


def build_loop_outcome(
    *,
    loop_id: str,
    theme_key: str,
    verdict: str,
    actor: str,
    note: str,
    salience_at_close: float,
    features_at_close: dict[str, Any],
) -> AttentionLoopOutcomeV1:
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict}")
    from orion.core.ids import stable_hash_id

    return AttentionLoopOutcomeV1(
        outcome_id=stable_hash_id("loopoutcome", [loop_id, verdict, actor]),
        loop_id=loop_id,
        theme_key=theme_key,
        verdict=verdict,  # type: ignore[arg-type]
        actor=actor,
        note=(note or "")[:500],
        salience_at_close=max(0.0, min(1.0, float(salience_at_close))),
        features_at_close=dict(features_at_close or {}),
    )


def persist_loop_outcome(outcome: AttentionLoopOutcomeV1) -> bool:
    """Write one outcome label. Never raises; idempotent on outcome_id."""
    try:
        from sqlalchemy import text

        with _engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO attention_loop_outcome
                        (outcome_id, loop_id, theme_key, verdict, actor, note,
                         salience_at_close, weights_version, features_at_close, created_at)
                    VALUES
                        (:outcome_id, :loop_id, :theme_key, :verdict, :actor, :note,
                         :salience_at_close, :weights_version, CAST(:features AS jsonb), :created_at)
                    ON CONFLICT (outcome_id) DO NOTHING
                    """
                ),
                {
                    "outcome_id": outcome.outcome_id,
                    "loop_id": outcome.loop_id,
                    "theme_key": outcome.theme_key,
                    "verdict": outcome.verdict,
                    "actor": outcome.actor,
                    "note": outcome.note,
                    "salience_at_close": float(outcome.salience_at_close),
                    "weights_version": outcome.weights_version,
                    "features": json.dumps(outcome.features_at_close),
                    "created_at": outcome.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("loop outcome persist failed id=%s err=%s", outcome.outcome_id, exc)
        return False


def suppress_loop(theme_key: str, *, cooldown_sec: float = 86400.0) -> bool:
    """Suppress a closed loop so it exits the coalition (reuse refractory table).

    Resolves rather than pauses: the theme is refractory-suppressed for a long
    cooldown so it won't re-ignite. Never raises.
    """
    try:
        from datetime import timedelta

        from sqlalchemy import text

        until = datetime.now(timezone.utc) + timedelta(seconds=cooldown_sec)
        with _engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_refractory (theme_key, suppressed_until)
                    VALUES (:k, :until)
                    ON CONFLICT (theme_key)
                    DO UPDATE SET suppressed_until = EXCLUDED.suppressed_until, updated_at = now()
                    """
                ),
                {"k": theme_key, "until": until},
            )
        return True
    except Exception as exc:
        logger.warning("suppress_loop failed theme=%s err=%s", theme_key, exc)
        return False


SURFACE_MIN_SALIENCE = 0.5
SURFACE_MIN_AGE_SEC = 300.0


def load_pending_loops(limit: int = 50) -> list[tuple[OpenLoopV1, datetime, int, str]]:
    """Return (loop, first_seen, recurrence_count, narrative) worth a human's time.

    Surfacing policy (quiet panel): salience >= SURFACE_MIN_SALIENCE and age >=
    SURFACE_MIN_AGE_SEC, excluding themes already suppressed (resolved/dismissed).
    Reads the salience trace table; best-effort -> [] on any miss.
    """
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (t.theme_key)
                        t.theme_key, t.loop_id, t.salience, t.features, t.description,
                        t.created_at,
                        (SELECT count(*) FROM attention_salience_trace t2
                         WHERE t2.theme_key = t.theme_key) AS recurrence_count,
                        (SELECT min(created_at) FROM attention_salience_trace t3
                         WHERE t3.theme_key = t.theme_key) AS first_seen
                    FROM attention_salience_trace t
                    WHERE t.salience >= :min_sal
                      AND NOT EXISTS (
                        SELECT 1 FROM substrate_reverie_refractory r
                        WHERE r.theme_key = t.theme_key AND r.suppressed_until > now()
                      )
                    ORDER BY t.theme_key, t.created_at DESC
                    LIMIT :limit
                    """
                ),
                {"min_sal": SURFACE_MIN_SALIENCE, "limit": limit},
            ).mappings().all()
    except Exception as exc:
        logger.warning("load_pending_loops failed: %s", exc)
        return []

    out: list[tuple[OpenLoopV1, datetime, int, str]] = []
    now = datetime.now(timezone.utc)
    for r in rows:
        features = r["features"]
        if isinstance(features, str):
            features = json.loads(features or "{}")
        first_seen = r["first_seen"] or r["created_at"]
        fs = first_seen if first_seen.tzinfo else first_seen.replace(tzinfo=timezone.utc)
        if (now - fs).total_seconds() < SURFACE_MIN_AGE_SEC:
            continue
        description = (r["description"] or "").strip() or str(r["theme_key"])
        loop = OpenLoopV1(
            id=str(r["loop_id"]),
            description=description,
            salience=float(r["salience"]),
            salience_features=features or {},
        )
        out.append((loop, first_seen, int(r["recurrence_count"] or 1), ""))
    return out


def latest_salience_for_theme(theme_key: str) -> tuple[float, dict[str, Any]]:
    """Most-recent (salience, features) for a theme from the trace table.

    Best-effort: returns (0.0, {}) on any miss so closing a loop never fails.
    """
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT salience, features
                    FROM attention_salience_trace
                    WHERE theme_key = :k
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"k": theme_key},
            ).mappings().first()
    except Exception as exc:
        logger.warning("latest_salience_for_theme failed theme=%s err=%s", theme_key, exc)
        return (0.0, {})
    if not row:
        return (0.0, {})
    features = row["features"]
    if isinstance(features, str):
        try:
            features = json.loads(features or "{}")
        except Exception:
            features = {}
    return (float(row["salience"] or 0.0), dict(features or {}))
