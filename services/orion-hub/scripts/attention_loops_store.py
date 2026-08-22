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

from orion.schemas.attention_frame import AttentionTargetTypeV1, OpenLoopV1
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    PendingAttentionCardV1,
)

logger = logging.getLogger("orion-hub.attention_loops")

_VALID_TARGET_TYPES = set(AttentionTargetTypeV1.__args__)


def _safe_target_type(raw: str) -> str:
    """Old/malformed rows shouldn't be able to take the whole panel down on a
    Pydantic Literal mismatch -- fall back to 'other' rather than raising."""
    return raw if raw in _VALID_TARGET_TYPES else "other"


_FEATURE_LABELS = {
    "evidence_strength": "strong evidence",
    "evidence_breadth": "corroborated across detectors",
}
# 2026-07-31: recurrence/recency/novelty_vs_known/dwell/habituation removed
# (killed with nothing put back -- see orion.substrate.attention.salience's
# module docstring and orion/sentience_striving_program/README.md's
# 2026-07-31 entry). A card built from an OLD, pre-kill trace row may still
# carry those keys in its stored `features` JSON; `_top_features` below
# simply won't find labels for them anymore and they drop out of the
# displayed list -- not an error, just older telemetry aging out.


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


def card_kind_for_scope(scope: str) -> str:
    """'resolvable' ONLY for scope=='chat' (a discrete, turn-scoped candidate a
    human can actually close); everything else -- 'reverie', the schema-documented
    but not-yet-produced 'broadcast', an unrecognized future value, or 'unknown'
    (a failed/missing lookup, see latest_trace_for_theme) -- is 'chronic_pressure'.
    Inverted deliberately (allowlist the safe case, not denylist the known one):
    review caught that a scope=='reverie' check alone would silently treat any
    future 'broadcast' producer as resolvable, reopening the exact false-closure-
    of-live-system-pressure failure this split exists to prevent. See
    PendingCardKindV1's docstring in orion/schemas/attention_salience.py for the
    full rationale."""
    return "resolvable" if scope == "chat" else "chronic_pressure"


def build_pending_card(
    loop: OpenLoopV1,
    *,
    first_seen: datetime,
    recurrence_count: int,
    narrative: str,
    now: datetime | None = None,
    scope: str = "chat",
) -> PendingAttentionCardV1:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    age = max(0.0, (now - first_seen).total_seconds())

    title = (loop.description or "").strip() or f"An unresolved {loop.target_type} loop"
    # Real reasoning now survives storage (attention_salience_trace.why_it_matters,
    # 2026-08-21) -- this generic sentence is a true last-resort fallback, not the
    # only text every card shows.
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
        weights_version=str((loop.salience_features or {}).get("weights_version") or "gwt-coalition-v1"),
        top_contributing_features=_top_features(loop.salience_features or {}),
        status="pending",
        card_kind=card_kind_for_scope(scope),
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
    """Suppress a closed loop out of live reverie coalition re-selection for
    `cooldown_sec` (reuse refractory table). Never raises.

    This is genuinely only a temporary cooldown, NOT what keeps a closed loop
    off the Hub panel permanently -- that guarantee now comes from
    `load_pending_loops()`'s own `attention_loop_outcome` check (added
    2026-08-22 after a live incident: this docstring used to claim the theme
    "won't re-ignite", which was only true for 24h -- ~22 loops Juniper
    resolved/dismissed on 2026-08-20 silently reappeared once this cooldown
    lapsed, because nothing else remembered they'd been judged).
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


def load_pending_loops(limit: int = 50) -> list[tuple[OpenLoopV1, datetime, int, str, str]]:
    """Return (loop, first_seen, recurrence_count, narrative, scope) worth a human's time.

    Surfacing policy (quiet panel): salience >= SURFACE_MIN_SALIENCE and age >=
    SURFACE_MIN_AGE_SEC, excluding themes already suppressed (resolved/dismissed).
    Reads the salience trace table; best-effort -> [] on any miss. `scope` is the
    most recent trace row's scope for that theme -- feeds build_pending_card's
    card_kind split (chat vs reverie/chronic).

    Verdict exclusion (2026-08-22, live-caught): `substrate_reverie_refractory`
    (checked below) is only a 24h COOLDOWN -- `suppress_loop`'s own name and
    docstring claim a human Resolve/Dismiss "won't re-ignite", but that was only
    ever true for 24 hours. Confirmed live: Juniper resolved/dismissed ~22 cards
    in one sitting on 2026-08-20; by 2026-08-22 every one of them had silently
    reappeared with the exact same stale evidence, because nothing here ever
    checked `attention_loop_outcome` directly -- once the refractory window
    lapsed, the SAME already-judged trace row just qualified again. Now also
    excludes a trace row if the loop's most recent verdict (human resolved/
    dismissed, or a system decayed_unattended) is at least as new as that row --
    i.e. "this specific piece of evidence was already judged, nothing new has
    arrived since." A row that legitimately postdates the verdict (fresh
    activity after a close) still surfaces normally -- this is a real reopen,
    not the same stale evidence again. Mirrors
    `orion/substrate/attention/verdicts.py::load_terminal_verdict_loop_ids`'s
    exclusion semantics for the live reverie coalition, which the Hub panel's
    own query never applied to itself.
    """
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (t.theme_key)
                        t.theme_key, t.loop_id, t.salience, t.features, t.description,
                        t.why_it_matters, t.target_type, t.scope, t.created_at,
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
                      AND NOT EXISTS (
                        SELECT 1 FROM attention_loop_outcome o
                        WHERE o.loop_id = t.loop_id
                          AND o.verdict IN ('resolved', 'dismissed', 'decayed_unattended')
                          AND o.created_at >= t.created_at
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

    out: list[tuple[OpenLoopV1, datetime, int, str, str]] = []
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
            target_type=_safe_target_type(str(r["target_type"] or "other")),
            why_it_matters=str(r["why_it_matters"] or ""),
            salience=float(r["salience"]),
            salience_features=features or {},
        )
        # NOT NULL DB default is 'reverie' so this Python fallback is normally
        # dead, but kept as 'unknown' (not 'reverie') for consistency with
        # latest_trace_for_theme's fail-safe default -- both feed the same
        # card_kind_for_scope allowlist, so an ambiguous scope should always
        # resolve to the same (safe) branch in every reader.
        out.append((loop, first_seen, int(r["recurrence_count"] or 1), "", str(r["scope"] or "unknown")))
    return out


def latest_trace_for_theme(theme_key: str) -> dict[str, Any]:
    """Most-recent (salience, features, scope) for a theme, in ONE round-trip.

    `_close()` used to call two separate single-column lookups
    (`latest_salience_for_theme` + a since-removed `latest_scope_for_theme`) --
    review caught both the extra query on a user-facing click path AND that the
    two functions defaulted to DIFFERENT scopes on failure ('reverie' vs 'chat').
    One function, one query fixes both.

    Failure-path default is scope='chat' (the PERMISSIVE branch), not 'unknown'
    -- a second review pass caught that defaulting a DB-error/no-row miss to the
    restrictive branch was a real regression: `latest_salience_for_theme`'s
    original contract was "closing a loop never fails" (best-effort, always
    (0.0, {}) on a miss), and this is called from `_close()` on a `loop_id` the
    Hub panel already showed the user moments earlier -- by construction that
    row existed at load_pending_loops() time, so a miss here is virtually
    always a transient hiccup, not evidence the loop is actually chronic_pressure.
    Only a SUCCESSFULLY read, real scope value routes to chronic_pressure below
    (card_kind_for_scope's allowlist) -- that's the fix that actually mattered
    (a genuinely-read non-'chat' scope, e.g. a future 'broadcast' producer,
    must not fall through to resolvable); a failed lookup must not manufacture
    a false chronic_pressure block on a legitimate human Resolve/Dismiss click.
    """
    default: dict[str, Any] = {"salience": 0.0, "features": {}, "scope": "chat"}
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT salience, features, scope
                    FROM attention_salience_trace
                    WHERE theme_key = :k
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"k": theme_key},
            ).mappings().first()
    except Exception as exc:
        logger.warning("latest_trace_for_theme failed theme=%s err=%s", theme_key, exc)
        return default
    if not row:
        return default
    features = row.get("features")
    if isinstance(features, str):
        try:
            features = json.loads(features or "{}")
        except Exception:
            features = {}
    return {
        "salience": float(row.get("salience") or 0.0),
        "features": dict(features or {}),
        "scope": str(row.get("scope") or "unknown"),
    }


def latest_salience_for_theme(theme_key: str) -> tuple[float, dict[str, Any]]:
    """Most-recent (salience, features) for a theme. Thin wrapper over
    `latest_trace_for_theme` -- kept as its own function since it's a distinct,
    already-tested public contract, not because it needs a second query."""
    trace = latest_trace_for_theme(theme_key)
    return (trace["salience"], trace["features"])
