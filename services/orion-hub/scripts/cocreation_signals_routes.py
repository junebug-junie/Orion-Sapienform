"""Read-only Hub API for the "cocreation signals" (codebase-mass) operator tab.

Surfaces the real data behind the codebase-mass Predictive Processing domain
(``docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md``):
``orion-cocreation-signals`` publishes ``CodebaseDeltaV1`` events (git churn /
PR lifecycle / graph structural deltas) to
``orion:substrate:codebase_delta``, ``orion-substrate-runtime``'s
``_handle_codebase_delta_message()`` scores each one with
``codebase_prediction_error()`` and persists two things per real tick:

1. ``substrate_codebase_mass_baseline`` -- the running EWMA state (one row
   per tick, only the latest matters for "what does the baseline think is
   normal right now").
2. ``substrate_codebase_delta_log`` -- the raw per-tick payload (real commit
   counts / PR numbers / graph deltas) plus that tick's score, unconditional
   (not gated on score, unlike the separate ``ReductionReceiptV1`` audit
   trail feeding ``orion-field-digester``) -- a complete history, added
   2026-07-31 specifically so a tab like this one would have real data to
   show instead of only the current aggregate.

Everything here is a read. No writes, no bus publishes, no new channel or
schema. A number this tab shows is a number some real producer actually
produced -- there is no synthetic/demo data path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

router = APIRouter(prefix="/api/cocreation-signals", tags=["cocreation-signals"])

KNOWN_DOMAINS: tuple[str, ...] = ("git", "pr_lifecycle", "graph")

ALLOWED_HISTORY_HOURS: frozenset[int] = frozenset({1, 6, 24, 168})
DEFAULT_HISTORY_HOURS: int = 24
HISTORY_ROW_CAP: int = 4000

_engine_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


# --------------------------------------------------------------------------
# Pure helpers (no I/O) -- separated so shaping logic is testable against
# fixed rows, matching this repo's established route-module convention (see
# attention_organ_routes.py's build_domain_rows/summarize_history).
# --------------------------------------------------------------------------


def normalize_history_hours(hours: int | None) -> int:
    if hours in ALLOWED_HISTORY_HOURS:
        return int(hours)
    return DEFAULT_HISTORY_HOURS


def _coerce_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return {}
    return payload if isinstance(payload, dict) else {}


def build_domain_summary(
    latest_rows: list[dict[str, Any]],
    window_counts: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """One row per known domain: most recent real tick + this window's stats.

    Every domain in KNOWN_DOMAINS is always present, even with zero data --
    a domain with no events at all is a different, louder fact than a domain
    with a calm score, and must not be silently omitted from the list.
    """
    now = now or datetime.now(timezone.utc)
    latest_by_domain = {str(r.get("domain")): r for r in latest_rows}
    counts_by_domain = {str(r.get("domain")): r for r in window_counts}

    out: list[dict[str, Any]] = []
    for domain in KNOWN_DOMAINS:
        latest = latest_by_domain.get(domain)
        counts = counts_by_domain.get(domain)

        observed_at = latest.get("observed_at") if latest else None
        age_sec: float | None = None
        if isinstance(observed_at, datetime):
            ts = observed_at if observed_at.tzinfo else observed_at.replace(tzinfo=timezone.utc)
            age_sec = round((now - ts).total_seconds(), 1)

        out.append(
            {
                "domain": domain,
                "present": latest is not None,
                "latest_score": float(latest["score"]) if latest and latest.get("score") is not None else None,
                "latest_observed_at": observed_at.isoformat() if isinstance(observed_at, datetime) else None,
                "latest_age_sec": age_sec,
                "latest_payload": _coerce_json(latest.get("event_json")).get(domain) if latest else None,
                "window_event_count": int(counts["n"]) if counts else 0,
                "window_avg_score": float(counts["avg_score"]) if counts and counts.get("avg_score") is not None else None,
                "window_max_score": float(counts["max_score"]) if counts and counts.get("max_score") is not None else None,
            }
        )
    return out


def build_history_series(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split a flat, time-ordered row list into one series per domain.

    Kept as a plain dict-of-lists (not a merged/interpolated series) since
    the three domains tick on entirely different, unrelated cadences (git
    ~60s when there's a new commit, pr_lifecycle ~15min unconditional, graph
    ~5min) -- forcing them onto one shared x-axis would either lie about
    resolution or require inventing values no producer produced.
    """
    series: dict[str, list[dict[str, Any]]] = {domain: [] for domain in KNOWN_DOMAINS}
    for row in rows:
        domain = str(row.get("domain"))
        observed_at = row.get("observed_at")
        score = row.get("score")
        series.setdefault(domain, []).append(
            {
                "observed_at": observed_at.isoformat() if isinstance(observed_at, datetime) else str(observed_at),
                "score": float(score) if score is not None else None,
            }
        )
    return series


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def _load_latest_baseline() -> tuple[dict[str, Any] | None, datetime | None]:
    with _engine().connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT generated_at, baseline_json
                    FROM substrate_codebase_mass_baseline
                    ORDER BY generated_at DESC
                    LIMIT 1
                    """
                )
            )
            .mappings()
            .first()
        )
    if not row:
        return None, None
    return _coerce_json(row["baseline_json"]), row["generated_at"]


def _load_latest_per_domain() -> list[dict[str, Any]]:
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (domain) domain, score, observed_at, event_json
                    FROM substrate_codebase_delta_log
                    ORDER BY domain, observed_at DESC
                    """
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def _load_window_counts(*, hours: int) -> list[dict[str, Any]]:
    with _engine().connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                    SELECT domain, count(*) AS n, avg(score) AS avg_score, max(score) AS max_score
                    FROM substrate_codebase_delta_log
                    WHERE observed_at >= NOW() - (:hours * INTERVAL '1 hour')
                    GROUP BY domain
                    """
                ),
                {"hours": hours},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


@router.get("/snapshot")
def snapshot() -> dict[str, Any]:
    """One poll = one request. Baseline and per-domain summary degrade
    independently: either being unavailable leaves its own block flagged
    rather than 500ing the whole response.
    """
    now = datetime.now(timezone.utc)

    baseline: dict[str, Any] | None = None
    baseline_generated_at: datetime | None = None
    baseline_error: str | None = None
    try:
        baseline, baseline_generated_at = _load_latest_baseline()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        baseline_error = f"{type(exc).__name__}: {exc}"

    domains: list[dict[str, Any]] = []
    domains_error: str | None = None
    try:
        domains = build_domain_summary(
            _load_latest_per_domain(),
            _load_window_counts(hours=24),
            now=now,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        domains_error = f"{type(exc).__name__}: {exc}"

    return {
        "generated_at": now.isoformat(),
        "baseline": {
            "ok": baseline is not None,
            "error": baseline_error,
            "generated_at": baseline_generated_at.isoformat() if baseline_generated_at else None,
            "model": baseline,
        },
        "domains": {
            "ok": domains_error is None,
            "error": domains_error,
            "known_domains": list(KNOWN_DOMAINS),
            "rows": domains,
        },
    }


@router.get("/history")
def history(hours: int = Query(DEFAULT_HISTORY_HOURS)) -> dict[str, Any]:
    window_hours = normalize_history_hours(hours)
    with _engine().connect() as conn:
        # DESC + reverse (not ASC + LIMIT) so a window wide enough to hit the
        # cap keeps the NEWEST rows, not the oldest -- same reasoning as
        # attention_organ_routes.history().
        result = (
            conn.execute(
                text(
                    """
                    SELECT domain, score, observed_at
                    FROM substrate_codebase_delta_log
                    WHERE observed_at >= NOW() - (:hours * INTERVAL '1 hour')
                    ORDER BY observed_at DESC
                    LIMIT :row_cap
                    """
                ),
                {"hours": window_hours, "row_cap": HISTORY_ROW_CAP},
            )
            .mappings()
            .all()
        )
    rows = [dict(r) for r in reversed(result)]
    series = build_history_series(rows)
    return {
        "window_hours": window_hours,
        "row_cap": HISTORY_ROW_CAP,
        "truncated": len(rows) >= HISTORY_ROW_CAP,
        "sample_count": len(rows),
        "series": series,
    }
