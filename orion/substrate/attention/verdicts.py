"""Read-only lookup of human Resolve/Dismiss verdicts for the rung-3 workspace
competition (``attention_loop_outcome``, written by a human action in the Hub
via ``services/orion-hub/scripts/attention_loops_store.py``).

Same table, same connection/query style as the two existing readers of this
table (``attention_loops_store.py`` itself and
``services/orion-thought/app/store.py::load_recent_loop_outcomes``) -- direct
SQLAlchemy engine, ``POSTGRES_URI`` env with the same conjourney default,
``DISTINCT ON (loop_id) ... ORDER BY created_at DESC`` for "most recent verdict
per loop". This module only needs the verdict and its timestamp (not
note/features), and is bounded to the loop_ids actually competing in a given
tick -- never scans the whole table.

Read-only, fail-open: any error (missing table, connection failure, bad env)
returns an empty set so a DB hiccup never blocks a broadcast tick. Callers
must treat the result as "loops known to be closed" and proceed as if no
verdicts exist otherwise.

**TTL, added 2026-08-25 (root cause: 2026-08-19 68h reverie dead window).**
Exclusion used to be permanent -- a loop resolved/dismissed once could never
compete again, even if the situation it named genuinely got worse later. The
first fix idea (re-arm when the loop's channel enters `orion.field.regime`'s
`loaded_steady`) was a category error: `channel_regime()` reads
`field_json.node_vectors`, a completely different store from the substrate
belief-graph node metadata (`dynamic_pressure`/`prediction_error`) these
signals actually come from -- no producer bridges the two, confirmed by a
repo-wide grep. `attention_loop_outcome.salience_at_close` looked like the
right existing column to compare against instead, but live data ruled that
out too: the only two loop_ids ever re-verdicted (`open-loop-5038aeb46982`,
`open-loop-64730f9cfeda`) had IDENTICAL `salience_at_close` across both of
their verdicts, and both re-closes landed in ~20-second clusters shared with
several unrelated loop_ids (08-20 16:33:2x-16:33:4x; 08-22 04:07:09-04:07:12)
-- a bulk triage sweep, not organic per-loop reassessment. A salience-delta
threshold would have been calibrated against noise.

What the same two re-verdicted loops DO show, cleanly: real gaps of
~37h22m and ~35h34m between their first and second verdict.
``VERDICT_EXCLUSION_TTL_HOURS`` below rounds that up to 48h -- n=2, disclosed
as thin, but real inter-verdict data rather than a guess. No extra salience
gate is layered on top of the TTL: `substrate_pressure_signals()`'s own
`min_salience` filter already keeps a loop from re-entering the candidate
pool at all unless it is independently salient again *today*, so the TTL only
answers "how long does a closed verdict block re-entry," not "is this loop
still real."

**Independence check.** This is not a transform of any metric already in the
rung-3 scoring path -- `evidence_strength`/`evidence_breadth`
(`orion.substrate.attention.salience`) are per-tick evidence magnitudes with
no time dimension at all; this TTL is the only place `attention_loop_outcome`
timing feeds the exclusion decision.

**Reversibility.** Cheap: ``VERDICT_EXCLUSION_TTL_HOURS`` /
``ORION_ATTENTION_VERDICT_EXCLUSION_TTL_HOURS`` is a single named constant
with an env override (see ``_ttl_hours()`` below), read fresh on every call --
not baked into a schema, manifest, or persisted row. Recalibrating it once
more verdict data accumulates needs no migration.

**Known limitation, not addressed here.** This is a blind wall-clock timer,
not an evidence-based re-arm: a loop that gets genuinely fresh, salient
evidence 5 hours after being dismissed still can't compete until hour 48, and
a loop with zero new activity still gets re-admitted at hour 48 regardless of
whether anything real changed. `services/orion-hub/scripts/
attention_loops_store.py::load_pending_loops` already compares whether fresh
evidence postdates a verdict for a *different* question (is this trace row
stale review-panel evidence) -- reusing that pattern here, or the
`dormancy_updated_at` per-node change signal `orion.substrate.dynamics`
already tracks, is the natural next step if the wall-clock TTL proves too
blunt. Not done in this patch to keep it a thin, single-mechanism fix.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable

logger = logging.getLogger("orion.substrate.attention.verdicts")

# Verdicts that mean a human explicitly closed the loop; a third valid verdict,
# "decayed_unattended" (see attention_loops_store.py's _VALID_VERDICTS), is an
# implicit non-engagement signal, not an explicit closure -- left eligible to
# compete.
TERMINAL_VERDICTS = {"resolved", "dismissed"}

# See module docstring's "TTL, added 2026-08-25" section for the live-data
# derivation (n=2 real inter-verdict gaps, ~37h22m / ~35h34m, rounded up).
VERDICT_EXCLUSION_TTL_HOURS = 48.0

# Env key for overriding the constant above without a code change/redeploy --
# same convention as attention_frame.py's _env_int/_env_float (e.g.
# ORION_CURIOSITY_MIN_ASK_SCORE), needed here specifically because the n=2
# calibration behind VERDICT_EXCLUSION_TTL_HOURS is disclosed as thin and
# likely to warrant a real revisit once more verdict data accumulates.
TTL_ENV_KEY = "ORION_ATTENTION_VERDICT_EXCLUSION_TTL_HOURS"


def _ttl_hours() -> float:
    try:
        value = float(os.getenv(TTL_ENV_KEY) or VERDICT_EXCLUSION_TTL_HOURS)
    except (TypeError, ValueError):
        return VERDICT_EXCLUSION_TTL_HOURS
    return value if value > 0 else VERDICT_EXCLUSION_TTL_HOURS


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


def load_terminal_verdict_loop_ids(
    loop_ids: Iterable[str], *, now: datetime | None = None
) -> set[str]:
    """Return the subset of ``loop_ids`` whose most recent verdict is terminal
    AND still within ``VERDICT_EXCLUSION_TTL_HOURS`` of that verdict.

    Bounded to the given loop_ids (the loops actually competing this tick).
    Best-effort: returns an empty set on any failure, including an empty/None
    input, so a lookup failure never blocks frame-building. ``now`` defaults
    to the real wall clock; callers pass it explicitly (e.g.
    ``attention_broadcast.py``'s own ``resolved_now``) to keep one tick's
    "now" internally consistent and to make this deterministically testable.
    """
    ids = sorted({str(i) for i in (loop_ids or []) if i})
    if not ids:
        return set()
    resolved_now = now or datetime.now(timezone.utc)
    if resolved_now.tzinfo is None:
        # Coerced the same way `created_at` is below -- a naive `now` must
        # not raise inside the try/except further down, where it would be
        # swallowed by the blanket DB-failure handler and silently re-arm
        # every loop_id in this batch instead of just failing one comparison.
        resolved_now = resolved_now.replace(tzinfo=timezone.utc)
    ttl = timedelta(hours=_ttl_hours())
    try:
        from sqlalchemy import bindparam, text

        stmt = text(
            """
            SELECT DISTINCT ON (loop_id) loop_id, verdict, created_at
            FROM attention_loop_outcome
            WHERE loop_id IN :ids
            ORDER BY loop_id, created_at DESC
            """
        ).bindparams(bindparam("ids", expanding=True))
        with _engine().connect() as conn:
            rows = conn.execute(stmt, {"ids": ids}).mappings().all()
        excluded: set[str] = set()
        for row in rows:
            if str(row.get("verdict") or "") not in TERMINAL_VERDICTS:
                continue
            created_at = row.get("created_at")
            if created_at is None:
                # No timestamp to judge staleness by -- fail closed on the
                # side of the pre-TTL behavior (still exclude) rather than
                # silently letting a malformed row re-arm a loop.
                excluded.add(str(row["loop_id"]))
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if resolved_now - created_at <= ttl:
                excluded.add(str(row["loop_id"]))
        return excluded
    except Exception as exc:
        logger.warning("attention_loop_outcome_verdict_lookup_failed ids=%s err=%s", ids, exc)
        return set()
