"""Shared read-only dream SQL: narrative dreams and OFFERED sleep-cycle hypotheses.

Copied from the introspect slice-2 plan (branch feat/introspect-dreams,
docs/superpowers/plans/2026-09-28-orion-introspect-slice2-dreams.md, which puts
the same statements in services/orion-dream/app/introspect_dreams.py) so the
daily letter (orion/orion_day/gather.py) and that tool read dreams the same way.
That slice can import these constants instead of redefining them.

The blind rule, stated once: hypotheses are a blind experiment
(orion/dream/hypotheses.py) -- curiosity shows each one once with the arm hidden.
So only hypotheses already offered to Orion (``offered_at IS NOT NULL``) are ever
selected, from both arms, and ``arm`` / ``ref_a`` / ``ref_b`` / ``cycle_json`` are
never selected. Every statement here is a SELECT.

Two parameter styles: the ``*_RECENT_SQL`` pair keeps the plan's SQLAlchemy
``:name`` form; the ``*_WINDOW_SQL`` pair is asyncpg ``$n`` form over a half-open
``[$1, $2)`` window.
"""

from __future__ import annotations

# dreams.created_at is timestamp WITHOUT time zone written by now() on a UTC server;
# AT TIME ZONE 'UTC' turns it into the timestamptz it always meant.
N_OCCURRED = "(d.created_at AT TIME ZONE 'UTC')"
N_COLS = f"d.id, d.dream_date, d.tldr, d.themes, d.narrative, {N_OCCURRED} AS occurred_at"
H_COLS = "h.hypothesis_id, h.cycle_id, h.claim, h.why, h.offered_at AS occurred_at, h.expires_at"
HYPOTHESIS_BLIND_WHERE = "h.offered_at IS NOT NULL"
# Column names that must never be selected from dream_hypothesis / dream_cycle.
BLIND_FORBIDDEN_COLUMNS = ("arm", "ref_a", "ref_b", "cycle_json")

_SINCE = "(CAST(:since AS timestamptz) IS NULL OR {col} >= CAST(:since AS timestamptz))"

NARRATIVE_RECENT_SQL = f"""
SELECT {N_COLS}, count(*) OVER () AS total FROM dreams d
WHERE d.created_at IS NOT NULL AND {_SINCE.format(col=N_OCCURRED)}
ORDER BY occurred_at DESC, d.id DESC
LIMIT :limit
"""

HYPOTHESIS_RECENT_SQL = f"""
SELECT {H_COLS}, count(*) OVER () AS total FROM dream_hypothesis h
WHERE {HYPOTHESIS_BLIND_WHERE} AND {_SINCE.format(col="h.offered_at")}
ORDER BY h.offered_at DESC, h.hypothesis_id DESC
LIMIT :limit
"""

NARRATIVE_WINDOW_SQL = f"""
SELECT {N_COLS} FROM dreams d
WHERE d.created_at IS NOT NULL AND {N_OCCURRED} >= $1 AND {N_OCCURRED} < $2
ORDER BY occurred_at, d.id
"""

HYPOTHESIS_WINDOW_SQL = f"""
SELECT {H_COLS} FROM dream_hypothesis h
WHERE {HYPOTHESIS_BLIND_WHERE} AND h.offered_at >= $1 AND h.offered_at < $2
ORDER BY h.offered_at, h.hypothesis_id
"""
