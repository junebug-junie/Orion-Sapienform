"""The integration metric's reads, shared by orion-durable-runs ``GET /temporal-self/integration`` and
``scripts/report_integration_coverage.py`` (and the daily roundup, integration patch 4), so the number
has one definition. SQL strings only (psycopg ``%(name)s`` params, bounds ``lo``/``hi``); the arithmetic is
``orion.temporal_self.now.use_rate``.

Spec: docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md, "The integration metric".
"""

from __future__ import annotations

# Experiences recorded in the window (by when they happened), per source kind.
RATE_KINDS_SQL = """
SELECT source_kind, count(*) AS n FROM temporal_self_event
WHERE occurred_at >= %(lo)s AND occurred_at < %(hi)s GROUP BY source_kind
"""
# Valid uses of those experiences made within 24 h of when they could first be read (ended_at, else
# occurred_at; experience_use holds only valid cites).
RATE_USES_SQL = """
SELECT DISTINCT u.event_id, e.source_kind, u.citing_organ
FROM experience_use u JOIN temporal_self_event e ON e.event_id = u.event_id
WHERE e.occurred_at >= %(lo)s AND e.occurred_at < %(hi)s
  AND u.cited_at < COALESCE((e.payload_json->>'ended_at')::timestamptz, e.occurred_at) + interval '24 hours'
"""
# Valid cites each organ made in the window.
RATE_CITES_MADE_SQL = """
SELECT citing_organ, count(*) AS n FROM experience_use
WHERE cited_at >= %(lo)s AND cited_at < %(hi)s GROUP BY citing_organ
"""
# Now reads per reader organ in the window, split three ways (#3498): cited (a valid use), closed with no cite
# ("read the now and cited nothing": the reader's model ran and used none of it), and never closed (the model never
# saw it: a refusal, a timeout, a preempted turn; or a reader that does not close). A row with a use counts as
# cited even if it predates closed_at.
NOW_READS_SQL = """
SELECT s.reader_organ, count(*) AS reads,
  count(*) FILTER (WHERE s.closed_at IS NOT NULL
                     AND NOT EXISTS (SELECT 1 FROM experience_use u WHERE u.shown_id = s.shown_id)) AS reads_no_cite,
  count(*) FILTER (WHERE s.closed_at IS NULL
                     AND NOT EXISTS (SELECT 1 FROM experience_use u WHERE u.shown_id = s.shown_id)) AS reads_unclosed
FROM experience_shown s WHERE s.shown_at >= %(lo)s AND s.shown_at < %(hi)s GROUP BY s.reader_organ
"""
