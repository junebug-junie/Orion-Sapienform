"""Act downstream use: was an act's output used by a later decision? (body economy, Organ 3b (a)).

Spec: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Organ 3b weight source (a)
"downstream use by decision sites", and Patch A results item 7. Read-only over its sources; it
writes only ``act_downstream_use`` (one row per ``grammar_acts`` act).

Today there is ONE working output -> decision link, and it is arc-level:

    act -> its output row ``<table>:<id>``
        -> a Temporal Self arc whose ``arc_json.evidence_refs[]`` holds that ref
        -> a ``thought_decision`` whose ``day_citations.cited_valid[]`` cites ``arc:<arc_id>``

An act is ``used`` when such a decision was made within USE_WINDOW_HOURS of the act's output
(``closed_at``, else ``opened_at``), and not before the act opened. ``first_use_at`` is the first
such decision; ``site`` names the decision site (only ``thought_decision.day_citations`` exists).

How an act finds its output row (``link_path``):

- ``direct``: the act's ``object_ref`` or ``test_before`` already is ``<arc source table>:<id>``
  (e.g. ``hub.retro_coplay``'s ``retro_play_sessions:<key>``). A producer that names its output
  this way links with no change here.
- ``painting``: ``thought.painting`` -> ``reverie_visual_chain`` whose
  ``chain_json.run_request.dispatch_id`` is the act's ``visual_dispatch:<dispatch_id>``.
- ``chat_row``: ``hub.chat_reply`` -> ``chat_history_log.id`` = the act's correlation id (the whole
  turn row: Juniper's message and Orion's reply together).
- ``retro_leisure``: ``hub.retro_leisure`` -> ``retro_play_sessions.dispatch_id`` = correlation id.

``link_status``:

- ``linked``: the output row was found; ``used`` is true or false.
- ``no_output``: the family has a path but this act's output row was not found (a failed or
  expired act); ``used`` is false (nothing existed to use).
- ``no_link_path``: the family's output can never reach an arc (e.g. ``hub.world_pulse_read``:
  no reading table is an arc source; ``thought.dream_hop``: dream carry pictures write no chain
  row; ``hub.outreach``: its chat row exists, but Temporal Self binds Orion's own chat rows as
  context, never as arc evidence (orion/temporal_self/sources.py chat_turn, danger mode 9), so
  live 0 of 13 outreach rows ever appear in any arc's evidence_refs). ``used`` is NULL, never
  false: "cannot be measured" is not "0% used".

``window_closed`` is false while a later decision could still flip ``used`` to true; a reader
must not count an open-window false as a final "not used".

Known limits (stated, not hidden):

- Snapshot: arcs are rewritten in place (a nightly rebuild plus intra-day updates) and keep no
  version history, so the link reads the arc's CURRENT evidence_refs, not what it held when the
  decision cited it. Guarded only by: the decision is after the act started and after its output
  row existed.
- Arc-granular: one citation of an arc credits every act in its evidence. ``arc_evidence_n`` is
  stored so a reader can give each 1 / N (the family report's ``credit_closed``).
- One decision site (the chat stance), so use is gated on Juniper chatting.

``started_at`` is the act's ``opened_at``, or its ``created_at`` for a close-before-open act.

Incremental: each pass recomputes every act with no row yet, every act whose window is still
open, and acts that opened in the last RECOMPUTE_HOURS. An empty table backfills every act.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text

from app.settings import Settings

logger = logging.getLogger("sql-writer.act_downstream_use")

USE_WINDOW_HOURS = 72
# Longest act horizon today is outreach's 36000 s (10 h); 72 h window + 10 h + a day of slack.
RECOMPUTE_HOURS = 24 * 5
SITE = "thought_decision.day_citations"

# The tables Temporal Self arcs cite as evidence (orion/temporal_self/sources.py ``table=`` and the
# ``f"<table>:{...}"`` refs in orion/temporal_self/arcs.py). Pinned by tests/test_act_downstream_use.py
# against both files: a new arc source shows up here.
ARC_SOURCE_TABLES = (
    "attention_loop_outcome", "attention_salience_trace", "chat_history_log", "curiosity_run_outcomes",
    "dream_cycle", "dream_hypothesis", "episode_memory", "field_dominance_run", "gpu_pool_events",
    "memory_consolidation_windows", "orion_metacog", "regulation", "retro_play_sessions",
    "reverie_visual_attempt", "reverie_visual_chain", "substrate_action_outcomes", "substrate_attention_schema",
    "substrate_attention_broadcast_log", "substrate_reverie_chain", "substrate_reverie_thought", "vision_events", "vision_handoff_street_outbox",
    "vision_motion_story", "vision_presence_transition",
)

# Families with an explicit (non-direct) join path. Any other actor links only through a
# ``direct`` ref, else it is ``no_link_path``.
PATH_ACTORS = ("thought.painting", "hub.chat_reply", "hub.retro_leisure")

ROLLUP_SQL = text(
    """
WITH acts AS (
  -- Recomputed: every act not yet materialized, every act whose window is still open (so a stalled
  -- writer or a late-replayed act is caught up, never frozen at window_closed=false), and the
  -- last :recompute_hours of acts. An empty table backfills every act.
  SELECT a.act_id, a.actor, a.verb, a.object_ref, a.test_before, a.correlation_id,
         COALESCE(a.opened_at, a.created_at) AS started_at,
         COALESCE(a.closed_at, a.opened_at, a.created_at) AS output_at,
         COALESCE(a.closed_at, a.opened_at, a.created_at) + make_interval(hours => :use_window_hours) AS window_end
  FROM grammar_acts a
  LEFT JOIN act_downstream_use x ON x.act_id = a.act_id
  WHERE x.act_id IS NULL OR NOT x.window_closed
     OR COALESCE(a.opened_at, a.created_at) >= now() - make_interval(hours => :recompute_hours)
), since AS (
  SELECT COALESCE(min(started_at), now()) AS t FROM acts
), outputs AS (
  -- output_at: when the output row existed. A decision before it cannot have used it.
  SELECT a.act_id, v.r AS output_ref, 'direct' AS link_path, a.output_at
  FROM acts a CROSS JOIN LATERAL (VALUES (a.object_ref), (a.test_before)) v(r)
  WHERE position(':' IN v.r) > 0 AND split_part(v.r, ':', 1) = ANY(CAST(:arc_tables AS text[]))
  UNION ALL
  SELECT a.act_id, 'reverie_visual_chain:' || c.chain_id, 'painting', c.created_at
  FROM acts a JOIN reverie_visual_chain c
    ON c.chain_json->'run_request'->>'dispatch_id' = substr(a.test_before, length('visual_dispatch:') + 1)
  WHERE a.actor = 'thought.painting' AND a.test_before LIKE 'visual_dispatch:%'
    AND c.created_at >= a.started_at - interval '1 hour'
  UNION ALL
  SELECT a.act_id, 'chat_history_log:' || h.id, 'chat_row', (h.created_at AT TIME ZONE 'UTC')
  FROM acts a JOIN chat_history_log h ON h.id = a.correlation_id
  WHERE a.actor = 'hub.chat_reply'
  UNION ALL
  SELECT a.act_id, 'retro_play_sessions:' || s.session_id, 'retro_leisure', s.created_at
  FROM acts a JOIN retro_play_sessions s ON s.dispatch_id = a.correlation_id
  WHERE a.actor = 'hub.retro_leisure'
), cited AS (
  SELECT d.id AS decision_id, (d.created_at AT TIME ZONE 'UTC') AS decided_at, arc.arc_id, ref.r AS evidence_ref,
         jsonb_array_length(arc.arc_json->'evidence_refs') AS arc_evidence_n
  FROM thought_decision d
  CROSS JOIN LATERAL jsonb_array_elements_text(
      CASE WHEN jsonb_typeof(d.day_citations->'cited_valid') = 'array'
           THEN d.day_citations->'cited_valid' ELSE '[]'::jsonb END) c(cite)
  JOIN temporal_self_arc arc ON c.cite LIKE 'arc:%' AND arc.arc_id = substr(c.cite, 5)
  CROSS JOIN LATERAL jsonb_array_elements_text(
      CASE WHEN jsonb_typeof(arc.arc_json->'evidence_refs') = 'array'
           THEN arc.arc_json->'evidence_refs' ELSE '[]'::jsonb END) ref(r)
  WHERE d.created_at >= ((SELECT t FROM since) AT TIME ZONE 'UTC')
), first_use AS (
  SELECT DISTINCT ON (o.act_id) o.act_id, o.link_path, u.decided_at, u.decision_id, u.arc_id, u.evidence_ref,
         u.arc_evidence_n
  FROM outputs o
  JOIN acts a ON a.act_id = o.act_id
  JOIN cited u ON u.evidence_ref = o.output_ref
             AND u.decided_at >= a.started_at AND u.decided_at >= o.output_at
             AND u.decided_at <= a.window_end
  ORDER BY o.act_id, u.decided_at, u.decision_id
), one_output AS (
  SELECT DISTINCT ON (act_id) act_id, output_ref, link_path FROM outputs ORDER BY act_id, output_ref
)
INSERT INTO act_downstream_use (act_id, actor, verb, started_at, link_status, link_path, output_ref, used,
                                first_use_at, site, evidence_ref, arc_id, arc_evidence_n, decision_id,
                                window_end, window_closed, refreshed_at)
SELECT a.act_id, a.actor, a.verb, a.started_at,
       CASE WHEN o.act_id IS NOT NULL THEN 'linked'
            WHEN a.actor = ANY(CAST(:path_actors AS text[])) THEN 'no_output'
            ELSE 'no_link_path' END,
       COALESCE(f.link_path, o.link_path),
       COALESCE(f.evidence_ref, o.output_ref),
       CASE WHEN o.act_id IS NOT NULL THEN f.act_id IS NOT NULL
            WHEN a.actor = ANY(CAST(:path_actors AS text[])) THEN false
            ELSE NULL END,
       f.decided_at, CASE WHEN f.act_id IS NOT NULL THEN :site END, f.evidence_ref, f.arc_id, f.arc_evidence_n,
       f.decision_id, a.window_end, a.window_end <= now(), now()
FROM acts a
LEFT JOIN one_output o ON o.act_id = a.act_id
LEFT JOIN first_use f ON f.act_id = a.act_id
ON CONFLICT (act_id) DO UPDATE
SET actor = EXCLUDED.actor, verb = EXCLUDED.verb, started_at = EXCLUDED.started_at,
    link_status = EXCLUDED.link_status, link_path = EXCLUDED.link_path, output_ref = EXCLUDED.output_ref,
    used = EXCLUDED.used, first_use_at = EXCLUDED.first_use_at, site = EXCLUDED.site,
    evidence_ref = EXCLUDED.evidence_ref, arc_id = EXCLUDED.arc_id, arc_evidence_n = EXCLUDED.arc_evidence_n,
    decision_id = EXCLUDED.decision_id, window_end = EXCLUDED.window_end,
    window_closed = EXCLUDED.window_closed, refreshed_at = EXCLUDED.refreshed_at
"""
)

# Per family over the last :days days. Families are actors.
# - no_link_path family: status no_link_path, every rate NULL (never 0).
# - use_rate: used / linked, closed windows only (did an output that existed get used).
# - use_rate_all: used / (linked + no_output), closed windows only (did an act of this family
#   end up used at all; failed acts count against it).
# - credit_closed: sum of 1 / arc_evidence_n over used closed acts. Use is arc-granular: one
#   citation of an arc credits every act in its evidence. A conversation arc holding 12 chat rows
#   gives each 1/12 here, so weight consumers can tell one cited arc from twelve.
FAMILY_REPORT_SQL = text(
    """
SELECT actor,
       count(*) AS acts,
       count(*) FILTER (WHERE link_status = 'linked') AS linked,
       count(*) FILTER (WHERE link_status = 'linked' AND window_closed) AS linked_closed,
       count(*) FILTER (WHERE link_status = 'no_output' AND window_closed) AS no_output_closed,
       count(*) FILTER (WHERE used) AS used,
       count(*) FILTER (WHERE used AND window_closed) AS used_closed,
       count(DISTINCT arc_id) FILTER (WHERE used AND window_closed) AS citing_arcs_closed,
       round(COALESCE(sum(1.0 / NULLIF(arc_evidence_n, 0)) FILTER (WHERE used AND window_closed), 0)::numeric, 3)
         AS credit_closed,
       CASE WHEN bool_and(link_status = 'no_link_path') THEN 'no_link_path'
            WHEN count(*) FILTER (WHERE link_status = 'linked') = 0 THEN 'no_output'
            ELSE 'linked' END AS status
FROM act_downstream_use
WHERE started_at >= now() - make_interval(days => :days)
GROUP BY actor
ORDER BY actor
"""
)


def _params() -> dict:
    return {
        "recompute_hours": RECOMPUTE_HOURS,
        "use_window_hours": USE_WINDOW_HOURS,
        "arc_tables": list(ARC_SOURCE_TABLES),
        "path_actors": list(PATH_ACTORS),
        "site": SITE,
    }


def run_one_rollup(engine=None) -> int:
    if engine is None:
        from app.db import engine

    with engine.begin() as conn:
        result = conn.execute(ROLLUP_SQL, _params())
        return int(result.rowcount or 0)


def family_report(engine=None, days: int = 14) -> list[dict]:
    """Per family: acts, linked acts, used, use_rate (used / linked), use_rate_all (used / linked +
    no_output) and credit_closed, all over closed windows; None for a no_link_path family."""
    if engine is None:
        from app.db import engine

    with engine.connect() as conn:
        rows = conn.execute(FAMILY_REPORT_SQL, {"days": days}).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        measurable = d["status"] != "no_link_path"
        both = d["linked_closed"] + d["no_output_closed"]
        d["use_rate"] = round(d["used_closed"] / d["linked_closed"], 3) if measurable and d["linked_closed"] else None
        d["use_rate_all"] = round(d["used_closed"] / both, 3) if measurable and both else None
        d["credit_closed"] = float(d["credit_closed"]) if measurable else None
        out.append(d)
    return out


async def act_downstream_use_loop(settings: Settings) -> None:
    interval = float(getattr(settings, "act_downstream_use_interval_sec", 0.0) or 0.0)
    if interval <= 0:
        logger.warning(
            "act_downstream_use_loop DISABLED (ACT_DOWNSTREAM_USE_INTERVAL_SEC=%s); weight source (a) "
            "has no rows", interval,
        )
        return
    logger.info("act_downstream_use_loop starting interval_sec=%.0f", interval)
    while True:
        try:
            rows = await asyncio.to_thread(run_one_rollup)
            logger.info("[ACT_DOWNSTREAM_USE] tick_complete rows_upserted=%d", rows)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("act_downstream_use_failed error=%s", exc)
        await asyncio.sleep(interval)


if __name__ == "__main__":  # read-only report: python -m app.act_downstream_use [days]
    import sys

    for row in family_report(days=int(sys.argv[1]) if len(sys.argv) > 1 else 14):
        print(row)
