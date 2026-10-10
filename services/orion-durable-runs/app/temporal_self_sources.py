"""The chronicle's reads: every bound Temporal Self source, by the reducer's own clock.

``orion/temporal_self/evals/export_fixture_day.sql`` is the reference for every query and every
flag column here (``has_prompt``, ``unsolicited``, the ``chat_turn`` EXISTS, the nested thought
list, ``trigger_timestamp``); the replay eval loads that export into these same table shapes and
checks that this module reproduces it.

How a window is read. The reducer orders items by when they became AVAILABLE (a process event at
its end, ``arcs._event_available_at``). Each query selects a SUPERSET by an indexed natural column
(widened by ``slack`` where available time can sit after it), every row goes through the reducer's
own adapter, and only events whose available time is in ``[lo, hi)`` are kept. So a window never
double-reads and never needs SQL to restate an adapter's time rule.

Rows that change after insert (stated, so a live/replay difference is never a surprise):

* ``reverie_visual_attempt.outcome`` reads ``active``/``unknown`` while in flight (the table's own
  partial index); those are skipped here. ``abandoned`` lands ~90 min after ``started_at``, so it
  is always a late row live (the chronicle probes 3 h back for it): stored ``late_unfolded`` in
  ``temporal_self_event``, counted, never folded.
* ``attention_salience_trace``'s ``chat_turn`` EXISTS becomes true when sql-writer lands the chat
  row: up to 141 s after the trace (live, 7 days to 10-10). The read lag covers it.
* ``episode_memory`` rows are written a median 13 h after ``occurred_at`` (live 10-10), so live
  they arrive late by construction and are counted, never bound. A replay binds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from orion.schemas.temporal_self import TemporalSelfEventV1
from orion.temporal_self.broadcast import BroadcastTickView, tick_from_log_row
from orion.temporal_self.day import as_utc
from orion.temporal_self.sources import ADAPTERS

STATEMENT_TIMEOUT = "30s"  # a stalled read fails the window (retried next step), never hangs it

# Outcomes of an attempt still running (reverie_visual_attempt's own "active" partial index).
VISUAL_IN_FLIGHT = ("active", "unknown")

BROADCAST_SQL = """
SELECT log_id, generated_at,
  json_build_object('selected_open_loop_id', projection_json->'selected_open_loop_id',
    'frame', json_build_object('open_loops', COALESCE((SELECT json_agg(json_build_object(
        'id', l->'id', 'source_refs', l->'source_refs', 'description', l->'description'))
      FROM jsonb_array_elements(projection_json->'frame'->'open_loops') l), '[]'::json))) AS projection_json
FROM substrate_attention_broadcast_log
WHERE generated_at >= %(lo)s AND generated_at < %(hi)s
"""


@dataclass(frozen=True)
class SourceQuery:
    kind: str
    sql: str
    # How far before ``lo`` the natural column may sit while the event is still available
    # inside the window (a process that started earlier, a joined timestamp that precedes it).
    slack: timedelta = timedelta(0)


# %(lo)s / %(hi)s are the superset bounds (already widened by ``slack``); timestamptz params.
QUERIES: tuple[SourceQuery, ...] = (
    SourceQuery("chat_turn", """
SELECT id, correlation_id, session_id, source, created_at,
  btrim(coalesce(prompt, '')) <> '' AS has_prompt,
  coalesce((client_meta->>'unsolicited')::boolean, false) AS unsolicited
FROM chat_history_log
WHERE created_at >= (%(lo)s::timestamptz AT TIME ZONE 'UTC') AND created_at < (%(hi)s::timestamptz AT TIME ZONE 'UTC')
"""),
    SourceQuery("curiosity_run", """
SELECT d.run_id, d.turn_started_at, d.decided_at, d.arm, d.offered, o.completed_at, o.turn_ok,
  o.n_tested, o.n_moved, o.n_formed
FROM curiosity_run_outcomes o JOIN curiosity_offer_decisions d USING (run_id)
WHERE o.completed_at >= %(lo)s AND o.completed_at < %(hi)s
""", slack=timedelta(days=1)),
    # reverie_chain is read in two steps (chains, then their thoughts): see Reader._reverie.
    SourceQuery("expectation_verdict", """
SELECT thought_id, correlation_id, thought_json->>'chain_id' AS chain_id, created_at,
  expectation_verdict, expectation_scored_at
FROM substrate_reverie_thought
WHERE expectation_scored_at >= %(lo)s AND expectation_scored_at < %(hi)s
"""),
    SourceQuery("visual_run", """
SELECT c.chain_id, c.created_at, c.theme_key, c.terminal_reason, a.started_at AS attempt_started_at,
  a.result_json->'detail'->>'state' AS thermal_state
FROM reverie_visual_chain c LEFT JOIN LATERAL (
  SELECT started_at, result_json FROM reverie_visual_attempt a
  WHERE a.result_json->>'chain_id' = c.chain_id ORDER BY started_at DESC LIMIT 1) a ON true
WHERE c.created_at >= %(lo)s AND c.created_at < %(hi)s
""", slack=timedelta(days=1)),
    SourceQuery("visual_deferral", """
SELECT attempt_id, started_at, outcome,
  json_build_object('reason', result_json->>'reason', 'refused', result_json->'refused',
    'detail', json_build_object('state', result_json->'detail'->>'state')) AS result_json
FROM reverie_visual_attempt
WHERE started_at >= %(lo)s AND started_at < %(hi)s
  AND outcome IS NOT NULL AND outcome <> 'produced' AND outcome <> ALL(%(in_flight)s)
"""),
    SourceQuery("gpu_wait", """
SELECT event_id, generated_at, event, holder, priority, work_class, waited_ms, turn_correlation_id
FROM gpu_pool_events
WHERE generated_at >= %(lo)s AND generated_at < %(hi)s
  AND priority = 'background' AND holder NOT LIKE 'http:%%'
  AND (event = 'unavailable' OR (event = 'granted' AND waited_ms >= 500))
"""),
    SourceQuery("dream_cycle", """
SELECT cycle_id, trigger, status, started_at, ended_at, pressure, replay_count, hypothesis_count,
  json_build_object('pressure', json_build_object('since', cycle_json->'pressure'->>'since')) AS cycle_json
FROM dream_cycle
WHERE status = 'completed' AND ended_at >= %(lo)s AND ended_at < %(hi)s
""", slack=timedelta(days=1)),
    SourceQuery("dream_hypothesis", """
SELECT hypothesis_id, cycle_id, arm, created_at, expires_at, offered_at, offered_run_id
FROM dream_hypothesis WHERE created_at >= %(lo)s AND created_at < %(hi)s
"""),
    # Time is observed_at, else created_at; observed_at is indexed (and never null live: 0 of
    # 144,909 on 10-10), so the two branches keep the read off a 201 MB seq scan.
    SourceQuery("action_outcome", """
SELECT id, observed_at, created_at, claim_upheld, dispatch_kind, target_id, prediction_error
FROM substrate_action_outcomes WHERE observed_at >= %(lo)s AND observed_at < %(hi)s
UNION ALL
SELECT id, observed_at, created_at, claim_upheld, dispatch_kind, target_id, prediction_error
FROM substrate_action_outcomes WHERE observed_at IS NULL AND created_at >= %(lo)s AND created_at < %(hi)s
"""),
    # Time is the trigger's naive-UTC timestamp; the metacog row is written up to ~9 min later
    # (live p99 12.7 s, max 532 s), so the TEXT row time is only a lower-bounded superset. The
    # indexed TEXT prefix (a day wider, either separator) keeps it off a 611 MB seq scan; the
    # cast then decides exactly.
    SourceQuery("metacog_observation", """
SELECT m.id, m.correlation_id, m.severity, m.trigger_kind, m.timestamp, t.timestamp AS trigger_timestamp
FROM orion_metacog m LEFT JOIN LATERAL (
  SELECT timestamp FROM metacog_trigger t WHERE t.correlation_id = m.correlation_id
  ORDER BY timestamp LIMIT 1) t ON true
WHERE m.timestamp >= %(lo_key)s AND m.severity IN ('degraded', 'critical') AND m.timestamp::timestamptz >= %(lo)s
""", slack=timedelta(days=1)),
    SourceQuery("consolidation_window_close", """
SELECT memory_window_id, closed_at, turn_correlation_ids, close_reason, source_platform
FROM memory_consolidation_windows WHERE closed_at >= %(lo)s AND closed_at < %(hi)s
"""),
    SourceQuery("attention_row", """
SELECT entry_id, generated_at, process, correlation_id, attention_reason
FROM substrate_attention_schema WHERE generated_at >= %(lo)s AND generated_at < %(hi)s
"""),
    SourceQuery("attention_loop_raised", """
SELECT trace_id, loop_id, scope, correlation_id, created_at,
  EXISTS (SELECT 1 FROM chat_history_log c WHERE c.correlation_id = t.correlation_id) AS chat_turn
FROM attention_salience_trace t
WHERE scope = 'chat' AND created_at >= %(lo)s AND created_at < %(hi)s
"""),
    SourceQuery("attention_loop_verdict", """
SELECT outcome_id, loop_id, verdict, actor, created_at
FROM attention_loop_outcome WHERE created_at >= %(lo)s AND created_at < %(hi)s
"""),
    SourceQuery("field_dominance_run", """
SELECT run_id, target_id, target_kind, started_at, ended_at, tick_count, min_streak_at_run, left_censored
FROM field_dominance_run WHERE ended_at >= %(lo)s AND ended_at < %(hi)s
""", slack=timedelta(hours=1)),
    SourceQuery("vision_percept", """
SELECT event_id, stream_id, event_type, entities, created_at
FROM vision_events
WHERE created_at >= %(lo)s AND created_at < %(hi)s
  AND jsonb_typeof(entities::jsonb) = 'array' AND jsonb_array_length(entities::jsonb) > 0
"""),
    SourceQuery("memory_episode", """
SELECT memory_id::text AS memory_id, episode_id, purpose, occurred_at
FROM episode_memory WHERE occurred_at >= %(lo)s AND occurred_at < %(hi)s
"""),
)

REVERIE_CHAINS_SQL = """
SELECT chain_id, created_at, theme_key, terminal_reason FROM substrate_reverie_chain
WHERE created_at >= %(lo)s AND created_at < %(hi)s AND terminal_reason IS NOT NULL AND terminal_reason <> ''
"""
# No upper bound: a chain's available time is its LAST thought, so every thought is needed.
REVERIE_THOUGHTS_SQL = """
SELECT thought_id, created_at, correlation_id, thought_json->>'chain_id' AS chain_id
FROM substrate_reverie_thought
WHERE created_at >= %(since)s AND thought_json->>'chain_id' = ANY(%(ids)s)
"""
REVERIE_SLACK = timedelta(hours=6)  # chains run minutes (live max 0.6 min, 3 days to 10-10)
REVERIE_THOUGHT_LOOKBACK = timedelta(days=1)

BODY_CLUSTER_SQL = """
SELECT observed_at, chassis_watts FROM orion_biometrics_cluster
WHERE observed_at >= %(lo)s AND observed_at <= %(hi)s
"""
# TEXT timestamps with ' ' or 'T' separators: the indexed text range is widened by a day on each
# side (so either separator sorts inside it), then the cast decides exactly (PR #2597 drift 7).
BODY_CABINET_SQL = """
SELECT timestamp, measurements->'cabinet_temp_c' AS cabinet_temp_c FROM orion_biometrics_summary
WHERE node = 'athena' AND timestamp >= %(lo_key)s AND timestamp <= %(hi_key)s
  AND measurements ? 'cabinet_temp_c'
  AND timestamp::timestamptz >= %(lo)s AND timestamp::timestamptz <= %(hi)s
"""
BODY_SPIKE_SQL = """
SELECT spike_id, timestamp FROM cabinet_ambient_spike WHERE timestamp >= %(lo)s AND timestamp <= %(hi)s
"""


def available_at(e: TemporalSelfEventV1) -> datetime:
    """The reducer's order key time (``arcs._event_available_at``), restated as a public helper."""
    return e.ended_at or e.occurred_at


def _text_key(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")


async def _fetch(conn: Any, sql: str, params: dict) -> list[dict]:
    cur = await conn.execute(sql, params)
    return list(await cur.fetchall())


class SourceReader:
    """Reads one window from Postgres on the service's psycopg pool (dict rows)."""

    def __init__(self, pool: Any, tz_name: str) -> None:
        self._pool = pool
        self._tz = tz_name

    async def read(
        self, lo: datetime, hi: datetime, *, ticks: bool = True, kinds: Optional[Iterable[str]] = None,
    ) -> tuple[list[BroadcastTickView], list[TemporalSelfEventV1]]:
        """Broadcast ticks with ``generated_at`` in ``[lo, hi)`` and every adapted event whose
        available time is in ``[lo, hi)``, from one read-only transaction."""
        want = set(kinds) if kinds is not None else None
        events: list[TemporalSelfEventV1] = []
        out_ticks: list[BroadcastTickView] = []
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                if ticks:
                    out_ticks = [tick_from_log_row(r) for r in await _fetch(conn, BROADCAST_SQL, {"lo": lo, "hi": hi})]
                for q in QUERIES:
                    if want is not None and q.kind not in want:
                        continue
                    qlo = lo - q.slack
                    rows = await _fetch(conn, q.sql, {"lo": qlo, "hi": hi, "in_flight": list(VISUAL_IN_FLIGHT),
                                                      "lo_key": _text_key(qlo - timedelta(days=1))})
                    events.extend(ADAPTERS[q.kind](r, self._tz) for r in rows)
                if want is None or "reverie_chain" in want:
                    events.extend(await self._reverie(conn, lo, hi))
        kept = [e for e in events if e is not None and lo <= available_at(e) < hi]
        return out_ticks, kept

    async def _reverie(self, conn: Any, lo: datetime, hi: datetime) -> list[Optional[TemporalSelfEventV1]]:
        chains = await _fetch(conn, REVERIE_CHAINS_SQL, {"lo": lo - REVERIE_SLACK, "hi": hi})
        if not chains:
            return []
        since = min(as_utc(c["created_at"]) for c in chains) - REVERIE_THOUGHT_LOOKBACK
        thoughts = await _fetch(conn, REVERIE_THOUGHTS_SQL, {"since": since, "ids": [c["chain_id"] for c in chains]})
        by_chain: dict[str, list[dict]] = {}
        for t in thoughts:
            by_chain.setdefault(t["chain_id"], []).append(t)
        return [ADAPTERS["reverie_chain"](dict(c, thoughts=by_chain.get(c["chain_id"], [])), self._tz) for c in chains]

    async def read_body(self, lo: datetime, hi: datetime) -> dict[str, list[dict]]:
        """Body sensor rows in the closed interval ``[lo, hi]``, each with a UTC ``_t``."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY")
                await conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
                cluster = await _fetch(conn, BODY_CLUSTER_SQL, {"lo": lo, "hi": hi})
                cabinet = await _fetch(conn, BODY_CABINET_SQL, {
                    "lo": lo, "hi": hi, "lo_key": _text_key(lo - timedelta(days=1)),
                    "hi_key": _text_key(hi + timedelta(days=1)) + "~"})
                spikes = await _fetch(conn, BODY_SPIKE_SQL, {"lo": lo, "hi": hi})
        return {
            "cluster": [dict(r, _t=as_utc(r["observed_at"])) for r in cluster],
            "cabinet": [dict(r, _t=as_utc(r["timestamp"])) for r in cabinet],
            "spike": [dict(r, _t=as_utc(r["timestamp"])) for r in spikes],
        }
