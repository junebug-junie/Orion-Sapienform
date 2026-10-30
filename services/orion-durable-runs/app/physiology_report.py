"""GET /physiology/report: one ``PhysiologyReportV1`` over a window (body economy, read-only).

Each section is one bounded SQL read plus a pure shaping function, so a test can feed rows and
check the shape without a database. Every section reads on its own: a failed read empties only
its own section and adds a ``missing`` entry with the reason, never a 500 and never a silent gap.

Speed (design H5, "under 2 s on 30 days"): Postgres does the hourly grouping, so Python only sees
aggregated rows. ``physiology_state`` is already a 2-minute rollup and ``orion_energy_hourly`` an
hourly one. ``gpu_pool_events`` has no persisted rollup today, so the waste, rhythm and queue-wait
reads group raw lease events in SQL on ``idx_gpu_pool_events_generated``. The route caches each
query for ``ORION_PHYSIOLOGY_REPORT_CACHE_SEC``.

Sources (all read-only):

* limits      -- ``physiology_state`` (prices from 2026-10-27; earlier ticks have levels only)
* wants       -- ``want_ledger`` (shadow bids from 2026-10-27), hand-run probes left out
* rhythm      -- ``gpu_pool_events`` lease closes: background (deferrable) vs all lease-seconds
* money       -- the newest ``energy`` constraint reading + ``orion_energy_hourly`` per local day
* unit_costs  -- the energy rate, ``config/physiology/heat_sensitivity.yaml``, ``energy_run_cost``,
                 ``grammar_acts.cost_json`` (``wh_by_host``/``usd``/``heat_c_by_host`` when filled),
                 ``orion_energy_attribution_hourly`` (per host per UTC day: energy above the pinned rest
                 that no act was charged, "work not yet metered", and coverage)
* waste       -- lease closes whose reason is not a success
* queue_wait  -- ``waited_ms`` on granted leases, family joined from ``want_ledger`` for durable runs
* decisions   -- ``want_ledger`` rows, newest first
* claude      -- ``dev_economics_ledger_log`` token deltas + the newest ``claude`` constraint
* vram        -- ``orion_biometrics_summary.measurements`` ``gpuN_mem_used_mb``/``gpuN_mem_total_mb``
                 per host, card -> roles from ``config/gpu_pool.yaml``
* value_per_cost -- the market's own family weights evidence (``Market.family_weights``)
* services    -- mesh-guardian ``stopped_container`` cards in ``notify_requests``
* hosts       -- per machine: newest fresh ``orion_biometrics_summary`` row (chassis, GPU, CPU, fans,
                 per card, per container) against its PINNED rest (``config/physiology/rest_body.yaml``,
                 else dated defaults), card -> lease holders from ``gpu_pool_events.cards``, daily
                 chassis average; container ids -> names from circe's container census
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from orion.schemas.physiology_report import (
    REPORT_SECTIONS,
    REPORT_WINDOWS,
    PhysiologyReportV1,
)

logger = logging.getLogger("orion-durable-runs.physiology_report")

WINDOW_DELTAS = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
assert set(WINDOW_DELTAS) == set(REPORT_WINDOWS)

#: Hand-run verification bids, same set Choice leaves out of the floor (orion/physiology/choice.py).
SYNTHETIC_PRODUCERS = ("deploy-verification", "probe", "replay", "test")
#: A lease close whose reason is one of these did its job. Anything else (attempt_failed, failed,
#: timeouts, recalls, release_failed:*) is lease time that produced nothing.
OK_CLOSE_REASONS = ("ok", "completed", "generated", "text_done")
WASTE_CLOSE_EVENTS = ("released", "expired", "aborted")
RHYTHM_CLOSE_EVENTS = ("released", "expired", "aborted")
#: An expiry or recall abort writes TWO close rows for one lease ~2 ms apart (an ``expired``/
#: ``aborted`` row and a ``released`` row with reason ``expire:*``/``abort:*``, same held_ms; 64 pairs
#: in the 30 days to 2026-10-29). The ``released`` twin is left out so a lease counts once.
DUP_CLOSE_SQL = "NOT (event = 'released' AND (reason LIKE 'expire:%%' OR reason LIKE 'abort:%%'))"
#: Prices were first posted on this tick (Patch B deploy, 2026-10-27 ~05:25Z).
PRICES_BEGAN_AT = datetime(2026, 10, 27, 5, 26, 40, tzinfo=timezone.utc)
PRICES_BEGAN_NOTE = "prices began 2026-10-27 (Patch B); earlier ticks have levels only"
DEFAULT_DECISION_LIMIT = 50
MAX_DECISION_LIMIT = 500
WASTE_TOP_N = 10
SERVICES_LOOKBACK = timedelta(days=30)
BIOMETRICS_FRESH = timedelta(minutes=15)
#: Hosts whose biometrics rows can carry GPU memory (athena's T10s are outside the pool).
VRAM_HOSTS = ("circe", "hecate", "athena")

CONSTRAINT_LABELS = {
    "heat": "Cabinet heat (my share against the soft line)",
    "energy": "Energy money this month",
    "attention:juniper": "Messages to Juniper today",
    "claude": "Claude usage limits",
}


def constraint_label(name: str) -> str:
    if name in CONSTRAINT_LABELS:
        return CONSTRAINT_LABELS[name]
    if name.startswith("lane:"):
        return f"GPU lane {name.split(':', 1)[1]} (lease-seconds per hour)"
    return name


# --------------------------------------------------------------------------------------------
# query parameters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportQuery:
    window: str = "24h"
    sections: tuple[str, ...] = REPORT_SECTIONS
    family: Optional[str] = None
    outcome: Optional[str] = None
    limit: int = DEFAULT_DECISION_LIMIT

    @property
    def cache_key(self) -> tuple:
        return (self.window, self.sections, self.family, self.outcome, self.limit)


def parse_query(window: Optional[str], sections: Optional[str], family: Optional[str],
                outcome: Optional[str], limit: Optional[int]) -> ReportQuery:
    """Raises ValueError with a plain reason on a bad parameter (the route answers 422)."""
    w = (window or "24h").strip()
    if w not in WINDOW_DELTAS:
        raise ValueError(f"window must be one of {', '.join(REPORT_WINDOWS)} (got {w!r})")
    if sections and sections.strip():
        asked = tuple(dict.fromkeys(s.strip() for s in sections.split(",") if s.strip()))
        unknown = [s for s in asked if s not in REPORT_SECTIONS]
        if unknown:
            raise ValueError(f"unknown section(s) {', '.join(unknown)}; known: {', '.join(REPORT_SECTIONS)}")
        secs = asked
    else:
        secs = REPORT_SECTIONS
    o = (outcome or "").strip() or None
    if o is not None and o not in ("funded", "deferred"):
        raise ValueError(f"outcome must be funded or deferred (got {o!r})")
    n = DEFAULT_DECISION_LIMIT if limit is None else int(limit)
    if n < 1 or n > MAX_DECISION_LIMIT:
        raise ValueError(f"limit must be 1..{MAX_DECISION_LIMIT} (got {n})")
    return ReportQuery(window=w, sections=secs, family=(family or "").strip() or None, outcome=o, limit=n)


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def _utc(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(v, datetime):
        return None
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def _f(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        return x if x == x else None  # NaN is not a reading
    except (TypeError, ValueError):
        return None


def _json(v: Any) -> Any:
    if isinstance(v, (str, bytes)):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def _r(v: Optional[float], nd: int = 3) -> Optional[float]:
    return None if v is None else round(v, nd)


def _day(d: datetime) -> str:
    return d.date().isoformat()


# --------------------------------------------------------------------------------------------
# SQL (every read bounded by the window; grouping happens in Postgres)
# --------------------------------------------------------------------------------------------

LIMITS_HOURLY_SQL = """
SELECT date_trunc('hour', s.generated_at) AS h,
       c->>'constraint' AS constraint_name,
       AVG(NULLIF(c->>'level', '')::float) AS level,
       AVG(NULLIF(c->'price'->>'value', '')::float) AS price,
       (ARRAY_AGG(COALESCE(c->'price'->>'bind_state', CASE WHEN c->>'no_reading' IS NOT NULL THEN 'no_reading' END)
                  ORDER BY s.generated_at DESC))[1] AS bind_state
FROM physiology_state s, jsonb_array_elements(s.constraints) c
WHERE s.generated_at >= %(since)s AND s.generated_at < %(until)s
GROUP BY 1, 2
ORDER BY 1, 2
"""
LIMITS_LATEST_SQL = """
SELECT generated_at, constraints FROM physiology_state
WHERE generated_at < %(until)s ORDER BY generated_at DESC LIMIT 1
"""
LIMITS_FIRST_SQL = "SELECT MIN(generated_at) AS first_at FROM physiology_state"

WANTS_HOURLY_SQL = """
SELECT date_trunc('hour', decided_at) AS h, family, outcome, COUNT(*) AS n, MAX(tier) AS tier
FROM want_ledger
WHERE decided_at >= %(since)s AND decided_at < %(until)s
  AND producer <> ALL(%(synthetic)s)
  AND (%(family)s::text IS NULL OR family = %(family)s)
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""
WANTS_FIRST_SQL = "SELECT MIN(decided_at) AS first_at FROM want_ledger WHERE producer <> ALL(%(synthetic)s)"

# Each lease is spread over the local hours it was held ([close - held, close]), clipped to the
# window, so a 2.4 h durable-run hold counts in the hours it ran, not only the hour it closed.
# A lease still open at the window's end is not counted (it has no close row yet).
RHYTHM_SQL = """
WITH leases AS (
  SELECT GREATEST(generated_at - held_ms * interval '1 millisecond', %(since)s) AS s,
         generated_at AS e, priority = 'background' AS heavy
  FROM gpu_pool_events
  WHERE generated_at >= %(since)s AND generated_at < %(until)s
    AND event = ANY(%(events)s) AND held_ms > 0 AND """ + DUP_CLOSE_SQL + """
), parts AS (
  SELECT h, heavy,
         EXTRACT(EPOCH FROM LEAST(l.e, h + interval '1 hour') - GREATEST(l.s, h)) AS sec
  FROM leases l,
       generate_series(date_trunc('hour', l.s), date_trunc('hour', l.e), interval '1 hour') AS h
)
SELECT to_char(h AT TIME ZONE %(tz)s, 'YYYY-MM-DD') AS day,
       EXTRACT(HOUR FROM h AT TIME ZONE %(tz)s)::int AS hour,
       COALESCE(SUM(sec) FILTER (WHERE heavy), 0) AS heavy_s,
       COALESCE(SUM(sec), 0) AS total_s
FROM parts WHERE sec > 0
GROUP BY 1, 2
ORDER BY 1, 2
"""

ENERGY_DAILY_SQL = """
SELECT to_char(hour_start AT TIME ZONE %(tz)s, 'YYYY-MM-DD') AS d, SUM(kwh) AS kwh,
       COUNT(DISTINCT hour_start) AS hours
FROM orion_energy_hourly
WHERE hour_start >= %(since)s AND hour_start < %(until)s
GROUP BY 1 ORDER BY 1
"""

RUN_COST_SQL = """
SELECT workload_kind, COUNT(*) AS n, AVG(energy_kwh) AS kwh, AVG(estimated_run_cost_usd) AS usd
FROM energy_run_cost
WHERE window_start >= %(since)s AND window_start < %(until)s AND energy_kwh IS NOT NULL
GROUP BY 1 ORDER BY 1
"""
ACT_COST_SQL = """
SELECT actor, cost_json->'wh_by_host' AS wh_by_host, cost_json->'usd' AS usd,
       cost_json->'heat_c_by_host' AS heat_c_by_host
FROM grammar_acts
WHERE opened_at >= %(since)s AND opened_at < %(until)s AND cost_json IS NOT NULL
  AND (%(family)s::text IS NULL OR actor = %(family)s)
"""

# Hourly attribution (sql-writer app/orion_energy_attribution.py): per host per UTC day, the energy
# above the pinned rest split into sponsored / metered (acts) / unattributed. Newest days first.
ATTRIBUTION_PRESENT_SQL = "SELECT to_regclass('orion_energy_attribution_hourly') IS NOT NULL AS present"
ATTRIBUTION_DAILY_SQL = """
SELECT host, to_char(hour_start AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, COUNT(*) AS hours,
       SUM(rest_wh) AS rest_wh, SUM(above_rest_wh) AS above_rest_wh, SUM(sponsored_wh) AS sponsored_wh,
       SUM(metered_wh) AS metered_wh, SUM(leased_unmetered_wh) AS leased_unmetered_wh,
       SUM(unattributed_wh) AS unattributed_wh, MIN(unattributed_wh) AS min_hour_unattributed_wh,
       SUM(covered_sec) / 3600.0 AS covered_h, SUM(split_unknown_sec) / 3600.0 AS split_unknown_h
FROM orion_energy_attribution_hourly
WHERE hour_start >= %(since)s AND hour_start < %(until)s
GROUP BY 1, 2 ORDER BY 2 DESC, 1
"""
# What acts were actually CHARGED per host per UTC day (whole-machine meter only: acts whose cost_json
# carries wh_gpu_by_host), to set beside the attributable ``metered_wh``.
ATTRIBUTION_CHARGED_SQL = """
SELECT e.key AS host, to_char(a.opened_at AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day,
       SUM(e.value::float) AS charged_wh, COUNT(*) AS acts
FROM grammar_acts a, jsonb_each_text(a.cost_json->'wh_by_host') AS e(key, value)
WHERE a.opened_at >= %(since)s AND a.opened_at < %(until)s AND a.cost_json ? 'wh_gpu_by_host'
GROUP BY 1, 2
"""
#: Days of "work not yet metered" rows in unit_costs (newest first); the table holds every day.
ATTRIBUTION_DAYS_SHOWN = 3

# Per-call token ledger (llm_call_usage, PR #2974; services/orion-sql-db/manual_migration_llm_call_usage_v1.sql).
# Count recorder='gateway' only, for totals AND per act. The warm relay's harness_relay recorder is retired
# (2026-10-29): the FCC adapter now forwards X-Orion-Correlation-Id, so the gateway row of an FCC turn carries
# the turn id and a leftover relay row would count that call twice.
TOKENS_PRESENT_SQL = "SELECT to_regclass('llm_call_usage') IS NOT NULL AS present"
TOKENS_TOTAL_SQL = """
SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens), 0) AS prompt, COALESCE(SUM(completion_tokens), 0) AS completion,
       MIN(recorded_at) AS first_at
FROM llm_call_usage
WHERE recorder = 'gateway' AND recorded_at >= %(since)s AND recorded_at < %(until)s
"""
TOKENS_ACT_JOIN_SQL = """
WITH acts AS (
  SELECT correlation_id,
         SUM((SELECT SUM(v::float) FROM jsonb_each_text(cost_json->'wh_by_host') AS e(k, v))) AS wh,
         SUM(NULLIF(cost_json->>'usd', '')::float) AS usd
  FROM grammar_acts
  WHERE opened_at >= %(since)s AND opened_at < %(until)s AND correlation_id IS NOT NULL
    AND jsonb_typeof(cost_json->'wh_by_host') = 'object'
  GROUP BY 1
), tok AS (
  SELECT correlation_id, SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) AS tokens
  FROM llm_call_usage
  WHERE recorder = 'gateway' AND correlation_id IN (SELECT correlation_id FROM acts)
  GROUP BY 1
)
SELECT (SELECT COUNT(*) FROM acts) AS acts_with_energy, COUNT(*) AS joined, SUM(a.wh) AS wh, SUM(a.usd) AS usd,
       SUM(t.tokens) AS tokens
FROM acts a JOIN tok t USING (correlation_id)
WHERE t.tokens > 0
"""

WASTE_BY_LANE_SQL = """
SELECT COALESCE(role, '?') AS lane, SUM(held_ms) / 1000.0 AS held_s
FROM gpu_pool_events
WHERE generated_at >= %(since)s AND generated_at < %(until)s
  AND event = ANY(%(events)s) AND held_ms IS NOT NULL
  AND (reason IS NULL OR (reason <> ALL(%(ok)s) AND reason NOT LIKE 'operator:%%'))
  AND """ + DUP_CLOSE_SQL + """
GROUP BY 1 ORDER BY 2 DESC
"""
WASTE_TOP_SQL = """
SELECT holder, role, held_ms / 1000.0 AS held_s, reason, generated_at AS at
FROM gpu_pool_events
WHERE generated_at >= %(since)s AND generated_at < %(until)s
  AND event = ANY(%(events)s) AND held_ms IS NOT NULL
  AND (reason IS NULL OR (reason <> ALL(%(ok)s) AND reason NOT LIKE 'operator:%%'))
  AND """ + DUP_CLOSE_SQL + """
ORDER BY held_ms DESC
LIMIT %(top)s
"""

QUEUE_WAIT_SQL = """
WITH fam AS (
  SELECT DISTINCT ON (want_id) want_id, family FROM want_ledger
  WHERE producer = 'orion-durable-runs.admission'
  ORDER BY want_id, decided_at DESC
), g AS (
  SELECT COALESCE(e.role, '?') AS lane,
         COALESCE(fam.family, split_part(e.holder, ':', 1)) AS family,
         CASE WHEN fam.family IS NULL THEN 'holder' ELSE 'want_ledger' END AS family_source,
         e.waited_ms
  FROM gpu_pool_events e
  LEFT JOIN fam ON e.holder LIKE 'durable-runs:%%' AND fam.want_id = substr(e.holder, 14)
  WHERE e.generated_at >= %(since)s AND e.generated_at < %(until)s
    AND e.event = 'granted' AND e.waited_ms IS NOT NULL
)
SELECT lane, family, family_source,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY waited_ms) / 1000.0 AS p50_s,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY waited_ms) / 1000.0 AS p90_s,
       COUNT(*) AS n
FROM g
WHERE (%(family)s::text IS NULL OR family = %(family)s)
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
"""

DECISIONS_SQL = """
SELECT decided_at, family, verb, tier, outcome, reason, net, binding_constraint,
       decision->>'why' AS why, decision->'prices_seen' AS prices_seen
FROM want_ledger
WHERE decided_at >= %(since)s AND decided_at < %(until)s
  AND producer <> ALL(%(synthetic)s)
  AND (%(family)s::text IS NULL OR family = %(family)s)
  AND (%(outcome)s::text IS NULL OR outcome = %(outcome)s)
ORDER BY decided_at DESC
LIMIT %(limit)s
"""

CLAUDE_TOKENS_SQL = """
SELECT COALESCE(SUM(total_input_tokens), 0) AS input,
       COALESCE(SUM(total_output_tokens), 0) AS output,
       COALESCE(SUM(total_cache_read_input_tokens), 0) AS cache_read,
       COALESCE(SUM(total_cache_creation_input_tokens), 0) AS cache_write,
       SUM(total_estimated_cost_usd) AS usd,
       COALESCE(SUM(priced_session_count), 0) AS priced,
       COALESCE(SUM(unpriced_session_count), 0) AS unpriced,
       COUNT(*) AS ticks,
       MAX(observed_at) AS newest
FROM dev_economics_ledger_log
WHERE observed_at >= %(since)s AND observed_at < %(until)s
"""

VRAM_SQL = """
SELECT n.node, b.ts, b.measurements
FROM unnest(%(hosts)s::text[]) AS n(node)
CROSS JOIN LATERAL (
  SELECT s."timestamp" AS ts, s.measurements FROM orion_biometrics_summary s
  WHERE s.node = n.node ORDER BY s."timestamp" DESC LIMIT 1
) b
"""

#: "Rest" is PINNED and dated, not rolling: a rolling low percentile would quietly count new always-on
#: work as rest (the Zelda dojo lifted athena's floor ~350 W from 2026-10-27). Source of truth:
#: config/physiology/rest_body.yaml (allowance proposal, PR #2989); until it exists, these defaults,
#: measured as the 5th percentile of chassis_watts over 2026-10-01..2026-10-13.
REST_BODY_FILE = "rest_body.yaml"
DEFAULT_REST_BODY = {
    "hosts": {
        "circe": {"rest_w": 541, "measured": "2026-10-01..2026-10-13", "method": "p05 chassis_watts"},
        "athena": {"rest_w": 340, "measured": "2026-10-01..2026-10-13", "method": "p05 chassis_watts"},
        "hecate": {"rest_w": 96, "measured": "standby", "method": "p50 chassis_watts when asleep"},
    },
    "sponsored": [{"name": "zelda_dojo", "containers": "zelda-dojo-live-*", "host": "athena", "until": None,
                   "note": "Juniper runs it until she kills it (pending her decision)"}],
}
#: The parts (GPUs, CPU, other) and each card are measured against the same dated window as the
#: pinned rest, so the split is comparable. This low percentile, not the minimum: one odd sample is not rest.
HOST_REST_PERCENTILE = 0.05
DEFAULT_REST_WINDOW = ("2026-10-01", "2026-10-13")
#: The recent floor (same percentile, trailing 7 days) shows drift: recent floor above pinned rest is
#: new always-on draw.
HOST_FLOOR_WINDOW = timedelta(days=7)
HOST_MAX_CARDS = 8
#: A 730-hour month (8,760 h / 12), so $/month figures do not change with the calendar month.
HOURS_PER_MONTH = 730.0
HOST_TOP_HOLDERS = 4
HOST_TOP_CONTAINERS = 5
#: The per-act meter fix that will publish "above-rest energy no act is charged for".
UNATTRIBUTED_NOT_YET = ("above-rest watts no act is charged for (unattributed_above_rest_w): unknown until the "
                        "whole-machine per-act meter (branch fix/act-energy-whole-machine) merges")


def _bio_ts(d: datetime) -> str:
    """``orion_biometrics_summary."timestamp"`` is varchar ('2026-10-30 03:09:37.46+00'). Comparing it to
    a string in the same shape keeps the (node, timestamp) index usable; a ::timestamptz cast does not."""
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


_PARTS = (("chassis", "(measurements->>'chassis_watts')::float"),
          ("gpu", "(measurements->>'gpu_watts_total')::float"),
          ("cpu", "(measurements->>'cpu_watts_total')::float"),
          ("other", "(measurements->>'chassis_watts')::float - (measurements->>'gpu_watts_total')::float"
                    " - (measurements->>'cpu_watts_total')::float"),
          *((f"gpu{i}", f"(measurements->>'gpu{i}_watts')::float") for i in range(HOST_MAX_CARDS)))
#: Per host: each part's low percentile over the pinned rest window (``<part>_rest``). The window is
#: in the past, so the answer never changes: it is cached per window (PINNED_PARTS_CACHE).
HOSTS_REST_SQL = """
SELECT node,
""" + ",\n".join(f"       percentile_cont(%(p)s) WITHIN GROUP (ORDER BY {expr}) AS {name}_rest" for name, expr in _PARTS) + """
FROM orion_biometrics_summary
WHERE node = ANY(%(hosts)s) AND "timestamp" >= %(rest_since)s AND "timestamp" < %(rest_until)s
GROUP BY node
"""
#: Per host: the same low percentile over the trailing 7 days, for the chassis (``recent_floor``:
#: above the pinned rest = new always-on draw) and each card (its idle when the pinned window
#: predates per-card watts, which began 2026-10-29).
HOSTS_FLOOR_SQL = """
SELECT node,
""" + ",\n".join(f"       percentile_cont(%(p)s) WITHIN GROUP (ORDER BY {expr}) AS {name}_floor"
                 for name, expr in _PARTS if name == "chassis" or name.startswith("gpu") and name != "gpu") + """
FROM orion_biometrics_summary
WHERE node = ANY(%(hosts)s) AND "timestamp" >= %(floor_since)s AND "timestamp" < %(until_ts)s
GROUP BY node
"""
PINNED_PARTS_TTL = timedelta(hours=6)
_PINNED_PARTS_CACHE: dict[tuple, tuple[datetime, list[dict]]] = {}


async def pinned_parts_rows(pool: Any, params: Mapping[str, Any], now: datetime) -> list[dict]:
    key = (id(pool), params["p"], params["rest_since"], params["rest_until"], tuple(params["hosts"]))
    hit = _PINNED_PARTS_CACHE.get(key)
    if hit and now - hit[0] < PINNED_PARTS_TTL:
        return hit[1]
    rows = await _rows(pool, HOSTS_REST_SQL, params)
    _PINNED_PARTS_CACHE.clear()
    _PINNED_PARTS_CACHE[key] = (now, rows)
    return rows


def load_rest_body(path: str) -> tuple[dict, str, Optional[str]]:
    """(rest body doc, source words, error). Missing file -> the dated defaults (said, never silent)."""
    doc, err = _load_yaml(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("hosts"), dict):
        why = err or "no hosts in it"
        return DEFAULT_REST_BODY, "pinned default (config/physiology/rest_body.yaml not present yet)", (
            None if (err or "").startswith("missing:") else f"rest_body.yaml unreadable ({why}): using the pinned defaults")
    return doc, "pinned in config/physiology/rest_body.yaml", None


def rest_window(doc: Mapping) -> tuple[str, str]:
    """The dated window the pinned rest was measured over ('2026-10-01..2026-10-13', end inclusive),
    as [since, until) biometrics timestamps. A host with no dated window (hecate 'standby') uses the default."""
    for spec in (doc.get("hosts") or {}).values():
        if not isinstance(spec, Mapping):
            continue
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", str(spec.get("measured") or ""))
        if m:
            try:
                datetime.fromisoformat(m.group(1))
                end = (datetime.fromisoformat(m.group(2)) + timedelta(days=1)).date().isoformat()
            except ValueError:
                continue  # not a real date: the defaults below, never a 500
            return m.group(1), end
    end = (datetime.fromisoformat(DEFAULT_REST_WINDOW[1]) + timedelta(days=1)).date().isoformat()
    return DEFAULT_REST_WINDOW[0], end


HOSTS_DAILY_SQL = """
SELECT node, to_char("timestamp"::timestamptz AT TIME ZONE %(tz)s, 'YYYY-MM-DD') AS d,
       AVG((measurements->>'chassis_watts')::float) AS w
FROM orion_biometrics_summary
WHERE node = ANY(%(hosts)s) AND "timestamp" >= %(since_ts)s AND "timestamp" < %(until_ts)s
  AND measurements ? 'chassis_watts'
GROUP BY 1, 2
ORDER BY 1, 2
"""

#: Lease-seconds per (card, holder) over the window, top HOST_TOP_HOLDERS per card. A durable run's
#: holder (``durable-runs:<run_id>``) becomes its workflow from ``durable_run_workflow``
#: (``durable-runs:curiosity.investigate``); a run id with no workflow row is folded by shape
#: (durable-runs:reading-<uuid> -> durable-runs:reading; a bare id -> durable-runs:run), so one kind
#: of run is one row. Lease time is time HELD (a run's calls attach to its hold), not time computing.
HOSTS_HOLDERS_SQL = """
WITH l AS (
  SELECT c AS card,
         CASE WHEN e.holder LIKE 'durable-runs:%%'
              THEN 'durable-runs:' || COALESCE(w.workflow,
                   substring(e.holder from '^durable-runs:([a-z_]+(?:[-:][a-z_]+)*)(?=[-:]|$)'), 'run')
              ELSE COALESCE(e.holder, '?') END AS holder,
         e.held_ms
  FROM gpu_pool_events e
  LEFT JOIN durable_run_workflow w ON e.holder LIKE 'durable-runs:%%' AND w.run_id = substr(e.holder, 14),
       json_array_elements_text(COALESCE(e.cards, '[]'::json)) AS c
  WHERE e.generated_at >= %(since)s AND e.generated_at < %(until)s
    AND e.event = ANY(%(events)s) AND e.held_ms > 0
    AND NOT (e.event = 'released' AND (e.reason LIKE 'expire:%%' OR e.reason LIKE 'abort:%%'))
), g AS (
  SELECT card, holder, SUM(held_ms) / 1000.0 AS lease_s FROM l GROUP BY 1, 2
), r AS (
  SELECT card, holder, lease_s, row_number() OVER (PARTITION BY card ORDER BY lease_s DESC) AS rk FROM g
)
SELECT card, holder, lease_s FROM r WHERE rk <= %(top)s ORDER BY card, rk
"""

SERVICES_SQL = """
SELECT created_at, context FROM notify_requests
WHERE source_service = 'orion-mesh-guardian' AND created_at >= %(since_naive)s
  AND context->>'event' = 'stopped_container'
ORDER BY created_at DESC
LIMIT 200
"""


async def _rows(pool: Any, sql: str, params: Mapping[str, Any]) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(sql, dict(params))
        return [dict(r) for r in await cur.fetchall()]


# --------------------------------------------------------------------------------------------
# pure shaping (tests feed rows straight into these)
# --------------------------------------------------------------------------------------------


def shape_limits(hourly: Sequence[Mapping], latest: Optional[Mapping], first_at: Optional[datetime],
                 since: datetime) -> tuple[list[dict], list[dict]]:
    missing: list[dict] = []
    latest_by: dict[str, dict] = {}
    for c in (_json(latest.get("constraints")) if latest else None) or []:
        if isinstance(c, dict) and c.get("constraint"):
            latest_by[str(c["constraint"])] = c
    by: dict[str, list[dict]] = {}
    any_price = False
    for r in hourly:
        name = r.get("constraint_name")
        if not name:
            continue
        price = _f(r.get("price"))
        any_price = any_price or price is not None
        by.setdefault(str(name), []).append({
            "h": _utc(r.get("h")), "level": _r(_f(r.get("level")), 4), "price": _r(price, 4),
            "bind_state": r.get("bind_state")})
    limits: list[dict] = []
    for name in sorted(set(by) | set(latest_by), key=lambda n: (n.startswith("lane:"), n)):
        c = latest_by.get(name) or {}
        price = c.get("price") if isinstance(c.get("price"), dict) else None
        if price:
            now = {"level": _f(c.get("level")), "bind_state": price.get("bind_state"), "why": str(price.get("why") or "")}
        elif c.get("no_reading"):
            now = {"level": None, "bind_state": "no_reading", "why": f"no reading: {c.get('no_reading')}"}
        else:
            now = {"level": _f(c.get("level")), "bind_state": None, "why": "no price on the newest tick"}
        limits.append({"constraint": name, "label": constraint_label(name), "unit": c.get("unit"),
                       "line": _f(c.get("budget")), "now": now, "hourly": by.get(name, [])})
    if not limits:
        first = f"; first tick {first_at.isoformat()}" if first_at else "; the table is empty"
        missing.append({"section": "limits", "reason": f"physiology_state has no tick in the window{first}"})
    elif since < PRICES_BEGAN_AT:
        missing.append({"section": "limits", "reason": f"partial: {PRICES_BEGAN_NOTE}"})
    elif not any_price:
        missing.append({"section": "limits", "reason": "no price in the window (ORION_PHYSIOLOGY_PRICES_ENABLED off?)"})
    return limits, missing


def shape_wants(rows: Sequence[Mapping], first_at: Optional[datetime], tiers: Mapping[str, int],
                family: Optional[str] = None) -> tuple[dict, list[dict]]:
    hourly: dict[tuple, dict] = {}
    fam: dict[str, dict] = {}
    for r in rows:
        f = str(r.get("family") or "?")
        h = _utc(r.get("h"))
        n = int(r.get("n") or 0)
        o = r.get("outcome")
        if o not in ("funded", "deferred"):
            continue
        cell = hourly.setdefault((h, f), {"h": h, "family": f, "funded": 0, "deferred": 0})
        cell[o] += n
        tot = fam.setdefault(f, {"family": f, "tier": tiers.get(f, r.get("tier")), "funded": 0, "deferred": 0})
        tot[o] += n
    out = {"families": sorted(fam.values(), key=lambda x: -(x["funded"] + x["deferred"])),
           "hourly": [hourly[k] for k in sorted(hourly, key=lambda k: (k[0], k[1]))]}
    missing = []
    if not fam:
        first = (f"first decision {first_at.isoformat()}" if first_at
                 else "want_ledger is empty (shadow bids began 2026-10-27)")
        scope = f" for family {family}" if family else ""
        missing.append({"section": "wants", "reason": f"no bid decision{scope} in the window; {first}"})
    return out, missing


def shape_rhythm(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    cells = [{"day": str(r["day"]), "hour": int(r["hour"]), "heavy": round(float(r.get("heavy_s") or 0.0), 1),
              "total": round(float(r.get("total_s") or 0.0), 1)} for r in rows]
    missing = [] if cells else [{"section": "rhythm", "reason": "no GPU lease closed in the window (gpu_pool_events); "
                                 "leases still open are counted once they close"}]
    return cells, missing


_FLOOR_RE = re.compile(r"always-on floor alone \(\$[0-9.]+/h\) reaches \$([0-9.]+)")


def shape_money(latest: Optional[Mapping], daily: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    """The newest tick's ``energy`` reading. ``floor_usd`` is not a field on the reading; it is the
    number the Budget organ writes into its own ``why`` (orion/physiology/budget.py energy_ask)."""
    missing: list[dict] = []
    money: dict = {"mtd_usd": None, "forecast_usd": None, "envelope_usd": None, "floor_usd": None,
                   "rate_usd_per_kwh": None,
                   "daily_kwh": [{"d": str(r["d"]), "kwh": round(float(r["kwh"]), 3),
                                  "partial": int(r.get("hours") or 24) < 23} for r in daily
                                 if r.get("kwh") is not None]}
    energy = None
    for c in (_json(latest.get("constraints")) if latest else None) or []:
        if isinstance(c, dict) and c.get("constraint") == "energy":
            energy = c
    if energy is None:
        missing.append({"section": "money", "reason": "no energy reading on the newest physiology tick"})
    elif energy.get("level") is None:
        missing.append({"section": "money", "reason": f"energy has no reading: {energy.get('no_reading')}"})
    else:
        price = energy.get("price") if isinstance(energy.get("price"), dict) else {}
        detail = energy.get("detail") if isinstance(energy.get("detail"), dict) else {}
        money["mtd_usd"] = _r(_f(energy.get("level")), 2)
        money["envelope_usd"] = _f(energy.get("budget"))
        money["forecast_usd"] = _r(_f(price.get("use")), 2)
        money["rate_usd_per_kwh"] = _r(_f(detail.get("latest_rate_usd_per_kwh")), 4)
        m = _FLOOR_RE.search(str(price.get("why") or ""))
        money["floor_usd"] = float(m.group(1)) if m else None
        if money["forecast_usd"] is None:
            missing.append({"section": "money", "reason": "no forecast on the newest tick (prices off or too few priced hours)"})
    if not money["daily_kwh"]:
        missing.append({"section": "money", "reason": "orion_energy_hourly has no hour in the window"})
    return money, missing


def _sum_json_numbers(v: Any) -> Optional[float]:
    v = _json(v)
    if isinstance(v, Mapping):
        vals = [_f(x) for x in v.values()]
        vals = [x for x in vals if x is not None]
        return sum(vals) if vals else None
    return _f(v)


def act_costs_by_family(rows: Sequence[Mapping]) -> dict[str, dict]:
    """Per act-ledger actor: mean measured W·h, USD and heat over acts that carry each field.
    Fields are the ActCostV2 additions being filled now (``wh_by_host``, ``usd``, ``heat_c_by_host``);
    an act without them is not counted (absent is unmeasured, never 0)."""
    acc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        actor = str(r.get("actor") or "?")
        a = acc.setdefault(actor, {"wh": [], "usd": [], "heat": []})
        for key, src in (("wh", "wh_by_host"), ("usd", "usd"), ("heat", "heat_c_by_host")):
            v = _sum_json_numbers(r.get(src))
            if v is not None:
                a[key].append(v)
    out = {}
    for actor, a in acc.items():
        n = max(len(a["wh"]), len(a["usd"]), len(a["heat"]))
        if n == 0:
            continue
        out[actor] = {"n": n, "wh_total": sum(a["wh"]) if a["wh"] else None, "wh_n": len(a["wh"]),
                      "wh": sum(a["wh"]) / len(a["wh"]) if a["wh"] else None,
                      "usd": sum(a["usd"]) / len(a["usd"]) if a["usd"] else None,
                      "heat_c": sum(a["heat"]) / len(a["heat"]) if a["heat"] else None}
    return out


@dataclass(frozen=True)
class HeatHost:
    host: str
    point: float
    fitted_at: str
    horizon: str


def load_heat_hosts(path: str) -> tuple[list[HeatHost], Optional[str]]:
    from orion.physiology.heat_config import load_heat_sensitivity

    cfg, err = load_heat_sensitivity(path)
    if cfg is None:
        return [], err
    return [HeatHost(h, e.point, e.fitted_at.date().isoformat(), e.horizon) for h, e in sorted(cfg.hosts.items())], None


def shape_unit_costs(rate: Optional[float], heat_hosts: Sequence[HeatHost], heat_err: Optional[str],
                     run_costs: Sequence[Mapping], act_costs: Optional[Mapping[str, dict]],
                     tokens: Optional[Mapping] = None,
                     attribution: Optional[Mapping] = None) -> tuple[list[dict], list[dict]]:
    """``act_costs`` None: the act-cost read failed (its reason is already in ``missing``)."""
    rows: list[dict] = []
    missing: list[dict] = []
    if rate is not None:
        rows.append({"unit": "1 kWh", "usd": round(rate, 4), "wh": 1000.0, "heat_c": None, "status": "derived",
                     "source": "newest marginal tariff rate (energy reading, energy_cost_accrued)"})
        rows.append({"unit": "1 MJ", "usd": round(rate / 3.6, 4), "wh": round(1000.0 / 3.6, 1), "heat_c": None,
                     "status": "derived", "source": "1 MJ = 277.8 W·h at the newest tariff rate"})
    else:
        rows.append({"unit": "1 kWh", "usd": None, "wh": 1000.0, "heat_c": None, "status": "not_measured",
                     "source": "no energy rate on the newest physiology tick"})
    for h in heat_hosts:
        rows.append({"unit": f"100 W for 2 h on {h.host}", "usd": _r(0.2 * rate, 4) if rate is not None else None,
                     "wh": 200.0, "heat_c": round(h.point, 3), "status": "derived",
                     "source": f"heat_sensitivity.yaml fitted {h.fitted_at}: cabinet degC per 100 W step within {h.horizon}"})
    if not heat_hosts:
        rows.append({"unit": "100 W for 2 h", "usd": None, "wh": 200.0, "heat_c": None, "status": "not_measured",
                     "source": f"heat sensitivity unavailable ({heat_err or 'no hosts fitted'})"})
    for r in run_costs:
        kwh = _f(r.get("kwh"))
        rows.append({"unit": f"one {r.get('workload_kind')} run", "usd": _r(_f(r.get("usd")), 6),
                     "wh": _r(kwh * 1000.0 if kwh is not None else None, 3), "heat_c": None, "status": "measured",
                     "source": f"energy_run_cost mean over {int(r.get('n') or 0)} runs in the window"})
    if not run_costs:
        rows.append({"unit": "one measured run", "usd": None, "wh": None, "heat_c": None, "status": "not_measured",
                     "source": "energy_run_cost has no run in the window (only reverie diffusion is measured today)"})
    for actor in sorted(act_costs or {}):
        a = act_costs[actor]
        total = f"family total {a['wh_total']:.3g} W·h over {a['wh_n']} acts" if a.get("wh_total") is not None \
            else f"{a['n']} acts"
        rows.append({"unit": f"one {actor} act (estimate: family mean)", "usd": _r(a["usd"], 6), "wh": _r(a["wh"], 3),
                     "heat_c": _r(a["heat_c"], 4), "status": "measured",
                     "source": f"grammar_acts.cost_json, whole machine above rest (chassis) from 2026-10-30, GPU "
                               f"above idle only before; {total} in the window. The family total is the honest "
                               "number; one act is an estimate"})
    if act_costs is not None and not act_costs:
        rows.append({"unit": "one act (any family)", "usd": None, "wh": None, "heat_c": None,
                     "status": "not_measured",
                     "source": "no act in the window carries cost_json wh_by_host / usd / heat_c_by_host yet "
                               "(the per-act energy fill is being added)"})
    rows.append(token_cost_row(tokens))
    if attribution is not None:
        if not attribution.get("present"):
            rows.append({"unit": "work not yet metered (per host per day)", "usd": None, "wh": None, "heat_c": None,
                         "status": "not_measured", "source": "orion_energy_attribution_hourly does not exist yet "
                         "(the sql-writer that creates it is not deployed)"})
        else:
            rows.extend(attribution_rows(attribution.get("rows"), rate, attribution.get("charged")))
    if not any(r["status"] == "measured" for r in rows):
        missing.append({"section": "unit_costs", "reason": "nothing measured in the window: every row is derived "
                        "from the tariff and the heat fit, or not measured yet"})
    return rows, missing


def attribution_rows(rows: Optional[Sequence[Mapping]], rate: Optional[float],
                     charged: Optional[Sequence[Mapping]] = None) -> list[dict]:
    """Per host per UTC day: the energy above the pinned rest that no act was charged ("work not yet
    metered"), with the split it closes on. ``rows`` None: the read failed (reason in ``missing``);
    {"present": False} sentinel handled by the caller."""
    if rows is None:
        return [{"unit": "work not yet metered (per host per day)", "usd": None, "wh": None, "heat_c": None,
                 "status": "not_measured", "source": "orion_energy_attribution_hourly read failed (see missing)"}]
    if not rows:
        return [{"unit": "work not yet metered (per host per day)", "usd": None, "wh": None, "heat_c": None,
                 "status": "not_measured",
                 "source": "orion_energy_attribution_hourly has no row in the window (the sql-writer hourly "
                           "attribution has not run, or no host had a chassis reading)"}]
    days = sorted({str(r["day"]) for r in rows}, reverse=True)[:ATTRIBUTION_DAYS_SHOWN]
    paid = {(str(c["host"]), str(c["day"])): c for c in (charged or [])}
    out = []
    for r in rows:
        if str(r["day"]) not in days:
            continue
        above, sponsored = _f(r.get("above_rest_wh")) or 0.0, _f(r.get("sponsored_wh")) or 0.0
        metered, un = _f(r.get("metered_wh")) or 0.0, _f(r.get("unattributed_wh")) or 0.0
        leased_un = _f(r.get("leased_unmetered_wh")) or 0.0
        activity = above - sponsored
        cov = f"{metered / activity:.1%}" if activity > 0 else "n/a"
        neg = _f(r.get("min_hour_unattributed_wh"))
        warn = (f" OVER-ATTRIBUTED: an hour reads {neg:.2f} W·h unattributed"
                if neg is not None and neg < -0.01 else "")
        c = paid.get((str(r["host"]), str(r["day"])))
        charged_wh = _f(c.get("charged_wh")) if c else None
        if charged_wh is not None and charged_wh > activity + 0.5:
            warn += (f" OVER-CHARGED: acts were charged {charged_wh:.0f} W·h, more than the "
                     f"{activity:.0f} W·h of activity above rest (double counting)")
        paid_txt = (f"acts actually charged {charged_wh:.0f} ({int(c.get('acts') or 0)} acts)"
                    if charged_wh is not None else "no act charged by the whole-machine meter")
        unknown = _f(r.get("split_unknown_h")) or 0.0
        partial = (f" PARTIAL: for {unknown:.1f} h a CPU or card reading was missing, so their above-rest "
                   "energy could not be split and is all in unattributed") if unknown > 0.05 else ""
        out.append({
            "unit": f"work not yet metered on {r['host']}, {r['day']} UTC", "wh": round(un, 1),
            "usd": _r(un / 1000.0 * rate, 4) if rate is not None else None, "heat_c": None, "status": "measured",
            "source": (f"chassis above the pinned rest {above:.0f} W·h over {_f(r.get('covered_h')) or 0:.1f} h "
                       f"(rest {_f(r.get('rest_wh')) or 0:.0f} W·h); sponsored {sponsored:.0f}; attributable to "
                       f"acts {metered:.0f} (coverage {cov} of activity); {paid_txt}; GPU leases whose turn has "
                       f"no act {leased_un:.0f} of the unmetered; orion_energy_attribution_hourly{partial}{warn}")})
    return out


def token_cost_row(tokens: Optional[Mapping]) -> dict:
    """Energy (and USD) per 1k tokens: per-act measured energy (grammar_acts.cost_json.wh_by_host)
    divided by the tokens of that act's calls (llm_call_usage on correlation_id). ``tokens`` None: the
    read failed (its reason is already in ``missing``)."""
    row = {"unit": "per 1k tokens", "usd": None, "wh": None, "heat_c": None, "status": "not_measured"}
    if tokens is None:
        return {**row, "source": "token ledger read failed (see missing)"}
    if not tokens.get("present"):
        return {**row, "source": "the per-call token table llm_call_usage does not exist yet (PR #2974 and its "
                                 "migration not applied)"}
    calls = int(tokens.get("calls") or 0)
    total = int(tokens.get("prompt") or 0) + int(tokens.get("completion") or 0)
    if calls == 0:
        return {**row, "source": "llm_call_usage has no gateway call in the window"}
    seen = f"{calls} gateway calls, {total:,} tokens in the window"
    j = tokens.get("join") or {}
    n_acts, joined, jt = int(j.get("acts_with_energy") or 0), int(j.get("joined") or 0), int(j.get("tokens") or 0)
    if n_acts == 0:
        return {**row, "source": f"{seen}; no act in the window carries measured energy (cost_json.wh_by_host), "
                                 "so there is nothing to divide"}
    if joined == 0 or jt <= 0:
        return {**row, "source": f"{seen}; none of the {n_acts} acts with measured energy has a call with its "
                                 "correlation_id in llm_call_usage"}
    wh = _f(j.get("wh"))
    usd = _f(j.get("usd"))
    return {**row, "status": "measured",
            "wh": _r(wh * 1000.0 / jt, 4) if wh is not None else None,
            "usd": _r(usd * 1000.0 / jt, 6) if usd is not None else None,
            "source": f"{joined} of {n_acts} acts with measured energy joined to {jt:,} tokens on correlation_id "
                      f"(J/token {wh * 3600.0 / jt:.3g}); {seen}" if wh is not None else
                      f"{joined} acts joined to {jt:,} tokens; {seen}"}


def shape_waste(by_lane: Sequence[Mapping], top: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    out = {"failed_lease_s_by_lane": {str(r["lane"]): round(float(r["held_s"] or 0.0), 1) for r in by_lane},
           "top": [{"holder": r.get("holder"), "role": r.get("role"), "held_s": round(float(r.get("held_s") or 0.0), 1),
                    "reason": r.get("reason"), "at": _utc(r.get("at"))} for r in top]}
    missing = [] if by_lane else [{"section": "waste", "reason": "no GPU lease closed with a failure in the window"}]
    return out, missing


def shape_queue_wait(rows: Sequence[Mapping], family: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    out = [{"lane": str(r["lane"]), "family": str(r["family"] or "?"), "family_source": r.get("family_source"),
            "p50_s": _r(_f(r.get("p50_s")), 1), "p90_s": _r(_f(r.get("p90_s")), 1), "n": int(r.get("n") or 0)}
           for r in rows]
    missing = []
    if not out and family:
        missing.append({"section": "queue_wait", "reason": f"no granted lease is attributable to family {family} in "
                        "the window: only durable-run admissions carry a want family; other leases are named by "
                        "their holder's service"})
    elif not out:
        missing.append({"section": "queue_wait", "reason": "no GPU lease was granted in the window"})
    elif any(r["family_source"] == "holder" for r in out):
        missing.append({"section": "queue_wait", "reason": "partial: only durable-run admissions join to a want "
                        "family; other rows are named by the holder's service"})
    return out, missing


def shape_decisions(rows: Sequence[Mapping], first_at: Optional[datetime],
                    family: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    out = []
    for r in rows:
        prices = _json(r.get("prices_seen"))
        out.append({"decided_at": _utc(r.get("decided_at")), "family": str(r.get("family") or "?"),
                    "verb": r.get("verb"), "tier": r.get("tier"), "outcome": str(r.get("outcome") or "?"),
                    "reason": r.get("reason"), "net": _r(_f(r.get("net")), 4),
                    "binding_constraint": r.get("binding_constraint"), "why": str(r.get("why") or ""),
                    "prices_seen": {str(k): float(v) for k, v in (prices or {}).items() if _f(v) is not None}
                    if isinstance(prices, dict) else {}})
    missing = []
    if not out:
        first = (f"first decision {first_at.isoformat()}" if first_at
                 else "first decision unknown (shadow bids began 2026-10-27)")
        scope = f" for family {family}" if family else ""
        missing.append({"section": "decisions", "reason": f"no bid decision{scope} matches in the window; {first}"})
    return out, missing


CLAUDE_TOKEN_SCOPE = ("tokens are Claude Code sessions on athena (dev_economics, ~/.claude/projects: Juniper's and "
                      "her agents' sessions); Orion's own harness sessions are not in this count")


def shape_claude(tokens: Optional[Mapping], latest: Optional[Mapping]) -> tuple[dict, list[dict]]:
    missing: list[dict] = []
    t = tokens or {}
    has_rows = int(t.get("ticks") or 0) > 0
    tok = (lambda k: int(t.get(k) or 0)) if has_rows else (lambda k: None)
    out: dict = {"window_tokens": {"input": tok("input"), "output": tok("output"),
                                   "cache_read": tok("cache_read"), "cache_write": tok("cache_write")},
                 "usd_est": None, "limit_events": [], "status": ""}
    priced = int(t.get("priced") or 0)
    usd = _f(t.get("usd"))
    out["usd_est"] = round(usd, 2) if (has_rows and usd is not None and priced > 0) else None
    claude = None
    for c in (_json(latest.get("constraints")) if latest else None) or []:
        if isinstance(c, dict) and c.get("constraint") == "claude":
            claude = c
    parts = [CLAUDE_TOKEN_SCOPE]
    if has_rows and (usd is None or priced == 0):
        parts.append("no priced session in the window, so no USD estimate")
    elif has_rows and int(t.get("unpriced") or 0):
        parts.append("some sessions had no price and are left out of usd_est (an API-price estimate, not a bill)")
    elif has_rows:
        parts.append("usd_est is an API-price estimate, not a bill")
    if claude is None:
        parts.append("limits: no claude reading on the newest physiology tick")
    elif claude.get("no_reading"):
        parts.append(f"limits: unknown ({claude.get('no_reading')}): no ClaudeLimitObservationV1 has arrived on "
                     "orion:substrate:claude_limit; its producer in orion-cocreation-signals is off by default "
                     "(COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false)")
    else:
        # orion/physiology/body.py claude_reading: per kind `<kind>_state`, `<kind>_events` (a count, no
        # per-event times), `<kind>_resets_at`. One entry per kind that saw limit events, at the reading's time.
        detail = claude.get("detail") if isinstance(claude.get("detail"), dict) else {}
        at = _utc(claude.get("source_at"))
        for key, n in sorted(detail.items()):
            if key.endswith("_events") and (_f(n) or 0) > 0 and at is not None:
                kind = key[: -len("_events")]
                out["limit_events"].append({"at": at, "kind": f"{kind} x{int(_f(n))} ({detail.get(kind + '_state', '?')})"})
        parts.append(f"limits: {claude.get('state') or 'read'}, {int(_f(claude.get('level')) or 0)} limit event(s)")
    out["status"] = "; ".join(parts)
    if not has_rows:
        missing.append({"section": "claude", "reason": "dev_economics_ledger_log has no row in the window "
                        "(orion-cocreation-signals dev_economics producer down or off): token counts unknown"})
    return out, missing


def _card_map(gpu_pool_doc: Mapping) -> list[tuple[str, str, int, list[str]]]:
    """(host, card name, device index, roles on it) from config/gpu_pool.yaml."""
    default_host = str(((gpu_pool_doc.get("host") or {}).get("name")) or "circe")
    roles_on: dict[str, list[str]] = {}
    for role, spec in (gpu_pool_doc.get("roles") or {}).items():
        for card in (spec or {}).get("cards") or []:
            roles_on.setdefault(str(card), []).append(str(role))
    out = []
    for name, spec in (gpu_pool_doc.get("cards") or {}).items():
        spec = spec or {}
        out.append((str(spec.get("host") or default_host), str(name), int(spec.get("index", 0)),
                    sorted(roles_on.get(str(name), []))))
    return out


def shape_vram(gpu_pool_doc: Optional[Mapping], pool_err: Optional[str], bio_rows: Sequence[Mapping],
               now: datetime) -> tuple[list[dict], list[dict]]:
    missing: list[dict] = []
    meas: dict[str, dict] = {}
    stale: list[str] = []
    for r in bio_rows:
        ts = _utc(r.get("ts"))
        m = _json(r.get("measurements"))
        if not isinstance(m, dict):
            continue
        if ts is None or now - ts > BIOMETRICS_FRESH:
            stale.append(str(r.get("node")))
            continue
        meas[str(r.get("node"))] = m
    stale = sorted(set(stale) | {h for h in VRAM_HOSTS if h not in meas})
    cards = _card_map(gpu_pool_doc) if gpu_pool_doc else []
    pooled = {(h, i) for h, _, i, _ in cards}
    # Cards a host reports that the pool does not schedule (athena's T10s): no roles.
    for host, m in sorted(meas.items()):
        n = int(_f(m.get("gpu_count")) or 0)
        for i in range(n):
            if (host, i) not in pooled:
                cards.append((host, f"{host}-gpu{i}" if host != "circe" else f"gpu{i}", i, []))
    out = []
    reported = 0
    for host, card, idx, roles in cards:
        m = meas.get(host) or {}
        used = _f(m.get(f"gpu{idx}_mem_used_mb"))
        total = _f(m.get(f"gpu{idx}_mem_total_mb"))
        watts = _f(m.get(f"gpu{idx}_watts"))
        reported += used is not None and total is not None
        out.append({"host": host, "card": card, "used_mb": used, "total_mb": total,
                    "headroom_mb": (total - used) if (used is not None and total is not None) else None,
                    "roles": roles, "watts": _r(watts, 1)})
    if pool_err:
        missing.append({"section": "vram", "reason": f"card -> role map unavailable ({pool_err})"})
    if reported == 0:
        missing.append({"section": "vram", "reason": "per-card GPU memory is not reported yet: "
                        "orion_biometrics_summary.measurements has no gpuN_mem_used_mb / gpuN_mem_total_mb "
                        "(being added beside gpuN_watts)"})
    elif reported < len(out):
        missing.append({"section": "vram", "reason": f"partial: {len(out) - reported} of {len(out)} cards report "
                        "no memory figures"})
    if stale:
        missing.append({"section": "vram", "reason": f"no biometrics row in the last 15 min for "
                        f"{', '.join(sorted(stale))} (asleep or its reporter is down): its cards read unknown"})
    return out, missing


CONTAINER_CPU_PREFIX = "container_cpu_watts."
PROCESSES_TOP_N = 12


_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def container_label(name: str, names: Optional[Mapping[str, str]]) -> str:
    """orion-biometrics publishes a container by name when it can read the docker root, else by its
    12-char id (circe, 2026-10-30: ``container_cpu_watts.d590a9d48af2``). An id the census knows is
    shown as its name; an unknown id stays an id (never guessed)."""
    if names and _CONTAINER_ID_RE.match(name):
        return names.get(name, name)
    return name


def container_names_from_census(raw_states: Mapping[Any, Any]) -> dict[str, str]:
    """id12 -> container name, from every heartbeat that carries a container census
    (``SystemHealthV1.details["container_restarts"].counts`` keys are ``<name>@<id12>``; circe's comes
    from orion-gpu-lane-controller). ``raw_states`` is the equilibrium service's per-service state
    hash (field -> JSON). Anything unreadable is skipped: an id without a name stays an id."""
    out: dict[str, str] = {}
    for v in raw_states.values():
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if not isinstance(v, str) or "container_restarts" not in v:
            continue
        doc = _json(v)
        census = ((doc or {}).get("details") or {}).get("container_restarts") if isinstance(doc, dict) else None
        for key in ((census or {}).get("counts") or {}) if isinstance(census, dict) else ():
            name, _, cid = str(key).rpartition("@")
            if name and _CONTAINER_ID_RE.match(cid):
                out[cid] = name
    return out


def shape_processes(bio_rows: Sequence[Mapping], now: datetime,
                    names: Optional[Mapping[str, str]] = None) -> tuple[list[dict], list[dict]]:
    """Top containers by CPU watts above idle, per host, from the newest fresh biometrics row
    (orion-biometrics container_cpu_measurements, PR #2977). Absent keys are a reason, never 0 W."""
    rows: list[dict] = []
    notes: list[str] = []
    for r in bio_rows:
        host = str(r.get("node"))
        ts = _utc(r.get("ts"))
        m = _json(r.get("measurements"))
        if not isinstance(m, dict) or ts is None or now - ts > BIOMETRICS_FRESH:
            continue
        named = {k[len(CONTAINER_CPU_PREFIX):]: _f(v) for k, v in m.items() if k.startswith(CONTAINER_CPU_PREFIX)}
        named = {k: v for k, v in named.items() if v is not None}
        if not named and m.get("container_cpu_watts_small") is None:
            if m.get("cpu_busy_cores") is not None:
                notes.append(f"{host}: CPU below the fitted idle floor this sample, so no honest split")
            continue
        for name, w in named.items():
            rows.append({"host": host, "name": container_label(name, names), "watts": round(w, 1), "cores": None})
        for key, label in (("container_cpu_watts_small", "(small containers)"),
                           ("cpu_watts_host_other", "(host, not in a container)")):
            w = _f(m.get(key))
            if w is not None:
                rows.append({"host": host, "name": label, "watts": round(w, 1), "cores": None})
    rows.sort(key=lambda x: -x["watts"])
    missing: list[dict] = []
    if not rows:
        missing.append({"section": "processes", "reason": "per-container CPU watts are not reported yet: no fresh "
                        "orion_biometrics_summary row carries container_cpu_watts.* (orion-biometrics, PR #2977)"
                        + ("; " + "; ".join(notes) if notes else "")})
    elif notes:
        missing.append({"section": "processes", "reason": "partial: " + "; ".join(notes)})
    if len(rows) > PROCESSES_TOP_N:
        missing.append({"section": "processes", "reason": f"top {PROCESSES_TOP_N} of {len(rows)} shown"})
        rows = rows[:PROCESSES_TOP_N]
    return rows, missing


def fold_holder(holder: Optional[str], workflow: Optional[str] = None) -> str:
    """Python twin of the SQL fold in HOSTS_HOLDERS_SQL (tests check the two agree on examples)."""
    h = str(holder or "?")
    if not h.startswith("durable-runs:"):
        return h
    if workflow:
        return "durable-runs:" + workflow
    m = re.match(r"^durable-runs:([a-z_]+(?:[-:][a-z_]+)*)(?=[-:]|$)", h)
    return "durable-runs:" + (m.group(1) if m else "run")


#: Plain words for the two gateway HTTP holders (same map as orion/physiology/heat_archaeology.py
#: HOLDER_LABELS; orion-llm-gateway pool_placement.py names them).
HOLDER_WORDS = {"http:anthropic": "harness turns", "http:openai": "AI Town"}


def holder_label(holder: str) -> str:
    if holder in HOLDER_WORDS:
        return HOLDER_WORDS[holder]
    if holder.startswith("durable-runs:"):
        return holder[len("durable-runs:"):] + " runs"
    return holder


def merge_by_node(*row_sets: Sequence[Mapping]) -> list[dict]:
    out: dict[str, dict] = {}
    for rows in row_sets:
        for r in rows:
            out.setdefault(str(r.get("node")), {}).update(dict(r))
    return list(out.values())


def _pinned_rest(spec: Any) -> tuple[Optional[float], Optional[str]]:
    """(rest in use, refusal note). ``rest_w`` is the value in use; ``approved_rest_w`` is Juniper's
    ceiling (allowance proposal PR #2989: lowering rest is always allowed, raising it needs her
    sign-off). A rest_w above the approved number is refused: the approved number is used."""
    if not isinstance(spec, Mapping):
        return _f(spec), None   # a bare number: the rest itself, no ceiling
    rest, ceiling = _f(spec.get("rest_w")), _f(spec.get("approved_rest_w"))
    if rest is None:
        return ceiling, None
    if ceiling is not None and rest > ceiling:
        return ceiling, f"rest_w {rest:.0f} W is above Juniper's approved {ceiling:.0f} W: the approved number is used"
    return rest, None


def shape_hosts(bio_rows: Sequence[Mapping], rest_rows: Sequence[Mapping], daily_rows: Sequence[Mapping],
                holder_rows: Sequence[Mapping], gpu_pool_doc: Optional[Mapping], pool_err: Optional[str],
                rate_usd_per_kwh: Optional[float], now: datetime,
                names: Optional[Mapping[str, str]] = None,
                rest_body: Optional[Mapping] = None, rest_source: str = "", rest_err: Optional[str] = None,
                hosts: Sequence[str] = VRAM_HOSTS) -> tuple[list[dict], list[dict]]:
    """What is spinning each machine: whole-machine watts now vs its PINNED rest, split into GPU /
    CPU / other (each against its own rest over the same dated window), each card with the lease
    holders that held it in the window, the top containers by CPU watts, sponsored work apart.

    The split never sums parts with the whole: gpu_w and cpu_w are INSIDE chassis_w_now, so
    other_w = chassis - gpu - cpu (fans, PSU loss, memory, disks). A part that is not measured is
    None and other_w is then None too (a remainder of an unmeasured part would be a guess)."""
    missing: list[dict] = []
    body = rest_body or DEFAULT_REST_BODY
    pinned = body.get("hosts") or {}
    since_d, until_d = rest_window(body)
    window_words = f"{since_d}..{(datetime.fromisoformat(until_d) - timedelta(days=1)).date().isoformat()}"
    if rest_err:
        missing.append({"section": "hosts", "reason": rest_err})
    meas: dict[str, tuple[datetime, dict]] = {}
    for r in bio_rows:
        ts = _utc(r.get("ts"))
        m = _json(r.get("measurements"))
        if isinstance(m, dict) and ts is not None and now - ts <= BIOMETRICS_FRESH:
            meas[str(r.get("node"))] = (ts, m)
    rest = {str(r.get("node")): r for r in rest_rows}
    daily: dict[str, list[dict]] = {}
    for r in daily_rows:
        w = _f(r.get("w"))
        if w is not None:
            daily.setdefault(str(r.get("node")), []).append({"d": str(r.get("d")), "chassis_w_avg": round(w, 1)})
    holders: dict[str, list[dict]] = {}
    for r in holder_rows:
        holders.setdefault(str(r.get("card")), []).append(
            {"holder": str(r.get("holder") or "?"), "label": holder_label(str(r.get("holder") or "?")),
             "lease_s": round(float(r.get("lease_s") or 0.0), 1)})
    cards = _card_map(gpu_pool_doc) if gpu_pool_doc else []
    if pool_err:
        missing.append({"section": "hosts", "reason": f"card -> role map unavailable ({pool_err}): cards show no roles"})
    sponsored_raw = body.get("sponsored")
    sponsors = [sp for sp in (sponsored_raw if isinstance(sponsored_raw, list) else []) if isinstance(sp, Mapping)]
    asleep: list[str] = []
    recent_idle_hosts: list[str] = []
    out: list[dict] = []
    for host in hosts:
        if host not in meas:
            asleep.append(host)
            continue
        ts, m = meas[host]
        chassis = _f(m.get("chassis_watts"))
        gpu = _f(m.get("gpu_watts_total"))
        cpu = _f(m.get("cpu_watts_total"))
        rr = rest.get(host) or {}
        raw_spec = pinned.get(host)
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        rest_w, refused = _pinned_rest(raw_spec)
        if refused:
            missing.append({"section": "hosts", "reason": f"{host}: {refused}"})
        above = (chassis - rest_w) if (chassis is not None and rest_w is not None) else None
        per_month = lambda w: (_r(w * HOURS_PER_MONTH / 1000.0 * rate_usd_per_kwh, 2)  # noqa: E731
                               if (w is not None and rate_usd_per_kwh is not None) else None)
        host_cards = [(name, idx, roles) for h, name, idx, roles in cards if h == host]
        pooled = {idx for _, idx, _ in host_cards}
        for i in range(min(int(_f(m.get("gpu_count")) or 0), HOST_MAX_CARDS)):
            if i not in pooled:
                host_cards.append((f"{host}-gpu{i}" if host != "circe" else f"gpu{i}", i, []))
        card_rows, recent_idle = [], False
        for name, idx, roles in sorted(host_cards, key=lambda c: c[1]):
            watts = _f(m.get(f"gpu{idx}_watts"))
            if watts is None and _f(m.get(f"gpu{idx}_mem_used_mb")) is None:
                continue  # a pooled card this host does not report (asleep twin): not a 0 W card
            idle = _f(rr.get(f"gpu{idx}_rest"))
            if idle is None:
                idle = _f(rr.get(f"gpu{idx}_floor"))
                recent_idle |= idle is not None
            card_rows.append({"card": name, "watts": _r(watts, 1), "idle_w": _r(idle, 1),
                              "temp_c": _f(m.get(f"gpu{idx}_temp_c")), "mem_used_mb": _f(m.get(f"gpu{idx}_mem_used_mb")),
                              "roles": roles, "top_holders": holders.get(name, [])})
        containers, all_named = [], []
        for k, v in m.items():
            w = _f(v)
            if w is None:
                continue
            if k.startswith(CONTAINER_CPU_PREFIX):
                c = {"name": container_label(k[len(CONTAINER_CPU_PREFIX):], names), "cpu_w": round(w, 1)}
                containers.append(c)
                all_named.append(c)
            elif k == "cpu_watts_host_other":
                containers.append({"name": "(host, not in a container)", "cpu_w": round(w, 1)})
        sponsored, sponsored_names = [], set()
        for sp in sponsors:
            if str(sp.get("host") or host) != host:
                continue
            pat = str(sp.get("containers") or "")
            hit = [c for c in all_named if pat and fnmatch.fnmatchcase(c["name"], pat)]
            sponsored_names |= {c["name"] for c in hit}
            sponsored.append({"name": str(sp.get("name") or pat), "note": str(sp.get("note") or ""),
                              "containers": len(hit), "cpu_w": round(sum(c["cpu_w"] for c in hit), 1) if hit else None})
        containers = [c for c in containers if c["name"] not in sponsored_names]
        containers.sort(key=lambda c: -c["cpu_w"])
        floor = _f(rr.get("chassis_floor"))
        recent_idle_hosts += [host] if recent_idle else []
        out.append({
            "host": host, "at": ts, "chassis_w_now": _r(chassis, 1), "chassis_w_rest": _r(rest_w, 1),
            "rest_basis": (f"{rest_source}: {spec.get('method') or 'p05 chassis_watts'} over {spec.get('measured') or '?'}"
                           if rest_w is not None else ""),
            "recent_floor_w": _r(floor, 1),
            "above_rest_w": _r(above, 1), "usd_per_month_at_this_rate": per_month(above),
            "rest_usd_per_month": per_month(rest_w), "gpu_w": _r(gpu, 1), "cpu_w": _r(cpu, 1),
            "other_w": _r(chassis - gpu - cpu, 1) if None not in (chassis, gpu, cpu) else None,
            "gpu_w_rest": _r(_f(rr.get("gpu_rest")), 1), "cpu_w_rest": _r(_f(rr.get("cpu_rest")), 1),
            "other_w_rest": _r(_f(rr.get("other_rest")), 1), "parts_rest_window": window_words,
            "fan_pct": _r(_f(m.get("fan_pct_max")), 1), "unattributed_above_rest_w": None,
            "cards": card_rows, "top_containers": containers[:HOST_TOP_CONTAINERS], "sponsored": sponsored,
            "daily": daily.get(host, []),
        })
        if chassis is None:
            missing.append({"section": "hosts", "reason": f"{host}: no chassis_watts on its newest row (no BMC or "
                            "PDU reading): its draw now and above-rest are unknown"})
        if rest_w is None:
            missing.append({"section": "hosts", "reason": f"{host}: no pinned rest in rest_body.yaml, so above-rest is unknown"})
        if any(_CONTAINER_ID_RE.match(c["name"]) for c in containers[:HOST_TOP_CONTAINERS]):
            missing.append({"section": "hosts", "reason": f"{host}: some containers show as 12-char ids (orion-biometrics "
                            "could not read their names, and the container census did not list them)"})
    out.sort(key=lambda h: -(h["chassis_w_now"] or 0.0))
    if recent_idle_hosts:
        missing.append({"section": "hosts", "reason": f"card rests on {', '.join(recent_idle_hosts)} are the last 7 days' "
                        f"5th percentile (per-card watts began 2026-10-29, after the pinned window {window_words})"})
    if asleep:
        missing.append({"section": "hosts", "reason": f"no biometrics row in the last 15 min for {', '.join(asleep)} "
                        "(asleep or its reporter is down): not shown"})
    if out:
        missing.append({"section": "hosts", "reason": UNATTRIBUTED_NOT_YET})
    if out and rate_usd_per_kwh is None:
        missing.append({"section": "hosts", "reason": "no electricity rate on the newest physiology tick: $/month unknown"})
    return out, missing


def shape_value_per_cost(weights: Optional[Mapping[str, Any]], weights_err: Optional[str],
                         act_costs: Mapping[str, dict], family: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    """One row per family with graded evidence: the same evidence the market's family weights use
    (orion/physiology/weights.py: (a) downstream use rate, (b) Juniper reply rate)."""
    if weights is None:
        return [], [{"section": "value_per_cost", "reason": f"family weights unavailable ({weights_err or 'market off'})"}]
    rows: list[dict] = []
    no_evidence: list[str] = []
    for fam in sorted(weights):
        if family and fam != family:
            continue
        w = weights[fam]
        ev = getattr(w, "evidence", None)
        n = int(getattr(ev, "n", 0) or 0) if ev is not None else 0
        rate = getattr(ev, "mean_rate", None) if ev is not None else None
        if n == 0 or rate is None:
            no_evidence.append(fam)
            continue
        parts = []
        if getattr(ev, "engagement_rate", None) is not None:
            parts.append(f"juniper_reply_rate {ev.engagement_rate:.2f} (n={ev.engagement_n})")
        if getattr(ev, "use_rate", None) is not None:
            parts.append(f"downstream_use_rate {ev.use_rate:.2f} (n={ev.use_n})")
        if getattr(w, "basis", None) == "prior":
            parts.append("under 10 graded acts, so bids still use the prior weight")
        usd = (act_costs.get(fam) or {}).get("usd")
        rows.append({"family": fam, "evidence_n": n,
                     "value_metric": "30-day evidence the bids use: mean of " + ", ".join(parts),
                     "value": round(float(rate), 4), "usd_per_act": _r(usd, 6),
                     "status": "measured" if usd is not None else "not_measured"})
    missing = []
    if no_evidence:
        missing.append({"section": "value_per_cost",
                        "reason": f"no graded evidence yet for {len(no_evidence)} famil"
                                  f"{'y' if len(no_evidence) == 1 else 'ies'}: {', '.join(no_evidence)}"})
    if family and not rows and not no_evidence:
        missing.append({"section": "value_per_cost", "reason": f"family {family} is not a bidding family"})
    if rows and all(r["usd_per_act"] is None for r in rows):
        missing.append({"section": "value_per_cost",
                        "reason": "usd_per_act not measured: no act in the window carries cost_json.usd yet"})
    return rows, missing


_HOST_RE = re.compile(r"^orion-(athena|circe|hecate|carbon|janus)-")


def shape_services(rows: Sequence[Mapping], generated_at: datetime) -> tuple[dict, list[dict]]:
    """Mesh-guardian ``stopped_container`` cards (one card per outage, PR #2968). There is no
    recovery event: a service started again after its card still shows here until the lookback
    ends, so ``basis`` says so."""
    seen: set[str] = set()
    down = []
    for r in rows:
        ctx = _json(r.get("context")) or {}
        name = str(ctx.get("service_id") or "?")
        if name in seen:
            continue
        seen.add(name)
        containers = ctx.get("containers") or []
        host = None
        for c in containers:
            m = _HOST_RE.match(str(c))
            if m:
                host = m.group(1)
                break
        created = _utc(r.get("created_at"))
        hours = _f(ctx.get("stopped_hours"))
        since = (created - timedelta(hours=hours)) if (created is not None and hours is not None) else created
        states = ctx.get("states") or []
        down.append({"name": name, "host": host, "state": ",".join(str(s) for s in states) or None, "since": since})
    out = {"checked_at": generated_at, "down": down,
           "basis": "mesh-guardian stopped_container cards from the last 30 days (one card per outage). No "
                    "recovery event exists, so a listed service may be running again."}
    missing = []
    if not down:
        missing.append({"section": "services", "reason": "no stopped_container card from the mesh guardian in 30 "
                        "days (the check is live from PR #2968, 2026-10-29): nothing reported stopped"})
    return out, missing


# --------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------


@dataclass
class ReportDeps:
    pool: Any
    tz: str = "America/Denver"
    heat_sensitivity_path: str = "/app/config/physiology/heat_sensitivity.yaml"
    gpu_pool_path: str = "/app/config/gpu_pool.yaml"
    tiers: Mapping[str, int] = field(default_factory=dict)
    #: async () -> mapping family -> FamilyWeight, or raises. None: the market is off.
    family_weights: Optional[Callable[[], Awaitable[Mapping[str, Any]]]] = None
    #: async () -> {id12: container name} (circe's container census). None: ids stay ids.
    container_names: Optional[Callable[[], Awaitable[Mapping[str, str]]]] = None
    concurrency: int = 3


def _load_yaml(path: str) -> tuple[Optional[dict], Optional[str]]:
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text()) or {}, None
    except FileNotFoundError:
        return None, f"missing:{Path(path).name}"
    except Exception as exc:  # noqa: BLE001 -- a broken file is a reason, not a 500
        return None, f"unreadable:{Path(path).name}:{type(exc).__name__}"


#: Sections the ``family=`` filter narrows; every other section is whole-body and says so.
FAMILY_FILTERED = ("wants", "decisions", "queue_wait", "value_per_cost")
#: The newest physiology tick is "now" only if it is this fresh (ticks every ~120 s).
LATEST_MAX_AGE = timedelta(minutes=10)
#: One semaphore for every report read in this process, so concurrent requests (the Hub page's
#: 24h/7d/30d plus a skill call) share ``concurrency`` slots of the pool, not ``concurrency`` each.
_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _shared_semaphore(n: int) -> asyncio.Semaphore:
    n = max(1, int(n))
    sem = _SEMAPHORES.get(n)
    if sem is None:
        sem = _SEMAPHORES[n] = asyncio.Semaphore(n)
    return sem


async def build_report(deps: ReportDeps, q: ReportQuery, now: Optional[datetime] = None) -> PhysiologyReportV1:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = now - WINDOW_DELTAS[q.window]
    base = {"since": since, "until": now}
    sem = _shared_semaphore(deps.concurrency)
    timing: dict[str, float] = {}
    missing: list[dict] = []
    wanted = set(q.sections)

    def fail(sections: Iterable[str], what: str, exc: BaseException) -> None:
        for s in sections:
            missing.append({"section": s, "reason": f"read failed{what}: {type(exc).__name__}: {str(exc)[:160]}"})

    async def timed(name: str, coro_fn: Callable[[], Awaitable[Any]], sections: Iterable[str]) -> Any:
        """A failed read is recorded under every asked section that needed it (never a 500)."""
        async with sem:
            t0 = time.perf_counter()
            try:
                return await coro_fn()
            except Exception as exc:  # noqa: BLE001 -- one read failing never fails the report
                logger.warning("physiology_report_read_failed read=%s", name, exc_info=True)
                fail([s for s in sections if s in wanted], f" ({name})" if name not in wanted else "", exc)
                return None
            finally:
                timing[name] = round((time.perf_counter() - t0) * 1000.0, 1)

    pool = deps.pool
    syn = list(SYNTHETIC_PRODUCERS)

    async def latest():
        return ((await _rows(pool, LIMITS_LATEST_SQL, base)) or [None])[0]

    async def limits():
        h = await _rows(pool, LIMITS_HOURLY_SQL, base)
        first = ((await _rows(pool, LIMITS_FIRST_SQL, {})) or [{}])[0].get("first_at")
        return h, _utc(first)

    async def wants():
        r = await _rows(pool, WANTS_HOURLY_SQL, {**base, "synthetic": syn, "family": q.family})
        first = ((await _rows(pool, WANTS_FIRST_SQL, {"synthetic": syn})) or [{}])[0].get("first_at")
        return r, _utc(first)

    async def act_costs():
        return act_costs_by_family(await _rows(pool, ACT_COST_SQL, {**base, "family": q.family}))

    tasks: dict[str, Awaitable[Any]] = {}
    if wanted & {"limits", "money", "unit_costs", "claude", "hosts"}:
        tasks["latest"] = timed("latest", latest, ("limits", "money", "unit_costs", "claude", "hosts"))
    if "limits" in wanted:
        tasks["limits"] = timed("limits", limits, ("limits",))
    if wanted & {"wants", "decisions"}:
        tasks["wants"] = timed("wants", wants, ("wants",))
    if "rhythm" in wanted:
        tasks["rhythm"] = timed("rhythm", lambda: _rows(pool, RHYTHM_SQL, {
            **base, "tz": deps.tz, "events": list(RHYTHM_CLOSE_EVENTS)}), ("rhythm",))
    if "money" in wanted:
        tasks["money"] = timed("money", lambda: _rows(pool, ENERGY_DAILY_SQL, {**base, "tz": deps.tz}), ("money",))
    if "unit_costs" in wanted:
        tasks["run_costs"] = timed("unit_costs", lambda: _rows(pool, RUN_COST_SQL, base), ("unit_costs",))
    if wanted & {"unit_costs", "value_per_cost"}:
        tasks["act_costs"] = timed("act_costs", act_costs, ("unit_costs", "value_per_cost"))
    if "unit_costs" in wanted:
        async def tokens():
            present = bool(((await _rows(pool, TOKENS_PRESENT_SQL, {})) or [{}])[0].get("present"))
            if not present:
                return {"present": False}
            total = ((await _rows(pool, TOKENS_TOTAL_SQL, base)) or [{}])[0]
            join = ((await _rows(pool, TOKENS_ACT_JOIN_SQL, base)) or [{}])[0]
            return {"present": True, **total, "join": join}
        tasks["tokens"] = timed("tokens", tokens, ("unit_costs",))

        async def attribution():
            present = bool(((await _rows(pool, ATTRIBUTION_PRESENT_SQL, {})) or [{}])[0].get("present"))
            if not present:
                return {"present": False}
            return {"present": True, "rows": await _rows(pool, ATTRIBUTION_DAILY_SQL, base),
                    "charged": await _rows(pool, ATTRIBUTION_CHARGED_SQL, base)}
        tasks["attribution"] = timed("attribution", attribution, ("unit_costs",))
    if "waste" in wanted:
        wp = {**base, "events": list(WASTE_CLOSE_EVENTS), "ok": list(OK_CLOSE_REASONS), "top": WASTE_TOP_N}

        async def waste():
            return await _rows(pool, WASTE_BY_LANE_SQL, wp), await _rows(pool, WASTE_TOP_SQL, wp)
        tasks["waste"] = timed("waste", waste, ("waste",))
    if "queue_wait" in wanted:
        tasks["queue_wait"] = timed("queue_wait", lambda: _rows(pool, QUEUE_WAIT_SQL, {**base, "family": q.family}),
                                    ("queue_wait",))
    if "decisions" in wanted:
        tasks["decisions"] = timed("decisions", lambda: _rows(pool, DECISIONS_SQL, {
            **base, "synthetic": syn, "family": q.family, "outcome": q.outcome, "limit": q.limit}), ("decisions",))
    if "claude" in wanted:
        tasks["claude"] = timed("claude", lambda: _rows(pool, CLAUDE_TOKENS_SQL, base), ("claude",))
    if wanted & {"vram", "processes", "hosts"}:
        tasks["vram"] = timed("biometrics", lambda: _rows(pool, VRAM_SQL, {"hosts": list(VRAM_HOSTS)}),
                              ("vram", "processes", "hosts"))
    if wanted & {"processes", "hosts"} and deps.container_names is not None:
        async def names():
            try:
                return dict(await deps.container_names())
            except Exception:  # noqa: BLE001 -- names are a nicety: an id is still a true label
                logger.warning("physiology_report_container_names_failed", exc_info=True)
                return {}
        tasks["names"] = names()
    if "hosts" in wanted:
        # Three reads, run side by side (30 d: ~0.5 s, ~1 s, ~0.9 s on 2026-10-30).
        hp = {"hosts": list(VRAM_HOSTS), "until_ts": _bio_ts(now)}
        rest_body, rest_source, rest_err = load_rest_body(str(Path(deps.heat_sensitivity_path).with_name(REST_BODY_FILE)))
        rs, ru = rest_window(rest_body)
        tasks["hosts_rest"] = timed("hosts_rest", lambda: pinned_parts_rows(pool, {
            "hosts": list(VRAM_HOSTS), "p": HOST_REST_PERCENTILE, "rest_since": rs, "rest_until": ru}, now), ("hosts",))
        tasks["hosts_floor"] = timed("hosts_floor", lambda: _rows(pool, HOSTS_FLOOR_SQL, {
            **hp, "p": HOST_REST_PERCENTILE, "floor_since": _bio_ts(now - HOST_FLOOR_WINDOW)}), ("hosts",))
        tasks["hosts_daily"] = timed("hosts_daily", lambda: _rows(pool, HOSTS_DAILY_SQL, {
            **hp, "tz": deps.tz, "since_ts": _bio_ts(since)}), ("hosts",))
        tasks["hosts_holders"] = timed("hosts_holders", lambda: _rows(pool, HOSTS_HOLDERS_SQL, {
            **base, "events": list(RHYTHM_CLOSE_EVENTS), "top": HOST_TOP_HOLDERS}), ("hosts",))
    if "services" in wanted:
        tasks["services"] = timed("services", lambda: _rows(pool, SERVICES_SQL, {
            "since_naive": (now - SERVICES_LOOKBACK).replace(tzinfo=None)}), ("services",))
    if "value_per_cost" in wanted and deps.family_weights is not None:
        tasks["weights"] = timed("value_per_cost", deps.family_weights, ("value_per_cost",))

    t_all = time.perf_counter()
    results = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    latest_row = results.get("latest")
    out: dict[str, Any] = {"generated_at": now, "window": {"start": since, "end": now},
                           "query": {"window": q.window, "sections": list(q.sections), "family": q.family,
                                     "outcome": q.outcome, "limit": q.limit}}

    if latest_row is not None:
        at = _utc(latest_row.get("generated_at"))
        if at is None or now - at > LATEST_MAX_AGE:
            age = "unknown" if at is None else f"{(now - at).total_seconds() / 60.0:.0f} min"
            for s in ("limits", "money", "unit_costs", "claude"):
                if s in wanted:
                    missing.append({"section": s, "reason": f"stale: the newest physiology tick is {age} old "
                                    f"(ticks run every ~2 min), so 'now' values are from {at.isoformat() if at else '?'}"})

    def add(section: str, fn: Callable[[], tuple[Any, list[dict]]]) -> None:
        """Shaping errors are a reason too: a bad row empties its own section, never the report."""
        try:
            value, miss = fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("physiology_report_shape_failed section=%s", section, exc_info=True)
            missing.append({"section": section, "reason": f"shaping failed: {type(exc).__name__}: {str(exc)[:160]}"})
            return
        out[section] = value
        missing.extend(miss)

    ac = results.get("act_costs") if "act_costs" in results else {}
    if "limits" in wanted and results.get("limits") is not None:
        add("limits", lambda: shape_limits(results["limits"][0], latest_row, results["limits"][1], since))
    if "wants" in wanted and results.get("wants") is not None:
        add("wants", lambda: shape_wants(results["wants"][0], results["wants"][1], deps.tiers, q.family))
    if "rhythm" in wanted and results.get("rhythm") is not None:
        add("rhythm", lambda: shape_rhythm(results["rhythm"]))
    if "money" in wanted and results.get("money") is not None:
        add("money", lambda: shape_money(latest_row, results["money"]))
    if "unit_costs" in wanted and results.get("run_costs") is not None:
        def _unit():
            rate = shape_money(latest_row, [])[0]["rate_usd_per_kwh"] if latest_row else None
            hosts, herr = load_heat_hosts(deps.heat_sensitivity_path)
            att = results.get("attribution") if "attribution" in results else None
            if "attribution" in results and att is None:
                att = {"present": True, "rows": None}  # the read failed; its reason is in missing
            return shape_unit_costs(rate, hosts, herr, results["run_costs"], ac, results.get("tokens"), att)
        add("unit_costs", _unit)
    if "waste" in wanted and results.get("waste") is not None:
        add("waste", lambda: shape_waste(*results["waste"]))
    if "queue_wait" in wanted and results.get("queue_wait") is not None:
        add("queue_wait", lambda: shape_queue_wait(results["queue_wait"], q.family))
    if "decisions" in wanted and results.get("decisions") is not None:
        first = results["wants"][1] if results.get("wants") else None
        add("decisions", lambda: shape_decisions(results["decisions"], first, q.family))
    if "claude" in wanted and results.get("claude") is not None:
        add("claude", lambda: shape_claude((results["claude"] or [None])[0], latest_row))
    if "vram" in wanted and results.get("vram") is not None:
        def _vram():
            doc, perr = _load_yaml(deps.gpu_pool_path)
            return shape_vram(doc, perr, results["vram"], now)
        add("vram", _vram)
    names_map = results.get("names") or {}
    if "processes" in wanted and results.get("vram") is not None:
        add("processes", lambda: shape_processes(results["vram"], now, names_map))
    if "hosts" in wanted and results.get("vram") is not None and results.get("hosts_rest") is not None:
        def _hosts():
            doc, perr = _load_yaml(deps.gpu_pool_path)
            rate = shape_money(latest_row, [])[0]["rate_usd_per_kwh"] if latest_row else None
            # The daily and holder reads are extras: a failure there is already a "read failed"
            # note, and the machines still show their draw now and rest.
            rest_rows = merge_by_node(results["hosts_rest"], results.get("hosts_floor") or [])
            return shape_hosts(results["vram"], rest_rows, results.get("hosts_daily") or [],
                               results.get("hosts_holders") or [], doc, perr, rate, now, names_map,
                               rest_body=rest_body, rest_source=rest_source, rest_err=rest_err)
        add("hosts", _hosts)
    if "value_per_cost" in wanted:
        if deps.family_weights is None:
            add("value_per_cost", lambda: shape_value_per_cost(
                None, "the bid market is off (ORION_PHYSIOLOGY_BID_ENABLED)", ac or {}, q.family))
        elif results.get("weights") is not None:
            add("value_per_cost", lambda: shape_value_per_cost(results["weights"], None, ac or {}, q.family))
    if "services" in wanted and results.get("services") is not None:
        add("services", lambda: shape_services(results["services"], now))
    if q.family:
        whole = [s for s in q.sections if s not in FAMILY_FILTERED and s != "unit_costs"]
        if whole:
            missing.append({"section": "*", "reason": f"family={q.family} narrows wants, decisions, queue_wait, "
                            f"value_per_cost and the per-act unit costs only; {', '.join(whole)} are whole-body"})
    not_asked = [s for s in REPORT_SECTIONS if s not in wanted]
    if not_asked and len(not_asked) < len(REPORT_SECTIONS):
        missing.append({"section": "*", "reason": f"not asked for: {', '.join(not_asked)}"})
    timing["total"] = round((time.perf_counter() - t_all) * 1000.0, 1)
    out["missing"] = missing
    out["timing_ms"] = timing
    return PhysiologyReportV1.model_validate(out)


def has_read_failure(report: PhysiologyReportV1) -> bool:
    return any(m.reason.startswith(("read failed", "shaping failed")) for m in report.missing)


class ReportCache:
    """Same query inside ``ttl_sec`` -> the same report. A report with a failed read is not cached
    (the next request retries). Concurrent misses on one key share one build (single flight)."""

    def __init__(self, ttl_sec: float) -> None:
        self.ttl = ttl_sec
        self._d: dict[tuple, tuple[float, PhysiologyReportV1]] = {}
        self._inflight: dict[tuple, asyncio.Future] = {}

    def get(self, key: tuple) -> Optional[PhysiologyReportV1]:
        hit = self._d.get(key)
        if hit and time.monotonic() - hit[0] <= self.ttl:
            return hit[1]
        return None

    def put(self, key: tuple, report: PhysiologyReportV1) -> None:
        if self.ttl <= 0 or has_read_failure(report):
            return
        if len(self._d) > 64:
            self._d.clear()
        self._d[key] = (time.monotonic(), report)

    async def get_or_build(self, key: tuple, build: Callable[[], Awaitable[PhysiologyReportV1]]) -> PhysiologyReportV1:
        hit = self.get(key)
        if hit is not None:
            return hit
        fut = self._inflight.get(key)
        if fut is not None:
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            report = await build()
            self.put(key, report)
            fut.set_result(report)
            return report
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # retrieved: no "never retrieved" warning when nobody else waited
            raise
        finally:
            self._inflight.pop(key, None)


def tiers_from_families(families: Any) -> dict[str, int]:
    try:
        return {f: int(c.tier) for f, c in (families.families or {}).items()}
    except Exception:  # noqa: BLE001
        return {}


__all__ = [
    "ReportCache",
    "ReportDeps",
    "ReportQuery",
    "build_report",
    "parse_query",
    "tiers_from_families",
]
