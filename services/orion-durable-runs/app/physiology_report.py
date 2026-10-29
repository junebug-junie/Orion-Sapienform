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
                 ``grammar_acts.cost_json`` (``wh_by_host``/``usd``/``heat_c_by_host`` when filled)
* waste       -- lease closes whose reason is not a success
* queue_wait  -- ``waited_ms`` on granted leases, family joined from ``want_ledger`` for durable runs
* decisions   -- ``want_ledger`` rows, newest first
* claude      -- ``dev_economics_ledger_log`` token deltas + the newest ``claude`` constraint
* vram        -- ``orion_biometrics_summary.measurements`` ``gpuN_mem_used_mb``/``gpuN_mem_total_mb``
                 per host, card -> roles from ``config/gpu_pool.yaml``
* value_per_cost -- the market's own family weights evidence (``Market.family_weights``)
* services    -- mesh-guardian ``stopped_container`` cards in ``notify_requests``
"""

from __future__ import annotations

import asyncio
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
RHYTHM_CLOSE_EVENTS = ("released", "expired", "aborted", "retried")
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

RHYTHM_SQL = """
SELECT to_char(generated_at AT TIME ZONE %(tz)s, 'YYYY-MM-DD') AS day,
       EXTRACT(HOUR FROM generated_at AT TIME ZONE %(tz)s)::int AS hour,
       COALESCE(SUM(held_ms) FILTER (WHERE priority = 'background'), 0) / 1000.0 AS heavy_s,
       COALESCE(SUM(held_ms), 0) / 1000.0 AS total_s
FROM gpu_pool_events
WHERE generated_at >= %(since)s AND generated_at < %(until)s
  AND event = ANY(%(events)s) AND held_ms IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2
"""

ENERGY_DAILY_SQL = """
SELECT to_char(hour_start AT TIME ZONE %(tz)s, 'YYYY-MM-DD') AS d, SUM(kwh) AS kwh
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

WASTE_BY_LANE_SQL = """
SELECT COALESCE(role, '?') AS lane, SUM(held_ms) / 1000.0 AS held_s
FROM gpu_pool_events
WHERE generated_at >= %(since)s AND generated_at < %(until)s
  AND event = ANY(%(events)s) AND held_ms IS NOT NULL
  AND (reason IS NULL OR reason <> ALL(%(ok)s))
GROUP BY 1 ORDER BY 2 DESC
"""
WASTE_TOP_SQL = """
SELECT holder, role, held_ms / 1000.0 AS held_s, reason, generated_at AS at
FROM gpu_pool_events
WHERE generated_at >= %(since)s AND generated_at < %(until)s
  AND event = ANY(%(events)s) AND held_ms IS NOT NULL
  AND (reason IS NULL OR reason <> ALL(%(ok)s))
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
         e.waited_ms
  FROM gpu_pool_events e
  LEFT JOIN fam ON e.holder LIKE 'durable-runs:%%' AND fam.want_id = substr(e.holder, 14)
  WHERE e.generated_at >= %(since)s AND e.generated_at < %(until)s
    AND e.event = 'granted' AND e.waited_ms IS NOT NULL
)
SELECT lane, family,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY waited_ms) / 1000.0 AS p50_s,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY waited_ms) / 1000.0 AS p90_s,
       COUNT(*) AS n
FROM g
WHERE (%(family)s::text IS NULL OR family = %(family)s)
GROUP BY 1, 2
ORDER BY 1, 2
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


def shape_wants(rows: Sequence[Mapping], first_at: Optional[datetime], tiers: Mapping[str, int]
                ) -> tuple[dict, list[dict]]:
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
        missing.append({"section": "wants", "reason": f"no bid decision in the window; {first}"})
    return out, missing


def shape_rhythm(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    cells = [{"day": str(r["day"]), "hour": int(r["hour"]), "heavy": round(float(r.get("heavy_s") or 0.0), 1),
              "total": round(float(r.get("total_s") or 0.0), 1)} for r in rows]
    missing = [] if cells else [{"section": "rhythm", "reason": "no GPU lease closed in the window (gpu_pool_events)"}]
    return cells, missing


_FLOOR_RE = re.compile(r"always-on floor alone \(\$[0-9.]+/h\) reaches \$([0-9.]+)")


def shape_money(latest: Optional[Mapping], daily: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    """The newest tick's ``energy`` reading. ``floor_usd`` is not a field on the reading; it is the
    number the Budget organ writes into its own ``why`` (orion/physiology/budget.py energy_ask)."""
    missing: list[dict] = []
    money: dict = {"mtd_usd": None, "forecast_usd": None, "envelope_usd": None, "floor_usd": None,
                   "rate_usd_per_kwh": None,
                   "daily_kwh": [{"d": str(r["d"]), "kwh": round(float(r["kwh"]), 3)} for r in daily
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
        out[actor] = {"n": n,
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
                     run_costs: Sequence[Mapping], act_costs: Mapping[str, dict]) -> tuple[list[dict], list[dict]]:
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
    for actor in sorted(act_costs):
        a = act_costs[actor]
        rows.append({"unit": f"one {actor} act", "usd": _r(a["usd"], 6), "wh": _r(a["wh"], 3),
                     "heat_c": _r(a["heat_c"], 4), "status": "measured",
                     "source": f"grammar_acts.cost_json mean over {a['n']} acts in the window"})
    if not act_costs:
        rows.append({"unit": "one act (any family)", "usd": None, "wh": None, "heat_c": None,
                     "status": "not_measured",
                     "source": "no act in the window carries cost_json wh_by_host / usd / heat_c_by_host yet "
                               "(the per-act energy fill is being added)"})
    # TODO(per-call token table): a sibling agent is adding a per-LLM-call token table. Read it here
    # (USD and W·h per 1k tokens) once it exists and its name is on main. Until then: not_measured.
    rows.append({"unit": "per 1k tokens", "usd": None, "wh": None, "heat_c": None, "status": "not_measured",
                 "source": "no per-call token table yet (being added); tokens are not joined to energy"})
    if not any(r["status"] == "measured" for r in rows):
        missing.append({"section": "unit_costs", "reason": "nothing measured in the window: every row is derived "
                        "from the tariff and the heat fit, or not measured yet"})
    return rows, missing


def shape_waste(by_lane: Sequence[Mapping], top: Sequence[Mapping]) -> tuple[dict, list[dict]]:
    out = {"failed_lease_s_by_lane": {str(r["lane"]): round(float(r["held_s"] or 0.0), 1) for r in by_lane},
           "top": [{"holder": r.get("holder"), "role": r.get("role"), "held_s": round(float(r.get("held_s") or 0.0), 1),
                    "reason": r.get("reason"), "at": _utc(r.get("at"))} for r in top]}
    missing = [] if by_lane else [{"section": "waste", "reason": "no GPU lease closed with a failure in the window"}]
    return out, missing


def shape_queue_wait(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    out = [{"lane": str(r["lane"]), "family": str(r["family"] or "?"), "p50_s": _r(_f(r.get("p50_s")), 1),
            "p90_s": _r(_f(r.get("p90_s")), 1), "n": int(r.get("n") or 0)} for r in rows]
    missing = [] if out else [{"section": "queue_wait", "reason": "no GPU lease was granted in the window"}]
    return out, missing


def shape_decisions(rows: Sequence[Mapping], first_at: Optional[datetime]) -> tuple[list[dict], list[dict]]:
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
                 else "want_ledger is empty (shadow bids began 2026-10-27)")
        missing.append({"section": "decisions", "reason": f"no bid decision matches in the window; {first}"})
    return out, missing


CLAUDE_TOKEN_SCOPE = ("tokens are Claude Code sessions on athena (dev_economics, ~/.claude/projects: Juniper's and "
                      "her agents' sessions); Orion's own harness sessions are not in this count")


def shape_claude(tokens: Optional[Mapping], latest: Optional[Mapping]) -> tuple[dict, list[dict]]:
    missing: list[dict] = []
    t = tokens or {}
    out: dict = {"window_tokens": {"input": int(t.get("input") or 0), "output": int(t.get("output") or 0),
                                   "cache_read": int(t.get("cache_read") or 0),
                                   "cache_write": int(t.get("cache_write") or 0)},
                 "usd_est": None, "limit_events": [], "status": ""}
    priced = int(t.get("priced") or 0)
    usd = _f(t.get("usd"))
    out["usd_est"] = round(usd, 2) if (usd is not None and priced > 0) else None
    claude = None
    for c in (_json(latest.get("constraints")) if latest else None) or []:
        if isinstance(c, dict) and c.get("constraint") == "claude":
            claude = c
    parts = [CLAUDE_TOKEN_SCOPE]
    if usd is None or priced == 0:
        parts.append("no priced session in the window, so no USD estimate")
    elif int(t.get("unpriced") or 0):
        parts.append(f"{int(t.get('unpriced'))} session(s) had no price and are left out of usd_est")
    if claude is None:
        parts.append("limits: no claude reading on the newest physiology tick")
    elif claude.get("no_reading"):
        parts.append(f"limits: unknown ({claude.get('no_reading')}): no ClaudeLimitObservationV1 has arrived on "
                     "orion:substrate:claude_limit; its producer in orion-cocreation-signals is off by default "
                     "(COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false)")
    else:
        detail = claude.get("detail") if isinstance(claude.get("detail"), dict) else {}
        for ev in detail.get("events") or detail.get("limit_events") or []:
            at = _utc((ev or {}).get("at") or (ev or {}).get("observed_at"))
            if at is not None:
                out["limit_events"].append({"at": at, "kind": str(ev.get("kind") or "?")})
        parts.append(f"limits: {claude.get('state') or 'read'}, level {claude.get('level')}")
    out["status"] = "; ".join(parts)
    if int(t.get("ticks") or 0) == 0:
        missing.append({"section": "claude", "reason": "dev_economics_ledger_log has no row in the window "
                        "(orion-cocreation-signals dev_economics producer down or off)"})
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
        reported += used is not None and total is not None
        out.append({"host": host, "card": card, "used_mb": used, "total_mb": total,
                    "headroom_mb": (total - used) if (used is not None and total is not None) else None,
                    "roles": roles})
    if pool_err:
        missing.append({"section": "vram", "reason": f"card -> role map unavailable ({pool_err})"})
    if reported == 0:
        why = ("per-card GPU memory is not reported yet: orion_biometrics_summary.measurements has no "
               "gpuN_mem_used_mb / gpuN_mem_total_mb (being added beside gpuN_watts)")
        if stale:
            why += f"; no fresh biometrics row for {', '.join(sorted(stale))}"
        missing.append({"section": "vram", "reason": why})
    return out, missing


def shape_value_per_cost(weights: Optional[Mapping[str, Any]], weights_err: Optional[str],
                         act_costs: Mapping[str, dict]) -> tuple[list[dict], list[dict]]:
    """One row per family with graded evidence: the same evidence the market's family weights use
    (orion/physiology/weights.py: (a) downstream use rate, (b) Juniper reply rate)."""
    if weights is None:
        return [], [{"section": "value_per_cost", "reason": f"family weights unavailable ({weights_err or 'market off'})"}]
    rows: list[dict] = []
    no_evidence: list[str] = []
    for fam in sorted(weights):
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
        rows.append({"family": fam, "evidence_n": n, "value_metric": "mean of " + ", ".join(parts),
                     "value": round(float(rate), 4), "usd_per_act": _r(usd, 6),
                     "status": "measured" if usd is not None else "not_measured"})
    missing = []
    if no_evidence:
        missing.append({"section": "value_per_cost",
                        "reason": f"no graded evidence yet for {len(no_evidence)} famil"
                                  f"{'y' if len(no_evidence) == 1 else 'ies'}: {', '.join(no_evidence)}"})
    if rows and all(r["usd_per_act"] is None for r in rows):
        missing.append({"section": "value_per_cost",
                        "reason": "usd_per_act not measured: no act carries cost_json.usd yet"})
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
    concurrency: int = 3


def _load_yaml(path: str) -> tuple[Optional[dict], Optional[str]]:
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text()) or {}, None
    except FileNotFoundError:
        return None, f"missing:{Path(path).name}"
    except Exception as exc:  # noqa: BLE001 -- a broken file is a reason, not a 500
        return None, f"unreadable:{Path(path).name}:{type(exc).__name__}"


async def build_report(deps: ReportDeps, q: ReportQuery, now: Optional[datetime] = None) -> PhysiologyReportV1:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = now - WINDOW_DELTAS[q.window]
    base = {"since": since, "until": now}
    sem = asyncio.Semaphore(max(1, deps.concurrency))
    timing: dict[str, float] = {}
    missing: list[dict] = []
    wanted = set(q.sections)

    async def timed(name: str, coro_fn: Callable[[], Awaitable[Any]]) -> Any:
        async with sem:
            t0 = time.perf_counter()
            try:
                return await coro_fn()
            except Exception as exc:  # noqa: BLE001 -- one section failing never fails the report
                logger.warning("physiology_report_read_failed section=%s", name, exc_info=True)
                missing.append({"section": name, "reason": f"read failed: {type(exc).__name__}: {str(exc)[:160]}"})
                return None
            finally:
                timing[name] = round((time.perf_counter() - t0) * 1000.0, 1)

    pool = deps.pool
    syn = list(SYNTHETIC_PRODUCERS)

    async def latest():
        return (await _rows(pool, LIMITS_LATEST_SQL, base) or [None])[0]

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

    need_latest = wanted & {"limits", "money", "unit_costs", "claude"}
    need_act_costs = wanted & {"unit_costs", "value_per_cost"}
    tasks: dict[str, Awaitable[Any]] = {}
    if need_latest:
        tasks["latest"] = timed("latest", latest)
    if "limits" in wanted:
        tasks["limits"] = timed("limits", limits)
    if "wants" in wanted or "decisions" in wanted:
        tasks["wants"] = timed("wants", wants)
    if "rhythm" in wanted:
        tasks["rhythm"] = timed("rhythm", lambda: _rows(pool, RHYTHM_SQL, {**base, "tz": deps.tz,
                                                                          "events": list(RHYTHM_CLOSE_EVENTS)}))
    if "money" in wanted:
        tasks["money"] = timed("money", lambda: _rows(pool, ENERGY_DAILY_SQL, {**base, "tz": deps.tz}))
    if "unit_costs" in wanted:
        tasks["run_costs"] = timed("unit_costs", lambda: _rows(pool, RUN_COST_SQL, base))
    if need_act_costs:
        tasks["act_costs"] = timed("act_costs", act_costs)
    if "waste" in wanted:
        wp = {**base, "events": list(WASTE_CLOSE_EVENTS), "ok": list(OK_CLOSE_REASONS), "top": WASTE_TOP_N}

        async def waste():
            return await _rows(pool, WASTE_BY_LANE_SQL, wp), await _rows(pool, WASTE_TOP_SQL, wp)
        tasks["waste"] = timed("waste", waste)
    if "queue_wait" in wanted:
        tasks["queue_wait"] = timed("queue_wait", lambda: _rows(pool, QUEUE_WAIT_SQL, {**base, "family": q.family}))
    if "decisions" in wanted:
        tasks["decisions"] = timed("decisions", lambda: _rows(pool, DECISIONS_SQL, {
            **base, "synthetic": syn, "family": q.family, "outcome": q.outcome, "limit": q.limit}))
    if "claude" in wanted:
        tasks["claude"] = timed("claude", lambda: _rows(pool, CLAUDE_TOKENS_SQL, base))
    if "vram" in wanted:
        tasks["vram"] = timed("vram", lambda: _rows(pool, VRAM_SQL, {"hosts": list(VRAM_HOSTS)}))
    if "services" in wanted:
        tasks["services"] = timed("services", lambda: _rows(pool, SERVICES_SQL, {
            "since_naive": (now - SERVICES_LOOKBACK).replace(tzinfo=None)}))
    if "value_per_cost" in wanted and deps.family_weights is not None:
        tasks["weights"] = timed("value_per_cost", deps.family_weights)

    t_all = time.perf_counter()
    results = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    latest_row = results.get("latest")
    out: dict[str, Any] = {"generated_at": now, "window": {"start": since, "end": now}}

    def add(section: str, value_missing: tuple[Any, list[dict]]):
        value, miss = value_missing
        out[section] = value
        missing.extend(miss)

    if "limits" in wanted and results.get("limits") is not None:
        h, first = results["limits"]
        add("limits", shape_limits(h, latest_row, first, since))
    if "wants" in wanted and results.get("wants") is not None:
        r, first = results["wants"]
        add("wants", shape_wants(r, first, deps.tiers))
    if "rhythm" in wanted and results.get("rhythm") is not None:
        add("rhythm", shape_rhythm(results["rhythm"]))
    if "money" in wanted and results.get("money") is not None:
        add("money", shape_money(latest_row, results["money"]))
    ac = results.get("act_costs") or {}
    if "unit_costs" in wanted and results.get("run_costs") is not None:
        rate = None
        if latest_row:
            m, _ = shape_money(latest_row, [])
            rate = m["rate_usd_per_kwh"]
        hosts, herr = load_heat_hosts(deps.heat_sensitivity_path)
        add("unit_costs", shape_unit_costs(rate, hosts, herr, results["run_costs"], ac))
    if "waste" in wanted and results.get("waste") is not None:
        add("waste", shape_waste(*results["waste"]))
    if "queue_wait" in wanted and results.get("queue_wait") is not None:
        add("queue_wait", shape_queue_wait(results["queue_wait"]))
    if "decisions" in wanted and results.get("decisions") is not None:
        first = results["wants"][1] if results.get("wants") else None
        add("decisions", shape_decisions(results["decisions"], first))
    if "claude" in wanted and results.get("claude") is not None:
        add("claude", shape_claude((results["claude"] or [None])[0], latest_row))
    if "vram" in wanted and results.get("vram") is not None:
        doc, perr = _load_yaml(deps.gpu_pool_path)
        add("vram", shape_vram(doc, perr, results["vram"], now))
    if "value_per_cost" in wanted:
        if deps.family_weights is None:
            add("value_per_cost", shape_value_per_cost(None, "the bid market is off (ORION_PHYSIOLOGY_BID_ENABLED)", ac))
        elif results.get("weights") is not None:
            add("value_per_cost", shape_value_per_cost(results["weights"], None, ac))
    if "services" in wanted and results.get("services") is not None:
        add("services", shape_services(results["services"], now))
    not_asked = [s for s in REPORT_SECTIONS if s not in wanted]
    if not_asked and len(not_asked) < len(REPORT_SECTIONS):
        missing.append({"section": "*", "reason": f"not asked for: {', '.join(not_asked)}"})
    timing["total"] = round((time.perf_counter() - t_all) * 1000.0, 1)
    out["missing"] = missing
    out["timing_ms"] = timing
    return PhysiologyReportV1.model_validate(out)


class ReportCache:
    """Same query inside ``ttl_sec`` -> the same report. A report is a few hundred KB at most."""

    def __init__(self, ttl_sec: float) -> None:
        self.ttl = ttl_sec
        self._d: dict[tuple, tuple[float, PhysiologyReportV1]] = {}

    def get(self, key: tuple) -> Optional[PhysiologyReportV1]:
        hit = self._d.get(key)
        if hit and time.monotonic() - hit[0] <= self.ttl:
            return hit[1]
        return None

    def put(self, key: tuple, report: PhysiologyReportV1) -> None:
        if self.ttl <= 0:
            return
        if len(self._d) > 64:
            self._d.clear()
        self._d[key] = (time.monotonic(), report)


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
