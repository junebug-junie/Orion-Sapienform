"""Fill an act's energy, money and heat on close (ActCostV2 ``wh_by_host``, ``wh_gpu_by_host``,
``usd``, ``heat_c_by_host``, ``basis``). Called by orion/grammar/act_ledger.py ``fill_close_cost``.

Reads, never writes: the act's GPU pool leases (gpu_pool_events, joined by the act's
correlation_id exactly as ``gpu_slot_s_by_lane`` is), the other leases on the same cards, the
host's readings (orion_biometrics_summary ``chassis_watts``, ``cpu_watts_total``,
``gpu{index}_watts``), the house tariff's marginal rate (energy_cost_accrued) and the heat fit
(config/physiology/heat_sensitivity.yaml). The arithmetic is orion/energy/act_energy.py.

``wh_by_host`` is the act's share of the whole machine's power above rest (chassis, at the wall).
A host with no chassis meter or no rest (hecate today) falls back to GPU above idle, and ``basis``
says so for that host. ``wh_gpu_by_host`` is always the GPU-above-idle part alone.

Known limits, said here rather than hidden:
* GPU work that takes no pool lease still draws on the card; it lands on whoever holds a lease then
  on that card, and in the host's unmetered energy otherwise.
* A durable hold and the child call it wraps are two leases on one card: each pays half.
* Acts sharing one correlation_id would each be charged the turn's whole lease energy (same exposure
  as ``gpu_slot_s_by_lane``; live 2026-10-29: 1103 acts, 1103 distinct correlation ids).
* A lease's cards are taken from one terminal row.
* Per-act Wh from ~30 s samples scatters widely; the p05 idle floor reads ~20% above the 1 Hz meter
  on reverie images (eval: services/orion-sql-writer/evals/act_energy_vs_power_intent.py).
* The chassis meter (PDU) and the cards are read in the same ~30 s row but not at the same instant,
  so a short act can see a chassis/load ratio under 1. The split still sums to the measured total
  (eval: services/orion-sql-writer/evals/act_energy_closure.py).

Rest: the host's PINNED chassis rest from config/physiology/rest_body.yaml (never rolling). CPU:
the packages' watts above their fitted idle (the row's ``cpu_idle_watts``) are in the split, so the
overhead they cause is not landed on GPU acts, but no act is charged them: an act does not name the
containers it ran in. That share stays in the host's unmetered energy (``basis["cpu"]``).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from orion.energy.act_energy import (
    IDLE_MIN_SAMPLES,
    IDLE_QUANTILE,
    HostFloors,
    HostSeries,
    card_energy,
    heat_c,
    machine_energy,
    usd,
)

logger = logging.getLogger("orion.grammar.act_energy_fill")

_REPO_ROOT = Path(__file__).resolve().parents[2]
HEAT_SENSITIVITY_PATH = _REPO_ROOT / "config" / "physiology" / "heat_sensitivity.yaml"
SAMPLE_PAD = timedelta(seconds=120)
RATE_MAX_AGE = timedelta(days=7)
# Leases held at once on one card: a durable hold is capped at 9000 s (gpu_pool.yaml max_hold_sec).
OTHER_LEASE_LOOKBACK = timedelta(hours=3)
MAX_HOLD = timedelta(seconds=9000)
CPU_BASIS = ("CPU above its fitted idle (cpu_idle_watts) is in the whole-machine split, so GPU acts do not pay for its overhead, but no "
             "act is charged it: an act does not name its containers. It stays in the host's unmetered energy")
WHOLE_MACHINE = "whole machine above rest (chassis)"
GPU_ONLY = "GPU above idle only"

_TERMINAL = "('released', 'expired', 'aborted')"

_ACT_LEASES_SQL = text(
    f"""
    SELECT DISTINCT ON (lease_id) lease_id, cards::text, held_ms, generated_at
      FROM gpu_pool_events
     WHERE turn_correlation_id = :corr
       AND generated_at >= :since AND generated_at <= :until
       AND event IN {_TERMINAL}
       AND held_ms IS NOT NULL AND cards IS NOT NULL
     ORDER BY lease_id, held_ms DESC
    """
)

# Every lease on the card that overlaps [ws, we]: finished ones by their terminal event, and the
# ones still held at close (granted, no terminal yet: they run to "now").
_CARD_LEASES_SQL = text(
    f"""
    SELECT lease_id, start_at, end_at, turn_correlation_id FROM (
        SELECT DISTINCT ON (lease_id) lease_id,
               generated_at - held_ms * interval '1 millisecond' AS start_at, generated_at AS end_at,
               turn_correlation_id
          FROM gpu_pool_events
         WHERE event IN {_TERMINAL} AND held_ms IS NOT NULL
           AND cards::jsonb ? :card
           AND generated_at >= :ws AND generated_at <= :we_plus
         ORDER BY lease_id, held_ms DESC
    ) done
    UNION ALL
    SELECT lease_id, start_at, end_at, turn_correlation_id FROM (
        SELECT DISTINCT ON (g.lease_id) g.lease_id, g.generated_at AS start_at, NULL::timestamptz AS end_at,
               g.turn_correlation_id
          FROM gpu_pool_events g
         WHERE g.event = 'granted' AND g.cards::jsonb ? :card
           AND g.generated_at >= :ws_minus AND g.generated_at <= :we
           AND NOT EXISTS (SELECT 1 FROM gpu_pool_events t
                            WHERE t.lease_id = g.lease_id AND t.event IN {_TERMINAL})
         ORDER BY g.lease_id, g.generated_at
    ) held
    """
)

# Which of these turns have an act (the act fill charges a lease to the act whose correlation_id is
# the lease's turn_correlation_id).
_TURNS_WITH_ACT_SQL = text("SELECT DISTINCT correlation_id FROM grammar_acts WHERE correlation_id = ANY(:corrs)")

# orion_biometrics_summary.timestamp is a VARCHAR ("2026-10-29 16:41:23.287434+00") indexed with
# node: filtered as a string on that prefix, parsed as timestamptz (same as cabinet_heat.py).
# One read per host per act: the chassis, the CPU packages and every card, row by row.
_HOST_WINDOW_COLUMNS = (
    "SELECT timestamp::timestamptz AS ts, (measurements->>'chassis_watts')::float AS chassis, "
    "(measurements->>'cpu_watts_total')::float AS cpu, (measurements->>'cpu_idle_watts')::float AS cpu_idle, "
    "(SELECT jsonb_object_agg(k, v) FROM jsonb_each_text(measurements) AS e(k, v) "
    "  WHERE k ~ '^gpu[0-9]+_watts$') AS cards"
)
_HOST_WINDOW_WHERE = (
    " FROM orion_biometrics_summary WHERE node = :node AND timestamp >= :since AND timestamp <= :until "
    "ORDER BY timestamp"
)
_HOST_WINDOW_SQL = text(_HOST_WINDOW_COLUMNS + _HOST_WINDOW_WHERE)
# The same plus each container's CPU watts (container_cpu_watts.<name>), for the sponsored split.
_HOST_WINDOW_CONTAINERS_SQL = text(
    _HOST_WINDOW_COLUMNS + ", (SELECT jsonb_object_agg(k, v) FROM jsonb_each_text(measurements) AS e(k, v) "
    "  WHERE k LIKE 'container_cpu_watts.%') AS containers" + _HOST_WINDOW_WHERE
)

# A floor: one quantile of one key over [since, until). percentile_cont interpolates linearly,
# the same rule as act_energy.idle_floor.
_QUANTILE_SQL = text(
    "SELECT percentile_cont(:q) WITHIN GROUP (ORDER BY (measurements->>:key)::float) AS v, count(*) AS n "
    "FROM orion_biometrics_summary WHERE node = :node AND timestamp >= :since AND timestamp < :until "
    "AND measurements ? :key AND (measurements->>:key)::float >= 0"
)

_RATE_SQL = text(
    "SELECT marginal_usd_per_kwh, interval_start, tariff_version FROM energy_cost_accrued "
    "WHERE interval_start <= :at AND interval_start >= :since ORDER BY interval_start DESC, computed_at DESC LIMIT 1"
)


def _ts_key(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


# --- config, cached per process -------------------------------------------------------------
_POOL_CACHE: dict[str, Any] = {}
_HEAT_CACHE: dict[str, Any] = {}
_FLOOR_CACHE: dict[tuple[str, str, date, float], tuple[float, Optional[float]]] = {}
_CONFIG_TTL_S = 600.0
_IDLE_TTL_S = 3600.0


def _card_locations() -> tuple[dict[str, tuple[str, int]], Optional[str]]:
    """card -> (host, nvidia-smi index) from config/gpu_pool.yaml, or ({}, reason)."""
    now = time.monotonic()
    if _POOL_CACHE and now - _POOL_CACHE["at"] < _CONFIG_TTL_S:
        return _POOL_CACHE["cards"], _POOL_CACHE["err"]
    cards: dict[str, tuple[str, int]] = {}
    err = None
    try:
        from orion.gpu_pool.config import load_pool_config

        cfg = load_pool_config()
        for name, spec in cfg.cards.items():
            if spec.index is not None:
                cards[name] = (cfg.card_host(name), int(spec.index))
    except Exception as exc:  # noqa: BLE001 -- an unreadable pool config is a basis reason
        err = f"gpu_pool_config_unreadable:{type(exc).__name__}"
    _POOL_CACHE.update(at=now, cards=cards, err=err)
    return cards, err


_REST_CACHE: dict[str, Any] = {}


def rest_body_config() -> tuple[Any, Optional[str]]:
    """config/physiology/rest_body.yaml, cached per process (orion/physiology/rest_body.py)."""
    now = time.monotonic()
    if _REST_CACHE and now - _REST_CACHE["at"] < _CONFIG_TTL_S:
        return _REST_CACHE["cfg"], _REST_CACHE["err"]
    from orion.physiology.rest_body import load_rest_body

    cfg, err = load_rest_body()
    _REST_CACHE.update(at=now, cfg=cfg, err=err)
    return cfg, err


def _heat_config() -> tuple[Any, Optional[str]]:
    now = time.monotonic()
    if _HEAT_CACHE and now - _HEAT_CACHE["at"] < _CONFIG_TTL_S:
        return _HEAT_CACHE["cfg"], _HEAT_CACHE["err"]
    from orion.physiology.heat_config import load_heat_sensitivity

    cfg, err = load_heat_sensitivity(HEAT_SENSITIVITY_PATH)
    _HEAT_CACHE.update(at=now, cfg=cfg, err=err)
    return cfg, err


def _floor(conn: Any, host: str, key: str, day: date, *, q: float, days: int, min_n: int) -> Optional[float]:
    """``q`` of ``key`` over the ``days`` before the UTC ``day``. None with under ``min_n`` samples.
    Day-keyed: every act on the host that day, and the closure eval, read the same floor."""
    now = time.monotonic()
    ck = (host, key, day, q)
    hit = _FLOOR_CACHE.get(ck)
    if hit is not None and now - hit[0] < _IDLE_TTL_S:
        return hit[1]
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    row = conn.execute(_QUANTILE_SQL, {"q": q, "key": key, "node": host,
                                       "since": _ts_key(day_start - timedelta(days=days)),
                                       "until": _ts_key(day_start)}).first()
    value = float(row[0]) if row is not None and row[0] is not None and int(row[1] or 0) >= min_n else None
    if len(_FLOOR_CACHE) > 256:
        _FLOOR_CACHE.clear()
    _FLOOR_CACHE[ck] = (now, value)
    return value


def _days(ws: datetime, we: datetime) -> list[date]:
    d, last, out = ws.astimezone(timezone.utc).date(), we.astimezone(timezone.utc).date(), []
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out


def read_host_series(conn: Any, host: str, ws: datetime, we: datetime, *,
                     sponsored_globs: Optional[list[str]] = None) -> HostSeries:
    """The host's readings over [ws, we] (padded). ``sponsored_globs``: also sum those containers'
    CPU watts per row (the hourly attribution); None skips the container read (the act fill)."""
    from orion.physiology.rest_body import is_sponsored

    sql = _HOST_WINDOW_CONTAINERS_SQL if sponsored_globs is not None else _HOST_WINDOW_SQL
    rows = conn.execute(sql, {"node": host, "since": _ts_key(ws - SAMPLE_PAD),
                              "until": _ts_key(we + SAMPLE_PAD) + "~"}).fetchall()
    chassis: list[tuple[datetime, float]] = []
    cpu_above: list[tuple[datetime, float]] = []
    sponsored: list[tuple[datetime, float]] = []
    cards: dict[int, list[tuple[datetime, float]]] = {}
    prefix = "container_cpu_watts."
    for row in rows:
        t, ch, c, idle, raw = _aware(row[0]), row[1], row[2], row[3], row[4]
        if ch is not None and ch >= 0:
            chassis.append((t, float(ch)))
        if c is not None and idle is not None and c >= 0 and idle >= 0:
            cpu_above.append((t, max(0.0, float(c) - float(idle))))
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        for k, v in data.items():
            try:
                idx, w = int(k[3:-6]), float(v)
            except (TypeError, ValueError):
                continue
            if w >= 0:
                cards.setdefault(idx, []).append((t, w))
        if sponsored_globs is not None and row[5] is not None:
            cont = row[5]
            cont = json.loads(cont) if isinstance(cont, str) else (cont or {})
            total = 0.0
            for k, v in cont.items():
                try:
                    if is_sponsored(k[len(prefix):], sponsored_globs):
                        total += max(0.0, float(v))
                except (TypeError, ValueError):
                    continue
            sponsored.append((t, total))
    return HostSeries(chassis=chassis, cpu_above=cpu_above, cards=cards, sponsored_cpu=sponsored)


def host_floors(conn: Any, host: str, days: list[date], card_indexes: set[int],
                ) -> tuple[dict[date, dict[int, float]], dict[date, HostFloors], Optional[str]]:
    """Per day: each card's idle floor, and the whole-machine floors (or why not)."""
    idle: dict[date, dict[int, float]] = {}
    out: dict[date, HostFloors] = {}
    cfg, err = rest_body_config()
    rest, why = (cfg.rest_w(host) if cfg is not None else (None, f"rest_body.yaml unreadable ({err})"))
    for day in days:
        idle[day] = {}
        for idx in sorted(card_indexes):
            v = _floor(conn, host, f"gpu{idx}_watts", day, q=IDLE_QUANTILE, days=1, min_n=IDLE_MIN_SAMPLES)
            if v is not None:
                idle[day][idx] = v
        missing_cards = sorted(card_indexes - set(idle[day]))
        if rest is None:
            continue
        if missing_cards:
            why = why or f"no idle floor for card(s) {missing_cards} on {day}"
            continue
        out[day] = HostFloors(chassis_w=rest, card_idle_w=dict(idle[day]))
    return idle, out, why


def _parse_cards(raw: Any) -> list[str]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [str(c) for c in data] if isinstance(data, list) else []


def _card_leases(conn: Any, card: str, ws: datetime, we: datetime, now: datetime,
                 ) -> dict[str, tuple[datetime, datetime, Optional[str]]]:
    """Every lease on ``card`` overlapping [ws, we]: lease_id -> (start, end, turn_correlation_id)."""
    out: dict[str, tuple[datetime, datetime, Optional[str]]] = {}
    for oid, o_start, o_end, corr in conn.execute(_CARD_LEASES_SQL, {
            "card": card, "ws": ws, "we": we, "we_plus": we + OTHER_LEASE_LOOKBACK,
            "ws_minus": ws - OTHER_LEASE_LOOKBACK}).fetchall():
        if o_start is None:
            continue
        o_s = _aware(o_start)
        # A grant with no terminal event yet: held until now, but never past the pool's longest
        # hold (live: ~8 grants a day never get a terminal event).
        o_e = _aware(o_end) if o_end is not None else min(now, o_s + MAX_HOLD)
        out.setdefault(str(oid), (o_s, o_e, str(corr) if corr else None))
    return out


def _other_leases(conn: Any, card: str, ws: datetime, we: datetime, now: datetime,
                  ) -> dict[str, tuple[datetime, datetime]]:
    return {k: (s, e) for k, (s, e, _) in _card_leases(conn, card, ws, we, now).items()}


def attribute_host_hour(conn: Any, host: str, start: datetime, end: datetime, *,
                        now: Optional[datetime] = None) -> tuple[Optional[Any], Optional[float], str]:
    """(HostAttribution, rest_w, rest_source) for ``host`` over [start, end]; (None, None, reason)
    when the host has no pinned rest. The same split, readers and floors as the act fill."""
    from orion.energy.act_energy import LeaseSpan, attribute_host_window

    now = _aware(now or datetime.now(timezone.utc))
    cfg, err = rest_body_config()
    if cfg is None:
        return None, None, f"rest_body.yaml unreadable ({err})"
    rest, why = cfg.rest_w(host)
    if rest is None:
        return None, None, why or "no rest"
    hr = cfg.hosts[host]
    source = (f"rest_body.yaml {host} rest_w {hr.rest_w:g} W ({hr.method}, {hr.measured}; "
              f"approved_at {hr.approved_at or 'not yet'})")
    globs = cfg.sponsored_containers(host, start)
    series = read_host_series(conn, host, start, end, sponsored_globs=globs)
    days = _days(start, end - timedelta(microseconds=1))
    locations, _ = _card_locations()
    # The pool's cards on this host are always in the split: a card that stops reporting is a hole
    # (split unknown), never silently dropped from the denominator.
    pool_cards = {index for (h, index) in locations.values() if h == host}
    _, floors, _ = host_floors(conn, host, days, set(series.cards) | pool_cards)
    raw: dict[int, list[tuple[datetime, datetime, Optional[str]]]] = {}
    for card, (h, index) in locations.items():
        if h != host:
            continue
        raw[index] = [(s, e, c) for s, e, c in _card_leases(conn, card, start, end, now).values()
                      if s < end and e > start]
    corrs = sorted({c for spans in raw.values() for _, _, c in spans if c})
    with_act = {str(r[0]) for r in conn.execute(_TURNS_WITH_ACT_SQL, {"corrs": corrs}).fetchall()} if corrs else set()
    leases = {i: [LeaseSpan(s, e, c in with_act) for s, e, c in spans] for i, spans in raw.items()}
    return attribute_host_window(series, floors, leases, start=start, end=end, rest_w=rest), rest, source


def act_energy_cost(conn: Any, *, correlation_id: str, since: datetime, until: datetime,
                    now: Optional[datetime] = None) -> dict[str, Any]:
    """ActCostV2 field updates for one act, or {} when the act held no GPU lease.

    ``since``/``until``: the window the act's leases are looked up in (the same lookback and close
    grace as ``gpu_slot_s_by_lane``). Every field that cannot be measured stays absent, and
    ``basis`` says why."""
    now = _aware(now or datetime.now(timezone.utc))
    leases = conn.execute(_ACT_LEASES_SQL, {"corr": correlation_id, "since": since, "until": until}).fetchall()
    if not leases:
        return {}
    basis: dict[str, str] = {"cpu": CPU_BASIS}
    locations, loc_err = _card_locations()
    wh_by_host: dict[str, float] = {}
    wh_gpu_by_host: dict[str, float] = {}
    host_lines: list[str] = []
    failed: Optional[str] = loc_err
    # Group the act's lease windows by host and card, so each host is read once (one readings read
    # over the union window) and each card's other leases once, however many leases the turn took.
    by_host: dict[str, dict[int, tuple[str, list[tuple[str, datetime, datetime]]]]] = {}
    for lease_id, cards_raw, held_ms, end_at in leases:
        end = _aware(end_at)
        start = end - timedelta(milliseconds=float(held_ms))
        if (end - start).total_seconds() <= 0:
            continue
        for card in _parse_cards(cards_raw):
            loc = locations.get(card)
            if loc is None:
                failed = failed or f"card_not_in_pool_config:{card}"
                continue
            host, index = loc
            by_host.setdefault(host, {}).setdefault(index, (card, []))[1].append((str(lease_id), start, end))
    for host, cards in by_host.items():
        if failed:
            break
        all_windows = [m for _, mine in cards.values() for m in mine]
        ws, we = min(m[1] for m in all_windows), max(m[2] for m in all_windows)
        series = read_host_series(conn, host, ws, we)
        days = _days(ws, we)
        # Every pool card on the host is in the split's denominator (same as attribute_host_hour): a
        # card that stops reporting is a hole (GPU-only fallback), never dropped from the load.
        pool_cards = {i for (h, i) in locations.values() if h == host}
        idle, floors, why = host_floors(conn, host, days, set(series.cards) | set(cards) | pool_cards)
        if why is None and not series.chassis:
            why = "no chassis_watts reading in the window"
        notes: list[str] = []
        gpu_wh = machine_wh = machine_gpu_wh = 0.0
        for index, (card, mine) in cards.items():
            if failed:
                break
            key = f"gpu{index}_watts"
            if any(index not in idle[d] for d in days):
                failed = f"no_idle_floor:{host}/{key}"
                break
            points = list(series.cards.get(index, ()))
            card_leases = _other_leases(conn, card, min(m[1] for m in mine), max(m[2] for m in mine), now)
            for lease_id, start, end in mine:
                # The act's OTHER leases on this card count as sharers too: two concurrent slots
                # of one turn each take their own share, and the shares still add up.
                others = [(o_s, o_e) for oid, (o_s, o_e) in card_leases.items()
                          if oid != lease_id and o_s < end and o_e > start]
                got = card_energy(points, idle_w=idle[_days(start, start)[0]][index], start=start, end=end,
                                  others=others)
                if got is None:
                    failed = f"no_watts_reading:{host}/{key}"
                    break
                gpu_wh += got.wh
                if why is None:
                    whole = machine_energy(series, floors, card=index, start=start, end=end, others=others)
                    if whole is None:
                        why = ("a second in the window has no chassis, CPU (cpu_watts_total + cpu_idle_watts) or "
                               "card reading within 90 s")
                    else:
                        machine_wh += whole.wh
                        machine_gpu_wh += whole.wh_gpu
                quality = "" if got.samples_in_window else " (interpolated: no sample inside the lease)"
                notes.append(f"{card}: {got.wh:.4f} Wh GPU, {got.samples_in_window} samples in "
                             f"{(end - start).total_seconds():.0f} s{quality}, share {got.mean_share:.2f}, "
                             f"idle {idle[_days(start, start)[0]][index]:.1f} W")
        if failed:
            break
        # The machine path's own GPU part when it ran (per-day card floors across midnight), so
        # wh_gpu_by_host and the GPU inside wh_by_host use the same floors.
        if why is None:
            gpu_wh = machine_gpu_wh
        wh_gpu_by_host[host] = gpu_wh
        if why is None:
            wh_by_host[host] = machine_wh
            mult = f", x{machine_wh / gpu_wh:.2f} of GPU" if gpu_wh > 0 else ""
            host_lines.append(f"{host}: {WHOLE_MACHINE}, {machine_wh:.4f} Wh{mult}; chassis rest "
                              f"{floors[days[0]].chassis_w:.0f} W (pinned, rest_body.yaml), CPU above its "
                              "fitted idle; " + "; ".join(notes))
        else:
            wh_by_host[host] = gpu_wh
            host_lines.append(f"{host}: {GPU_ONLY}, not whole machine ({why}); " + "; ".join(notes))
    if failed:
        basis["wh"] = f"unmeasured: {failed}"
        return {"basis": basis}
    if not wh_by_host:
        return {}
    basis["wh"] = (f"{WHOLE_MACHINE}: chassis watts above the host's rest, split by each card's above-idle "
                   "share of the host's load (cards + CPU), then by lease share. " + " | ".join(host_lines))
    out: dict[str, Any] = {"wh_by_host": {h: round(v, 6) for h, v in wh_by_host.items()},
                           "wh_gpu_by_host": {h: round(v, 6) for h, v in wh_gpu_by_host.items()}}
    total_wh = sum(wh_by_host.values())

    rate = conn.execute(_RATE_SQL, {"at": now, "since": now - RATE_MAX_AGE}).first()
    if rate is not None and rate[0] is not None:
        out["usd"] = round(usd(total_wh, float(rate[0])), 8)
        basis["usd"] = (f"marginal {float(rate[0]):.4f} $/kWh, energy_cost_accrued interval "
                        f"{_aware(rate[1]).isoformat()} ({rate[2]})")
    else:
        basis["usd"] = "unmeasured: no energy_cost_accrued rate in the last 7 days"

    from orion.physiology.heat_config import host_sensitivity

    heat_cfg, heat_err = _heat_config()
    heat: dict[str, float] = {}
    heat_notes: list[str] = []
    # The heat fit's regressor is COMPUTE watts (fit_heat_sensitivity.py), so its slope already
    # carries the fans and PSU loss that follow compute: priced on the GPU part, not the chassis part.
    for host, wh in wh_gpu_by_host.items():
        entry, sens, reason = host_sensitivity(heat_cfg, host, now)
        if sens is None:
            heat_notes.append(f"{host} unpriced ({reason or heat_err})")
            continue
        heat[host] = round(heat_c(wh, sens), 8)
        heat_notes.append(f"{host} {sens:.3f} C/100W fitted {entry.fitted_at.date().isoformat()}")
    if heat:
        out["heat_c_by_host"] = heat
    basis["heat"] = "sensitivity x GPU Wh / 100 (the fit is per compute watt): " + "; ".join(heat_notes)
    out["basis"] = basis
    return out


__all__ = ["act_energy_cost", "attribute_host_hour", "CPU_BASIS", "GPU_ONLY", "HEAT_SENSITIVITY_PATH", "WHOLE_MACHINE", "host_floors",
           "read_host_series", "rest_body_config"]
