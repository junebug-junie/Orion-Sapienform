"""Fill an act's energy, money and heat on close (ActCostV2 ``wh_by_host``, ``usd``,
``heat_c_by_host``, ``basis``). Called by orion/grammar/act_ledger.py ``fill_close_cost``.

Reads, never writes: the act's GPU pool leases (gpu_pool_events, joined by the act's
correlation_id exactly as ``gpu_slot_s_by_lane`` is), the other leases on the same cards, the
cards' watts (orion_biometrics_summary ``gpu{index}_watts``), the house tariff's marginal rate
(energy_cost_accrued) and the heat fit (config/physiology/heat_sensitivity.yaml). The arithmetic
is orion/energy/act_energy.py.

Known limits, said here rather than hidden:
* GPU work that takes no pool lease still draws on the card; it lands on whoever holds a lease then.
* A durable hold and the child call it wraps are two leases on one card: each pays half.
* Acts sharing one correlation_id would each be charged the turn's whole lease energy (same exposure
  as ``gpu_slot_s_by_lane``; live 2026-10-29: 1103 acts, 1103 distinct correlation ids).
* A lease's cards are taken from one terminal row.
* Per-act Wh from ~30 s samples scatters widely; the p05 idle floor reads ~20% above the 1 Hz meter
  on reverie images (eval: services/orion-sql-writer/evals/act_energy_vs_power_intent.py).

What is not charged: CPU. Per-container CPU watts are measured
(``container_cpu_watts.<name>`` in orion_biometrics_summary.measurements), but an act does not
name the containers it ran in, so CPU is said in ``basis["cpu"]`` instead of charged at 0.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from orion.energy.act_energy import card_energy, heat_c, idle_floor, usd

logger = logging.getLogger("orion.grammar.act_energy_fill")

_REPO_ROOT = Path(__file__).resolve().parents[2]
HEAT_SENSITIVITY_PATH = _REPO_ROOT / "config" / "physiology" / "heat_sensitivity.yaml"
IDLE_WINDOW = timedelta(hours=24)
SAMPLE_PAD = timedelta(seconds=120)
RATE_MAX_AGE = timedelta(days=7)
# Leases held at once on one card: a durable hold is capped at 9000 s (gpu_pool.yaml max_hold_sec).
OTHER_LEASE_LOOKBACK = timedelta(hours=3)
MAX_HOLD = timedelta(seconds=9000)
CPU_BASIS = "not charged per act: CPU watts are measured per container (container_cpu_watts.*), not per act"

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
    SELECT lease_id, start_at, end_at FROM (
        SELECT DISTINCT ON (lease_id) lease_id,
               generated_at - held_ms * interval '1 millisecond' AS start_at, generated_at AS end_at
          FROM gpu_pool_events
         WHERE event IN {_TERMINAL} AND held_ms IS NOT NULL
           AND cards::jsonb ? :card
           AND generated_at >= :ws AND generated_at <= :we_plus
         ORDER BY lease_id, held_ms DESC
    ) done
    UNION ALL
    SELECT lease_id, start_at, end_at FROM (
        SELECT DISTINCT ON (g.lease_id) g.lease_id, g.generated_at AS start_at, NULL::timestamptz AS end_at
          FROM gpu_pool_events g
         WHERE g.event = 'granted' AND g.cards::jsonb ? :card
           AND g.generated_at >= :ws_minus AND g.generated_at <= :we
           AND NOT EXISTS (SELECT 1 FROM gpu_pool_events t
                            WHERE t.lease_id = g.lease_id AND t.event IN {_TERMINAL})
         ORDER BY g.lease_id, g.generated_at
    ) held
    """
)

# orion_biometrics_summary.timestamp is a VARCHAR ("2026-10-29 16:41:23.287434+00") indexed with
# node: filtered as a string on that prefix, parsed as timestamptz (same as cabinet_heat.py).
_CARD_WATTS_SQL = text(
    "SELECT timestamp::timestamptz AS ts, (measurements->>:key)::float AS w FROM orion_biometrics_summary "
    "WHERE node = :node AND timestamp >= :since AND timestamp <= :until AND measurements ? :key "
    "ORDER BY timestamp"
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
_IDLE_CACHE: dict[tuple[str, str], tuple[float, Optional[float]]] = {}
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


def _heat_config() -> tuple[Any, Optional[str]]:
    now = time.monotonic()
    if _HEAT_CACHE and now - _HEAT_CACHE["at"] < _CONFIG_TTL_S:
        return _HEAT_CACHE["cfg"], _HEAT_CACHE["err"]
    from orion.physiology.heat_config import load_heat_sensitivity

    cfg, err = load_heat_sensitivity(HEAT_SENSITIVITY_PATH)
    _HEAT_CACHE.update(at=now, cfg=cfg, err=err)
    return cfg, err


def _card_idle(conn: Any, host: str, key: str, before: datetime) -> Optional[float]:
    now = time.monotonic()
    hit = _IDLE_CACHE.get((host, key))
    if hit is not None and now - hit[0] < _IDLE_TTL_S:
        return hit[1]
    rows = conn.execute(_CARD_WATTS_SQL, {"key": key, "node": host, "since": _ts_key(before - IDLE_WINDOW),
                                          "until": _ts_key(before) + "~"}).fetchall()
    value = idle_floor([float(r[1]) for r in rows if r[1] is not None])
    if len(_IDLE_CACHE) > 64:
        _IDLE_CACHE.clear()
    _IDLE_CACHE[(host, key)] = (now, value)
    return value


def _parse_cards(raw: Any) -> list[str]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [str(c) for c in data] if isinstance(data, list) else []


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
    notes: list[str] = []
    failed: Optional[str] = loc_err
    # Group the act's lease windows by card, so each card is read once (one watts read and one
    # other-leases read over the union window), however many leases the turn took.
    by_card: dict[str, list[tuple[str, datetime, datetime]]] = {}
    for lease_id, cards_raw, held_ms, end_at in leases:
        end = _aware(end_at)
        start = end - timedelta(milliseconds=float(held_ms))
        if (end - start).total_seconds() <= 0:
            continue
        for card in _parse_cards(cards_raw):
            by_card.setdefault(card, []).append((str(lease_id), start, end))
    for card, mine in by_card.items():
        if failed:
            break
        loc = locations.get(card)
        if loc is None:
            failed = f"card_not_in_pool_config:{card}"
            break
        host, index = loc
        key = f"gpu{index}_watts"
        ws, we = min(m[1] for m in mine), max(m[2] for m in mine)
        idle = _card_idle(conn, host, key, ws)
        if idle is None:
            failed = f"no_idle_floor:{host}/{key}"
            break
        rows = conn.execute(_CARD_WATTS_SQL, {"key": key, "node": host, "since": _ts_key(ws - SAMPLE_PAD),
                                              "until": _ts_key(we + SAMPLE_PAD) + "~"}).fetchall()
        points = [(_aware(r[0]), float(r[1])) for r in rows if r[1] is not None]
        card_leases: dict[str, tuple[datetime, datetime]] = {}
        for oid, o_start, o_end in conn.execute(_CARD_LEASES_SQL, {
                "card": card, "ws": ws, "we": we, "we_plus": we + OTHER_LEASE_LOOKBACK,
                "ws_minus": ws - OTHER_LEASE_LOOKBACK}).fetchall():
            if o_start is None:
                continue
            o_s = _aware(o_start)
            # A grant with no terminal event yet: held until now, but never past the pool's longest
            # hold (live: ~8 grants a day never get a terminal event).
            o_e = _aware(o_end) if o_end is not None else min(now, o_s + MAX_HOLD)
            card_leases.setdefault(str(oid), (o_s, o_e))
        for lease_id, start, end in mine:
            # The act's OTHER leases on this card count as sharers too: two concurrent slots
            # of one turn each take their own share, and the shares still add up.
            others = [(o_s, o_e) for oid, (o_s, o_e) in card_leases.items()
                      if oid != lease_id and o_s < end and o_e > start]
            got = card_energy(points, idle_w=idle, start=start, end=end, others=others)
            if got is None:
                failed = f"no_watts_reading:{host}/{key}"
                break
            wh_by_host[host] = wh_by_host.get(host, 0.0) + got.wh
            quality = "" if got.samples_in_window else " (interpolated: no sample inside the lease)"
            notes.append(f"{host}/{card}: {got.wh:.4f} Wh, {got.samples_in_window} samples in "
                         f"{(end - start).total_seconds():.0f} s{quality}, share {got.mean_share:.2f}, "
                         f"idle {idle:.1f} W")
    if failed:
        basis["wh"] = f"unmeasured: {failed}"
        return {"basis": basis}
    if not wh_by_host:
        return {}
    basis["wh"] = ("GPU watts above each card's idle floor (p05 of its trailing 24 h), split by "
                   "lease share. " + "; ".join(notes))
    out: dict[str, Any] = {"wh_by_host": {h: round(v, 6) for h, v in wh_by_host.items()}}
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
    for host, wh in wh_by_host.items():
        entry, sens, reason = host_sensitivity(heat_cfg, host, now)
        if sens is None:
            heat_notes.append(f"{host} unpriced ({reason or heat_err})")
            continue
        heat[host] = round(heat_c(wh, sens), 8)
        heat_notes.append(f"{host} {sens:.3f} C/100W fitted {entry.fitted_at.date().isoformat()}")
    if heat:
        out["heat_c_by_host"] = heat
    basis["heat"] = "sensitivity x Wh / 100: " + "; ".join(heat_notes)
    out["basis"] = basis
    return out


__all__ = ["act_energy_cost", "CPU_BASIS", "HEAT_SENSITIVITY_PATH"]
