#!/usr/bin/env python3
"""Replay, read-only: the last N days of Juniper's messages through PR A's door rules AND PR B's one
attention budget (every door on one daily cap of 6, the pacing price on attention:juniper).

    PYTHONPATH=.:services/orion-hub .venv/bin/python -P \
        services/orion-hub/evals/run_outreach_attention_budget_replay.py [--days 14] [--dsn ...] [--json]

Input: every message PR A would still have sent (``run_outreach_door_policy_replay.replay``, verdict
``kept``), at the moment it was really sent: that is when the door wanted to speak. Each one is offered
to the new market in time order (``simulate``):

* value = family weight x the drive at that moment: 1 - exp(-t / tau), t = time since the last message
  that went out in the replay (Appetite's hunger, ``orion/physiology/appetite.py``; rank ~ intensity).
  The weight (0.648) and tau (4,000 s) are the live hub.outreach values (want_ledger, 7 days to 11-10).
  So a long silence makes Orion want to speak more, and a message right after another wants less.
* price = ``attention_pacing.message_price(sent / allowance(now))`` per message, the allowance growing
  to ``pace_budget`` (the cap less one).
* goes when value > price and the cap has room; curiosity Door-A does not bid (its run already did),
  so it only needs cap room. The strain report is exempt.
* held: re-offered at its relief time (``relief_sec``; 10 min when none), and composed anyway at the
  end of its window: office visits 5 min (``OFFICE_WINDOW_S``), everything else the end of Juniper's
  waking day. A held message still waiting when quiet hours begin is ``expired`` (the door re-offers
  it the next day as a new want; the replay does not carry it over).
* quiet hours (23:00-08:00 local) hold every door but curiosity Door-A, as the Hub gate does.

What this cannot replay: Orion's own words (PASS, the Why Juniper gate), the GPU lanes' small charge,
and Prime's regulate charge. SELECT only. Exit 0 on a completed replay; 2 when Postgres is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

_HUB = Path(__file__).resolve().parents[1]
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

from orion.physiology import attention_pacing as ap  # noqa: E402

TZ = "America/Denver"
WAKE_START, WAKE_END = 8, 23           # Hub quiet hours 23 -> 8
FAMILY_WEIGHT = 0.648                  # hub.outreach family weight, 2026-11-10
TAU_SEC = 4000.0                       # hub.outreach hunger tau (want_ledger 'tau' median, 7 days)
DEFAULT_RETRY_SEC = 600.0              # orion.physiology.binding.REBID_DEFAULT_SEC
OFFICE_WINDOW_SEC = 300.0              # scripts.outreach_bind.OFFICE_WINDOW_S
NO_BID = {"curiosity_outreach"}        # composed inside its run, which already bid (#3010)
EXEMPT = set(ap.CAP_EXEMPT_SOURCES)


@dataclass
class Offer:
    at: datetime                      # when the door first wanted to speak
    source: str
    text: str = ""
    tries: int = 0
    first_at: Optional[datetime] = None
    trail: list = field(default_factory=list)


def _local(t: datetime) -> datetime:
    return t.astimezone(ZoneInfo(TZ))


def _quiet(t: datetime) -> bool:
    h = _local(t).hour
    return h >= WAKE_END or h < WAKE_START


def _next_morning(t: datetime) -> datetime:
    loc = _local(t)
    day = loc if loc.hour < WAKE_START else loc + timedelta(days=1)
    return day.replace(hour=WAKE_START, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def drive_value(since_last_sec: Optional[float], *, weight: float = FAMILY_WEIGHT, tau: float = TAU_SEC) -> float:
    """weight x (1 - exp(-t / tau)); no message yet today-or-ever -> a full drive. Pure."""
    if since_last_sec is None:
        return weight
    return weight * (1.0 - math.exp(-max(0.0, since_last_sec) / tau))


def simulate(offers: Iterable[Offer], *, cap: int = ap.JUNIPER_DAILY_CAP) -> list[dict[str, Any]]:
    """Every offer's outcome: sent (with when, value and price) / expired / cap. Pure."""
    queue = sorted(offers, key=lambda o: o.at)
    sent_by_day: Counter = Counter()
    last_sent: Optional[datetime] = None
    out: list[dict[str, Any]] = []
    pending: list[tuple[datetime, int, Offer]] = [(o.at, i, o) for i, o in enumerate(queue)]
    seq = len(pending)
    while pending:
        pending.sort(key=lambda x: (x[0], x[1]))
        t, _, o = pending.pop(0)
        o.first_at = o.first_at or o.at
        if o.source not in NO_BID and _quiet(t):
            if o.tries:      # a held want still waiting at quiet hours: it lapses (a new want tomorrow)
                out.append({"source": o.source, "first_at": o.first_at, "outcome": "expired", "tries": o.tries})
                continue
            pending.append((_next_morning(t), seq, o)); seq += 1   # the quiet-hours gate delays it
            continue
        day = _local(t).date()
        counted = o.source not in EXEMPT
        if counted and sent_by_day[day] >= cap:
            out.append({"source": o.source, "first_at": o.first_at, "outcome": "cap", "at": t})
            continue
        frac = ap.waking_fraction(t, tz=TZ, start_hour=WAKE_START, end_hour=WAKE_END)
        budget = ap.pace_budget(cap)
        price = ap.message_price(ap.pace(sent_by_day[day], frac, budget))
        value = drive_value(None if last_sent is None else (t - last_sent).total_seconds())
        window_end = (o.first_at + timedelta(seconds=OFFICE_WINDOW_SEC) if o.source == "office_reach_out"
                      else ap.waking_end(o.first_at, tz=TZ, start_hour=WAKE_START, end_hour=WAKE_END))
        go = o.source in NO_BID or o.source in EXEMPT or value > price or (o.tries and t >= window_end)
        if go:
            if counted:
                sent_by_day[day] += 1
            last_sent = t
            out.append({"source": o.source, "first_at": o.first_at, "outcome": "sent", "at": t, "value": round(value, 3),
                        "price": round(price, 3), "held_min": round((t - o.first_at).total_seconds() / 60.0, 1),
                        "why": "window_end" if (o.tries and t >= window_end and value <= price) else "bid",
                        "text": o.text})
            continue
        o.tries += 1
        rate = (budget - 1) / ap.waking_span_hours(t, tz=TZ, start_hour=WAKE_START, end_hour=WAKE_END) if 0 < frac < 1 else 0.0
        relief = ap.relief_sec(sent_by_day[day], value, allowance_now=ap.allowance(frac, budget), rate_per_hour=rate)
        wait = DEFAULT_RETRY_SEC if not relief else min(3600.0, max(120.0, relief))
        nxt = min(t + timedelta(seconds=wait), window_end) if window_end > t else t + timedelta(seconds=wait)
        o.trail.append((t, round(value, 3), round(price, 3)))
        pending.append((nxt, seq, o)); seq += 1
    return out


def summarize(results: list[dict[str, Any]], days: int) -> dict[str, Any]:
    sent = [r for r in results if r["outcome"] == "sent"]
    per_day: dict[str, Counter] = defaultdict(Counter)
    for r in sent:
        per_day[_local(r["at"]).date().isoformat()][r["source"]] += 1
    hours = Counter(_local(r["at"]).hour for r in sent)
    held = [r for r in sent if r["held_min"] > 0]
    return {
        "days": days,
        "sent_per_day": round(len(sent) / days, 2),
        "by_door_per_day": {k: round(v / days, 2) for k, v in Counter(r["source"] for r in sent).items()},
        "per_day": {d: dict(c) | {"total": sum(c.values())} for d, c in sorted(per_day.items())},
        "outcomes": dict(Counter(f"{r['source']}:{r['outcome']}" for r in results)),
        "held_then_sent": len(held),
        "held_minutes_median": (sorted(r["held_min"] for r in held)[len(held) // 2] if held else 0.0),
        "window_end_sends": sum(1 for r in sent if r["why"] == "window_end"),
        "send_hours_local": dict(sorted(hours.items())),
    }


def _load_door_replay():
    path = Path(__file__).with_name("run_outreach_door_policy_replay.py")
    spec = importlib.util.spec_from_file_location("run_outreach_door_policy_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


async def _main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--days", type=int, default=14)
    ap_.add_argument("--dsn", default=os.environ.get(
        "RECALL_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"))
    ap_.add_argument("--json", action="store_true")
    a = ap_.parse_args()
    try:
        import asyncpg

        conn = await asyncpg.connect(a.dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"postgres unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    door = _load_door_replay()
    now = datetime.now(timezone.utc)
    # Whole local days only: start at local midnight a.days ago, end at the last local midnight.
    end = _local(now).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=a.days)
    try:
        await conn.execute("SET default_transaction_read_only = on")
        rows = await door.replay(conn, days=a.days + 2, now=now)
    finally:
        await conn.close()
    kept = [r for r in rows if r["verdict"] == "kept" and start <= _local(r["decided_at"]) < end]
    before = Counter(r["source"] for r in rows if start <= _local(r["decided_at"]) < end)
    offers = [Offer(at=r["decided_at"], source=r["source"], text=r.get("text") or "") for r in kept]
    results = simulate(offers)
    out = {"window": [start.isoformat(), end.isoformat()],
           "before_pr_a_per_day": {k: round(v / a.days, 2) for k, v in before.items()},
           "after_pr_a_per_day": round(len(kept) / a.days, 2),
           "after_pr_b": summarize(results, a.days)}
    late = [r for r in results if r["outcome"] == "sent" and _local(r["at"]).hour >= 19 and r["why"] == "bid"]
    waited = [r for r in results if r["outcome"] == "sent" and r["held_min"] >= 30]
    out["examples"] = {
        "late_strong_sends": [{"at": _local(r["at"]).strftime("%m-%d %H:%M"), "door": r["source"], "value": r["value"],
                               "price": r["price"]} for r in late[:5]],
        "weak_waits": [{"first": _local(r["first_at"]).strftime("%m-%d %H:%M"), "sent": _local(r["at"]).strftime("%H:%M"),
                        "door": r["source"], "held_min": r["held_min"], "value_at_send": r["value"],
                        "price_at_send": r["price"]} for r in waited[:5]],
    }
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    b = out["after_pr_b"]
    print(f"window {out['window'][0][:10]} .. {out['window'][1][:10]} ({a.days} days)")
    print(f"before PR A: {json.dumps(out['before_pr_a_per_day'])}")
    print(f"after PR A: {out['after_pr_a_per_day']}/day")
    print(f"after PR B: {b['sent_per_day']}/day  by door {json.dumps(b['by_door_per_day'])}")
    print(f"outcomes: {json.dumps(b['outcomes'])}")
    print(f"held then sent: {b['held_then_sent']} (median {b['held_minutes_median']} min); window-end sends: {b['window_end_sends']}")
    print(f"send hours (local): {json.dumps(b['send_hours_local'])}")
    for d, c in b["per_day"].items():
        print(f"  {d}: {json.dumps(c)}")
    print("late-day strong sends:", json.dumps(out["examples"]["late_strong_sends"]))
    print("weak ones that waited:", json.dumps(out["examples"]["weak_waits"]))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
