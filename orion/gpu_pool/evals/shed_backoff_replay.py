"""Replay eval: how many leases reach the GPU pool per hour during a sustained heat shed, before
and after the shed backoff (orion/gpu_pool/shed_backoff.py), and how long work stays idle after
the heat clears.

Two layers, exactly as deployed: the caller's own ShedBackoff (vision-council, topic-foundry:
skips the gateway RPC) and the gateway's pool-client ShedBackoff keyed by lease priority (skips
the pool). "Before" = every arrival is one pool attempt (what gpu_pool_events showed live).

    python -m orion.gpu_pool.evals.shed_backoff_replay                 # synthetic arrivals
    python -m orion.gpu_pool.evals.shed_backoff_replay --from-db 14    # real 'admitted' times, last 14 h

``--from-db`` reads the live gpu_pool_events table (read-only) through
``docker exec orion-athena-sql-db psql``: every 'admitted' row of a holder that was shed is one
arrival in the old world.
"""
from __future__ import annotations

import argparse
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime

from orion.gpu_pool.shed_backoff import ShedBackoff

HOT = "shed:cabinet_hot"
# Callers that run their own caller-side ShedBackoff in front of the gateway (this patch).
CALLER_GATED = frozenset({"vision-council", "orion-topic-foundry"})
PRIORITY = {"vision-council": "background"}   # everything else seen shedding was system priority


@dataclass(frozen=True)
class Arrival:
    t: float
    caller: str


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def replay(arrivals: list[Arrival], *, clear_at: float | None = None, seed: int = 7) -> dict:
    """Pool attempts per caller with the backoff, the heat shed active until ``clear_at`` (None =
    the whole replay). Also the seconds from ``clear_at`` to each caller's first grant."""
    clock = _Clock()
    rng = random.Random(seed).random
    quiet = lambda _m: None  # noqa: E731
    gateway = ShedBackoff(name="gateway", clock=clock, rng=rng, log=quiet)
    callers: dict[str, ShedBackoff] = {}
    pool_attempts: dict[str, int] = {}
    first_grant: dict[str, float] = {}
    skipped_after_clear: dict[str, int] = {}
    for a in sorted(arrivals, key=lambda x: x.t):
        clock.t = a.t
        shedding = clear_at is None or a.t < clear_at
        cb = callers.setdefault(a.caller, ShedBackoff(name=a.caller, clock=clock, rng=rng, log=quiet)) \
            if a.caller in CALLER_GATED else None
        if cb is not None and cb.check("llm") is not None:
            if not shedding:
                skipped_after_clear[a.caller] = skipped_after_clear.get(a.caller, 0) + 1
            continue                                        # caller skipped: no gateway RPC
        prio = PRIORITY.get(a.caller, "system")
        if gateway.check(prio) is not None:                 # gateway refused locally: no pool lease
            if not shedding:
                skipped_after_clear[a.caller] = skipped_after_clear.get(a.caller, 0) + 1
            if cb is not None:
                cb.resolve("llm", HOT, hint_sec=gateway.remaining(prio))
            continue
        pool_attempts[a.caller] = pool_attempts.get(a.caller, 0) + 1
        if shedding:
            gateway.resolve(prio, HOT)
            if cb is not None:
                cb.resolve("llm", HOT, hint_sec=gateway.remaining(prio))
        else:
            gateway.resolve(prio, None)
            if cb is not None:
                cb.resolve("llm", None)
            first_grant.setdefault(a.caller, a.t - clear_at)
    # Backoff-caused delay: first grant minus the caller's first arrival after the clear (the time
    # it would have been served without any backoff).
    extra: dict[str, float] = {}
    if clear_at is not None:
        for caller, g in first_grant.items():
            first_arrival = min(a.t for a in arrivals if a.caller == caller and a.t >= clear_at) - clear_at
            extra[caller] = g - first_arrival
    return {"pool_attempts": pool_attempts, "first_grant_after_clear_sec": first_grant,
            "backoff_delay_after_clear_sec": extra, "skipped_after_clear": skipped_after_clear}


def synthetic(hours: float) -> list[Arrival]:
    """Live 2026-10-20 cadences: vision-council ~1 window / 6 s (2 streams); topic-foundry a batch
    burst of ~400 calls in 60 s once an hour; cortex-exec ~25/h."""
    out: list[Arrival] = []
    end = hours * 3600
    t = 0.0
    while t < end:
        out.append(Arrival(t, "vision-council"))
        t += 6.0
    for h in range(int(hours)):
        out += [Arrival(h * 3600 + 600 + i * 0.15, "orion-topic-foundry") for i in range(400)]
    t = 0.0
    while t < end:
        out.append(Arrival(t + 7.0, "cortex-exec"))
        t += 144.0
    return out


def from_db(hours: int) -> list[Arrival]:
    sql = ("select extract(epoch from created_at), split_part(holder,':',1) from gpu_pool_events "
           "where event='admitted' and holder in (select distinct holder from gpu_pool_events "
           f"where event='unavailable' and reason like 'shed:%' and created_at > now()-interval '{int(hours)} hours') "
           f"and created_at > now()-interval '{int(hours)} hours' order by 1")
    rows = subprocess.run(["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
                           "-At", "-F", "|", "-c", sql], check=True, capture_output=True, text=True).stdout
    pts = [ln.split("|") for ln in rows.splitlines() if "|" in ln]
    if not pts:
        return []
    t0 = float(pts[0][0])
    return [Arrival(float(ts) - t0, holder) for ts, holder in pts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-db", type=int, default=0, metavar="HOURS")
    ap.add_argument("--hours", type=float, default=14.0)
    args = ap.parse_args()
    arrivals = from_db(args.from_db) if args.from_db else synthetic(args.hours)
    if not arrivals:
        print("no arrivals")
        return
    span_h = max(arrivals[-1].t / 3600.0, 1e-9)
    before: dict[str, int] = {}
    for a in arrivals:
        before[a.caller] = before.get(a.caller, 0) + 1
    after = replay(arrivals)["pool_attempts"]
    print(f"source={'db:' + str(args.from_db) + 'h' if args.from_db else 'synthetic'} span_h={span_h:.1f} "
          f"at={datetime.now().isoformat(timespec='seconds')}")
    print(f"{'caller':28} {'before/h':>9} {'after/h':>8}")
    for caller in sorted(before, key=lambda c: -before[c]):
        print(f"{caller:28} {before[caller] / span_h:9.1f} {after.get(caller, 0) / span_h:8.1f}")
    print(f"{'TOTAL':28} {sum(before.values()) / span_h:9.1f} {sum(after.values()) / span_h:8.1f}")
    worst: dict[str, float] = {}
    lost: dict[str, int] = {}
    for clear in range(2 * 3600, 3 * 3600, 37):     # many clear moments across the backoff cycle
        r = replay(synthetic(4.0), clear_at=clear)
        for k, v in r["backoff_delay_after_clear_sec"].items():
            worst[k] = max(worst.get(k, 0.0), v)
        for k, v in r["skipped_after_clear"].items():
            lost[k] = max(lost.get(k, 0), v)
    print("worst delay after heat clears caused by backoff (synthetic, s; bursty callers wait for their next "
          "burst): " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(worst.items())))
    print("worst calls skipped after heat cleared (synthetic): "
          + ", ".join(f"{k}={v}" for k, v in sorted(lost.items())))

if __name__ == "__main__":
    main()
