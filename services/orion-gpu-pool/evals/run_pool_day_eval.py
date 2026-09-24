#!/usr/bin/env python3
"""GPU pool policy eval: replay two simulated hours of mixed traffic through the real scheduler
and the real lease transition table, with the actuator simulated (swaps take 60s).

Scenario (deterministic, seeded):
  - chat, agent, metacog (system + background), fast, world and diffusion traffic
  - 2% of calls fail upstream (retry path), metacog worker outage 30:00-40:00
  - gpu0 lent 50:00-70:00 while chat keeps arriving (recall path)

Hard targets (exit 1 if missed; spec acceptance check 4):
  - owner starvation beyond grace: 0s  (an owner waiting while a borrower holds its role
    longer than clawback_grace_sec after the owner arrived)
  - leases lost: 0  (anything neither finished nor explicitly unavailable/dead-lettered at end)
  - small-role violations: 0  (a chat/agent lease granted on metacog/fast)

Also measured: lease-graph checkpoint throughput through the real PoolRuntime + MemorySaver.
That number is an in-memory ceiling; Postgres checkpoint throughput is UNVERIFIED until live.

Run: python services/orion-gpu-pool/evals/run_pool_day_eval.py
"""
from __future__ import annotations

import asyncio
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE.parents[1]), str(SERVICE)]

from orion.gpu_pool.config import load_pool_config  # noqa: E402
from orion.gpu_pool.lease_graph import initial_state, transition  # noqa: E402
from orion.gpu_pool.scheduler import (  # noqa: E402
    Abort, Backlog, CardLive, DeadLetter, Expire, Grant, LeaseView, Recall, Requeue, RoleLive,
    SwapLoad, SwapUnload, Unavailable, schedule,
)

CFG = load_pool_config()
T0 = datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc)
DURATION = 2 * 3600
SWAP_SEC = 60
_EV = {Grant: "grant", Recall: "recall", Abort: "abort", Expire: "expire", Unavailable: "unavailable",
       Backlog: "backlog", Requeue: "requeue", DeadLetter: "dead_letter"}

# class, priority, mean inter-arrival s, (min, max) work seconds, kind, deadline s
TRAFFIC = [
    ("chat", "interactive", 45, (15, 60), "request", 120),
    ("agent", "system", 120, (60, 300), "hold", None),
    ("metacog", "system", 6, (3, 10), "request", None),
    ("metacog", "background", 10, (3, 10), "request", None),
    ("fast", "system", 4, (1, 4), "request", 60),
    ("world", "system", 3, (1, 1), "request", None),
    ("diffusion", "system", 900, (40, 90), "request", None),
]
LIVE = {
    "chat": RoleLive("chat", True, 1, 65536), "agent": RoleLive("agent", True, 1, 131072),
    "agent-gpu2": RoleLive("agent-gpu2", True, 1, 131072), "metacog": RoleLive("metacog", True, 4, 4096),
    "fast": RoleLive("fast", True, 4, 4096), "world": RoleLive("world", True, 2),
    "diffusion": RoleLive("diffusion", True, 1), "experiment": RoleLive("experiment", True, 1, 8192),
}


def simulate(seed: int = 7) -> dict:
    rng = random.Random(seed)
    leases: dict[str, dict] = {}
    work_left: dict[str, float] = {}
    cards = {c: CardLive(c) for c in CFG.cards}
    pending_swaps: list[tuple[int, object]] = []
    stats = defaultdict(list)
    counts = defaultdict(int)
    starvation = 0.0
    violations = 0
    next_arrival = {i: rng.expovariate(1 / t[2]) for i, t in enumerate(TRAFFIC)}
    n = 0

    for sec in range(DURATION):
        now = T0 + timedelta(seconds=sec)
        cards["gpu0"].lent = 3000 <= sec < 4200
        live = dict(LIVE)
        if 1800 <= sec < 2400:
            live["metacog"] = RoleLive("metacog", False, 4, 4096)

        for i, (cls, prio, gap, work, kind, dl) in enumerate(TRAFFIC):
            while next_arrival[i] <= sec:
                n += 1
                lid = f"L{n}"
                # Durable/background work comes back for re-grants; interactive callers do not.
                retryable = cls in ("agent", "world", "diffusion") or prio == "background"
                req = {"work_class": cls, "kind": kind, "priority": prio, "request_id": lid,
                       "retryable": retryable}
                st = dict(initial_state(lid, req, now))
                st["deadline_at"] = (now + timedelta(seconds=dl)).isoformat() if dl else None
                leases[lid] = st
                work_left[lid] = rng.uniform(*work)
                next_arrival[i] += rng.expovariate(1 / gap)

        for due, swap in [p for p in pending_swaps if p[0] <= sec]:
            pending_swaps.remove((due, swap))
            for c in CFG.roles[swap.role].cards:
                card = cards[c]
                card.swap_state = "idle"
                if isinstance(swap, SwapLoad):
                    card.swapped_in.add(swap.role)
                else:
                    card.swapped_in.discard(swap.role)
                    card.cooldown_until = now + timedelta(seconds=CFG.defaults.swap_cooldown_sec)

        # granted work progresses; heartbeats keep leases alive; 2% fail upstream
        for lid, st in leases.items():
            if st["status"] in ("granted", "recalling"):
                if st["request"]["work_class"] in ("chat", "agent") and st["role"] in ("metacog", "fast"):
                    violations += 1
                st.update(transition(st, {"type": "heartbeat", "at": now.isoformat()}, CFG), history=[])
                work_left[lid] -= 1
                if st["role"] and CFG.roles[st["role"]].swap:
                    for c in CFG.roles[st["role"]].cards:
                        cards[c].last_active_at = now
                if work_left[lid] <= 0:
                    fail = rng.random() < 0.02
                    ev = "release_failed" if fail else "release_ok"
                    upd = transition(st, {"type": ev, "at": now.isoformat(), "reason": "upstream"}, CFG)
                    st.update(upd, history=[])
                    counts["failed_calls" if fail else "completed"] += 1
                    if fail:
                        work_left[lid] = rng.uniform(1, 5)

        views = [LeaseView(
            lease_id=lid, work_class=st["request"]["work_class"], priority=st["request"]["priority"],
            status=st["status"], created_at=datetime.fromisoformat(st["created_at"]), role=st.get("role"),
            deadline_at=datetime.fromisoformat(st["deadline_at"]) if st.get("deadline_at") else None,
            recall_by=datetime.fromisoformat(st["recall_by"]) if st.get("recall_by") else None,
            not_before=datetime.fromisoformat(st["not_before"]) if st.get("not_before") else None,
            queued_since=datetime.fromisoformat(st["queued_since"]) if st.get("queued_since") else None,
            granted_at=datetime.fromisoformat(st["granted_at"]) if st.get("granted_at") else None,
            expires_at=datetime.fromisoformat(st["expires_at"]) if st.get("expires_at") else None,
            retryable=bool(st["request"].get("retryable")),
        ) for lid, st in leases.items() if st["status"] not in ("released", "unavailable", "dead_letter")]

        # owner starvation: owner queued past grace while a borrower holds that role
        for v in views:
            if v.status != "queued":
                continue
            waited = (now - (v.queued_since or v.created_at)).total_seconds()
            if waited <= CFG.defaults.clawback_grace_sec + 1:
                continue
            for role in CFG.classes[v.work_class].roles:
                if CFG.owns(v.work_class, role) and live[role].healthy and any(
                        o.role == role and o.status in ("granted", "recalling") and not CFG.owns(o.work_class, role)
                        for o in views):
                    starvation += 1
                    break

        for d in schedule(CFG, live, cards, views, now):
            if isinstance(d, (SwapLoad, SwapUnload)):
                for c in CFG.roles[d.role].cards:
                    cards[c].swap_state = "loading" if isinstance(d, SwapLoad) else "unloading"
                pending_swaps.append((sec + SWAP_SEC, d))
                counts["swap_load" if isinstance(d, SwapLoad) else "swap_unload"] += 1
                continue
            st = leases[d.lease_id]
            ev = {"type": _EV[type(d)], "at": now.isoformat(), "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                ev["role"] = d.role
                qs = datetime.fromisoformat(st["queued_since"])
                stats[(st["request"]["work_class"], st["request"]["priority"])].append((now - qs).total_seconds())
                counts[f"grant:{st['request']['work_class']}->{d.role}"] += 1
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
            counts[_EV[type(d)]] += 1
            st.update(transition(st, ev, CFG), history=[])

    lost = sum(1 for st in leases.values() if st["status"] in ("queued", "retry_wait", "granted", "recalling")
               and datetime.fromisoformat(st["created_at"]) < T0 + timedelta(seconds=DURATION - 900))
    waits = {f"{c}/{p}": {"n": len(w), "p50": round(statistics.median(w), 1),
                          "p95": round(sorted(w)[int(0.95 * (len(w) - 1))], 1)} for (c, p), w in sorted(stats.items())}
    return {"leases": n, "waits_sec": waits, "counts": dict(sorted(counts.items())),
            "owner_starvation_sec": starvation, "leases_lost": lost, "small_role_violations": violations}


async def checkpoint_throughput(n: int = 300) -> float:
    from langgraph.checkpoint.memory import MemorySaver

    from app.runtime import PoolRuntime
    from app.store import MemoryStore
    from orion.gpu_pool.discovery import Probe, load_profiles
    from orion.gpu_pool.lease_graph import build_lease_graph
    from orion.schemas.gpu_pool import GpuLeaseRequestV1

    async def prober(role, url, kind, health):
        return Probe(kind == "service")

    rt = PoolRuntime(cfg=CFG, profiles=load_profiles(), store=MemoryStore(),
                     graph=build_lease_graph(lambda: CFG, MemorySaver()), prober=prober, probe_interval_sec=1e9)
    await rt.start()
    started = time.perf_counter()
    for i in range(n):
        r = await rt.acquire(GpuLeaseRequestV1(verb="acquire", work_class="world", holder="eval", request_id=f"r{i}"))
        await rt.release(r.lease_id, "ok")
    return n / (time.perf_counter() - started)


def main() -> int:
    report = simulate()
    report["leases_per_sec_inmemory"] = round(asyncio.run(checkpoint_throughput()), 1)
    import json

    print(json.dumps(report, indent=2))
    failures = [k for k in ("owner_starvation_sec", "leases_lost", "small_role_violations") if report[k]]
    print("VERDICT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
