#!/usr/bin/env python3
"""GPU pool policy eval: replay two simulated hours of mixed traffic through the real scheduler
and the real lease transition table, with the actuator simulated (swaps take 60s).

Scenario (deterministic, seeded):
  - chat, agent (cortex-exec system calls), metacog (system + background), fast, world and diffusion
  - 2% of calls fail upstream (retry path), metacog worker outage 30:00-40:00
  - gpu0 lent 50:00-70:00 while chat keeps arriving (recall path)
  - stage 4.3: four durable runs on pool HOLDS (two of 35 min, one of 7.8 h, one of 35 min
    arriving later), each alternating LLM calls made as children of its hold (attach) with tool
    gaps (~40% of the time, as measured live); the second run waits past agent-gpu2's
    after_wait_sec, the FIRST gpu2 load fails with the residents restored (cooldown), a later one
    succeeds; agent-gpu2's max_hold_sec then recalls its hold with hold_clawback_grace_sec and the
    run re-requests at its next node boundary.

Hard targets (exit 1 if missed; spec acceptance check 4 + stage 4 checks 2/3):
  - owner starvation beyond grace: 0s  (an owner waiting while a borrower holds its role
    longer than clawback_grace_sec after the owner arrived)
  - leases lost: 0  (waiting or granted-but-idle at the end, older than LOST_WINDOW_SEC -- a lease
    still running its work, e.g. a 7.8 h hold, is not lost)
  - small-role violations: 0  (a chat/agent lease granted on metacog/fast)
  - run blocked behind itself: 0s  (a run's call waiting while nothing but its own hold is on the
    slot -- the self-deadlock "attach, not lease" exists to prevent)
  - run call wait above one interleaved inference: 0  (gaps are shared, but a run's next call
    waits at most for the one higher-priority call that used its gap)
  - interleaved grants: > 0  (a system agent call used a run's tool gap)

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
    SwapBlocked, SwapLoad, SwapUnload, Unavailable, schedule,
)

CFG = load_pool_config()
T0 = datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc)
DURATION = 2 * 3600
SWAP_SEC = 60
_EV = {Grant: "grant", Recall: "recall", Abort: "abort", Expire: "expire", Unavailable: "unavailable",
       Backlog: "backlog", Requeue: "requeue", DeadLetter: "dead_letter"}

# class, priority, mean inter-arrival s, (min, max) work seconds, kind, deadline s
INTERLEAVE_MAX_SEC = 60   # the longest agent single call (cortex-exec system), below
TRAFFIC = [
    ("chat", "interactive", 45, (15, 60), "request", 120),
    ("agent", "system", 240, (20, INTERLEAVE_MAX_SEC), "request", None),
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


# Durable runs on holds: (start s, total run seconds). 35 min twice, 7.8 h, and a late 35 min.
RUNS = [(30, 35 * 60), (90, int(7.8 * 3600)), (150, 35 * 60), (2400, 35 * 60)]
CALL_SEC = (30, 90)     # one LLM call under the hold
GAP_SEC = (20, 60)      # a tool phase between calls (~40% of the run)
# The actuator: the first gpu2 load fails with the residents restored, every later one succeeds.
FAILED_LOADS = 1
LOST_WINDOW_SEC = 900 + max(CFG.swap_after_wait_sec(r) for r in CFG.roles if CFG.roles[r].swap)


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
    failed_loads_left = FAILED_LOADS
    runs = [{"id": f"run{i}", "start": start, "left": total, "hold": None, "child": None, "phase": "gap",
             "phase_left": 0.0, "attempt": 0, "done": False} for i, (start, total) in enumerate(RUNS)]
    child_waits: list[float] = []
    self_block = 0.0
    over_one_inference = 0
    interleaved = 0
    run_ends: dict[str, float] = {}

    def new_lease(req: dict, now: datetime) -> str:
        nonlocal n
        n += 1
        lid = f"L{n}"
        st = dict(initial_state(lid, {"request_id": lid, **req}, now))
        st["deadline_at"] = None
        leases[lid] = st
        return lid

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
            fail = isinstance(swap, SwapLoad) and failed_loads_left > 0
            if fail:
                failed_loads_left -= 1
                counts["swap_failed_restored"] += 1
            for c in CFG.roles[swap.role].cards:     # as app/runtime.py _finish() settles a card
                card = cards[c]
                card.swap_state = "idle"
                if fail:
                    card.cooldown_until = now + timedelta(seconds=CFG.defaults.swap_cooldown_sec)
                elif isinstance(swap, SwapLoad):
                    card.swapped_in.add(swap.role)
                    card.loaded_at = card.last_active_at = now
                else:
                    card.swapped_in.discard(swap.role)
                    card.loaded_at = None
                    card.residency_until = now + timedelta(seconds=CFG.defaults.swap_min_residency_sec)

        # durable runs: hold -> (call as a child of the hold | tool gap)* -> release
        for run in runs:
            if run["done"] or sec < run["start"]:
                continue
            hold = leases.get(run["hold"]) if run["hold"] else None
            if hold is None or hold["status"] in ("released", "unavailable", "dead_letter"):
                run["attempt"] += 1
                run["hold"] = new_lease({"work_class": "agent", "kind": "hold", "priority": "background",
                                         "retryable": True, "holder": f"durable-runs:{run['id']}"}, now)
                run["child"], run["phase"], run["phase_left"] = None, "gap", 0.0
                continue
            if hold["status"] not in ("granted", "recalling"):
                continue
            child = leases.get(run["child"]) if run["child"] else None
            if run["phase"] == "call" and child is not None:
                if child["status"] == "queued":
                    waited = (now - datetime.fromisoformat(child["queued_since"])).total_seconds()
                    others = sum(1 for o in leases.values() if o["status"] in ("granted", "recalling")
                                 and o.get("role") == hold["role"] and o["lease_id"] not in (hold["lease_id"],)
                                 and o["request"].get("hold_lease_id") != hold["lease_id"])
                    if others == 0 and waited >= 1:
                        self_block += 1          # nothing but its own run on the slot, and it waits
                    continue
                if child["status"] in ("granted", "recalling"):
                    continue                       # the call runs (work_left below)
                # the call finished (released): tool gap, or give the hold back if recalled
                run["child"] = None
                if hold["status"] == "recalling":
                    hold.update(transition(hold, {"type": "release_ok", "at": now.isoformat(),
                                                  "reason": "recalled_node_boundary"}, CFG), history=[])
                    counts["run_released_on_recall"] += 1
                    continue
                run["phase"], run["phase_left"] = "gap", rng.uniform(*GAP_SEC)
                continue
            # tool gap
            run["phase_left"] -= 1
            run["left"] -= 1
            if run["left"] <= 0:
                hold.update(transition(hold, {"type": "release_ok", "at": now.isoformat()}, CFG), history=[])
                run["done"] = True
                run_ends[run["id"]] = sec
                continue
            if hold["status"] == "recalling":
                hold.update(transition(hold, {"type": "release_ok", "at": now.isoformat(),
                                              "reason": "recalled_node_boundary"}, CFG), history=[])
                counts["run_released_on_recall"] += 1
                continue
            if run["phase_left"] <= 0:
                call = rng.uniform(*CALL_SEC)
                cid = new_lease({"work_class": "agent", "kind": "request", "priority": "background",
                                 "holder": "orion-llm-gateway", "hold_lease_id": hold["lease_id"]}, now)
                work_left[cid] = call
                run["left"] -= call
                run["child"], run["phase"] = cid, "call"

        # granted work progresses; heartbeats keep leases alive; 2% fail upstream
        for lid, st in leases.items():
            if st["status"] in ("granted", "recalling"):
                if st["request"]["work_class"] in ("chat", "agent") and st["role"] in ("metacog", "fast"):
                    violations += 1
                st.update(transition(st, {"type": "heartbeat", "at": now.isoformat()}, CFG), history=[])
                if lid not in work_left:
                    continue                      # a durable-run hold: its run drives it (above)
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
            kind=st["request"].get("kind", "request"), hold_lease_id=st["request"].get("hold_lease_id"),
        ) for lid, st in leases.items() if st["status"] not in ("released", "unavailable", "dead_letter")]

        # owner starvation: owner queued past the borrower's bound while a borrower holds that role.
        # A single-call borrower's bound is clawback_grace_sec. A durable-run hold's is ONE of its
        # calls: the owner uses the run's gaps at once (strictly higher priority), so it waits only
        # while a call is in flight -- never the hold's 600 s grace, which is for the run's node.
        for v in views:
            if v.status != "queued" or v.hold_lease_id:
                continue
            waited = (now - (v.queued_since or v.created_at)).total_seconds()
            for role in CFG.classes[v.work_class].roles:
                if not (CFG.owns(v.work_class, role) and live[role].healthy):
                    continue
                bounds = [(CALL_SEC[1] if (o.kind == "hold" or o.hold_lease_id) else CFG.defaults.clawback_grace_sec) + 1
                          for o in views if o.role == role and o.status in ("granted", "recalling")
                          and not CFG.owns(o.work_class, role)]
                if bounds and waited > max(bounds):
                    starvation += 1
                    break

        holds_on = {v.role: v for v in views if v.kind == "hold" and v.status in ("granted", "recalling")}
        for d in schedule(CFG, live, cards, views, now, guards={"thermal": None, "visual_baseline": None}):
            if isinstance(d, SwapBlocked):
                counts[f"swap_blocked:{d.reason}"] += 1
                continue
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
                if st["request"].get("hold_lease_id"):
                    wait = (now - qs).total_seconds()
                    child_waits.append(wait)
                    if wait > INTERLEAVE_MAX_SEC + 2:
                        over_one_inference += 1
                elif st["request"].get("kind") != "hold" and d.role in holds_on:
                    interleaved += 1               # a single call used a run's gap
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
                counts[f"recall:{d.reason}"] += 1
            counts[_EV[type(d)]] += 1
            st.update(transition(st, ev, CFG), history=[])

    def running(lid: str, st: dict) -> bool:
        if st["request"].get("kind") == "hold":
            return any(r["hold"] == lid and not r["done"] for r in runs)
        return work_left.get(lid, 0) > 0
    lost = sum(1 for lid, st in leases.items()
               if st["status"] in ("queued", "retry_wait", "granted", "recalling")
               and not (st["status"] in ("granted", "recalling") and running(lid, st))
               and datetime.fromisoformat(st["created_at"]) < T0 + timedelta(seconds=DURATION - LOST_WINDOW_SEC))
    waits = {f"{c}/{p}": {"n": len(w), "p50": round(statistics.median(w), 1),
                          "p95": round(sorted(w)[int(0.95 * (len(w) - 1))], 1)} for (c, p), w in sorted(stats.items())}
    return {"leases": n, "waits_sec": waits, "counts": dict(sorted(counts.items())),
            "runs": {r["id"]: {"attempts": r["attempt"], "finished_at_sec": run_ends.get(r["id"]),
                               "left_sec": round(max(0.0, r["left"]))} for r in runs},
            "run_call_wait_sec": {"n": len(child_waits),
                                  "p50": round(statistics.median(child_waits), 1) if child_waits else None,
                                  "max": round(max(child_waits), 1) if child_waits else None},
            "interleaved_grants": interleaved,
            "owner_starvation_sec": starvation, "leases_lost": lost, "small_role_violations": violations,
            "run_blocked_behind_itself_sec": self_block, "run_call_waits_over_one_inference": over_one_inference}


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
    failures = [k for k in ("owner_starvation_sec", "leases_lost", "small_role_violations",
                            "run_blocked_behind_itself_sec", "run_call_waits_over_one_inference") if report[k]]
    if not report["interleaved_grants"]:
        failures.append("interleaved_grants")
    if not report["counts"].get("swap_failed_restored"):
        failures.append("failed_load_not_exercised")
    print("VERDICT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
