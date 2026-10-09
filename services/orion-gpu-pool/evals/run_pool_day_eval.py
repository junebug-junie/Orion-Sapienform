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
  - world/diffusion grant overlap: 0s  (a world lease and a diffusion lease GRANTED together on gpu2;
    a scheduler property, not proof of physical non-overlap -- that needs callers to hold their lease
    for as long as their GPU work runs, which the world-model/thought tests pin --
    the mutex the durable-runs /capacity permit gave, kept by serialize_with since stage 5.4), and
    serialize_with actually exercised (some serialized:<role> report)

Urgent scenario (separate short replay, same real scheduler + lease table): background holds on
agent and the loaded agent-gpu2 seat, chat traffic on chat, then an urgent hold. Hard targets:
  - the urgent hold is granted within urgent_preempt_grace_sec + one tick
  - exactly one hold is paused, and it is the most recently granted background hold
  - the paused hold is re-granted before a background hold created after it, with no attempt spent
  - no chat/interactive lease is ever recalled
  - with urgent_max_concurrent: 0 (rollback) nothing is paused

Shed scenario (U4, docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md):
metacog system/background and chat traffic, a background durable-run hold already running, and an
urgent hold arriving while cooling_incident sheds background+system. Hard targets:
  - zero new grants to background/system leases (other than a running hold's own calls) while shed
  - shed recalls nothing (the only recall is U1 pausing the running hold for the urgent one), and
    the running hold keeps getting its calls granted until then
  - chat (interactive) and the urgent hold are granted while shed
  - after the shed clears, waiting background/system work is granted within one tick
  - D3 (docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md): a one-shot
    (non-retryable) background/system request arriving while shed is refused in the tick it
    arrives with Unavailable("shed:*") -- it never waits out its deadline

Concurrency scenario (stage 7.3, docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md):
agent-gpu2 loaded with 2 slots (Bonsai, 2 x 131072), gpu1's agent at 1 slot, nothing lent. Durable
runs arrive faster than one run per card can drain them (six at once, then one every ~10 min), with
the live hold shape (calls 120-240 s of single-slot work, 30-90 s tool gaps: ~75% busy), plus one-off
system agent calls. Service time follows an OCCUPANCY SLOWDOWN MODEL measured live on agent-gpu2
(2026-10-09): Bonsai decodes 46 tok/s alone and 26 tok/s per slot with both slots busy, so a call's
work advances 1.0 s per second alone and 26/46 = 0.565 s per second while the other slot also runs
a call. The same replay runs at max_holds 1 (stage 7.2), max_holds 2 (7.3) and max_holds 2 with
reserve_one_off_slots 1. Hard targets (7.3 = max_holds 2):
  - two holds granted on agent-gpu2 at once for some time (acceptance check 4)
  - zero seconds where one gap-sharing call stalls two runs (two idle-run calls waiting, one
    one-off on the seat), and a scripted tick (check 9's eval case): two idle runs, one call in a gap,
    both runs' next calls plus a queued one-off arrive -> exactly one run's call is granted at once
  - a run's call never waits longer than one slowed one-off inference (+2 s)
  - queued-hold wait p90 and mean waiting holds below the max_holds 1 replay
  - with reserve_one_off_slots 1 the seat never carries two holds (the reserve is in force)
Reported, not gated: one-off system agent call waits per variant (acceptance check 6's question,
which decides the reserve default -- see the stage 7.3 PR report).

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
    Serialized, Shed, SwapBlocked, SwapLoad, SwapUnload, Unavailable, schedule,
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


def _views(leases: dict[str, dict]) -> list[LeaseView]:
    """Lease rows as the scheduler sees them (as app/runtime.py _view builds them)."""
    def ts(st: dict, key: str) -> datetime | None:
        return datetime.fromisoformat(st[key]) if st.get(key) else None
    return [LeaseView(
        lease_id=lid, work_class=st["request"]["work_class"], priority=st["request"]["priority"],
        status=st["status"], created_at=datetime.fromisoformat(st["created_at"]), role=st.get("role"),
        deadline_at=ts(st, "deadline_at"), recall_by=ts(st, "recall_by"), not_before=ts(st, "not_before"),
        queued_since=ts(st, "queued_since"), granted_at=ts(st, "granted_at"), expires_at=ts(st, "expires_at"),
        retryable=bool(st["request"].get("retryable")),
        kind=st["request"].get("kind", "request"), hold_lease_id=st["request"].get("hold_lease_id"),
        operator=bool(st["request"].get("operator")),
        reason=st.get("reason"),
    ) for lid, st in leases.items() if st["status"] not in ("released", "unavailable", "dead_letter")]


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
    gpu2_overlap = 0   # seconds a world lease and a diffusion lease/hold were granted together (stage 5.4)
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

        views = _views(leases)
        on = {v.role for v in views if v.status in ("granted", "recalling")}
        if "world" in on and "diffusion" in on:
            gpu2_overlap += 1

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
        for d in schedule(CFG, live, cards, views, now, guards={"thermal": None}):
            if isinstance(d, SwapBlocked):
                counts[f"swap_blocked:{d.reason}"] += 1
                continue
            if isinstance(d, Serialized):
                counts[d.reason] += 1   # serialized:<role>; a report, not a transition
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
            "world_diffusion_grant_overlap_sec": gpu2_overlap,
            "run_blocked_behind_itself_sec": self_block, "run_call_waits_over_one_inference": over_one_inference}


# Urgent scenario (docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md
# Part 1): name -> (arrives at s, background/urgent hold, released at s or after N s granted).
URGENT_SEC = 400
URGENT_HOLDS = {
    "bg_a": (0, "background"),    # granted on agent
    "bg_b": (2, "background"),    # granted on agent-gpu2 (seat loaded): the most recent grant
    "urgent": (20, "urgent"),     # no free slot -> pauses bg_b
    "bg_c": (25, "background"),   # a background hold created after bg_b
}
URGENT_RUN_SEC = 120              # the urgent hold gives its slot back this long after its grant
BG_A_ENDS_SEC = 300               # then bg_a finishes, so bg_c gets a slot and the order shows
CHAT_EVERY_SEC, CHAT_WORK_SEC = 30, 25   # interactive chat keeps arriving on the chat role


def urgent_scenario(cfg=CFG) -> dict:
    """Two background durable-run holds (agent + the loaded agent-gpu2 seat), then an urgent hold,
    through the real scheduler and lease transition table. A paused hold is mid-node: it never
    gives the slot back itself, so the pool's grace runs out and the abort re-queues it (U2)."""
    leases: dict[str, dict] = {}
    work_left: dict[str, float] = {}
    names: dict[str, str] = {}
    cards = {c: CardLive(c) for c in cfg.cards}
    cards["gpu2"].swapped_in.add("agent-gpu2")
    cards["gpu2"].loaded_at = cards["gpu2"].last_active_at = T0
    grants: dict[str, list[int]] = defaultdict(list)
    recalls: list[dict] = []
    granted_at_urgent_arrival: dict[str, str] = {}

    def add(lid: str, req: dict, now: datetime) -> None:
        deadline = req.pop("_deadline", None)
        st = dict(initial_state(lid, {"request_id": lid, **req}, now))
        st["deadline_at"] = deadline
        leases[lid] = st

    def apply(st: dict, ev: dict) -> None:
        st.update(transition(st, ev, cfg), history=[])

    for sec in range(URGENT_SEC):
        now = T0 + timedelta(seconds=sec)
        at = now.isoformat()
        for name, (start, prio) in URGENT_HOLDS.items():
            if sec == start:
                if prio == "urgent":
                    granted_at_urgent_arrival.update(
                        {names[l]: st["granted_at"] for l, st in leases.items()
                         if st["request"]["kind"] == "hold" and st["status"] == "granted"})
                lid = f"H-{name}"
                names[lid] = name
                add(lid, {"work_class": "agent", "kind": "hold", "priority": prio, "retryable": True,
                          "holder": f"durable-runs:{name}"}, now)
        if sec % CHAT_EVERY_SEC == 5:
            lid = f"C{sec}"
            names[lid] = lid
            add(lid, {"work_class": "chat", "kind": "request", "priority": "interactive",
                      "_deadline": (now + timedelta(seconds=120)).isoformat()}, now)
            work_left[lid] = CHAT_WORK_SEC

        for lid, st in leases.items():
            if st["status"] not in ("granted", "recalling"):
                continue
            ends = (lid == "H-urgent" and sec >= grants[lid][-1] + URGENT_RUN_SEC) \
                or (lid == "H-bg_a" and sec >= BG_A_ENDS_SEC)
            if lid in work_left:
                work_left[lid] -= 1
                ends = work_left[lid] <= 0
            if ends:
                apply(st, {"type": "release_ok", "at": at})
                continue
            apply(st, {"type": "heartbeat", "at": at})
            if st["role"] and cfg.roles[st["role"]].swap:
                for c in cfg.roles[st["role"]].cards:
                    cards[c].last_active_at = now

        for d in schedule(cfg, LIVE, cards, _views(leases), now,
                          guards={"thermal": None}):
            if isinstance(d, (SwapLoad, SwapUnload, SwapBlocked, Serialized)):
                continue                  # the seat is already loaded; no swap is part of this story
            st = leases[d.lease_id]
            ev = {"type": _EV[type(d)], "at": at, "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                ev["role"] = d.role
                grants[d.lease_id].append(sec)
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
                recalls.append({"lease": names[d.lease_id], "reason": d.reason, "role": st["role"],
                                "priority": st["request"]["priority"], "kind": st["request"]["kind"]})
            apply(st, ev)

    def first_grant_after(lid: str, sec: int) -> int | None:
        return next((g for g in grants[lid] if g > sec), None)

    urgent_start = URGENT_HOLDS["urgent"][0]
    paused = [r["lease"] for r in recalls if r["reason"] == "urgent_preempt"]
    expected = max(granted_at_urgent_arrival, key=granted_at_urgent_arrival.get) \
        if granted_at_urgent_arrival else None
    victim = f"H-{paused[0]}" if paused else None
    urgent_grant = first_grant_after("H-urgent", urgent_start - 1)
    victim_regrant = first_grant_after(victim, urgent_start) if victim else None
    later_grant = first_grant_after("H-bg_c", URGENT_HOLDS["bg_c"][0] - 1)
    return {
        "urgent_grant_wait_sec": None if urgent_grant is None else urgent_grant - urgent_start,
        "urgent_grant_wait_limit_sec": cfg.defaults.urgent_preempt_grace_sec + 1,
        "paused": paused,
        "expected_victim": expected,
        "victim_regranted_at_sec": victim_regrant,
        "later_background_granted_at_sec": later_grant,
        "victim_attempts_spent": (leases[victim]["attempt"] - 1) if victim else None,
        "chat_or_interactive_recalls": sum(1 for r in recalls if r["priority"] == "interactive"
                                           or r["role"] == "chat"),
        "recalls": recalls,
    }


def urgent_failures(u: dict, rollback: dict) -> list[str]:
    out = []
    if u["urgent_grant_wait_sec"] is None or u["urgent_grant_wait_sec"] > u["urgent_grant_wait_limit_sec"]:
        out.append("urgent_grant_wait")
    if u["paused"] != [u["expected_victim"]]:
        out.append("urgent_victim")      # exactly one pause, of the most recently granted background hold
    if u["victim_regranted_at_sec"] is None or u["later_background_granted_at_sec"] is None \
            or u["victim_regranted_at_sec"] >= u["later_background_granted_at_sec"]:
        out.append("urgent_victim_not_resumed_first")
    if u["victim_attempts_spent"]:
        out.append("urgent_victim_attempt_spent")
    if u["chat_or_interactive_recalls"]:
        out.append("urgent_recalled_chat")
    if rollback["paused"]:
        out.append("urgent_rollback_still_pauses")
    return out


SHED_SEC, SHED_FROM, SHED_UNTIL = 900, 200, 600
SHED = {"background": "cooling_incident", "system": "cooling_incident"}


def shed_scenario(cfg=CFG) -> dict:
    """U4 through the real scheduler + lease table: shed on at SHED_FROM, off at SHED_UNTIL."""
    leases: dict[str, dict] = {}
    work_left: dict[str, float] = {}
    rng = random.Random(11)
    grants: list[tuple[int, str, str, bool]] = []     # (sec, lease, priority, is_child)
    recalls: list[str] = []
    shed_reported: set[str] = set()
    one_shot_refused: dict[str, int] = {}              # lease -> wait in ticks before shed refusal
    one_shot_created: dict[str, int] = {}
    cards = {c: CardLive(c) for c in cfg.cards}

    def add(lid: str, req: dict, now: datetime) -> None:
        leases[lid] = dict(initial_state(lid, {"request_id": lid, **req}, now))

    def apply(st: dict, ev: dict) -> None:
        st.update(transition(st, ev, cfg), history=[])

    for sec in range(SHED_SEC):
        now = T0 + timedelta(seconds=sec)
        at = now.isoformat()
        if sec == 0:
            add("H-run", {"work_class": "agent", "kind": "hold", "priority": "background", "retryable": True,
                          "holder": "durable-runs:run"}, now)
        if sec == 350:
            add("H-urgent", {"work_class": "agent", "kind": "hold", "priority": "urgent", "retryable": True,
                             "holder": "durable-runs:urgent"}, now)
        run = leases.get("H-run")
        if run and run["status"] == "granted" and sec % 40 == 10:   # the running hold's next call
            lid = f"K{sec}"
            add(lid, {"work_class": "agent", "kind": "request", "priority": "background",
                      "hold_lease_id": "H-run"}, now)
            work_left[lid] = 20
        if sec % 30 == 5:
            add(f"C{sec}", {"work_class": "chat", "kind": "request", "priority": "interactive"}, now)
            work_left[f"C{sec}"] = 20
        if sec % 5 == 0:
            prio = "system" if rng.random() < 0.5 else "background"
            add(f"M{sec}", {"work_class": "metacog", "kind": "request", "priority": prio, "retryable": True}, now)
            work_left[f"M{sec}"] = rng.randint(3, 10)
        if sec % 15 == 7:   # D3: a one-shot gateway call (orion-mind / memory annotation shape)
            add(f"F{sec}", {"work_class": "fast", "kind": "request", "priority": "background"}, now)
            work_left[f"F{sec}"] = 3
            one_shot_created[f"F{sec}"] = sec

        for lid, st in leases.items():
            if st["status"] not in ("granted", "recalling"):
                continue
            if lid in work_left:
                work_left[lid] -= 1
                if work_left[lid] <= 0:
                    apply(st, {"type": "release_ok", "at": at})
                    continue
            elif lid == "H-urgent" and sec >= 500:
                apply(st, {"type": "release_ok", "at": at})
                continue
            apply(st, {"type": "heartbeat", "at": at})

        shed = SHED if SHED_FROM <= sec < SHED_UNTIL else None
        for d in schedule(cfg, LIVE, cards, _views(leases), now, guards={"thermal": None, "visual_baseline": None},
                          shed=shed):
            if isinstance(d, Shed):
                shed_reported.add(d.lease_id)
                continue
            if isinstance(d, (SwapLoad, SwapUnload, SwapBlocked, Serialized)):
                continue
            st = leases[d.lease_id]
            if isinstance(d, Unavailable) and d.lease_id in one_shot_created and d.reason.startswith("shed:"):
                one_shot_refused[d.lease_id] = sec - one_shot_created[d.lease_id]
            ev = {"type": _EV[type(d)], "at": at, "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                ev["role"] = d.role
                grants.append((sec, d.lease_id, st["request"]["priority"],
                               bool(st["request"].get("hold_lease_id"))))
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
                recalls.append((d.lease_id, d.reason))
            apply(st, ev)

    during = [g for g in grants if SHED_FROM <= g[0] < SHED_UNTIL]
    after = [g for g in grants if g[0] >= SHED_UNTIL and g[2] in ("background", "system") and not g[3]]
    return {
        "shed_window_sec": [SHED_FROM, SHED_UNTIL],
        "new_low_priority_grants_while_shed": sum(1 for g in during if g[2] in ("background", "system") and not g[3]),
        "running_hold_calls_granted_while_shed": sum(1 for g in during if g[3]),
        # U1 may pause it for the urgent hold (urgent_preempt); shed itself must never recall anything.
        "running_hold_paused_for_urgent": ("H-run", "urgent_preempt") in recalls,
        "recalls_not_for_urgent": [r for r in recalls if r[1] != "urgent_preempt"],
        "chat_grants_while_shed": sum(1 for g in during if g[2] == "interactive"),
        "urgent_granted_at_sec": next((g[0] for g in grants if g[1] == "H-urgent"), None),
        "leases_reported_shed": len(shed_reported),
        "first_low_priority_grant_after_clear_sec": (after[0][0] - SHED_UNTIL) if after else None,
        "one_shots_arrived_while_shed": sum(1 for c in one_shot_created.values() if SHED_FROM <= c < SHED_UNTIL),
        "one_shots_refused_while_shed": len(one_shot_refused),
        "one_shot_max_wait_before_refusal_sec": max(one_shot_refused.values(), default=None),
    }


def shed_failures(s: dict) -> list[str]:
    out = []
    if s["new_low_priority_grants_while_shed"]:
        out.append("shed_granted_low_priority")
    if not s["running_hold_calls_granted_while_shed"] or s["recalls_not_for_urgent"]:
        out.append("shed_disturbed_running_work")
    if not s["chat_grants_while_shed"]:
        out.append("shed_blocked_chat")
    if s["urgent_granted_at_sec"] is None or not SHED_FROM <= s["urgent_granted_at_sec"] < SHED_UNTIL:
        out.append("shed_blocked_urgent")
    if not s["leases_reported_shed"]:
        out.append("shed_not_reported")
    if s["first_low_priority_grant_after_clear_sec"] is None or s["first_low_priority_grant_after_clear_sec"] > 1:
        out.append("shed_not_cleared")
    if not s["one_shots_arrived_while_shed"] \
            or s["one_shots_refused_while_shed"] != s["one_shots_arrived_while_shed"] \
            or s["one_shot_max_wait_before_refusal_sec"] != 0:
        out.append("shed_one_shot_waited")
    return out


ENFORCE_SEC = 1800
PAUSE_FROM, PAUSE_UNTIL = 600, 1200


def enforce_scenario(cfg=CFG) -> dict:
    """Stage 5.7 through the real scheduler + lease table, 30 min with the 27B loaded on gpu2.

    - An operator lease on ``experiment`` (no launch block: nothing can load it) sits in the queue
      the whole time. Before 5.7 it drained every resident it evicts; it must drain nothing.
    - Actuation is paused from PAUSE_FROM to PAUSE_UNTIL while a diffusion hold (gpu2's owner) waits:
      the 27B must keep serving and nothing may be recalled for the swap that cannot happen. After
      the resume the owner reclaim drains the seat as usual."""
    leases: dict[str, dict] = {}
    work_left: dict[str, float] = {}
    recalls: list[tuple[int, str, str, str]] = []     # (sec, lease, reason, role)
    grants: list[tuple[int, str, str]] = []           # (sec, lease, role)
    swaps_while_paused: list[str] = []
    cards = {c: CardLive(c) for c in cfg.cards}
    cards["gpu2"] = CardLive("gpu2", swapped_in={"agent-gpu2"}, loaded_at=T0, last_active_at=T0)
    seat = "agent-gpu2"

    def add(lid: str, req: dict, now: datetime) -> None:
        leases[lid] = dict(initial_state(lid, {"request_id": lid, **req}, now))

    def apply(st: dict, ev: dict) -> None:
        st.update(transition(st, ev, cfg), history=[])

    for sec in range(ENFORCE_SEC):
        now = T0 + timedelta(seconds=sec)
        at = now.isoformat()
        if sec == 0:
            add("X-exp", {"work_class": "experiment", "kind": "hold", "priority": "interactive",
                          "holder": "operator:eval", "operator": True}, now)
            add("H-home", {"work_class": "agent", "kind": "hold", "priority": "background", "retryable": True,
                           "holder": "durable-runs:home"}, now)
            add("H-gpu2", {"work_class": "agent", "kind": "hold", "priority": "background", "retryable": True,
                           "holder": "durable-runs:gpu2"}, now)
        if sec == PAUSE_FROM + 60:
            add("D-img", {"work_class": "diffusion", "kind": "hold", "priority": "background", "retryable": True,
                          "holder": "durable-runs:img"}, now)
        if sec % 30 == 5:
            add(f"C{sec}", {"work_class": "chat", "kind": "request", "priority": "interactive"}, now)
            work_left[f"C{sec}"] = 20
        if sec % 5 == 0:
            add(f"M{sec}", {"work_class": "metacog", "kind": "request", "priority": "system"}, now)
            work_left[f"M{sec}"] = 4
        for lid, st in leases.items():
            if st["status"] not in ("granted", "recalling"):
                continue
            if lid in work_left:
                work_left[lid] -= 1
                if work_left[lid] <= 0:
                    apply(st, {"type": "release_ok", "at": at})
                    continue
            elif st["status"] == "recalling" and lid.startswith("H-"):
                apply(st, {"type": "release_ok", "at": at})       # a recalled run gives its hold back
                continue
            apply(st, {"type": "heartbeat", "at": at})
            if st["role"] and cfg.roles[st["role"]].swap:
                for c in cfg.roles[st["role"]].cards:
                    cards[c].last_active_at = now
        paused = PAUSE_FROM <= sec < PAUSE_UNTIL
        for d in schedule(cfg, LIVE, cards, _views(leases), now, guards={"thermal": None},
                          frozen=cfg.actuated_seats() if paused else ()):
            if isinstance(d, (SwapLoad, SwapUnload, SwapBlocked)):
                if paused and not isinstance(d, SwapBlocked):
                    swaps_while_paused.append(type(d).__name__)
                elif isinstance(d, SwapUnload) and d.role == seat:
                    cards["gpu2"] = CardLive("gpu2")              # the actuator unloads at once here
                continue
            if isinstance(d, Serialized):
                continue
            st = leases[d.lease_id]
            ev = {"type": _EV[type(d)], "at": at, "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                ev["role"] = d.role
                grants.append((sec, d.lease_id, d.role))
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
                recalls.append((sec, d.lease_id, d.reason, st["role"]))
            apply(st, ev)

    residents = set(cfg.evicted_by("experiment"))
    return {
        "experiment_resident_recalls": [r for r in recalls if r[3] in residents and r[3] != seat],
        "experiment_granted": any(g[1] == "X-exp" for g in grants),
        "resident_grants": sum(1 for g in grants if g[2] in ("chat", "metacog", "fast")),
        "seat_recalls_while_paused": [r for r in recalls if PAUSE_FROM <= r[0] < PAUSE_UNTIL and r[3] == seat],
        "swap_decisions_reported_while_paused": len(swaps_while_paused),
        "seat_reclaimed_after_resume_sec": next((r[0] - PAUSE_UNTIL for r in recalls
                                                 if r[0] >= PAUSE_UNTIL and r[3] == seat), None),
        "diffusion_granted_at_sec": next((g[0] for g in grants if g[1] == "D-img"), None),
    }


def enforce_failures(e: dict) -> list[str]:
    out = []
    if e["experiment_resident_recalls"] or e["experiment_granted"]:
        out.append("experiment_drained_residents")     # stage 5 "Corrections from building 5.1" item 5
    if not e["resident_grants"]:
        out.append("residents_starved")
    if e["seat_recalls_while_paused"]:
        out.append("pause_drained_the_seat")
    if e["seat_reclaimed_after_resume_sec"] is None or e["diffusion_granted_at_sec"] is None \
            or e["diffusion_granted_at_sec"] < PAUSE_UNTIL:
        out.append("resume_did_not_restore_reclaim")
    return out


# --- stage 7.3 concurrency -------------------------------------------------------------------
CONC_SEC = 8000                     # under agent-gpu2's max_hold_sec (9000): no drain in the window
CONC_START_RUNS = 6                 # queued at once (live 2026-09-30: 7 waiting)
CONC_RUN_EVERY_SEC = 600            # then one new run about every 10 min
CONC_RUN_WORK_SEC = (1500, 3000)    # run length in single-slot seconds (live hold p50 2,225 s)
CONC_CALL_SEC = (120, 240)          # one call's single-slot work (live median call ~181 s)
CONC_GAP_SEC = (30, 90)             # tool phase between calls (runs busy ~75% of their hold)
CONC_ONE_OFF_EVERY_SEC = 240        # one-off system agent calls (cortex-exec shape)
CONC_ONE_OFF_SEC = (20, 60)
# Occupancy slowdown model, agent-gpu2 only (Bonsai, measured live 2026-10-09): tok/s per slot by the
# number of calls decoding on the seat at once. 1 -> 46, 2 -> 26.
BONSAI_TOK_S = {1: 46.0, 2: 26.0}
CONC_SEEDS = (3, 5, 8)


def _slowdown(busy: int) -> float:
    """Single-slot seconds of work one call gets through per wall second with ``busy`` calls on the seat."""
    if busy <= 1:
        return 1.0
    return BONSAI_TOK_S[min(busy, max(BONSAI_TOK_S))] / BONSAI_TOK_S[1]


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def concurrency_scenario(cfg=CFG, seed: int = 3, gpu2_slots: int = 2) -> dict:
    rng = random.Random(seed)
    leases: dict[str, dict] = {}
    work_left: dict[str, float] = {}
    roles = dict(LIVE, **{"agent-gpu2": RoleLive("agent-gpu2", True, gpu2_slots, 131072)})
    cards = {c: CardLive(c) for c in cfg.cards}
    cards["gpu2"] = CardLive("gpu2", swapped_in={"agent-gpu2"}, loaded_at=T0, last_active_at=T0)
    starts = [0] * CONC_START_RUNS
    t = 0.0
    while True:
        t += rng.expovariate(1 / CONC_RUN_EVERY_SEC)
        if t >= CONC_SEC - 1200:
            break
        starts.append(int(t))
    runs = [{"id": f"r{i}", "start": st, "left": rng.uniform(*CONC_RUN_WORK_SEC), "hold": None, "child": None,
             "phase": "gap", "phase_left": 0.0, "done": False} for i, st in enumerate(starts)]
    next_one_off = rng.expovariate(1 / CONC_ONE_OFF_EVERY_SEC)
    hold_waits: list[float] = []
    child_waits: list[float] = []
    one_off_waits: list[float] = []
    waiting_holds: list[int] = []
    two_holds_sec = 0
    max_holds_seen = 0
    pin_stall_sec = 0
    seat_work_sec = 0.0
    run_ends: list[float] = []
    n = 0

    def add(req: dict, now: datetime) -> str:
        nonlocal n
        n += 1
        lid = f"C{n}"
        st = dict(initial_state(lid, {"request_id": lid, **req}, now))
        st["deadline_at"] = None
        leases[lid] = st
        return lid

    def apply(st: dict, ev: dict) -> None:
        st.update(transition(st, ev, cfg), history=[])

    for sec in range(CONC_SEC):
        now = T0 + timedelta(seconds=sec)
        at = now.isoformat()
        while next_one_off <= sec:
            lid = add({"work_class": "agent", "kind": "request", "priority": "system", "retryable": True,
                       "holder": "cortex-exec"}, now)
            work_left[lid] = rng.uniform(*CONC_ONE_OFF_SEC)
            next_one_off += rng.expovariate(1 / CONC_ONE_OFF_EVERY_SEC)

        for run in runs:
            if run["done"] or sec < run["start"]:
                continue
            hold = leases.get(run["hold"]) if run["hold"] else None
            if hold is None or hold["status"] in ("released", "unavailable", "dead_letter"):
                run["hold"] = add({"work_class": "agent", "kind": "hold", "priority": "background",
                                   "retryable": True, "holder": f"durable-runs:{run['id']}"}, now)
                run["child"], run["phase"], run["phase_left"] = None, "gap", 0.0
                continue
            if hold["status"] not in ("granted", "recalling"):
                continue
            child = leases.get(run["child"]) if run["child"] else None
            if run["phase"] == "call" and child is not None:
                if child["status"] in ("queued", "granted", "recalling"):
                    continue
                run["child"], run["phase"], run["phase_left"] = None, "gap", rng.uniform(*CONC_GAP_SEC)
                continue
            run["phase_left"] -= 1
            if run["phase_left"] > 0:
                continue
            if run["left"] <= 0:
                apply(hold, {"type": "release_ok", "at": at})
                run["done"] = True
                run_ends.append(sec - run["start"])
                continue
            call = min(run["left"], rng.uniform(*CONC_CALL_SEC))
            run["left"] -= call
            cid = add({"work_class": "agent", "kind": "request", "priority": "system", "retryable": False,
                       "holder": "orion-llm-gateway", "hold_lease_id": hold["lease_id"]}, now)
            work_left[cid] = call
            run["child"], run["phase"] = cid, "call"

        active = [st for st in leases.values() if st["status"] in ("granted", "recalling")]
        calls_on = defaultdict(int)
        for st in active:
            if st["request"].get("kind") != "hold":
                calls_on[st["role"]] += 1
        for st in active:
            lid = st["lease_id"]
            if lid in work_left:
                rate = _slowdown(calls_on[st["role"]]) if st["role"] == "agent-gpu2" else 1.0
                work_left[lid] -= rate
                if st["role"] == "agent-gpu2":
                    seat_work_sec += rate
                if work_left[lid] <= 0:
                    apply(st, {"type": "release_ok", "at": at})
                    continue
            apply(st, {"type": "heartbeat", "at": at})
            if st["role"] == "agent-gpu2":
                cards["gpu2"].last_active_at = now

        views = _views(leases)
        seat_holds = [v for v in views if v.kind == "hold" and v.role == "agent-gpu2"
                      and v.status in ("granted", "recalling")]
        max_holds_seen = max(max_holds_seen, len(seat_holds))
        two_holds_sec += len(seat_holds) >= 2
        waiting_holds.append(sum(1 for v in views if v.kind == "hold" and v.status == "queued"))
        seat_ids = {h.lease_id for h in seat_holds}
        in_flight = {v.hold_lease_id for v in views if v.status in ("granted", "recalling") and v.hold_lease_id}
        one_offs = [v for v in views if v.role == "agent-gpu2" and v.status in ("granted", "recalling")
                    and v.kind != "hold" and v.hold_lease_id not in seat_ids]
        stalled = {v.hold_lease_id for v in views if v.status == "queued" and v.hold_lease_id in seat_ids
                   and v.hold_lease_id not in in_flight}
        if gpu2_slots == 2 and len(stalled) == 2 and len(one_offs) == 1:
            pin_stall_sec += 1        # one gap borrow holding up both idle runs' next calls

        for d in schedule(cfg, roles, cards, views, now, guards={"thermal": None}):
            if not isinstance(d, tuple(_EV)):
                continue                  # the seat stays loaded; no swap is part of this story
            st = leases[d.lease_id]
            ev = {"type": _EV[type(d)], "at": at, "reason": getattr(d, "reason", None)}
            if isinstance(d, Grant):
                ev["role"] = d.role
                waited = (now - datetime.fromisoformat(st["queued_since"])).total_seconds()
                req = st["request"]
                if req.get("kind") == "hold":
                    hold_waits.append(waited)
                elif req.get("hold_lease_id"):
                    child_waits.append(waited)
                else:
                    one_off_waits.append(waited)
            if isinstance(d, Recall):
                ev["recall_by"] = d.recall_by.isoformat()
            apply(st, ev)

    return {
        "runs_started": len(runs), "runs_finished": len(run_ends),
        "run_wall_sec_p50": _p(run_ends, 0.5),
        "hold_wait_sec": {"n": len(hold_waits), "p50": _p(hold_waits, 0.5), "p90": _p(hold_waits, 0.9)},
        "mean_waiting_holds": round(statistics.mean(waiting_holds), 2),
        "one_off_wait_sec": {"n": len(one_off_waits), "p50": _p(one_off_waits, 0.5),
                             "p90": _p(one_off_waits, 0.9), "max": _p(one_off_waits, 1.0)},
        "run_call_wait_sec": {"n": len(child_waits), "p90": _p(child_waits, 0.9), "max": _p(child_waits, 1.0)},
        "seat_two_holds_sec": two_holds_sec, "seat_max_holds_seen": max_holds_seen,
        "gap_borrow_stalled_two_runs_sec": pin_stall_sec,
        "seat_work_sec_per_hour": round(seat_work_sec / (CONC_SEC / 3600), 0),
    }


def _with_seat(cfg, **update):
    roles = dict(cfg.roles)
    roles["agent-gpu2"] = roles["agent-gpu2"].model_copy(update=update)
    return cfg.model_copy(update={"roles": roles})


def concurrency_report(cfg=CFG) -> dict:
    variants = {"max_holds_1": _with_seat(cfg, max_holds=1, reserve_one_off_slots=0),
                "max_holds_2": _with_seat(cfg, max_holds=2, reserve_one_off_slots=0),
                "max_holds_2_reserve_1": _with_seat(cfg, max_holds=2, reserve_one_off_slots=1)}
    out: dict = {"slowdown_model_tok_s": BONSAI_TOK_S, "seeds": list(CONC_SEEDS),
                 "gap_pinning_case": gap_pinning_case(variants["max_holds_2"])}
    for name, vcfg in variants.items():
        per_seed = [concurrency_scenario(vcfg, seed) for seed in CONC_SEEDS]
        out[name] = {"per_seed": per_seed,
                     "mean_hold_wait_p90": round(statistics.mean(r["hold_wait_sec"]["p90"] or 0 for r in per_seed), 1),
                     "mean_waiting_holds": round(statistics.mean(r["mean_waiting_holds"] for r in per_seed), 2),
                     "mean_one_off_wait_p90": round(statistics.mean(r["one_off_wait_sec"]["p90"] or 0 for r in per_seed), 1),
                     "runs_finished": sum(r["runs_finished"] for r in per_seed)}
    return out


def gap_pinning_case(cfg=CFG) -> dict:
    """Acceptance check 9 as one scripted tick through the real lease table: two runs hold the
    2-slot seat, both idle between calls; a system one-off is using one gap; then both runs' next
    calls and another one-off arrive. Exactly one run's call is granted at once; the one-off waits."""
    leases: dict[str, dict] = {}
    roles = dict(LIVE, **{"agent-gpu2": RoleLive("agent-gpu2", True, 2, 131072),
                          "agent": RoleLive("agent", True, 1, 131072)})
    cards = {c: CardLive(c) for c in cfg.cards}
    cards["gpu2"] = CardLive("gpu2", swapped_in={"agent-gpu2"}, loaded_at=T0, last_active_at=T0)

    def add(lid: str, req: dict, at: datetime, role: str | None = None) -> None:
        st = dict(initial_state(lid, {"request_id": lid, **req}, at))
        if role:
            st.update(transition(st, {"type": "grant", "at": at.isoformat(), "role": role}, cfg), history=[])
        leases[lid] = st

    # Times within the heartbeat TTLs (request 30 s, hold 90 s) so nothing expires in this one tick.
    hold = {"work_class": "agent", "kind": "hold", "priority": "background", "retryable": True}
    add("home", {**hold, "holder": "durable-runs:home"}, T0 - timedelta(seconds=25), "agent")
    add("home-call", {"work_class": "agent", "kind": "request", "priority": "system", "hold_lease_id": "home"},
        T0 - timedelta(seconds=5), "agent")
    add("A", {**hold, "holder": "durable-runs:A"}, T0 - timedelta(seconds=20), "agent-gpu2")
    add("B", {**hold, "holder": "durable-runs:B"}, T0 - timedelta(seconds=15), "agent-gpu2")
    add("gap", {"work_class": "agent", "kind": "request", "priority": "system"}, T0 - timedelta(seconds=8), "agent-gpu2")
    for lid, h in (("A-call", "A"), ("B-call", "B")):
        add(lid, {"work_class": "agent", "kind": "request", "priority": "system", "hold_lease_id": h}, T0)
    add("one-off", {"work_class": "agent", "kind": "request", "priority": "system"}, T0 - timedelta(seconds=5))
    got = {d.lease_id: d.role for d in schedule(cfg, roles, cards, _views(leases), T0, guards={"thermal": None})
           if isinstance(d, Grant)}
    return {"granted": got}


def concurrency_failures(c: dict) -> list[str]:
    out = []
    if c["gap_pinning_case"]["granted"] != {"A-call": "agent-gpu2"}:
        out.append("conc_gap_pinning_case")
    two, one, res = c["max_holds_2"], c["max_holds_1"], c["max_holds_2_reserve_1"]
    bound = max(CONC_ONE_OFF_SEC) / _slowdown(2) + 2
    if not all(r["seat_two_holds_sec"] for r in two["per_seed"]):
        out.append("conc_never_two_holds")
    if any(r["gap_borrow_stalled_two_runs_sec"] for r in two["per_seed"]):
        out.append("conc_gap_borrow_stalled_two_runs")
    if any((r["run_call_wait_sec"]["max"] or 0) > bound for r in two["per_seed"]):
        out.append("conc_run_call_waited_past_one_inference")
    if not two["mean_hold_wait_p90"] < one["mean_hold_wait_p90"]:
        out.append("conc_hold_wait_not_reduced")
    if not two["mean_waiting_holds"] < one["mean_waiting_holds"]:
        out.append("conc_waiting_holds_not_reduced")
    if any(r["seat_max_holds_seen"] > 1 for r in one["per_seed"] + res["per_seed"]):
        out.append("conc_hold_limit_not_in_force")
    return out


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
    report["urgent_scenario"] = urgent_scenario()
    rollback_cfg = CFG.model_copy(update={"defaults": CFG.defaults.model_copy(update={"urgent_max_concurrent": 0})})
    rollback = urgent_scenario(rollback_cfg)
    report["urgent_rollback_paused"] = rollback["paused"]
    report["shed_scenario"] = shed_scenario()
    report["enforce_scenario"] = enforce_scenario()
    report["concurrency_scenario"] = concurrency_report()
    report["leases_per_sec_inmemory"] = round(asyncio.run(checkpoint_throughput()), 1)
    import json

    print(json.dumps(report, indent=2))
    failures = [k for k in ("owner_starvation_sec", "leases_lost", "small_role_violations", "world_diffusion_grant_overlap_sec",
                            "run_blocked_behind_itself_sec", "run_call_waits_over_one_inference") if report[k]]
    failures += urgent_failures(report["urgent_scenario"], rollback)
    failures += shed_failures(report["shed_scenario"])
    failures += enforce_failures(report["enforce_scenario"])
    failures += concurrency_failures(report["concurrency_scenario"])
    if not report["interleaved_grants"]:
        failures.append("interleaved_grants")
    if not (report["counts"].get("serialized:diffusion") or report["counts"].get("serialized:world")):
        failures.append("serialize_with_not_exercised")   # the world/diffusion mutex never came up
    if not report["counts"].get("swap_failed_restored"):
        failures.append("failed_load_not_exercised")
    print("VERDICT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
