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
    Serialized, SwapBlocked, SwapLoad, SwapUnload, Unavailable, schedule,
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
    report["leases_per_sec_inmemory"] = round(asyncio.run(checkpoint_throughput()), 1)
    import json

    print(json.dumps(report, indent=2))
    failures = [k for k in ("owner_starvation_sec", "leases_lost", "small_role_violations", "world_diffusion_grant_overlap_sec",
                            "run_blocked_behind_itself_sec", "run_call_waits_over_one_inference") if report[k]]
    failures += urgent_failures(report["urgent_scenario"], rollback)
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
