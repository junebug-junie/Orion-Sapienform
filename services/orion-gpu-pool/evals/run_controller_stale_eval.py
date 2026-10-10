#!/usr/bin/env python3
"""Replay of the 2026-10-09/10 stale-controller incident through the real PoolRuntime.

Live: from 2026-10-09 03:56 to 2026-10-10 06:01 UTC the lane controller on circe refused every
agent-gpu2 load with ``config_unloadable:ValidationError`` (136 refusals, its image older than the
config it read). The pool backed off swap_cooldown_sec and retried; nobody was told for 26 h.

Scenario A (incident): real scheduler + lease graph, agent demand that wants agent-gpu2 loaded, an
actuator that refuses every load with the live reason until 136 refusals, then is "rebuilt" and
succeeds. Scenario B (healthy noise): the same demand, refusals a retry fixes (busy,
deadline_passed, upstream_not_idle, stale_generation) 136 times.

Hard targets (exit 1 if missed):
  - A: first alert within 2 x swap_cooldown_sec + 60 s of the first refusal (was: never)
  - A: exactly one "degraded" alert over the whole outage (no pager storm)
  - A: one "recovered" alert, in the same tick the rebuilt controller answers
  - A: /health-equivalent view names the seat degraded with the live reason throughout
  - B: zero alerts

Run: python services/orion-gpu-pool/evals/run_controller_stale_eval.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
# service first: the repo root has its own `tests` package, which would shadow the service fixtures
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]
os.environ.setdefault("ORION_BUS_URL", "redis://127.0.0.1:1/0")
os.environ.setdefault("POSTGRES_URI", "postgresql://unused@127.0.0.1:1/unused")

from tests.test_holds_and_actuation import SEAT, actuations, boot, demand_gpu2, make, result, step  # noqa: E402
from tests.test_runtime import CFG  # noqa: E402

logging.disable(logging.CRITICAL)   # 272 refusal log lines would bury the verdict

LIVE_REFUSALS = 136
STALE = "config_unloadable:ValidationError"
RETRYABLE = ("busy", "deadline_passed", "upstream_not_idle:agent-gpu2", "stale_generation")


async def replay(reasons: list[str], *, then_succeed: bool) -> dict:
    rt, clock = make()
    sent: list[tuple[str, object]] = []

    async def alert(seat, state, view):
        sent.append((state, clock()))

    rt.controller_alert = alert
    await boot(rt)
    home, _ = await demand_gpu2(rt, clock)
    first_refusal = clock()
    handled: set[str] = set()
    refused = 0
    degraded_views = 0
    cooldown = CFG.defaults.swap_cooldown_sec
    recovered_at = None
    for _ in range(len(reasons) * 3):
        loads = [m for m in actuations(rt) if m.action == "load" and m.action_id not in handled]
        for msg in loads:
            handled.add(msg.action_id)
            if refused < len(reasons):
                await result(rt, msg, "refused", reason=reasons[refused])
                refused += 1
            elif then_succeed:
                await result(rt, msg, "accepted")
                recovered_at = clock()
        for _ in range(3):
            await asyncio.sleep(0)
        if SEAT in rt.controller_health.degraded():
            v = rt.controller_health.view()[SEAT]
            assert v["reason"] == STALE and "circe" in v["advice"]
            degraded_views += 1
        if refused >= len(reasons) and (recovered_at or not then_succeed):
            break
        await step(rt, clock, cooldown + 1, beat=[home.lease_id], every=30)
    first = next((t for s, t in sent if s == "degraded"), None)
    return {"refused": refused, "alerts": [s for s, _ in sent],
            "first_alert_after_sec": (first - first_refusal).total_seconds() if first else None,
            "recovered_alert_lag_sec": (next((t for s, t in sent if s == "recovered"), clock()) - recovered_at
                                        ).total_seconds() if recovered_at else None,
            "degraded_views": degraded_views}


def main() -> int:
    a = asyncio.run(replay([STALE] * LIVE_REFUSALS, then_succeed=True))
    b = asyncio.run(replay([RETRYABLE[i % len(RETRYABLE)] for i in range(LIVE_REFUSALS)], then_succeed=False))
    budget = 2 * CFG.defaults.swap_cooldown_sec + 60
    checks = [
        ("A refusals replayed", a["refused"] == LIVE_REFUSALS, a["refused"]),
        ("A first alert within budget", a["first_alert_after_sec"] is not None
         and a["first_alert_after_sec"] <= budget, f"{a['first_alert_after_sec']} s (budget {budget} s)"),
        ("A one degraded alert", a["alerts"].count("degraded") == 1, a["alerts"]),
        ("A one recovered alert, same tick", a["alerts"][-1:] == ["recovered"]
         and a["recovered_alert_lag_sec"] == 0, a["recovered_alert_lag_sec"]),
        ("A degraded visible every cycle after the 2nd refusal", a["degraded_views"] >= LIVE_REFUSALS - 1,
         a["degraded_views"]),
        ("B refusals replayed", b["refused"] == LIVE_REFUSALS, b["refused"]),
        ("B zero alerts on retryable refusals", b["alerts"] == [], b["alerts"]),
    ]
    failed = 0
    for name, ok, value in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value}")
        failed += not ok
    print(f"\nincident: before this patch, {LIVE_REFUSALS} refusals over 26 h raised 0 alerts; "
          f"replayed now, the first alert lands {a['first_alert_after_sec']:.0f} s after the first refusal.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
