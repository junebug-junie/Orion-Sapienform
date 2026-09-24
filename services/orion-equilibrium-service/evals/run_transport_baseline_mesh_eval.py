#!/usr/bin/env python3
"""Eval: replay a synthetic 24h multi-hop mesh through the transport baseline gate.

No bus, no Redis: drives ``TransportBaselineGate`` (the exact object the service
uses) with RpcHealthSnapshotV1-shaped payloads carrying ``channel_latency``, and
compares what it would publish against what the legacy pooled-p95 >= 5000 ms
branch fired on the same windows.

Scenarios (one per hop, all in the same replay):
- metacog self-loop: 1 background draft per window at ~15 s, excluded label.
- calm state hop and calm recall hop: never fire; must rest at z~0, ratio~1.
- LLM chat hop: a 20-minute 5x spike at 04:00, then a timeout burst at 10:00.
- LLM background hop: slow creep to 2.5x over an hour from 12:00, held -> must be
  saturation then exactly one regime_shift (never cleared before it).
- state-service hop: 5-minute total outage (0 successes, timeouts) at 16:00.

Run:  python services/orion-equilibrium-service/evals/run_transport_baseline_mesh_eval.py
Exit code 0 = all checks pass. Prints a JSON report.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
for p in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.transport_baseline_gate import TransportBaselineGate  # noqa: E402
from orion.metacog.transport_baseline import TransportBaselineConfig  # noqa: E402

WIN = 30.0
HOURS = 24
T0 = datetime(2026, 9, 24, tzinfo=timezone.utc)
LEGACY_P95_MS = 5000.0

METACOG = "orion:cortex:exec:request:background#log_orion_metacognition"
STATE_ORCH = "orion:state:request"
LLM_CHAT = "orion:exec:request:LLMGatewayService#chat_general"
LLM_BG = "orion:exec:request:LLMGatewayService#background"
RECALL = "orion:exec:request:RecallService"
STATE_EXEC = "orion:state:request"


def _stats(lat: list[float], timeouts: int = 0) -> dict:
    logs = [math.log(x) for x in lat]
    return {
        "success_count": len(lat),
        "timeout_count": timeouts,
        "log_ms_sum": sum(logs),
        "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(lat) if lat else None,
    }


def _draw(rng: random.Random, center: float, n: int, sigma: float = 0.3) -> list[float]:
    return [center * math.exp(rng.gauss(0.0, sigma)) for _ in range(n)]


def _hour(i: int) -> float:
    return i * WIN / 3600.0


def _mesh_windows(seed: int = 7):
    rng = random.Random(seed)
    n_windows = int(HOURS * 3600 / WIN)
    for i in range(n_windows):
        h = _hour(i)
        start = T0 + timedelta(seconds=WIN * i)
        end = start + timedelta(seconds=WIN)
        # --- cortex-orch: metacog draft + state call (the self-loop shape) ---
        orch = {
            METACOG: _stats(_draw(rng, 15000.0, 1)),
            STATE_ORCH: _stats(_draw(rng, 300.0, 1)),
        }
        yield "cortex-orch", None, start, end, orch

        # --- cortex-exec chat lane ---
        chat_center, chat_to = 9000.0, 0
        if 4.0 <= h < 4.0 + 20 / 60:
            chat_center = 45000.0
        if 10.0 <= h < 10.0 + 5 / 60:
            chat_to = 2
        exec_chat = {
            LLM_CHAT: _stats(_draw(rng, chat_center, 8), timeouts=chat_to),
            RECALL: _stats(_draw(rng, 400.0, 6)),
        }
        # state-service outage
        if 16.0 <= h < 16.0 + 5 / 60:
            exec_chat[STATE_EXEC] = _stats([], timeouts=3)
        else:
            exec_chat[STATE_EXEC] = _stats(_draw(rng, 250.0, 5))
        yield "cortex-exec", "chat", start, end, exec_chat

        # --- cortex-exec background lane: slow creep then plateau ---
        frac = 0.0 if h < 12.0 else min(1.0, (h - 12.0) / 1.0)
        bg_center = 12000.0 * math.exp(frac * math.log(2.5))
        exec_bg = {LLM_BG: _stats(_draw(rng, bg_center, 6 + int(10 * frac)))}
        yield "cortex-exec", "background", start, end, exec_bg


def _payload(service, instance, start, end, cl) -> dict:
    succ = sum(v["success_count"] for v in cl.values())
    tos = sum(v["timeout_count"] for v in cl.values())
    maxes = [v["max_ms"] for v in cl.values() if v["max_ms"] is not None]
    return {
        "service": service,
        "instance": instance,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "success_count": succ,
        "timeout_count": tos,
        "success_latency_ms_p95": max(maxes) if maxes else None,
        "channel_latency": cl,
    }


def run_eval(seed: int = 7) -> dict:
    gate = TransportBaselineGate(TransportBaselineConfig(), ["log_orion_metacognition"])
    triggers = []
    excluded_events = 0
    legacy_fires = 0
    quiet_obs: dict[str, list] = {}
    all_obs = []
    for service, instance, start, end, cl in _mesh_windows(seed):
        payload = _payload(service, instance, start, end, cl)
        if payload["timeout_count"] > 0 or (payload["success_latency_ms_p95"] or 0) >= LEGACY_P95_MS:
            legacy_fires += 1
        result, trigs = gate.process(payload, zen_state="zen", pressure=0.0, recall_enabled=True)
        excluded_events += sum(1 for e in result.events if e.excluded)
        triggers.extend(trigs)
        for ob in result.observations:
            all_obs.append((start, ob))
            # quiet hours 01:00-03:30 (after warm-up, before any incident)
            if 1.0 <= (start - T0).total_seconds() / 3600 < 3.5 and ob.evaluated:
                quiet_obs.setdefault(f"{ob.service}|{ob.instance}|{ob.key}", []).append(ob)

    by = Counter((t.upstream["key"], t.upstream["condition"], t.upstream["phase"]) for t in triggers)
    checks: dict[str, bool] = {}

    checks["self_loop_never_triggers"] = not any(t.upstream["key"] == METACOG for t in triggers)
    checks["self_loop_was_measured"] = any(
        ob.key == METACOG and ob.evaluated for _, ob in all_obs
    )
    calm = {("cortex-orch", STATE_ORCH), ("cortex-exec", RECALL)}
    checks["calm_hops_never_trigger"] = not any(
        (t.upstream["service"], t.upstream["key"]) in calm for t in triggers
    )
    checks["chat_spike_open_and_close"] = (
        by[(LLM_CHAT, "spike", "open")] == 1 and by[(LLM_CHAT, "spike", "close")] == 1
    )
    checks["chat_timeout_one_episode"] = (
        by[(LLM_CHAT, "timeout", "open")] == 1 and by[(LLM_CHAT, "timeout", "close")] == 1
    )
    checks["creep_saturation_then_one_regime_shift"] = (
        by[(LLM_BG, "saturation", "open")] == 1
        and by[(LLM_BG, "saturation", "close")] == 0
        and by[(LLM_BG, "regime_shift", "open")] == 1
        and by[(LLM_BG, "spike", "open")] == 0
    )
    checks["state_outage_zero_success_episode"] = (
        by[(STATE_EXEC, "zero_success", "open")] == 1 and by[(STATE_EXEC, "zero_success", "close")] == 1
    )

    rest = {}
    rest_ok = True
    for k, obs in quiet_obs.items():
        zs = sorted(o.z for o in obs if o.z is not None)
        rs = sorted(o.saturation_ratio for o in obs if o.saturation_ratio is not None)
        if not zs:
            continue
        mz, mr = zs[len(zs) // 2], rs[len(rs) // 2]
        rest[k] = {"z_median": round(mz, 3), "ratio_median": round(mr, 3), "n": len(zs)}
        if not (abs(mz) <= 0.5 and 0.8 <= mr <= 1.3):
            rest_ok = False
    checks["quiet_hours_rest_state"] = rest_ok and bool(rest)
    # every key's final state: no episode left open except none expected
    final_open = {
        f"{ks.service}|{ks.instance}|{ks.hop}": sorted(ks.episodes)
        for ks in gate.state.keys.values()
        if ks.episodes
    }
    checks["nothing_left_open_at_end"] = not final_open

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "triggers_total": len(triggers),
        "legacy_pooled_p95_fires": legacy_fires,
        "excluded_events_suppressed": excluded_events,
        "triggers_by_key_condition_phase": {"|".join(k): v for k, v in sorted(by.items())},
        "quiet_hours_rest": rest,
        "final_open_episodes": final_open,
    }


def main() -> int:
    report = run_eval()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
