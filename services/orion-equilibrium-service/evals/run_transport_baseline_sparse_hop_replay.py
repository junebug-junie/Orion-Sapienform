#!/usr/bin/env python3
"""Eval: replay two real sparse hops' stored hourly rows through the reducer.

Why: 2026-10-13, live. On ``cortex-orch -> orion:cortex:exec:request:background``
and ``actions -> orion:cortex:request`` the floor fell from ~14 s to ~1.3 s on
2026-10-08 (cabinet-hot GPU shed: metacog drafts fell back to the canned
template and "succeeded" in 0.3-1 s; the floor follows improvements with a 90 s
half-life), then real calls came back at ~6-9 s. The gate opened and closed
saturation 44 times on each hop and never stated a regime_shift: silence (nights,
and the 25-47 min between this sparse hop's evaluations) aged every episode out
and reset its hot time, while the floor crept back up 1.3 s -> 4 s between
episodes. Drift absorbed, never stated (spec guard 3).

Fixture: ``fixtures/transport_baseline_hourly_sparse_hops_2026-10-13.jsonl`` --
the stored ``transport_baseline_hourly`` rows for those two hops, 2026-09-30 to
2026-10-13 (read-only export, trimmed to the columns used here).

Reconstruction (approximate, stated): per hour, ``windows_evaluated`` bursts of
``min_calls`` calls (evenly spaced, or packed 2.5 min apart -- both are run) plus
the hour's remaining ``success_count`` as single calls evenly spread; every call
in an hour reads the hour's median level, ``floor_ms * saturation_ratio_p50``.
Validated offline against the pre-fix reducer (origin/main 3f22b2f25): it
reproduced the stored floor (1.53 s -> 4.2-4.4 s replayed vs 1.26 s -> 3.99 s
stored), 37-55 saturation open/close pairs (44 stored) and 0 regime_shift
(0 stored).

Checks (current reducer, from 2026-10-08 18:00 UTC, after the floor collapsed):
- the fixture itself shows the bug: >= 30 saturation opens, 0 regime_shift;
- at most 2 saturation opens per hop (one paused episode, not ~44 pairs);
- exactly one regime_shift per hop, before 2026-10-10 00:00 UTC, stating the
  collapsed floor (< 2.5 s) against a window > 5 s;
- no went_silent close (the longest silence in the fixture is < 24 h).

Run: ``python services/orion-equilibrium-service/evals/run_transport_baseline_sparse_hop_replay.py``
Exit 0 = all checks pass. Prints a JSON report.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metacog.transport_baseline import (  # noqa: E402
    TransportBaselineConfig,
    fold_snapshot,
    new_state,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "transport_baseline_hourly_sparse_hops_2026-10-13.jsonl"
HOPS = (
    ("cortex-orch", "orion:cortex:exec:request:background"),
    ("actions", "orion:cortex:request"),
)
WIN = 30.0
AFTER = datetime(2026, 10, 8, 18, tzinfo=timezone.utc).timestamp()
DEADLINE = datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp()


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def load(service: str, key: str) -> dict[float, dict]:
    hours: dict[float, dict] = {}
    for line in FIXTURE.read_text().splitlines():
        r = json.loads(line)
        if (r["service"], r["key"]) != (service, key):
            continue
        a = hours.setdefault(_ts(r["hour_start"]), {"n": 0, "calls": 0, "lv": [], "opened": Counter()})
        n = int(r["windows_evaluated"] or 0)
        a["n"] += n
        a["calls"] += int(r["success_count"] or 0)
        if r["saturation_ratio_p50"] is not None and n and r["floor_ms"]:
            a["lv"].append((math.log(r["saturation_ratio_p50"] * r["floor_ms"]), n))
        a["opened"].update(r["conditions_opened"] or {})
    return hours


def schedule(hours: dict[float, dict], spacing: str, min_calls: int) -> dict[float, list]:
    """window_end -> [calls, log-ms per call]."""
    calls: dict[float, list] = defaultdict(lambda: [0, 0.0])
    m = None
    for h in sorted(hours):
        a = hours[h]
        if a["lv"]:
            m = sum(v * w for v, w in a["lv"]) / sum(w for _, w in a["lv"])
        if m is None:
            continue
        n = a["n"] if a["lv"] else 0
        for k in range(n):
            off = int(((k + 0.5) * 3600 / n) // WIN) * WIN if spacing == "even" else k * 150.0
            calls[h + off + WIN] = [calls[h + off + WIN][0] + min_calls, m]
        rest = max(a["calls"] - min_calls * n, 0)
        for k in range(rest):
            off = int(((k + 0.25) * 3600 / rest) // WIN) * WIN
            calls[h + off + WIN] = [calls[h + off + WIN][0] + 1, m]
    return calls


def replay(service: str, key: str, spacing: str) -> dict:
    cfg = TransportBaselineConfig()
    hours = load(service, key)
    calls = schedule(hours, spacing, cfg.min_calls)
    state = new_state(cfg)
    events = []
    w, end = min(hours), max(hours) + 3600
    while w < end:
        we = w + WIN
        cl = {}
        if we in calls:
            n, m = calls[we]
            cl = {key: {"success_count": n, "timeout_count": 0, "log_ms_sum": n * m,
                        "log_ms_sumsq": n * m * m, "max_ms": math.exp(m)}}
        payload = {
            "service": service, "instance": "main",
            "window_start": datetime.fromtimestamp(w, timezone.utc).isoformat(),
            "window_end": datetime.fromtimestamp(we, timezone.utc).isoformat(),
            "channel_latency": cl,
        }
        events.extend((we, e) for e in fold_snapshot(state, payload, config=cfg).events)
        w = we
    late = [(t, e) for t, e in events if t >= AFTER]
    shifts = [(t, e) for t, e in late if e.condition == "regime_shift"]
    stored_opens = sum(a["opened"].get("saturation", 0) for h, a in hours.items() if h >= AFTER)
    stored_shifts = sum(a["opened"].get("regime_shift", 0) for a in hours.values())
    sat_opens = sum(1 for _, e in late if e.condition == "saturation" and e.phase == "open")
    went_silent = sum(1 for _, e in events if e.close_reason == "went_silent")
    checks = {
        "fixture_shows_the_bug": stored_opens >= 30 and stored_shifts == 0,
        "at_most_two_saturation_opens": sat_opens <= 2,
        "exactly_one_regime_shift": len(shifts) == 1,
        "shift_before_deadline": bool(shifts) and shifts[0][0] < DEADLINE,
        "shift_states_collapsed_floor": bool(shifts)
        and (shifts[0][1].floor_ms or 1e9) < 2500.0
        and (shifts[0][1].window_mean_ms or 0.0) > 5000.0,
        "no_went_silent": went_silent == 0,
    }
    rs = shifts[0][1] if shifts else None
    return {
        "hop": f"{service}|{key}",
        "spacing": spacing,
        "stored_saturation_opens_since_10_08_18": stored_opens,
        "stored_regime_shifts": stored_shifts,
        "replayed_saturation_opens_since_10_08_18": sat_opens,
        "replayed_events_since_10_08_18": dict(Counter(f"{e.condition}:{e.phase}" for _, e in late)),
        "regime_shift_at": datetime.fromtimestamp(shifts[0][0], timezone.utc).isoformat() if shifts else None,
        "regime_shift_floor_ms": round(rs.floor_ms) if rs and rs.floor_ms else None,
        "regime_shift_window_ms": round(rs.window_mean_ms) if rs and rs.window_mean_ms else None,
        "regime_shift_episode_hours": round(rs.duration_s / 3600, 1) if rs and rs.duration_s else None,
        "checks": checks,
    }


def run_eval() -> list[dict]:
    return [replay(s, k, spacing) for s, k in HOPS for spacing in ("even", "packed")]


def main() -> int:
    reports = run_eval()
    print(json.dumps(reports, indent=2))
    return 0 if all(all(r["checks"].values()) for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
