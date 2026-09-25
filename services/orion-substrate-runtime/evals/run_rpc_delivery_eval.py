"""Replay live rpc-health data through the RPC delivery bridge's reducer.

Two fixtures, both captured 2026-09-25 from production. Every row is real; in the
first one, 46 probe timeouts were re-attributed to a different hop key (below).

- ``fixtures/rpc_health_obs_2026-09-25.jsonl.gz``: 2 h 03 min (03:54-05:58 UTC)
  of per-producer, per-hop success/timeout counts, recovered from
  orion-equilibrium-service's ``transport_baseline_obs`` log lines (one line per
  hop per folded ``RpcHealthSnapshotV1``; nothing else persists the snapshots).
  1,074 producer snapshots, 14 producer instances. Latency fields dropped.
  cortex-exec's current-turn probe did not carry its own hop label yet, so its
  timeouts were attributed from ``grammar_events`` (``rpc_transport_timeout``
  atoms with a 3.0 s deadline on ``LLMGatewayService``): 46 of 49 were moved to
  ``orion:exec:request:LLMGatewayService#current_turn_probe`` in the nearest
  following cortex-exec/background snapshot, which is what the labelled
  producer publishes after this patch. The probe's *successes* could not be
  separated and stay on the unlabelled hop (this makes the unlabelled hop's
  ratio slightly lower than it will be live). 3 probe timeouts were not
  matched and stay unlabelled (this makes it slightly higher). Net effect: the
  "shipped" zero-fraction is an optimistic bound until the cortex-exec label is
  deployed; resting at zero live is UNVERIFIED until then.
- ``fixtures/rpc_health_wire_2026-09-25.jsonl.gz``: 93 raw payloads captured
  off ``orion:rpc_health:snapshot`` (05:56-06:00 UTC), unmodified. Used to prove
  the reducer reads the real wire shape.

Run: ``python services/orion-substrate-runtime/evals/run_rpc_delivery_eval.py``
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.rpc_delivery import (  # noqa: E402
    RpcDeliveryConfig,
    RpcDeliveryWindow,
)

OBS_FIXTURE = HERE / "fixtures" / "rpc_health_obs_2026-09-25.jsonl.gz"
WIRE_FIXTURE = HERE / "fixtures" / "rpc_health_wire_2026-09-25.jsonl.gz"
TICK_S = 30.0


def load_obs() -> list[dict]:
    out = []
    with gzip.open(OBS_FIXTURE, "rt") as f:
        for line in f:
            row = json.loads(line)
            out.append(
                {
                    "service": row["service"],
                    "instance": row["instance"],
                    "window_end": row["window_end"],
                    "channel_latency": {
                        hop: {"success_count": s, "timeout_count": t}
                        for hop, (s, t) in row["hops"].items()
                    },
                }
            )
    return out


def load_wire() -> list[dict]:
    with gzip.open(WIRE_FIXTURE, "rt") as f:
        return [json.loads(line) for line in f]


def unlabel_probe(snapshots: list[dict]) -> list[dict]:
    """Merge ``...#current_turn_probe`` back into its channel's hop: the shape
    producers published before the probe carried its own label."""
    out = []
    for s in snapshots:
        lat: dict[str, dict] = {}
        for hop, st in s["channel_latency"].items():
            key = hop.split("#", 1)[0] if hop.endswith("#current_turn_probe") else hop
            acc = lat.setdefault(key, {"success_count": 0, "timeout_count": 0})
            acc["success_count"] += st["success_count"]
            acc["timeout_count"] += st["timeout_count"]
        out.append({**s, "channel_latency": lat})
    return out


def replay(snapshots: list[dict], config: RpcDeliveryConfig) -> list[dict]:
    """Fold snapshots in window_end order and take a reading every TICK_S,
    exactly as the worker's listener + tick do."""
    snaps = sorted(snapshots, key=lambda s: s["window_end"])
    t0 = datetime.fromisoformat(snaps[0]["window_end"])
    t_end = datetime.fromisoformat(snaps[-1]["window_end"])
    win = RpcDeliveryWindow(config)
    readings = []
    i = 0
    tick = t0 + timedelta(seconds=TICK_S)
    while tick <= t_end + timedelta(seconds=TICK_S):
        while i < len(snaps) and datetime.fromisoformat(snaps[i]["window_end"]) <= tick:
            win.fold(snaps[i])
            i += 1
        r = win.reading(tick.timestamp())
        readings.append({"at": tick.isoformat(), "reading": r.as_dict() if r else None})
        tick += timedelta(seconds=TICK_S)
    return readings


def summarize(readings: list[dict]) -> dict:
    measured = [r["reading"] for r in readings if r["reading"] is not None]
    values = [m["pressure"] for m in measured]
    nonzero = [m for m in measured if m["pressure"] > 0.0]
    return {
        "ticks": len(readings),
        "unmeasured_ticks": len(readings) - len(measured),
        "zero_ticks": sum(1 for v in values if v == 0.0),
        "zero_fraction": round(sum(1 for v in values if v == 0.0) / len(values), 3) if values else None,
        "p50": round(statistics.median(values), 4) if values else None,
        "p90": round(sorted(values)[int(0.9 * (len(values) - 1))], 4) if values else None,
        "max": max(values) if values else None,
        "nonzero_worst_hops": dict(Counter(m["worst_hop"] for m in nonzero).most_common()),
        "producers_max": max((m["producers"] for m in measured), default=0),
        "calls_per_window_median": statistics.median(m["total_calls"] for m in measured) if measured else 0,
    }


def run() -> dict:
    obs = load_obs()
    result = {
        "shipped": summarize(replay(obs, RpcDeliveryConfig())),
        # What the field would read WITHOUT the cortex-exec probe label:
        # probe timeouts folded back into the unlabelled LLMGatewayService hop.
        "probe_unlabelled": summarize(replay(unlabel_probe(obs), RpcDeliveryConfig())),
        # Without the min_timeouts=2 hysteresis (one lone timeout counts).
        "no_hysteresis": summarize(replay(obs, RpcDeliveryConfig(min_timeouts=1))),
        "floor_sensitivity": {
            n0: summarize(replay(obs, RpcDeliveryConfig(min_denominator=n0, min_timeouts=1)))["max"]
            for n0 in (1, 5, 10, 20)
        },
        "wire": summarize(replay(load_wire(), RpcDeliveryConfig())),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
