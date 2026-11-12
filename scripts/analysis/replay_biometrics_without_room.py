#!/usr/bin/env python3
"""Replay node:substrate.biometrics over real data: with the room keys (v2) vs without (v3).

Read-only. Rebuilds each node's pressure hints from ``orion_biometrics_summary`` (the same
fields ``services/orion-biometrics/app/grammar_emit.py`` turns into grammar atoms, which
``orion/substrate/biometrics_loop/grammar_extract.py`` turns into hints), then diffs them
at every stored ``node:substrate.biometrics`` tick in
``substrate_node_prediction_error_history``.

- v2 = the old mean over every hint key (room keys included).
- v3 = the production ``biometrics_prediction_error`` (room keys left out).

The v2 replay is checked against the stored values to show the reconstruction is faithful.

Usage:
  python3 scripts/analysis/replay_biometrics_without_room.py --fetch --days 7 --out-dir /tmp/x
  python3 scripts/analysis/replay_biometrics_without_room.py --summary-tsv s.tsv --pe-tsv p.tsv
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from orion.schemas.biometrics_projection import (  # noqa: E402
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.substrate.biometrics_loop.room_hints import is_room_hint  # noqa: E402
from orion.substrate.prediction_error import _THRESHOLD, biometrics_prediction_error  # noqa: E402

PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc"]

# summary field -> hint key (grammar_emit.py atom salience sources).
PRESSURE_TO_HINT = {
    "gpu_util": "gpu",
    "mem": "memory_pressure",
    "thermal": "thermal_pressure",
    "disk": "disk_pressure",
    "power": "power_pressure",
    "disk_capacity": "disk_capacity_pressure",
    "fan": "fan_pressure",
}


def _ts(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("+00"):
        raw = raw + ":00"
    dt = datetime.fromisoformat(raw.replace(" ", "T"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch(out_dir: Path, days: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summ, pe = out_dir / "summary.tsv", out_dir / "pe.tsv"
    q_pe = (
        "copy (select observed_at, value from substrate_node_prediction_error_history "
        "where node_id='node:substrate.biometrics' "
        f"and observed_at > now() - interval '{int(days)} days' order by observed_at) to stdout"
    )
    q_summ = (
        "copy (select timestamp, node, pressures, composites from orion_biometrics_summary "
        f"where timestamp::timestamptz > now() - interval '{int(days)} days 1 hour' "
        "order by timestamp) to stdout"
    )
    summ.write_text(subprocess.run(PSQL + [q_summ], check=True, capture_output=True, text=True).stdout)
    pe.write_text(subprocess.run(PSQL + [q_pe], check=True, capture_output=True, text=True).stdout)
    return summ, pe


def hints_from_summary(pressures: dict, composites: dict) -> dict[str, float]:
    hints = {
        "strain": float(composites.get("strain", 0.5)),
        "stability": float(composites.get("stability", 0.5)),
    }
    for src, key in PRESSURE_TO_HINT.items():
        hints[key] = float(pressures.get(src, 0.0))
    for key, val in pressures.items():
        if key.startswith("cabinet_"):
            hints[key] = float(val)
    return hints


def load_summaries(path: Path) -> dict[str, tuple[list[datetime], list[dict]]]:
    per_node: dict[str, tuple[list[datetime], list[dict]]] = {}
    merged: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or not parts[1]:
            continue
        ts, node = _ts(parts[0]), parts[1]
        pressures = json.loads(parts[2]) if parts[2] not in ("", "\\N") else {}
        composites = json.loads(parts[3]) if parts[3] not in ("", "\\N") else {}
        hints = hints_from_summary(pressures, composites)
        # The node reducer merges hints, so a key once seen stays on the node.
        # Staleness keys are 0.0 whenever the sensor board / mic report (always, live).
        state = {**merged.get(node, {}), **hints}
        if any(k.startswith("cabinet_") for k in hints):
            state.setdefault("cabinet_sensor_staleness", 0.0)
            state.setdefault("cabinet_ambient_audio_staleness", 0.0)
        merged[node] = state
        times, states = per_node.setdefault(node, ([], []))
        times.append(ts)
        states.append(dict(state))
    return per_node


def projection_at(per_node, at: datetime) -> NodeBiometricsProjectionV1:
    nodes = {}
    for node, (times, states) in per_node.items():
        i = bisect.bisect_right(times, at) - 1
        if i >= 0:
            nodes[node] = NodeBiometricsStateV1(node_id=node, last_seen_at=times[i], pressure_hints=states[i])
    return NodeBiometricsProjectionV1(projection_id="replay", generated_at=at, nodes=nodes)


def v2_error(prev: NodeBiometricsProjectionV1, curr: NodeBiometricsProjectionV1) -> tuple[float, float]:
    """Old mean over every key. Returns (error, share of summed |delta| from room keys)."""
    deltas, room = [], 0.0
    for node_id, c in curr.nodes.items():
        p = prev.nodes.get(node_id)
        if p is None:
            continue
        for key in set(p.pressure_hints) | set(c.pressure_hints):
            d = abs(float(c.pressure_hints.get(key, 0.0)) - float(p.pressure_hints.get(key, 0.0)))
            deltas.append(d)
            if is_room_hint(key):
                room += d
    total = sum(deltas)
    err = min(1.0, (total / len(deltas)) / _THRESHOLD) if deltas else 0.0
    return err, (room / total if total > 0 else 0.0)


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def _row(name: str, xs: list[float]) -> str:
    return (f"| {name} | {statistics.fmean(xs):.4f} | {statistics.median(xs):.4f} | "
            f"{_pct(xs, 0.9):.4f} | {max(xs):.4f} | {sum(1 for x in xs if x == 0.0) / len(xs):.1%} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/biometrics_without_room"))
    ap.add_argument("--summary-tsv", type=Path)
    ap.add_argument("--pe-tsv", type=Path)
    ap.add_argument("--lag-sec", type=float, default=0.0,
                    help="report-to-tick processing lag to subtract from each tick time")
    ap.add_argument("--emit-tsv", type=Path, help="write observed_at, stored, v2 replay, v3 per tick")
    args = ap.parse_args()

    if args.fetch:
        summ_path, pe_path = fetch(args.out_dir, args.days)
    else:
        summ_path, pe_path = args.summary_tsv, args.pe_tsv
    per_node = load_summaries(summ_path)
    ticks = []
    for line in pe_path.read_text().splitlines():
        a, b = line.split("\t")
        ticks.append((_ts(a), float(b)))

    # The live tick runs every ~10-20 s and each batch usually carries ONE node's new
    # report (nodes report every ~31 s). The history table keeps one tick every ~2 min.
    # So each stored value is "projection before that report" vs "after it": rebuild it
    # from the latest report at or before (tick time - lag).
    events = sorted(
        (t, node, i) for node, (times, _) in per_node.items() for i, t in enumerate(times)
    )
    event_times = [e[0] for e in events]
    lag = timedelta(seconds=args.lag_sec)
    stored, old, new, room_share, days, at_list = [], [], [], [], [], []
    for at, value in ticks:
        j = bisect.bisect_right(event_times, at - lag) - 1
        if j < 1:
            continue
        t_ev, node, i = events[j]
        if i < 1:
            continue
        curr = projection_at(per_node, t_ev)
        prev_nodes = dict(curr.nodes)
        times, states = per_node[node]
        prev_nodes[node] = NodeBiometricsStateV1(
            node_id=node, last_seen_at=times[i - 1], pressure_hints=states[i - 1]
        )
        prev = NodeBiometricsProjectionV1(projection_id="replay", generated_at=at, nodes=prev_nodes)
        e2, share = v2_error(prev, curr)
        stored.append(value)
        old.append(e2)
        new.append(biometrics_prediction_error(prev, curr))
        room_share.append(share)
        days.append(at.date().isoformat())
        at_list.append(at)

    if args.emit_tsv:
        args.emit_tsv.write_text("".join(
            f"{a.isoformat()}\t{s_:.6f}\t{o:.6f}\t{w:.6f}\n"
            for a, s_, o, w in zip(at_list, stored, old, new)
        ))

    n = len(stored)
    print(f"ticks: {n}  ({ticks[0][0].isoformat()} .. {ticks[-1][0].isoformat()})")
    keys_now = {k: len(s[-1]) for k, (_, s) in per_node.items()}
    print(f"keys per node at end: {keys_now}")
    corr = statistics.correlation(stored, old) if n > 2 else float("nan")
    print(f"replayed v2 vs stored: pearson r = {corr:.3f}, "
          f"median |diff| = {statistics.median(abs(a - b) for a, b in zip(stored, old)):.4f}, "
          f"exact (|diff| < 1e-4): {sum(1 for a, b in zip(stored, old) if abs(a - b) < 1e-4) / n:.1%}")
    print(f"pearson r(v2 replay, v3) = {statistics.correlation(old, new):.3f}")
    print(f"room keys' share of the summed change (v2): mean {statistics.fmean(room_share):.1%}, "
          f"median {statistics.median(room_share):.1%}, ticks where room > 50%: "
          f"{sum(1 for s in room_share if s > 0.5) / n:.1%}")
    print()
    print("| series | mean | median | p90 | max | exactly 0 |")
    print("|---|---|---|---|---|---|")
    print(_row("stored (live v2)", stored))
    print(_row("replay v2 (room in)", old))
    print(_row("replay v3 (room out)", new))
    # Which ticks would sit in the top 10% (the attention percentile line) under each.
    top_old = {i for i, x in enumerate(old) if x >= _pct(old, 0.9)}
    top_new = {i for i, x in enumerate(new) if x >= _pct(new, 0.9)}
    print(f"\ntop-10% ticks shared by v2 and v3: {len(top_old & top_new) / max(1, len(top_old)):.1%}")
    print("\n| day | ticks | v2 mean | v3 mean | v3/v2 |")
    print("|---|---|---|---|---|")
    for d in sorted(set(days)):
        o = [x for x, dd in zip(old, days) if dd == d]
        w = [x for x, dd in zip(new, days) if dd == d]
        print(f"| {d} | {len(o)} | {statistics.fmean(o):.4f} | {statistics.fmean(w):.4f} | "
              f"{statistics.fmean(w) / max(1e-12, statistics.fmean(o)):.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
