"""Replay real grammar atoms through a heartbeat build and report what H1 did.

Used for the 2026-11-24 heartbeat overhaul (#3838) to compare the shipped build
before the overhaul with the overhauled build, on the same atoms. Uses the
service's own routing, EnsembleSubstrate and compute_h1_ensemble -- not a
re-implementation -- so a result here is a result about that build's code.

It works against both APIs:
  - before: route_atom(source_service, atom_type, confidence, salience,
    uncertainty) and ensemble.decay_reheat_tick(reheat_prob);
  - after: route_atom(source_service, atom) and ensemble.decay_tick().

Input: CSV from the live ledger, one row per atom_emitted event:
    emitted_at_epoch,source_service,atom_type,semantic_role,confidence,salience,uncertainty

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "copy (
      select extract(epoch from emitted_at), source_service,
             event_json->'atom'->>'atom_type', event_json->'atom'->>'semantic_role',
             event_json->'atom'->>'confidence', event_json->'atom'->>'salience',
             event_json->'atom'->>'uncertainty'
      from grammar_events where event_kind='atom_emitted'
        and emitted_at >= '2026-11-24T18:00Z' and emitted_at < '2026-11-24T21:00Z'
      order by emitted_at) to stdout with csv" > grammar_day.csv

The simulated clock follows emitted_at: one dissipation tick every
HEARTBEAT_DECAY_INTERVAL_SEC (2 s), one H1 every 30 s, as in service.py.
After the last atom it runs --quiet-sec of pure silence (dissipation only) and
reports whether the substrate relaxed. That is the 07-28 thermalization check:
a substrate that cannot come down from a busy state has no rest state.

Run each build in its own process (both name their package ``app``):

    OMP_NUM_THREADS=2 python services/orion-heartbeat/evals/replay_overhaul_before_after.py \\
        --service-root <path-to-a-heartbeat-checkout> --csv grammar_day.csv --out after_day.json
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DECAY_INTERVAL_SEC = 2.0
H1_INTERVAL_SEC = 30.0


def _import(service_root: Path):
    repo_root = service_root.parents[1]
    for p in (str(repo_root), str(service_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # The repo root must resolve `orion`; when service_root is a snapshot
    # outside a checkout, fall back to this file's own repo for `orion`.
    here_repo = Path(__file__).resolve().parents[3]
    if str(here_repo) not in sys.path:
        sys.path.append(str(here_repo))
    from app.substrate import routing
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import compute_h1_ensemble

    return routing, EnsembleConfig, EnsembleSubstrate, compute_h1_ensemble


def _f(value: str):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[idx], 4)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 10:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    try:
        return round(statistics.correlation(rx, ry), 4)
    except statistics.StatisticsError:
        return None


def replay(service_root: Path, rows: list[list[str]], *, reheat_prob: float | None, warmup_sec: float, quiet_sec: float) -> dict:
    routing, EnsembleConfig, EnsembleSubstrate, compute_h1_ensemble = _import(service_root)
    new_api = "atom" in inspect.signature(routing.route_atom).parameters
    if not new_api and reheat_prob is None:
        raise SystemExit("--reheat-prob is required for a before-overhaul build (it ran with a live reheat)")
    ensemble = EnsembleSubstrate(config=EnsembleConfig(), base_seed=42)

    def dissipate() -> None:
        if hasattr(ensemble, "decay_tick"):
            ensemble.decay_tick()
        else:
            ensemble.decay_reheat_tick(reheat_prob)

    t0 = float(rows[0][0])
    next_decay = t0 + DECAY_INTERVAL_SEC
    next_h1 = t0 + H1_INTERVAL_SEC
    ticks: list[dict] = []
    window_atoms = 0
    window_gated = 0
    window_seats: Counter = Counter()
    seat_fires: Counter = Counter()
    skipped: Counter = Counter()

    def h1_tick(at: float, phase: str) -> None:
        nonlocal window_atoms, window_gated, window_seats
        h1 = compute_h1_ensemble(ensemble)
        ticks.append(
            {
                "t": at - t0,
                "phase": phase,
                "verdict": h1.verdict,
                "mean_ratio": h1.mean_ratio,
                "std_ratio": h1.std_ratio,
                "bulk": h1.bulk_penetration_depth,
                "atoms": window_atoms,
                "gated": window_gated,
                "seats": dict(window_seats),
            }
        )
        window_atoms = 0
        window_gated = 0
        window_seats = Counter()

    def advance(until: float, phase: str) -> None:
        nonlocal next_decay, next_h1
        while min(next_decay, next_h1) <= until:
            if next_decay <= next_h1:
                dissipate()
                next_decay += DECAY_INTERVAL_SEC
            else:
                h1_tick(next_h1, phase)
                next_h1 += H1_INTERVAL_SEC

    for row in rows:
        ts = float(row[0])
        source, atom_type, role = row[1], row[2], row[3]
        conf, sal, unc = _f(row[4]), _f(row[5]), _f(row[6])
        advance(ts, "replay")
        try:
            if new_api:
                assignment = routing.route_atom(
                    source_service=source,
                    atom={
                        "atom_type": atom_type,
                        "semantic_role": role,
                        "confidence": conf,
                        "salience": sal,
                        "uncertainty": unc,
                    },
                )
            else:
                assignment = routing.route_atom(
                    source_service=source,
                    atom_type=atom_type,
                    confidence=conf,
                    salience=sal,
                    uncertainty=unc,
                )
        except Exception as exc:  # noqa: BLE001 - count every kind of skip
            skipped[f"{source}:{type(exc).__name__}"] += 1
            continue
        ensemble.absorb(assignment)
        seat = routing.SITE_ORGAN_MAP.get(assignment.site_index, str(assignment.site_index))
        seat_fires[seat] += 1
        window_seats[seat] += 1
        window_atoms += 1
        level = getattr(assignment, "level", None)
        if level is None or level > 0:
            window_gated += 1

    t_end = float(rows[-1][0])
    advance(t_end, "replay")
    advance(t_end + quiet_sec, "quiet")

    replay_ticks = [t for t in ticks if t["phase"] == "replay" and t["t"] >= warmup_sec]
    quiet_ticks = [t for t in ticks if t["phase"] == "quiet"]
    last_busy = [t for t in ticks if t["phase"] == "replay"][-10:]

    def summary(ts: list[dict]) -> dict:
        verdicts = Counter(t["verdict"] for t in ts)
        n = max(1, len(ts))
        return {
            "n_h1_ticks": len(ts),
            "verdict_share": {k: round(v / n, 4) for k, v in sorted(verdicts.items())},
            "mean_ratio_p5_p50_p95": [_pct([t["mean_ratio"] for t in ts], q) for q in (0.05, 0.5, 0.95)],
            "std_ratio_p5_p50_p95": [_pct([t["std_ratio"] for t in ts], q) for q in (0.05, 0.5, 0.95)],
            "bulk_p5_p50_p95": [_pct([t["bulk"] for t in ts], q) for q in (0.05, 0.5, 0.95)],
        }

    atoms = [float(t["atoms"]) for t in replay_ticks]
    coupling = {
        metric: _spearman(atoms, [t[metric] for t in replay_ticks])
        for metric in ("mean_ratio", "std_ratio", "bulk")
    }
    quiet = {
        "quiet_sec": quiet_sec,
        "mean_ratio_last_busy": round(statistics.fmean(t["mean_ratio"] for t in last_busy), 4) if last_busy else None,
        "mean_ratio_quiet_end": round(statistics.fmean(t["mean_ratio"] for t in quiet_ticks[-5:]), 4) if quiet_ticks else None,
        "std_ratio_quiet_end": round(statistics.fmean(t["std_ratio"] for t in quiet_ticks[-5:]), 4) if quiet_ticks else None,
        "verdict_quiet_end": quiet_ticks[-1]["verdict"] if quiet_ticks else None,
        "mean_ratio_trace": [round(t["mean_ratio"], 4) for t in quiet_ticks[:: max(1, len(quiet_ticks) // 10)]],
    }
    return {
        "service_root": str(service_root),
        "api": "after" if new_api else "before",
        "reheat_prob": reheat_prob if not new_api else None,
        "rows": len(rows),
        "absorbed": sum(seat_fires.values()),
        "seat_fires": dict(seat_fires),
        "skipped": dict(skipped),
        "window_hours": round((t_end - t0) / 3600.0, 2),
        "warmup_sec": warmup_sec,
        "replay": summary(replay_ticks),
        "coupling_spearman_vs_atoms_per_tick": coupling,
        "quiet": quiet,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--service-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--reheat-prob",
        type=float,
        default=None,
        help="before-build only, required there (live value 2026-11-24: ~0.0035)",
    )
    parser.add_argument("--warmup-sec", type=float, default=1800.0)
    parser.add_argument("--quiet-sec", type=float, default=1800.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0]]
    result = replay(
        Path(args.service_root).resolve(),
        rows,
        reheat_prob=args.reheat_prob,
        warmup_sec=args.warmup_sec,
        quiet_sec=args.quiet_sec,
    )
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("api", "absorbed", "replay", "coupling_spearman_vs_atoms_per_tick", "quiet")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
