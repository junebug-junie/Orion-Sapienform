#!/usr/bin/env python3
"""Read-only calibration harness for orion-heartbeat's N-trajectory ensemble
dissipation mechanism (per-trajectory relaxation targets, spread-gated decay,
bus_synaptic-driven two-site reheat).

Design doc: docs/superpowers/specs/2026-07-28-precision-weighted-attention-
organ-and-heartbeat-discrimination-design.md.

Mechanism validated this session via throwaway spikes (not committed) before
this script existed:

1. Per-trajectory relaxation target: each of the N trajectories damps toward
   its OWN seed-derived initial per-site state, not a shared fixed target --
   preserves inter-trajectory diversity through the meaningful part of any
   relaxation curve instead of collapsing all trajectories to clones.
2. Spread-gated decay: the decay probability for the *next* tick is a
   decreasing function of the ensemble's current cross-trajectory spread --
   disagreement (something unresolved) suppresses decay; agreement (settled)
   allows full decay.
3. Two-site entangling reheat, NOT one-site: a single-site unitary is
   PROVABLY incapable of changing entanglement entropy across any cut (basic
   invariance theorem -- confirmed the hard way this session, a one-site
   "reheat" spike produced a null result for exactly this reason). Reheat
   must use the same two-site entangling gate family absorb() itself uses.
4. Reheat strength is driven by REAL live `bus_synaptic` data
   (orion_bus_synapse FalkorDB graph's gap_zscore edges), not synthetic RNG
   -- confirmed live this session: 421 real edges, 368 reliable (count>5),
   avg|gap_zscore|~1.0-1.1 at query time. Uses the RAW mean(|gap_zscore|)
   (no calm-floor subtraction) -- the calm floor
   (sqrt(2/pi), orion/substrate/prediction_error.py:31) that's a BUG to
   subtract out for the anomaly-detection use case is exactly the "always
   some real ambient jitter" signal this mechanism needs, read the opposite
   way.

Two replay modes:

  synthetic: a busy/quiet/busy pattern (same shape used to develop the
    mechanism) -- useful for quick sanity checks and regression after any
    parameter change, cheap and reproducible.
  grammar: replays REAL historical `grammar_events` rows (Postgres,
    orion-sql-writer's durable store, confirmed this session: 1.6M rows
    spanning 5 real days) through the ensemble in real chronological order,
    with decay/reheat opportunities driven by REAL wall-clock gaps between
    events, not a fixed synthetic tick count. This is the mode that actually
    answers Missing Question 9 (does the tuning hold up against real organ
    silence durations, not an invented pattern) and should be re-run any
    time the mechanism or its tuning changes, or periodically as more real
    history accumulates.

No writes, no events, no flag/config changes. Read-only against Postgres
(grammar_events) and FalkorDB (orion_bus_synapse), both via the same
connection conventions already established by measure_ast_hot_reducer.py /
services/orion-heartbeat's own code.

Environment: this script needs quimb (a services/orion-heartbeat-specific
dependency, not installed repo-wide) plus psycopg2 and redis clients. DO NOT
`pip install` these into the shared repo venv (`./venv`) -- that's an
undocumented, unreproducible side effect on a venv other sessions/agents
share (confirmed live 2026-07-28: this is exactly what happened developing
this script, and it should not be repeated). Use a dedicated venv scoped to
this script instead:

  python3 -m venv .harness-venv
  .harness-venv/bin/pip install quimb==1.14.0 psycopg2-binary redis

Usage (from repo root, with that dedicated venv):
  PYTHONPATH=.:services/orion-heartbeat .harness-venv/bin/python3 \\
    scripts/analysis/measure_heartbeat_ensemble_calibration.py --mode synthetic
  PYTHONPATH=.:services/orion-heartbeat .harness-venv/bin/python3 \\
    scripts/analysis/measure_heartbeat_ensemble_calibration.py --mode grammar --window-hours 24

  # For a long grammar-mode replay, detach from the current shell entirely --
  # confirmed live 2026-07-28: a run backgrounded only via an interactive
  # tool session's own job control died silently with zero output when that
  # session was disrupted (a concurrent mesh redeploy), losing all progress.
  # `setsid` gives the process its own session so it survives the launching
  # shell exiting or being disrupted; PROGRESS_PATH (below) gives a live,
  # append-only record to inspect/resume from even if it does die.
  setsid nohup .harness-venv/bin/python3 \\
    scripts/analysis/measure_heartbeat_ensemble_calibration.py --mode grammar --window-hours 24 \\
    > /tmp/heartbeat-ensemble-calibration/run.log 2>&1 < /dev/null &
  tail -f /tmp/heartbeat-ensemble-calibration/progress.log

  # Alternative: run inside the live orion-heartbeat container instead (has
  # real quimb==1.14.0 already, and real network access to Postgres/FalkorDB
  # over their internal hostnames/ports -- no --falkordb-port override
  # needed there, only when running from the host against the externally
  # mapped port):
  docker cp scripts/analysis/measure_heartbeat_ensemble_calibration.py \\
    orion-athena-heartbeat:/tmp/calibrate.py
  docker exec -e PYTHONPATH=/app orion-athena-heartbeat python3 /tmp/calibrate.py --mode grammar
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("measure_heartbeat_ensemble_calibration")

OUTPUT_DIR = Path("/tmp/heartbeat-ensemble-calibration")
REPORT_PATH = OUTPUT_DIR / "report.md"
PROGRESS_PATH = OUTPUT_DIR / "progress.log"
PROGRESS_EVERY_N_EVENTS = 500  # append-only checkpoint cadence, not tuned -- cheap enough to
# flush this often (one line of text), and frequent enough that a crash mid-run (confirmed
# live 2026-07-28: a session disruption killed a run with zero output anywhere) leaves a
# diagnosable trail instead of total silence.

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
DEFAULT_FALKORDB_HOST = "orion-athena-falkordb"
DEFAULT_FALKORDB_PORT = 6379
_BUS_SYNAPTIC_ZSCORE_SATURATION = 3.0  # mirrors orion/substrate/prediction_error.py


def _import_heartbeat_substrate():
    """Deferred import: heartbeat's app package isn't a top-level repo
    import, and quimb is a service-specific dependency not installed
    repo-wide. Import here (not at module load) so --help works even
    without quimb installed."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "orion-heartbeat"))
    from app.substrate.mps_state import HeartbeatSubstrate, _ONE_SITE_GENERATORS, _TWO_SITE_GENERATORS
    from app.substrate.routing import (
        ORGAN_SITE_MAP,
        N_SITES,
        BOND_DIM,
        BOUNDARY_BULK_CUT,
        SiteAssignment,
        UnroutableAtomTypeError,
        UnroutableOrganError,
        route_atom,
    )
    return {
        "HeartbeatSubstrate": HeartbeatSubstrate,
        "_ONE_SITE_GENERATORS": _ONE_SITE_GENERATORS,
        "_TWO_SITE_GENERATORS": _TWO_SITE_GENERATORS,
        "ORGAN_SITE_MAP": ORGAN_SITE_MAP,
        "N_SITES": N_SITES,
        "BOND_DIM": BOND_DIM,
        "BOUNDARY_BULK_CUT": BOUNDARY_BULK_CUT,
        "SiteAssignment": SiteAssignment,
        "UnroutableAtomTypeError": UnroutableAtomTypeError,
        "UnroutableOrganError": UnroutableOrganError,
        "route_atom": route_atom,
    }


# ---------------------------------------------------------------------------
# Real bus_synaptic query -- drives reheat strength from live ambient activity
# ---------------------------------------------------------------------------

def query_real_bus_synaptic_raw_mean_abs_z(
    host: str = DEFAULT_FALKORDB_HOST, port: int = DEFAULT_FALKORDB_PORT,
) -> float:
    """RAW mean(|gap_zscore|) across reliable orion_bus_synapse edges
    (count>5, per bus-mirror's documented cold-start floor). Deliberately
    NOT the calm-floor-corrected anomaly-detection version -- see module
    docstring point 4."""
    try:
        import redis
    except ImportError:
        logger.warning("redis client unavailable; falling back to 0.0 (no reheat)")
        return 0.0
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=5.0)
        result = r.execute_command(
            "GRAPH.QUERY", "orion_bus_synapse",
            "MATCH ()-[rel]->() WHERE rel.count > 5 RETURN avg(abs(rel.gap_zscore))",
        )
        return float(result[1][0][0])
    except Exception as exc:  # noqa: BLE001 - read-only diagnostic, must not crash the harness
        logger.warning("bus_synaptic query failed (%s); falling back to 0.0 (no reheat)", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Mechanism -- validated shape from this session's spikes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnsembleConfig:
    n_trajectories: int = 8
    gamma: float = 0.2
    base_decay_prob: float = 0.15
    decay_spread_sensitivity: float = 4.0
    reheat_strength: float = 0.08
    reheat_prob_scale: float = 0.02
    # Cadence assumption for real-time replay: how many real seconds one
    # decay/reheat "opportunity" represents. NOT independently derived --
    # matches HEARTBEAT_H1_INTERVAL_SEC's default (30s) order of magnitude
    # as a starting point (services/orion-heartbeat/app/settings.py:38);
    # Missing Question 7 in the design doc is still open on whether
    # production cadence should be tick-driven or timer-driven, so this
    # stays an explicit, documented, CLI-overridable choice.
    seconds_per_decay_tick: float = 2.0
    max_decay_ticks_per_gap: int = 300


def site_reduced_density_matrix(mps, site: int) -> np.ndarray:
    rho_tn = mps.make_reduced_density_matrix([site])
    t = rho_tn.contract()
    return np.asarray(t.to_dense([f"k{site}"], [f"b{site}"]))


class CalibrationTrajectory:
    def __init__(self, seed: int, cfg: EnsembleConfig, mods: dict) -> None:
        self.substrate = mods["HeartbeatSubstrate"](seed=seed)
        self.rng = random.Random(seed * 7919 + 1)
        phys_dim = mods["_ONE_SITE_GENERATORS"]["amplitude"].shape[0]
        D = np.diag([1.0] + [1.0 - cfg.gamma] * (phys_dim - 1)).astype("complex128")
        n_sites = mods["N_SITES"]
        ops = []
        for site in range(n_sites):
            rho0 = site_reduced_density_matrix(self.substrate._mps, site)
            eigvals, eigvecs = np.linalg.eigh(rho0)
            order = np.argsort(eigvals)[::-1]
            U = eigvecs[:, order].conj().T
            ops.append(U.conj().T @ D @ U)
        self.decay_ops = ops
        self.reheat_generator = mods["_TWO_SITE_GENERATORS"]["amplitude"]
        self.phys_dim = phys_dim
        self.n_sites = n_sites
        self.bond_dim = mods["BOND_DIM"]
        self.boundary_bulk_cut = mods["BOUNDARY_BULK_CUT"]
        self.max_possible_entropy = np.log2(self.bond_dim)

    def decay_reheat_tick(self, decay_prob: float, reheat_prob: float, cfg: EnsembleConfig) -> None:
        import scipy.linalg as sla
        for site in range(self.n_sites):
            if self.rng.random() < decay_prob:
                self.substrate._mps.gate_(self.decay_ops[site], site, contract=True)
        for left in range(self.n_sites - 1):
            if self.rng.random() < reheat_prob:
                U2 = sla.expm(1j * cfg.reheat_strength * self.reheat_generator).reshape(
                    self.phys_dim, self.phys_dim, self.phys_dim, self.phys_dim
                )
                self.substrate._mps.gate_(
                    U2, (left, left + 1), contract="swap+split", max_bond=self.bond_dim, cutoff=0.0,
                )
        self.substrate._mps.normalize()

    def absorb(self, assignment) -> None:
        self.substrate.absorb(assignment)

    def ratio(self) -> float:
        profile = self.substrate.entropy_profile()
        return max(0.0, min(1.0, profile[self.boundary_bulk_cut - 1] / self.max_possible_entropy))


def spread_gate(recent_spread: float, base_decay_prob: float, sensitivity: float) -> float:
    return base_decay_prob * float(np.exp(-sensitivity * recent_spread))


@dataclass
class TickRecord:
    label: str
    mean_ratio: float
    std_ratio: float


def run_ensemble(
    cfg: EnsembleConfig,
    mods: dict,
    reheat_prob: float,
    event_stream,  # iterable of (label, assignment_or_None, n_gap_ticks) triples
    total_events: Optional[int] = None,
) -> list[TickRecord]:
    seeds = [1000 + i for i in range(cfg.n_trajectories)]
    trajectories = [CalibrationTrajectory(seed=s, cfg=cfg, mods=mods) for s in seeds]
    records: list[TickRecord] = []
    recent_spread = 0.0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_fh = PROGRESS_PATH.open("a", encoding="utf-8")
    progress_fh.write(
        f"{datetime.now(timezone.utc).isoformat()} | start | total_events={total_events}\n"
    )
    progress_fh.flush()
    events_processed = 0

    try:
        for label, assignment, n_gap_ticks in event_stream:
            n_gap_ticks = min(n_gap_ticks, cfg.max_decay_ticks_per_gap)
            for _ in range(n_gap_ticks):
                eff_decay_prob = spread_gate(recent_spread, cfg.base_decay_prob, cfg.decay_spread_sensitivity)
                for traj in trajectories:
                    traj.decay_reheat_tick(eff_decay_prob, reheat_prob, cfg)
                ratios = [t.ratio() for t in trajectories]
                recent_spread = float(np.std(ratios))
                records.append(TickRecord(label=label, mean_ratio=float(np.mean(ratios)), std_ratio=recent_spread))
            if assignment is not None:
                for traj in trajectories:
                    traj.absorb(assignment)
                ratios = [t.ratio() for t in trajectories]
                recent_spread = float(np.std(ratios))
                records.append(TickRecord(label=label, mean_ratio=float(np.mean(ratios)), std_ratio=recent_spread))

            events_processed += 1
            if events_processed % PROGRESS_EVERY_N_EVENTS == 0:
                pct = (100.0 * events_processed / total_events) if total_events else None
                pct_str = f"{pct:.1f}%" if pct is not None else "n/a"
                progress_fh.write(
                    f"{datetime.now(timezone.utc).isoformat()} | events_processed={events_processed} "
                    f"pct={pct_str} label={label} mean_ratio={records[-1].mean_ratio:.4f} "
                    f"std_ratio={records[-1].std_ratio:.4f} ticks_so_far={len(records)}\n"
                )
                progress_fh.flush()
    finally:
        progress_fh.write(f"{datetime.now(timezone.utc).isoformat()} | done | events_processed={events_processed}\n")
        progress_fh.close()

    return records


# ---------------------------------------------------------------------------
# Synthetic mode
# ---------------------------------------------------------------------------

def synthetic_event_stream(mods: dict, rng: random.Random, busy_ticks: int, quiet_ticks: int):
    organs = list(mods["ORGAN_SITE_MAP"].keys())
    for _ in range(busy_ticks):
        organ = rng.choice(organs)
        assignment = mods["SiteAssignment"](
            site_index=mods["ORGAN_SITE_MAP"][organ],
            operator_kind=rng.choice(["amplitude", "phase", "rotation", "projection"]),
            confidence=rng.uniform(0.4, 0.9), salience=rng.uniform(0.4, 0.9), uncertainty=rng.uniform(0.1, 0.4),
        )
        yield "busy", assignment, 0
    for _ in range(quiet_ticks):
        yield "quiet", None, 1
    for _ in range(busy_ticks):
        organ = rng.choice(organs)
        assignment = mods["SiteAssignment"](
            site_index=mods["ORGAN_SITE_MAP"][organ],
            operator_kind=rng.choice(["amplitude", "phase", "rotation", "projection"]),
            confidence=rng.uniform(0.4, 0.9), salience=rng.uniform(0.4, 0.9), uncertainty=rng.uniform(0.1, 0.4),
        )
        yield "busy2", assignment, 0


# ---------------------------------------------------------------------------
# Grammar mode -- real historical replay
# ---------------------------------------------------------------------------

def fetch_real_grammar_rows(dsn: str, window_hours: float):
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT emitted_at, source_service, event_json
                FROM grammar_events
                WHERE event_kind = 'atom_emitted'
                  AND created_at > now() - (%s || ' hours')::interval
                ORDER BY emitted_at ASC
                """,
                (window_hours,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def grammar_event_stream(rows, mods: dict, cfg: EnsembleConfig):
    route_atom = mods["route_atom"]
    UnroutableOrganError = mods["UnroutableOrganError"]
    UnroutableAtomTypeError = mods["UnroutableAtomTypeError"]

    prev_ts: Optional[datetime] = None
    skipped_organ = 0
    skipped_atom_type = 0
    routed = 0

    for row in rows:
        emitted_at = row["emitted_at"]
        atom = (row["event_json"] or {}).get("atom") or {}
        try:
            assignment = route_atom(
                source_service=row["source_service"],
                atom_type=str(atom.get("atom_type") or ""),
                confidence=atom.get("confidence"),
                salience=atom.get("salience"),
                uncertainty=atom.get("uncertainty"),
            )
        except UnroutableOrganError:
            skipped_organ += 1
            continue
        except UnroutableAtomTypeError:
            skipped_atom_type += 1
            continue

        gap_ticks = 0
        if prev_ts is not None:
            gap_sec = (emitted_at - prev_ts).total_seconds()
            gap_ticks = max(0, int(gap_sec / cfg.seconds_per_decay_tick))
        prev_ts = emitted_at
        routed += 1
        yield "grammar", assignment, gap_ticks

    logger.info(
        "grammar_event_stream_summary routed=%d skipped_organ=%d skipped_atom_type=%d",
        routed, skipped_organ, skipped_atom_type,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(records: list[TickRecord]) -> str:
    lines = ["| label | n | mean(mean_ratio) | mean(std_ratio) | flatline_ticks (std<1e-6) |",
             "| --- | --- | --- | --- | --- |"]
    by_label: dict[str, list[TickRecord]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)
    for label, recs in by_label.items():
        means = [r.mean_ratio for r in recs]
        stds = [r.std_ratio for r in recs]
        flat = sum(1 for s in stds if s < 1e-6)
        lines.append(f"| {label} | {len(recs)} | {np.mean(means):.4f} | {np.mean(stds):.4f} | {flat} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["synthetic", "grammar"], default="synthetic")
    parser.add_argument("--window-hours", type=float, default=24.0, help="grammar mode: real history window")
    parser.add_argument("--busy-ticks", type=int, default=300, help="synthetic mode: busy-phase length")
    parser.add_argument("--quiet-ticks", type=int, default=500, help="synthetic mode: quiet-phase length")
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--base-decay-prob", type=float, default=0.15)
    parser.add_argument("--reheat-strength", type=float, default=0.08)
    parser.add_argument("--reheat-prob-scale", type=float, default=0.02)
    parser.add_argument("--seconds-per-decay-tick", type=float, default=2.0)
    parser.add_argument("--postgres-uri", default=os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI))
    parser.add_argument("--falkordb-host", default=os.environ.get("FALKORDB_HOST", DEFAULT_FALKORDB_HOST))
    parser.add_argument(
        "--falkordb-port", type=int,
        default=int(os.environ.get("FALKORDB_PORT", DEFAULT_FALKORDB_PORT)),
        help="6379 when run inside the docker network (container's internal port); "
             "6380 when run from the host against orion-athena-falkordb's externally "
             "mapped port -- confirmed live this session these are NOT interchangeable, "
             "a plain (non-FalkorDB) Redis instance also answers on host:6379.",
    )
    args = parser.parse_args(argv)

    cfg = EnsembleConfig(
        n_trajectories=args.n_trajectories, gamma=args.gamma, base_decay_prob=args.base_decay_prob,
        reheat_strength=args.reheat_strength, reheat_prob_scale=args.reheat_prob_scale,
        seconds_per_decay_tick=args.seconds_per_decay_tick,
    )

    mods = _import_heartbeat_substrate()
    real_z = query_real_bus_synaptic_raw_mean_abs_z(host=args.falkordb_host, port=args.falkordb_port)
    real_signal = min(1.0, real_z / _BUS_SYNAPTIC_ZSCORE_SATURATION)
    reheat_prob = cfg.reheat_prob_scale * real_signal
    logger.info("real_bus_synaptic_raw_mean_abs_z=%.4f -> reheat_prob=%.4f", real_z, reheat_prob)

    rng = random.Random(42)
    total_events: Optional[int] = None
    if args.mode == "synthetic":
        stream = synthetic_event_stream(mods, rng, args.busy_ticks, args.quiet_ticks)
        total_events = 2 * args.busy_ticks + args.quiet_ticks
    else:
        rows = fetch_real_grammar_rows(args.postgres_uri, args.window_hours)
        logger.info("fetched %d real grammar_events rows over %.1fh window", len(rows), args.window_hours)
        total_events = len(rows)
        stream = grammar_event_stream(rows, mods, cfg)

    records = run_ensemble(cfg, mods, reheat_prob, stream, total_events=total_events)
    table = summarize(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = (
        f"# Heartbeat ensemble calibration -- mode={args.mode}\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Config: {cfg}\n\n"
        f"real_bus_synaptic_raw_mean_abs_z={real_z:.4f} -> reheat_prob={reheat_prob:.4f}\n\n"
        f"{table}\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nartifact: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
