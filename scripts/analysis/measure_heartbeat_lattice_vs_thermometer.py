#!/usr/bin/env python3
"""Replay live grammar through production EnsembleSubstrate.

Pre-reg: docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer.md

Run inside orion-athena-heartbeat (has quimb). Do not pip-install into the
shared repo venv.

  docker cp scripts/analysis/measure_heartbeat_lattice_vs_thermometer.py \\
    orion-athena-heartbeat:/tmp/lattice_probe_run.py
  docker cp services/orion-heartbeat/app/substrate/lattice_probe.py \\
    orion-athena-heartbeat:/app/app/substrate/lattice_probe.py
  docker exec -e PYTHONPATH=/app -e POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney \\
    orion-athena-heartbeat python3 /tmp/lattice_probe_run.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("heartbeat_lattice_vs_thermometer")

OUTPUT_DIR = Path("/tmp/heartbeat-lattice-vs-thermometer")
MAX_POSSIBLE_ENTROPY = math.log2(4)
SECONDS_PER_DECAY_TICK = 2.0
MAX_DECAY_TICKS_PER_GAP = 20  # cap so a long quiet hole does not dominate
MAX_ATOMS = 800
KICK_N = 50
PRE_REG = "docs/research/preregistration/2026-09-19-heartbeat-lattice-vs-thermometer.md"


def _import_runtime():
    if Path("/app/app/substrate").is_dir():
        sys.path.insert(0, "/app")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "orion-heartbeat"))
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.reconstruction import compute_h1_ensemble
    from app.substrate.routing import (
        UnroutableAtomTypeError,
        UnroutableOrganError,
        route_atom,
    )
    from app.substrate.lattice_probe import (
        KICK_N as PROBE_KICK_N,
        MIN_ROUTED_ATOMS,
        decide_probe,
        score_kick,
        score_shuffle,
        shuffle_source_service,
    )
    from app.substrate.routing import SiteAssignment

    return {
        "EnsembleConfig": EnsembleConfig,
        "EnsembleSubstrate": EnsembleSubstrate,
        "compute_h1_ensemble": compute_h1_ensemble,
        "UnroutableAtomTypeError": UnroutableAtomTypeError,
        "UnroutableOrganError": UnroutableOrganError,
        "route_atom": route_atom,
        "SiteAssignment": SiteAssignment,
        "KICK_N": PROBE_KICK_N,
        "MIN_ROUTED_ATOMS": MIN_ROUTED_ATOMS,
        "decide_probe": decide_probe,
        "score_kick": score_kick,
        "score_shuffle": score_shuffle,
        "shuffle_source_service": shuffle_source_service,
    }


def _parse_emitted_at(value: Any):
    from datetime import datetime

    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_grammar_rows(path: Path) -> list[dict[str, Any]]:
    """Same columns as fetch_grammar_rows, from a JSON/JSONL dump.

    Heartbeat's production image has quimb but not psycopg2. Dump via
    sql-writer, then replay here. Scoring is unchanged.
    """
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = json.loads(raw)
    out = []
    for row in rows:
        out.append(
            {
                "emitted_at": _parse_emitted_at(row["emitted_at"]),
                "source_service": row["source_service"],
                "event_json": row["event_json"],
            }
        )
    return out


def fetch_grammar_rows(dsn: str, window_hours: float) -> list[dict[str, Any]]:
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
            return list(cur.fetchall())
    finally:
        conn.close()


def route_rows(rows, mods, *, shuffle: bool) -> tuple[list[tuple[Any, Any, int]], dict[str, int]]:
    route_atom = mods["route_atom"]
    stats = {"routed": 0, "skipped_organ": 0, "skipped_atom_type": 0}
    out: list[tuple[Any, Any, int]] = []
    prev_ts = None
    shuffle_fn = mods["shuffle_source_service"]
    for row in rows:
        source = row["source_service"]
        if shuffle:
            source = shuffle_fn(source)
        atom = (row["event_json"] or {}).get("atom") or {}
        try:
            assignment = route_atom(
                source_service=source,
                atom_type=str(atom.get("atom_type") or ""),
                confidence=atom.get("confidence"),
                salience=atom.get("salience"),
                uncertainty=atom.get("uncertainty"),
            )
        except mods["UnroutableOrganError"]:
            stats["skipped_organ"] += 1
            continue
        except mods["UnroutableAtomTypeError"]:
            stats["skipped_atom_type"] += 1
            continue
        emitted_at = row["emitted_at"]
        gap_ticks = 0
        if prev_ts is not None:
            gap_sec = (emitted_at - prev_ts).total_seconds()
            gap_ticks = max(0, min(MAX_DECAY_TICKS_PER_GAP, int(gap_sec / SECONDS_PER_DECAY_TICK)))
        prev_ts = emitted_at
        stats["routed"] += 1
        out.append((emitted_at, assignment, gap_ticks))
    return out, stats


def mean_ratio_profile(ensemble, mods) -> list[float]:
    profiles = [traj.entropy_profile() for traj in ensemble.trajectories]
    n_cuts = len(profiles[0])
    mean_entropy = [
        float(sum(p[i] for p in profiles) / len(profiles)) for i in range(n_cuts)
    ]
    return [e / MAX_POSSIBLE_ENTROPY for e in mean_entropy]


def replay(assignments_with_gaps, mods, *, reheat_prob: float, base_seed: int = 1000):
    cfg = mods["EnsembleConfig"]()
    ensemble = mods["EnsembleSubstrate"](config=cfg, base_seed=base_seed)
    n = len(assignments_with_gaps)
    for i, (_ts, assignment, gap_ticks) in enumerate(assignments_with_gaps):
        for _ in range(gap_ticks):
            ensemble.decay_reheat_tick(reheat_prob)
        ensemble.absorb(assignment)
        if (i + 1) % 100 == 0:
            logger.info("replay %d/%d tick_count=%d", i + 1, n, ensemble.tick_count())
    h1 = mods["compute_h1_ensemble"](ensemble)
    return ensemble, h1, mean_ratio_profile(ensemble, mods)


def query_reheat_prob(mods, host: str, port: int, scale: float) -> tuple[float, float]:
    """Return (raw_z, reheat_prob). 0,0 if unreachable — recorded, not guessed."""
    try:
        import redis
        r = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=5.0)
        result = r.execute_command(
            "GRAPH.QUERY",
            "orion_bus_synapse",
            "MATCH ()-[rel]->() WHERE rel.count > 5 RETURN avg(abs(rel.gap_zscore))",
        )
        raw_z = float(result[1][0][0])
        return raw_z, scale * min(1.0, raw_z / 3.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("bus_synaptic unavailable (%s); reheat_prob=0.0", exc)
        return 0.0, 0.0


def apply_kick(ensemble, mods, n: int) -> None:
    assignment = mods["route_atom"](
        source_service="orion-hub",
        atom_type="signal",
        confidence=1.0,
        salience=1.0,
        uncertainty=0.0,
    )
    if assignment.site_index != 0:
        raise RuntimeError(f"kick must land on site 0, got {assignment.site_index}")
    for _ in range(n):
        ensemble.absorb(assignment)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=float, default=2.0)
    parser.add_argument("--max-atoms", type=int, default=MAX_ATOMS)
    parser.add_argument(
        "--postgres-uri",
        default=os.environ.get(
            "POSTGRES_URI", "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
        ),
    )
    parser.add_argument("--falkordb-host", default=os.environ.get("FALKORDB_HOST", "orion-athena-falkordb"))
    parser.add_argument("--falkordb-port", type=int, default=int(os.environ.get("FALKORDB_PORT", "6379")))
    parser.add_argument(
        "--grammar-json",
        default="",
        help="JSON/JSONL dump of grammar_events (use when psycopg2 is unavailable)",
    )
    args = parser.parse_args(argv)

    mods = _import_runtime()
    raw_z, reheat_prob = query_reheat_prob(
        mods, args.falkordb_host, args.falkordb_port, mods["EnsembleConfig"]().reheat_prob_scale
    )
    logger.info("reheat raw_z=%.4f reheat_prob=%.4f", raw_z, reheat_prob)

    if args.grammar_json:
        raw_rows = load_grammar_rows(Path(args.grammar_json))
        logger.info("loaded grammar dump %s rows=%d", args.grammar_json, len(raw_rows))
    else:
        raw_rows = fetch_grammar_rows(args.postgres_uri, args.window_hours)
    as_is, stats_a = route_rows(raw_rows, mods, shuffle=False)
    cap_fired = False
    if len(as_is) > args.max_atoms:
        as_is = as_is[-args.max_atoms :]
        cap_fired = True
    shuffled, stats_b = route_rows(raw_rows, mods, shuffle=True)
    if cap_fired:
        shuffled = shuffled[-args.max_atoms :]

    logger.info(
        "window=%.1fh raw=%d routed_a=%d cap_fired=%s skipped_organ=%d skipped_type=%d",
        args.window_hours,
        len(raw_rows),
        len(as_is),
        cap_fired,
        stats_a["skipped_organ"],
        stats_a["skipped_atom_type"],
    )

    ensemble_a, h1_a, profile_a = replay(as_is, mods, reheat_prob=reheat_prob)
    _, h1_b, profile_b = replay(shuffled, mods, reheat_prob=reheat_prob)

    profile_before = mean_ratio_profile(ensemble_a, mods)
    apply_kick(ensemble_a, mods, mods["KICK_N"])
    h1_kick = mods["compute_h1_ensemble"](ensemble_a)
    profile_after = mean_ratio_profile(ensemble_a, mods)

    shuffle_score = mods["score_shuffle"](
        profile_a=profile_a,
        profile_b=profile_b,
        mean_ratio_a=h1_a.mean_ratio,
        mean_ratio_b=h1_b.mean_ratio,
        std_ratio_a=h1_a.std_ratio,
        std_ratio_b=h1_b.std_ratio,
        verdict_a=h1_a.verdict,
        verdict_b=h1_b.verdict,
    )
    kick_score = mods["score_kick"](profile_before=profile_before, profile_after=profile_after)
    decided = mods["decide_probe"](n_routed=len(as_is), shuffle=shuffle_score, kick=kick_score)

    payload = {
        "pre_reg": PRE_REG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.window_hours,
        "raw_rows": len(raw_rows),
        "routed": len(as_is),
        "cap_fired": cap_fired,
        "max_atoms": args.max_atoms,
        "skipped_organ": stats_a["skipped_organ"],
        "skipped_atom_type": stats_a["skipped_atom_type"],
        "reheat_raw_z": raw_z,
        "reheat_prob": reheat_prob,
        "arm_a": {
            "mean_ratio": h1_a.mean_ratio,
            "std_ratio": h1_a.std_ratio,
            "bulk": h1_a.bulk_penetration_depth,
            "verdict": h1_a.verdict,
            "profile": profile_a,
        },
        "arm_b": {
            "mean_ratio": h1_b.mean_ratio,
            "std_ratio": h1_b.std_ratio,
            "bulk": h1_b.bulk_penetration_depth,
            "verdict": h1_b.verdict,
            "profile": profile_b,
        },
        "kick": {
            "mean_ratio": h1_kick.mean_ratio,
            "std_ratio": h1_kick.std_ratio,
            "verdict": h1_kick.verdict,
            "profile_before": profile_before,
            "profile_after": profile_after,
            "near": kick_score.near,
            "far": kick_score.far,
            "smear": None if math.isinf(kick_score.smear) else kick_score.smear,
            "smeared": kick_score.smeared,
        },
        "shuffle": {
            "d_shuffle": shuffle_score.d_shuffle,
            "rel_shuffle": shuffle_score.rel_shuffle,
            "null": shuffle_score.null,
        },
        "decision": {
            "thermometer": decided.thermometer,
            "reason": decided.reason,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "profiles.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = (
        f"# Heartbeat lattice vs thermometer\n\n"
        f"Pre-reg: `{PRE_REG}`\n\n"
        f"Generated: {payload['generated_at']}\n\n"
        f"- window: {args.window_hours}h, raw atoms {len(raw_rows)}, routed {len(as_is)}, "
        f"800-cap {'yes' if cap_fired else 'no'}\n"
        f"- reheat_prob={reheat_prob:.4f} (raw_z={raw_z:.4f})\n\n"
        f"## Arm A (as-is)\n\n"
        f"mean={h1_a.mean_ratio:.4f} std={h1_a.std_ratio:.4f} bulk={h1_a.bulk_penetration_depth:.4f} "
        f"verdict={h1_a.verdict}\n\n"
        f"## Arm B (cyclic organ shuffle)\n\n"
        f"mean={h1_b.mean_ratio:.4f} std={h1_b.std_ratio:.4f} bulk={h1_b.bulk_penetration_depth:.4f} "
        f"verdict={h1_b.verdict}\n\n"
        f"rel_shuffle={shuffle_score.rel_shuffle:.4f} null={shuffle_score.null}\n\n"
        f"## Arm C (kick site 0 × {mods['KICK_N']})\n\n"
        f"near={kick_score.near:.4f} far={kick_score.far:.4f} smear={kick_score.smear} "
        f"smeared={kick_score.smeared}\n\n"
        f"## Decision\n\n"
        f"**{decided.reason}**\n\n"
        f"thermometer={decided.thermometer}\n"
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"artifact: {OUTPUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
