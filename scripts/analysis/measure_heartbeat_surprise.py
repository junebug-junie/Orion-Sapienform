#!/usr/bin/env python3
"""Replay chat sessions vs control windows; score surprise settle/jump.

Pre-reg: docs/research/preregistration/2026-09-20-heartbeat-surprise.md
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("heartbeat_surprise")

OUTPUT_DIR = Path("/tmp/heartbeat-surprise")
PRE_REG = "docs/research/preregistration/2026-09-20-heartbeat-surprise.md"
MAX_POSSIBLE_ENTROPY = math.log2(4)
SECONDS_PER_DECAY_TICK = 2.0
MAX_DECAY_TICKS_PER_GAP = 20
PROFILE_TIMEOUT_SEC = 2.0


def _import_runtime():
    if Path("/app/app/substrate").is_dir():
        sys.path.insert(0, "/app")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "orion-heartbeat"))
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.routing import UnroutableAtomTypeError, UnroutableOrganError, route_atom
    from app.substrate.surprise_probe import (
        SESSION_ATOMS,
        SNAPSHOT_EVERY,
        WARMUP_ATOMS,
        decide_surprise,
        l2,
        score_side,
        score_window,
    )

    return {
        "EnsembleConfig": EnsembleConfig,
        "EnsembleSubstrate": EnsembleSubstrate,
        "UnroutableAtomTypeError": UnroutableAtomTypeError,
        "UnroutableOrganError": UnroutableOrganError,
        "route_atom": route_atom,
        "SESSION_ATOMS": SESSION_ATOMS,
        "SNAPSHOT_EVERY": SNAPSHOT_EVERY,
        "WARMUP_ATOMS": WARMUP_ATOMS,
        "decide_surprise": decide_surprise,
        "l2": l2,
        "score_side": score_side,
        "score_window": score_window,
    }


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def load_atoms(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            {
                "emitted_at": _parse_ts(row["emitted_at"]),
                "source_service": row["source_service"],
                "event_json": row["event_json"],
            }
        )
    out.sort(key=lambda r: r["emitted_at"])
    deduped = []
    seen: set[tuple] = set()
    for row in out:
        key = (
            row["emitted_at"].isoformat(),
            row["source_service"],
            json.dumps(row["event_json"], sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def route_rows(rows, mods):
    out = []
    stats = {"routed": 0, "skipped_organ": 0, "skipped_atom_type": 0}
    for row in rows:
        atom = (row["event_json"] or {}).get("atom") or {}
        try:
            assignment = mods["route_atom"](
                source_service=row["source_service"],
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
        stats["routed"] += 1
        out.append((row["emitted_at"], assignment))
    return out, stats


def query_reheat_prob(host: str, port: int, scale: float) -> tuple[float, float]:
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


def mean_ratio_profile(ensemble) -> list[float]:
    t0 = time.perf_counter()
    profiles = [traj.entropy_profile() for traj in ensemble.trajectories]
    dt = time.perf_counter() - t0
    if dt > PROFILE_TIMEOUT_SEC:
        raise TimeoutError(f"entropy profile took {dt:.2f}s")
    n_cuts = len(profiles[0])
    mean_entropy = [float(sum(p[i] for p in profiles) / len(profiles)) for i in range(n_cuts)]
    return [e / MAX_POSSIBLE_ENTROPY for e in mean_entropy]


def in_range(routed, lo: datetime, hi: datetime):
    return [(ts, a) for ts, a in routed if lo <= ts < hi]


def replay_window(warmup, body, mods, reheat_prob: float) -> list[float]:
    cfg = mods["EnsembleConfig"]()
    ensemble = mods["EnsembleSubstrate"](config=cfg, base_seed=1000)
    warmup = warmup[-mods["WARMUP_ATOMS"] :]
    body = body[: mods["SESSION_ATOMS"]]
    prev_ts = None
    for ts, assignment in warmup:
        if prev_ts is not None:
            gap_sec = (ts - prev_ts).total_seconds()
            ticks = max(0, min(MAX_DECAY_TICKS_PER_GAP, int(gap_sec / SECONDS_PER_DECAY_TICK)))
            for _ in range(ticks):
                ensemble.decay_reheat_tick(reheat_prob)
        ensemble.absorb(assignment)
        prev_ts = ts
    surprises: list[float] = []
    prev_profile = None
    for i, (ts, assignment) in enumerate(body, start=1):
        if prev_ts is not None:
            gap_sec = (ts - prev_ts).total_seconds()
            ticks = max(0, min(MAX_DECAY_TICKS_PER_GAP, int(gap_sec / SECONDS_PER_DECAY_TICK)))
            for _ in range(ticks):
                ensemble.decay_reheat_tick(reheat_prob)
        ensemble.absorb(assignment)
        prev_ts = ts
        if i % mods["SNAPSHOT_EVERY"] == 0:
            profile = mean_ratio_profile(ensemble)
            if prev_profile is not None:
                surprises.append(mods["l2"](profile, prev_profile))
            prev_profile = profile
    return surprises


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True)
    parser.add_argument("--atoms", required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--falkordb-host", default="orion-athena-falkordb")
    parser.add_argument("--falkordb-port", type=int, default=6379)
    args = parser.parse_args(argv)

    mods = _import_runtime()
    raw_z, reheat_prob = query_reheat_prob(
        args.falkordb_host, args.falkordb_port, mods["EnsembleConfig"]().reheat_prob_scale
    )
    logger.info("reheat raw_z=%.4f reheat_prob=%.4f", raw_z, reheat_prob)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    atoms = load_atoms(Path(args.atoms))
    routed, stats = route_rows(atoms, mods)
    logger.info(
        "atom_rows=%d routed=%d skipped_organ=%d skipped_type=%d sessions=%d controls=%d",
        len(atoms),
        len(routed),
        stats["skipped_organ"],
        stats["skipped_atom_type"],
        len(meta["sessions"]),
        len(meta["controls"]),
    )

    session_scores = []
    control_scores = []
    details = []

    def run_one(kind: str, start_s: str, stop_s: str, extra: dict) -> None:
        start = _parse_ts(start_s)
        stop = _parse_ts(stop_s)
        warmup_lo = start - timedelta(seconds=600)
        warmup = in_range(routed, warmup_lo, start)
        body = in_range(routed, start, stop)
        logger.info("%s %s warmup=%d body=%d", kind, start.isoformat(), len(warmup), len(body))
        surprises = replay_window(warmup, body, mods, reheat_prob)
        scored = mods["score_window"](kind=kind, surprises=surprises)
        rec = {
            "kind": kind,
            "start": start.isoformat(),
            "stop": stop.isoformat(),
            "warmup": len(warmup),
            "body": len(body),
            "n_surprise": len(surprises),
            "scored": scored is not None,
            **extra,
        }
        if scored is not None:
            rec["slope"] = scored.slope
            rec["drop"] = scored.drop
            if kind == "session":
                session_scores.append(scored)
            else:
                control_scores.append(scored)
        details.append(rec)

    for sess in meta["sessions"]:
        run_one("session", sess["start"], sess["stop"], {"hub_atoms": sess.get("hub_atoms")})
    for ctrl in meta["controls"]:
        run_one("control", ctrl["start"], ctrl["stop"], {})

    sess_side = mods["score_side"](session_scores)
    ctrl_side = mods["score_side"](control_scores)
    decided = mods["decide_surprise"](sessions=sess_side, controls=ctrl_side)

    payload = {
        "pre_reg": PRE_REG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reheat_raw_z": raw_z,
        "reheat_prob": reheat_prob,
        "atom_rows": len(atoms),
        "routed": stats["routed"],
        "sessions": {
            "n": sess_side.n,
            "median_slope": sess_side.median_slope,
            "frac_negative": sess_side.frac_negative,
            "median_drop": sess_side.median_drop,
            "settles": sess_side.settles,
        },
        "controls": {
            "n": ctrl_side.n,
            "median_slope": ctrl_side.median_slope,
            "frac_negative": ctrl_side.frac_negative,
            "median_drop": ctrl_side.median_drop,
            "settles": ctrl_side.settles,
        },
        "decision": {
            "holds": decided.holds,
            "thermometer": decided.thermometer,
            "mixed": decided.mixed,
            "reason": decided.reason,
        },
        "windows": details,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "windows.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = (
        f"# Heartbeat surprise\n\n"
        f"Pre-reg: `{PRE_REG}`\n\n"
        f"Generated: {payload['generated_at']}\n\n"
        f"- reheat_prob={reheat_prob:.4f} (raw_z={raw_z:.4f})\n"
        f"- sessions scored={sess_side.n} median_slope={sess_side.median_slope:.4f} "
        f"frac_neg={sess_side.frac_negative:.2f} median_drop={sess_side.median_drop:.4f} "
        f"settles={sess_side.settles}\n"
        f"- controls scored={ctrl_side.n} median_slope={ctrl_side.median_slope:.4f} "
        f"frac_neg={ctrl_side.frac_negative:.2f} median_drop={ctrl_side.median_drop:.4f} "
        f"settles={ctrl_side.settles}\n\n"
        f"## Decision\n\n"
        f"**{decided.reason}**\n\n"
        f"holds={decided.holds} thermometer={decided.thermometer} mixed={decided.mixed}\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"artifact: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
