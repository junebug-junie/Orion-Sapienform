#!/usr/bin/env python3
"""Replay live grammar windows through a fresh production ensemble.

Pre-reg: docs/research/preregistration/2026-09-20-heartbeat-neighbor-links.md

Run inside orion-athena-heartbeat (has quimb). Do not pip-install into the
shared repo venv.

  docker cp services/orion-heartbeat/app/substrate/neighbor_links.py \\
    orion-athena-heartbeat:/app/app/substrate/neighbor_links.py
  docker cp scripts/analysis/measure_heartbeat_neighbor_links.py \\
    orion-athena-heartbeat:/tmp/neighbor_links_run.py
  docker exec -e PYTHONPATH=/app orion-athena-heartbeat \\
    python3 /tmp/neighbor_links_run.py --grammar-json /tmp/grammar_atoms_2h.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("heartbeat_neighbor_links")

OUTPUT_DIR = Path("/tmp/heartbeat-neighbor-links-v2")
PRE_REG = "docs/research/preregistration/2026-09-20-heartbeat-neighbor-links-v2.md"
MI_TIMEOUT_SEC = 2.0


def _import_runtime():
    if Path("/app/app/substrate").is_dir():
        sys.path.insert(0, "/app")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "orion-heartbeat"))
    from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate
    from app.substrate.routing import UnroutableAtomTypeError, UnroutableOrganError, route_atom
    from app.substrate.neighbor_links import (
        PAIRS,
        V0_ORGANS,
        WINDOW_SEC_V2,
        classify_window_firing,
        decide_probe_v2,
        score_pair,
    )

    return {
        "EnsembleConfig": EnsembleConfig,
        "EnsembleSubstrate": EnsembleSubstrate,
        "UnroutableAtomTypeError": UnroutableAtomTypeError,
        "UnroutableOrganError": UnroutableOrganError,
        "route_atom": route_atom,
        "PAIRS": PAIRS,
        "V0_ORGANS": V0_ORGANS,
        "WINDOW_SEC_V2": WINDOW_SEC_V2,
        "classify_window_firing": classify_window_firing,
        "decide_probe_v2": decide_probe_v2,
        "score_pair": score_pair,
    }


def _parse_emitted_at(value: Any):
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_grammar_rows(path: Path) -> list[dict[str, Any]]:
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


def route_rows(rows, mods) -> tuple[list[tuple[Any, str, Any]], dict[str, int]]:
    stats = {"routed": 0, "skipped_organ": 0, "skipped_atom_type": 0}
    out = []
    for row in rows:
        source = row["source_service"]
        atom = (row["event_json"] or {}).get("atom") or {}
        try:
            assignment = mods["route_atom"](
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
        stats["routed"] += 1
        out.append((row["emitted_at"], source, assignment))
    return out, stats


def von_neumann(rho) -> float:
    arr = np.asarray(rho)
    if arr.ndim == 4:
        d = arr.shape[0] * arr.shape[1]
        arr = arr.reshape(d, d)
    elif arr.ndim != 2:
        raise RuntimeError(f"unexpected rho ndim={arr.ndim} shape={arr.shape}")
    ev = np.linalg.eigvalsh((arr + arr.conj().T) / 2.0)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return 0.0
    return float(-np.sum(ev * np.log2(ev)))


def two_site_mi(substrate, site_a: int, site_b: int) -> float:
    mps = substrate._mps
    t0 = time.perf_counter()
    rho_a = mps.partial_trace_exact([site_a])
    rho_b = mps.partial_trace_exact([site_b])
    rho_ab = mps.partial_trace_exact([site_a, site_b])
    dt = time.perf_counter() - t0
    if dt > MI_TIMEOUT_SEC:
        raise TimeoutError(f"2-seat MI took {dt:.2f}s > {MI_TIMEOUT_SEC}")
    return von_neumann(rho_a) + von_neumann(rho_b) - von_neumann(rho_ab)


def mean_pair_mi(ensemble, site_a: int, site_b: int) -> float:
    values = [two_site_mi(traj, site_a, site_b) for traj in ensemble.trajectories]
    return float(sum(values) / len(values))


def bin_windows(routed, window_sec: float) -> dict[int, list]:
    buckets: dict[int, list] = defaultdict(list)
    for ts, source, assignment in routed:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        key = int(ts.timestamp() // window_sec)
        buckets[key].append((ts, source, assignment))
    return dict(buckets)


def replay_window(assignments, mods):
    cfg = mods["EnsembleConfig"]()
    ensemble = mods["EnsembleSubstrate"](config=cfg, base_seed=1000)
    for assignment in assignments:
        ensemble.absorb(assignment)
    return ensemble


def other_count(counts: dict[str, int], organ_a: str, organ_b: str, v0: tuple[str, ...]) -> int:
    return sum(counts.get(name, 0) for name in v0 if name not in (organ_a, organ_b))


def evaluate_pairs(window_rows: list[dict[str, Any]], mods) -> list:
    scores = []
    for organ_a, organ_b, site_a, site_b in mods["PAIRS"]:
        counts_a = [w["counts"].get(organ_a, 0) for w in window_rows]
        counts_b = [w["counts"].get(organ_b, 0) for w in window_rows]
        counts_other = [other_count(w["counts"], organ_a, organ_b, mods["V0_ORGANS"]) for w in window_rows]
        i_co: list[float] = []
        i_el: list[float] = []
        for w, n_a, n_b, n_o in zip(window_rows, counts_a, counts_b, counts_other):
            cell = mods["classify_window_firing"](n_a, n_b, n_o)
            key = f"{site_a}:{site_b}"
            if cell == "cofire":
                i_co.append(w["mi"][key])
            elif cell == "elsewhere":
                i_el.append(w["mi"][key])
        score = mods["score_pair"](
            organ_a=organ_a,
            organ_b=organ_b,
            site_a=site_a,
            site_b=site_b,
            i_cofire=i_co,
            i_elsewhere=i_el,
        )
        scores.append({"score": score})
    return scores


def run_windows(routed, mods, window_sec: float) -> tuple[list[dict[str, Any]], list]:
    buckets = bin_windows(routed, window_sec)
    window_rows: list[dict[str, Any]] = []
    keys = sorted(buckets)
    n = len(keys)
    for i, key in enumerate(keys, start=1):
        events = buckets[key]
        counts: dict[str, int] = defaultdict(int)
        assignments = []
        for _ts, source, assignment in events:
            counts[source] += 1
            assignments.append(assignment)
        ensemble = replay_window(assignments, mods)
        mi = {}
        for _oa, _ob, site_a, site_b in mods["PAIRS"]:
            mi[f"{site_a}:{site_b}"] = mean_pair_mi(ensemble, site_a, site_b)
        window_rows.append({"key": key, "n": len(events), "counts": dict(counts), "mi": mi})
        if i % 50 == 0 or i == n:
            logger.info("window %d/%d sec=%.0f atoms=%d", i, n, window_sec, len(events))
    scored = evaluate_pairs(window_rows, mods)
    return window_rows, scored


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=float, default=2.0)
    parser.add_argument("--grammar-json", required=True)
    args = parser.parse_args(argv)

    mods = _import_runtime()
    raw_rows = load_grammar_rows(Path(args.grammar_json))
    routed, stats = route_rows(raw_rows, mods)
    logger.info(
        "raw=%d routed=%d skipped_organ=%d skipped_type=%d",
        len(raw_rows),
        len(routed),
        stats["skipped_organ"],
        stats["skipped_atom_type"],
    )

    used_sec = mods["WINDOW_SEC_V2"]
    window_rows, scored = run_windows(routed, mods, used_sec)

    pair_scores = [item["score"] for item in scored]
    decided = mods["decide_probe_v2"](n_windows=len(window_rows), pairs=pair_scores)

    payload = {
        "pre_reg": PRE_REG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": args.window_hours,
        "window_sec": used_sec,
        "split": "firing_v2",
        "raw_rows": len(raw_rows),
        "routed": len(routed),
        "n_windows": len(window_rows),
        "skipped_organ": stats["skipped_organ"],
        "skipped_atom_type": stats["skipped_atom_type"],
        "pairs": [
            {
                "organ_a": item["score"].organ_a,
                "organ_b": item["score"].organ_b,
                "n_cofire": item["score"].n_cofire,
                "n_elsewhere": item["score"].n_elsewhere,
                "mean_i_cofire": item["score"].mean_i_cofire,
                "mean_i_elsewhere": item["score"].mean_i_elsewhere,
                "delta": item["score"].delta,
                "holds": item["score"].holds,
                "unverified": item["score"].unverified,
                "reason": item["score"].reason,
            }
            for item in scored
        ],
        "decision": {
            "thermometer": decided.thermometer,
            "relational": decided.relational,
            "mixed": decided.mixed,
            "reason": decided.reason,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "windows.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUTPUT_DIR / "windows.jsonl").open("w", encoding="utf-8") as fh:
        for row in window_rows:
            fh.write(json.dumps(row) + "\n")
    lines = [
        "# Heartbeat neighbor links v2",
        "",
        f"Pre-reg: `{PRE_REG}`",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        f"- window: {args.window_hours}h, raw {len(raw_rows)}, routed {len(routed)}, "
        f"{len(window_rows)} bins of {used_sec:.0f}s, quiet-means-quiet split",
        "",
    ]
    for item in scored:
        s = item["score"]
        lines.append(f"## {s.organ_a} — {s.organ_b} (seats {s.site_a}–{s.site_b})")
        lines.append("")
        lines.append(
            f"co-fire n={s.n_cofire} I={s.mean_i_cofire:.4f}; "
            f"elsewhere n={s.n_elsewhere} I={s.mean_i_elsewhere:.4f}; "
            f"delta={s.delta:.4f}"
        )
        lines.append("")
        lines.append(s.reason)
        lines.append("")
    lines.extend(["## Decision", "", f"**{decided.reason}**", ""])
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"artifact: {OUTPUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
