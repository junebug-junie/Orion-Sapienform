#!/usr/bin/env python3
"""PREVIEW: the trained-network field demon on per-service heartbeat seat activity. Read-only.

Spec: docs/superpowers/specs/2026-11-24-heartbeat-seats-field-demon-design.md (rev 3), "Replay".

Organs: the 5 heartbeat seats (services/orion-heartbeat/app/substrate/routing.py, after #3873).
Each seat's value in a 5-minute bucket is the summed gate level of the atoms the heartbeat would
absorb (``route_atom``): a "presence" atom adds its fixed level, so for an event-driven seat this
is its count times 0.5; a "measured" atom adds its measured level; a "skip" atom adds nothing.
Raw counts would be useless for the clockwork seats (biometrics and bus emit at a fixed cadence,
so their count is flat).

Values become time-of-day residuals (``whole_body_structure.tod_residuals``), then 3 levels per
seat.

Sources (issue #3886). ``grammar_events`` keeps about 3 days. orion-heartbeat now also writes the
same per-seat value for every closed 5-minute bucket to ``heartbeat_seat_rollup_5m`` (one row per
seat per bucket, kept). The replay reads the rollup first and uses ``grammar_events`` only for
buckets the rollup has no rows for. Each bucket is labelled with its source:

- ``rollup``: all 5 seats from complete rollup rows;
- ``rollup_partial``: the rollup has rows, but the heartbeat was not listening for the whole bucket
  (or a seat row is missing). Read as missing, not as zero, and not refilled from grammar_events;
- ``grammar_events``: no rollup rows; recomputed from the ledger;
- ``none``: neither source covers the bucket. Missing.

Reads both tables with ``docker exec orion-athena-sql-db psql``. Writes results.json to --out and
prints a summary. Aggregates only; no content leaves the database.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# The chains are tiny (bond dimension <= 16). Multithreaded BLAS on matrices this small spends
# ~30x more CPU on thread handoffs than on arithmetic (measured: 39 s wall, 19 min CPU).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "services" / "orion-heartbeat"))

from app.substrate import routing  # noqa: E402
from orion.substrate import born_machine as bm  # noqa: E402
from orion.substrate import demon  # noqa: E402
from orion.substrate import whole_body_structure as wbs  # noqa: E402

LABEL = "PREVIEW (per-service seats; not the stable-mode organs)"
BUCKET = 300
LAGS = (1, 3, 6, 12)
N_NULL = 20
NULL_GUARD = 24  # 2 h: above the largest lag (1 h). The 3 h default does not fit 25% of ~2 days.
TRAIN_FRACTION = 2 / 3
PIPELINE = [[0, 2, 3, 4], [1]]  # hub, exec, bus, harness | biometrics (#3795)
DB_CONTAINER = "orion-athena-sql-db"
ROLLUP_TABLE = "heartbeat_seat_rollup_5m"
SOURCES = ("rollup", "rollup_partial", "grammar_events", "none")
# Clockwork seats: biometrics (site 1) and bus (site 3) tick every ~11-16 s.
CLOCKWORK_SITES = (1, 3)


def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(
        ["docker", "exec", DB_CONTAINER, "psql", "-U", "postgres", "-d", "conjourney", "-v", "ON_ERROR_STOP=1",
         "-AtF", "\t", "-c", "set default_transaction_read_only = on; " + sql],
        check=True, capture_output=True, text=True,
    ).stdout
    # The first output line is the SET command's tag (-At still prints it).
    return [line.split("\t") for line in out.splitlines() if line and line != "SET"]


def _routed_roles() -> tuple[list[str], dict[str, list[str]]]:
    """Services whose default rule is not skip (all roles pulled), and for the clockwork services
    only the listed roles that are not skip."""
    open_services, listed = [], {}
    for svc, rule in routing.SERVICE_DEFAULT_RULES.items():
        if rule.kind != "skip":
            open_services.append(svc)
        else:
            listed[svc] = [r for (s, r), rr in routing.ROLE_RULES.items() if s == svc and rr.kind != "skip"]
    return open_services, listed


def grammar_sql(where_extra: str = "") -> str:
    """The ledger query: one row per routed-service atom, bucketed by emitted_at."""
    open_services, listed = _routed_roles()
    clauses = [f"source_service in ({','.join(repr(s) for s in open_services)})"]
    for svc, roles in listed.items():
        clauses.append(f"(source_service = '{svc}' and event_json->'atom'->>'semantic_role' in ({','.join(repr(r) for r in roles)}))")
    return (
        "select floor(extract(epoch from emitted_at) / 300)::bigint, source_service, "
        "event_json->'atom'->>'semantic_role', event_json->'atom'->>'atom_type', "
        "coalesce(event_json->'atom'->>'salience', ''), coalesce(event_json->'atom'->>'confidence', ''), "
        "coalesce(event_json->'atom'->>'uncertainty', '') "
        "from grammar_events where event_kind = 'atom_emitted' and (" + " or ".join(clauses) + ")" + where_extra
    )


def aggregate_grammar_rows(rows) -> tuple[dict[tuple[int, int], tuple[float, int]], dict]:
    """(bucket, site) -> (summed route_atom level, atom count), from ledger rows shaped like
    ``grammar_sql``. Levels are summed with math.fsum (exactly rounded, so order-independent):
    the heartbeat's rollup uses the same sum, so the two match exactly, not just closely."""
    levels: dict[tuple[int, int], list[float]] = {}
    stats: dict = {"atoms_read": 0, "skipped": {}, "unroutable": 0}
    for b, svc, role, atype, sal, conf, unc in rows:
        stats["atoms_read"] += 1
        atom = {"atom_type": atype, "semantic_role": role}
        for key, val in (("salience", sal), ("confidence", conf), ("uncertainty", unc)):
            if val != "":
                atom[key] = float(val)
        try:
            a = routing.route_atom(source_service=svc, atom=atom)
        except routing.SkippedInputError as exc:
            stats["skipped"][exc.reason] = stats["skipped"].get(exc.reason, 0) + 1
            continue
        except (routing.UnroutableOrganError, routing.UnroutableAtomTypeError, ValueError, TypeError):
            stats["unroutable"] += 1
            continue
        levels.setdefault((int(b), a.site_index), []).append(a.level)
    return {k: (math.fsum(v), len(v)) for k, v in levels.items()}, stats


def _rollup_rows() -> list[list[str]]:
    """(bucket, node, site_index, level_sum, atom_count, partial) per stored rollup row. Empty when
    the table does not exist yet (before the first deploy)."""
    if _psql(f"select to_regclass('{ROLLUP_TABLE}') is not null") != [["t"]]:
        return []
    return _psql(
        "select floor(extract(epoch from bucket_start) / 300)::bigint, node, site_index, "
        f"level_sum, atom_count, partial from {ROLLUP_TABLE}"
    )


def build_series(
    grammar: dict[tuple[int, int], tuple[float, int]],
    rollup_rows,
    n_seats: int = len(routing.SEAT_ORDER),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Merge the two sources onto one 5-minute grid. Returns level, count, bucket index, the source
    label per bucket (see module docstring), and stats. Missing buckets are NaN, never 0."""
    rollup: dict[int, dict[int, tuple[float, int, bool]]] = {}
    nodes: dict[tuple[int, int], set[str]] = {}
    for b, node, site, level_sum, atom_count, partial in rollup_rows:
        b, site = int(b), int(site)
        nodes.setdefault((b, site), set()).add(str(node))
        rollup.setdefault(b, {})[site] = (float(level_sum), int(atom_count), str(partial) in ("t", "true", "True", "1"))
    clash = {k: v for k, v in nodes.items() if len(v) > 1}
    if clash:
        k = min(clash)
        raise SystemExit(f"demon_network_replay: {len(clash)} rollup bucket/seat cells come from more than one "
                         f"node (first: bucket {k[0]} site {k[1]}: {sorted(clash[k])}); refusing to sum them")
    g_buckets = sorted({b for b, _ in grammar})
    all_buckets = g_buckets + sorted(rollup)
    if not all_buckets:
        raise SystemExit("demon_network_replay: no seat rows in the rollup or grammar_events")
    b0, b1 = min(all_buckets), max(all_buckets)
    g0, g1 = (g_buckets[0], g_buckets[-1]) if g_buckets else (None, None)
    n = b1 - b0 + 1
    level = np.full((n, n_seats), np.nan)
    count = np.zeros((n, n_seats))
    source = np.full(n, "none", dtype=object)
    for t in range(n):
        b = b0 + t
        r = rollup.get(b)
        if r is not None:
            complete = len(r) == n_seats and not any(p for _, _, p in r.values())
            source[t] = "rollup" if complete else "rollup_partial"
            for site, (lv, c, _p) in r.items():
                count[t, site] = c
                if complete:
                    level[t, site] = lv
        elif g0 is not None and g0 <= b <= g1:
            source[t] = "grammar_events"
            for site in range(n_seats):
                lv, c = grammar.get((b, site), (0.0, 0))
                level[t, site], count[t, site] = lv, c
            # The first and last ledger buckets are partial (retention cut, still filling).
            if b in (g0, g1):
                level[t] = np.nan
    # A bucket with no biometrics or no bus atom at all is an outage of the ledger or the
    # emitter, not calm: mark the whole bucket missing (both sources).
    outage = np.zeros(n, dtype=bool)
    for site in CLOCKWORK_SITES:
        outage |= count[:, site] == 0
    outage &= (source == "rollup") | (source == "grammar_events")
    level[outage] = np.nan
    idx = np.arange(b0, b1 + 1)
    stats = {
        "buckets": int(n),
        "outage_buckets": int(outage.sum()),
        "source_buckets": {k: int((source == k).sum()) for k in SOURCES},
        "source_spans": source_spans(source, idx),
    }
    return level, count, idx, source, stats


def source_spans(source: np.ndarray, idx: np.ndarray) -> list[dict]:
    """Run-length list of (first bucket, last bucket, source), as UTC times."""
    spans: list[dict] = []
    for t in range(len(source)):
        if spans and spans[-1]["source"] == source[t]:
            spans[-1]["last"] = _iso(idx[t])
            spans[-1]["buckets"] += 1
        else:
            spans.append({"source": str(source[t]), "first": _iso(idx[t]), "last": _iso(idx[t]), "buckets": 1})
    return spans


def _iso(bucket: int) -> str:
    return datetime.fromtimestamp(int(bucket) * BUCKET, timezone.utc).isoformat()


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    grammar, gstats = aggregate_grammar_rows(_psql(grammar_sql()))
    rows = _rollup_rows()
    level, count, idx, source, stats = build_series(grammar, rows)
    stats = {**gstats, "rollup_rows_read": len(rows), **stats}
    stats["counts_per_bucket_median"] = dict(zip(routing.SEAT_ORDER, np.median(count, axis=0).tolist()))
    stats["count_cv"] = dict(zip(routing.SEAT_ORDER, (count.std(0) / np.maximum(count.mean(0), 1e-9)).round(3).tolist()))
    return level, count, idx, source, stats


def longest_finite_run(x: np.ndarray) -> tuple[int, int]:
    ok = np.all(np.isfinite(x), axis=1)
    best, start = (0, 0), None
    for t in range(len(ok) + 1):
        if t < len(ok) and ok[t]:
            start = t if start is None else start
        elif start is not None:
            if t - start > best[1] - best[0]:
                best = (start, t)
            start = None
    return best


def _r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3837)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    names = list(routing.SEAT_ORDER)
    raw, count, idx, source, stats = load()
    res = wbs.tod_residuals(raw, idx, min_days=1)
    lo, hi = longest_finite_run(res)
    if hi - lo < 2:
        raise SystemExit("demon_network_replay: no run of complete buckets to score")
    x, raw_run, idx_run = res[lo:hi], raw[lo:hi], idx[lo:hi]
    run_sources = Counter(str(v) for v in source[lo:hi])
    first = datetime.fromtimestamp(int(idx_run[0]) * BUCKET, timezone.utc).isoformat()
    last = datetime.fromtimestamp(int(idx_run[-1]) * BUCKET, timezone.utc).isoformat()
    refusal = demon.refusal_reason(raw_run, names)
    if refusal:
        print(f"{LABEL}\nREFUSED by the hygiene gate: {refusal}. Nothing scored.")
        return 2
    out: dict = {
        "label": LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "organs": names,
        "seed": args.seed,
        "load": stats,
        "scored_span": {"first": first, "last": last, "rows": int(len(x)),
                        "hours": round(len(x) / 12, 1),
                        # Which table each scored row came from (issue #3886).
                        "rows_by_source": dict(run_sources),
                        "source_spans": source_spans(source[lo:hi], idx_run)},
        "refusal_reason_raw": refusal,
        "lags_steps": list(LAGS),
    }
    train_rows = int(TRAIN_FRACTION * len(x))

    # 1. Skill by k: one refit through the rolling-errors interface (window = training rows).
    errs = demon.network_rolling_errors(x, window=train_rows, step=len(x), lags=LAGS)
    skill = {}
    level_best = demon._best_of(errs.clock, errs.persistence)
    for name, base in (("clock", errs.clock), ("persistence", errs.persistence), ("linear", errs.linear),
                       ("best_of_clock_persistence", level_best), ("best_of_all_three", errs.baseline)):
        s = demon.skill_spread(errs.whole, base, block=12, n_boot=2000, rng=rng)
        skill[name] = {"skill": _r(s.skill), "bits_per_row": _r(s.bits), "p10": _r(s.p10), "p90": _r(s.p90),
                       "p_real": _r(s.p_real, 3)}
    lin_vs_base = demon.skill_spread(errs.linear, level_best, block=12, n_boot=2000, rng=rng)
    out["skill"] = {
        "scored_rows": int(errs.whole.shape[0]),
        "network_bits_per_row": _r(errs.whole.mean(0)),
        "clock_bits_per_row": _r(errs.clock.mean(0)),
        "persistence_bits_per_row": _r(errs.persistence.mean(0)),
        "linear_bits_per_row": _r(errs.linear.mean(0)),
        "network_vs": skill,
        "linear_vs_best_of_clock_persistence": {"skill": _r(lin_vs_base.skill), "p10": _r(lin_vs_base.p10),
                                          "p90": _r(lin_vs_base.p90), "p_real": _r(lin_vs_base.p_real, 3)},
    }

    # 2-3. Cut cost on all 52 partitions, network and linear, against the time-shift null.
    lat = demon.lattice_vs_null(x, lags=LAGS, train_fraction=TRAIN_FRACTION, n_null=args.n_null,
                                guard=NULL_GUARD, rng=rng)
    obs = lat.observed
    p_real, p_real_lin = lat.p_real, lat.p_real_linear
    p_mip, p_mip_lin = lat.p_real_mip("network"), lat.p_real_mip("linear")
    null_mean, null_sd = lat.null_network_bits.mean(0), lat.null_network_bits.std(0, ddof=1)

    def label(p):
        return " | ".join("+".join(names[i].replace("orion-", "") for i in part) for part in p)

    parts = []
    for pi, p in enumerate(obs.partitions):
        parts.append({
            "partition": label(p), "normalization_bits": round(float(obs.normalization[pi]), 4),
            "network_bits": _r(obs.network_bits[pi]), "network_null_mean": _r(null_mean[pi]),
            "network_null_sd": _r(null_sd[pi]), "network_p_real": _r(p_real[pi], 3),
            "linear_bits": _r(obs.linear_bits[pi]), "linear_null_mean": _r(lat.null_linear_bits.mean(0)[pi]),
            "linear_p_real": _r(p_real_lin[pi], 3),
        })
    mips = []
    for c, k in enumerate(LAGS):
        mn, ml = obs.mip("network", c), obs.mip("linear", c)
        mips.append({
            "k": k, "network_mip": label(obs.partitions[mn]), "network_mip_bits": round(float(obs.network_bits[mn, c]), 4),
            "network_mip_null_mean": round(float(null_mean[mn, c]), 4), "network_mip_null_sd": round(float(null_sd[mn, c]), 4),
            "network_mip_p_real_at_that_partition": round(float(p_real[mn, c]), 3),
            "network_mip_score_bits_per_bit": round(demon.mip_index(obs.network_bits, obs.normalization, c)[1], 4),
            "network_mip_p_real_vs_null_mips": round(float(p_mip[c]), 3),
            "linear_mip": label(obs.partitions[ml]), "linear_mip_bits": round(float(obs.linear_bits[ml, c]), 4),
            "linear_mip_p_real_at_that_partition": round(float(p_real_lin[ml, c]), 3),
            "linear_mip_p_real_vs_null_mips": round(float(p_mip_lin[c]), 3),
            "agree": mn == ml,
            "linear_mip_rank_in_network_order": int(np.argsort(np.argsort(np.where(obs.normalization > 0, obs.network_bits[:, c] / np.where(obs.normalization > 0, obs.normalization, 1), np.inf)))[ml]) + 1,
        })
    pipe_key = frozenset(map(frozenset, PIPELINE))
    pipe = next(i for i, p in enumerate(obs.partitions) if frozenset(map(frozenset, p)) == pipe_key)
    out["lattice"] = {
        "note": "scored on the first rows of the same length a shifted null copy keeps (75%)",
        "rows": int(len(x) - int(demon.DEFAULT_NULL_SPAN_FRACTION * len(x))),
        "train_rows": int(TRAIN_FRACTION * (len(x) - int(demon.DEFAULT_NULL_SPAN_FRACTION * len(x)))),
        "test_rows": obs.n_test, "n_null": args.n_null, "null_guard_steps": NULL_GUARD,
        "network_whole_bits_per_row": _r(obs.whole_bits), "linear_whole_bits_per_row": _r(obs.linear_whole_bits),
        "mip_by_k": mips,
        "pipeline_block": parts[pipe],
        "partitions": parts,
    }

    # Entanglement against classical mutual information, per contiguous cut of each trained chain.
    fc = lat.forecaster
    site_names = [f"{n.replace('orion-', '')}:{w}" for n in names for w in ("now", "later")]
    tangling = []
    for k in LAGS:
        ci = demon.cut_information(fc.machines[k], rng=rng)
        sel = fc.selections[k]
        tangling.append({
            "k": k, "bond_dim": sel.bond_dim, "sweeps": sel.n_sweeps,
            "val_ll_by_bond_dim": {str(d): round(v, 4) for d, v in sel.val_ll.items()},
            "cuts": [{"after": site_names[j], "entanglement_bits": round(float(ci.entanglement_bits[j]), 4),
                      "mutual_information_bits": round(float(ci.mutual_information_bits[j]), 4),
                      "gap_bits": round(float(ci.gap_bits[j]), 4)} for j in range(len(site_names) - 1)],
        })
    out["tangling"] = tangling
    out["runtime_s"] = round(time.time() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    print(LABEL)
    print(f"sources over the scored run: {dict(run_sources)}")
    print(f"scored {first} .. {last}: {len(x)} rows; skill: train {train_rows}, test {errs.whole.shape[0]}; "
          f"lattice: test {obs.n_test}; hygiene gate passed")
    for c, k in enumerate(LAGS):
        s = skill["linear"]
        print(f"k={k:2d}: network {errs.whole.mean(0)[c]:.3f} bits, linear {errs.linear.mean(0)[c]:.3f}, "
              f"clock {errs.clock.mean(0)[c]:.3f}, persistence {errs.persistence.mean(0)[c]:.3f}; "
              f"vs linear skill {s['skill'][c]:+.3f} [{s['p10'][c]:+.3f}, {s['p90'][c]:+.3f}] P(real) {s['p_real'][c]:.2f}")
    for m in mips:
        print(f"k={m['k']:2d} MIP network: {m['network_mip']} ({m['network_mip_bits']:+.3f} bits, null "
              f"{m['network_mip_null_mean']:+.3f}±{m['network_mip_null_sd']:.3f}, P(real vs null MIPs) {m['network_mip_p_real_vs_null_mips']:.2f}); "
              f"linear: {m['linear_mip']}; agree={m['agree']}")
    print("pipeline block:", json.dumps(parts[pipe]))
    for tg in tangling:
        print(f"k={tg['k']} D={tg['bond_dim']}: ent " + " ".join(f"{c['entanglement_bits']:.2f}" for c in tg["cuts"]))
        print("        mi  " + " ".join(f"{c['mutual_information_bits']:.2f}" for c in tg["cuts"]))
    print(f"runtime {out['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
