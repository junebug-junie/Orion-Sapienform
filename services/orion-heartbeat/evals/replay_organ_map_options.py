"""Replay real grammar atoms through the real heartbeat ensemble under different
organ->site maps and report the H1 verdict distribution for each.

Question it answers: if heartbeat routed the grammar producers it currently
drops (llm-gateway, sql-writer, vision-frame-router, ...), would the H1 verdict
mean something different? Uses the service's own EnsembleSubstrate and
compute_h1_ensemble -- not a re-implementation -- so a result here is a result
about the shipped code.

Input: a CSV exported from the live ledger (no DB driver needed), one row per
atom_emitted event, columns:
    emitted_at_epoch,source_service,atom_type,confidence,salience,uncertainty

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "copy (
      select extract(epoch from emitted_at), source_service,
             event_json->'atom'->>'atom_type', event_json->'atom'->>'confidence',
             event_json->'atom'->>'salience', event_json->'atom'->>'uncertainty'
      from grammar_events where event_json->>'event_kind'='atom_emitted'
        and created_at between now()-interval '4 hours' and now()-interval '1 hours'
      order by emitted_at) to stdout with csv" > grammar.csv

Simulated wall clock follows emitted_at: decay/reheat every
HEARTBEAT_DECAY_REHEAT_INTERVAL_SEC, H1 every HEARTBEAT_H1_INTERVAL_SEC, same
as service.py. Reheat probability is held at --reheat-prob for every option
(the bus_synaptic history it is derived from is not stored), so options differ
only in routing.

Options:
  current       -- routing.ORGAN_SITE_MAP as shipped (5 boundary organs).
  fold_boundary -- every other orion:grammar:event producer from
                   orion/bus/channels.yaml folded round-robin onto boundary
                   sites 0-4 (the "derive from the catalog" option that keeps
                   N_SITES fixed).
  extend_bulk   -- the same extra producers placed round-robin on bulk sites
                   5-8 (no atom ever lands there today; site 9 has no
                   right-neighbour so absorb() refuses it).

Slow on purpose (real quimb absorb, ~120 ms/atom at N=8): ~45 min per 3 h of
traffic per option. Run options as separate processes.

    OMP_NUM_THREADS=2 python services/orion-heartbeat/evals/replay_organ_map_options.py \
        --csv grammar.csv --option current --out current.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parents[1]
for _p in (str(_REPO_ROOT), str(_SERVICE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.settings import settings  # noqa: E402
from app.substrate.ensemble import EnsembleConfig, EnsembleSubstrate  # noqa: E402
from app.substrate.reconstruction import compute_h1_ensemble  # noqa: E402
from app.substrate.routing import (  # noqa: E402
    ATOM_TYPE_OPERATOR_KIND,
    BOUNDARY_SITES,
    BULK_SITES,
    ORGAN_SITE_MAP,
    SiteAssignment,
    catalog_grammar_producers,
)

SELF = "orion-heartbeat"


def build_site_map(option: str) -> dict[str, int]:
    site_map = dict(ORGAN_SITE_MAP)
    if option == "current":
        return site_map
    extras = sorted(p for p in catalog_grammar_producers() if p not in site_map and p != SELF)
    # absorb() needs a right-neighbour, so the last bulk site (9) cannot take
    # an organ at all -- itself evidence the bulk was never meant to be fed.
    targets = BOUNDARY_SITES if option == "fold_boundary" else BULK_SITES[:-1]
    for i, producer in enumerate(extras):
        site_map[producer] = targets[i % len(targets)]
    return site_map


def _f(value: str, default: float) -> float:
    try:
        return float(value) if value not in ("", None) else default
    except ValueError:
        return default


def replay(rows: list[list[str]], option: str, *, reheat_prob: float, warmup_sec: float) -> dict:
    site_map = build_site_map(option)
    ensemble = EnsembleSubstrate(
        config=EnsembleConfig(
            n_trajectories=settings.n_trajectories,
            gamma=settings.decay_gamma,
            base_decay_prob=settings.base_decay_prob,
            decay_spread_sensitivity=settings.decay_spread_sensitivity,
            reheat_strength=settings.reheat_strength,
            reheat_prob_scale=settings.reheat_prob_scale,
        ),
        base_seed=settings.substrate_seed,
    )
    t0 = float(rows[0][0])
    next_decay = t0 + settings.decay_reheat_interval_sec
    next_h1 = t0 + settings.h1_interval_sec
    verdicts: list[dict] = []
    absorbed: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    def advance(until: float) -> None:
        nonlocal next_decay, next_h1
        while min(next_decay, next_h1) <= until:
            if next_decay <= next_h1:
                ensemble.decay_reheat_tick(reheat_prob)
                next_decay += settings.decay_reheat_interval_sec
            else:
                h1 = compute_h1_ensemble(ensemble)
                if next_h1 - t0 >= warmup_sec:
                    verdicts.append(
                        {
                            "t": round(next_h1 - t0, 1),
                            "verdict": h1.verdict,
                            "mean_ratio": round(h1.mean_ratio, 4),
                            "std_ratio": round(h1.std_ratio, 4),
                            "bulk": round(h1.bulk_penetration_depth, 4),
                        }
                    )
                next_h1 += settings.h1_interval_sec

    for row in rows:
        ts, source, atom_type = float(row[0]), row[1], row[2]
        advance(ts)
        if source not in site_map or atom_type not in ATOM_TYPE_OPERATOR_KIND:
            dropped[source] += 1
            continue
        ensemble.absorb(
            SiteAssignment(
                site_index=site_map[source],
                operator_kind=ATOM_TYPE_OPERATOR_KIND[atom_type],
                confidence=_f(row[3], 1.0),
                salience=_f(row[4], 0.5),
                uncertainty=_f(row[5], 0.5),
            )
        )
        absorbed[source] += 1
    advance(float(rows[-1][0]))

    dist = Counter(v["verdict"] for v in verdicts)
    n = max(1, len(verdicts))
    changes = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a["verdict"] != b["verdict"])
    return {
        "option": option,
        "site_map": site_map,
        "h1_ticks": len(verdicts),
        "verdict_share": {k: round(v / n, 4) for k, v in sorted(dist.items())},
        "verdict_changes": changes,
        "mean_of_mean_ratio": round(sum(v["mean_ratio"] for v in verdicts) / n, 4),
        "mean_of_std_ratio": round(sum(v["std_ratio"] for v in verdicts) / n, 4),
        "mean_of_bulk": round(sum(v["bulk"] for v in verdicts) / n, 4),
        "absorbed": dict(absorbed),
        "dropped": dict(dropped),
        "verdicts": verdicts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--option", choices=["current", "fold_boundary", "extend_bulk"], required=True)
    parser.add_argument("--reheat-prob", type=float, default=0.0054, help="held constant across options")
    parser.add_argument("--warmup-sec", type=float, default=1800.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    with open(args.csv, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0]]
    result = replay(rows, args.option, reheat_prob=args.reheat_prob, warmup_sec=args.warmup_sec)
    Path(args.out).write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k != "verdicts"}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
