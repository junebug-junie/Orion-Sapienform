#!/usr/bin/env python3
"""Are dreams better than random? Live scorecard for dream cycle v2.

Joins offered dream hypotheses (Postgres `dream_hypothesis`, offered_at set)
to the priors Orion formed from them in its own graph (`formed_from` starts
with "dream_hypothesis:") and reports adoption / support per arm.

Read-only on both stores (GRAPH.RO_QUERY + SELECT).

Usage:
    POSTGRES_URI=postgresql://... GRAPH_HOST=... GRAPH_PORT=6379 \\
      python scripts/dream_hypothesis_scorecard.py [--graph orion_worldview] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# scripts/ on sys.path[0] shadows stdlib `platform` via scripts/platform/ --
# same fix as check_metric_lineage.py.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.curiosity.worldview import WorldviewReader  # noqa: E402
from orion.dream.hypotheses import score_hypotheses  # noqa: E402
from orion.schemas.dream_cycle import FORMED_FROM_PREFIX  # noqa: E402

OFFERED_SQL = "SELECT hypothesis_id, arm FROM dream_hypothesis WHERE offered_at IS NOT NULL"
PRIORS_CYPHER = (
    "MATCH (p:Prior) WHERE p.formed_from STARTS WITH '" + FORMED_FROM_PREFIX + "' "
    "RETURN p.prior_id AS prior_id, p.formed_from AS formed_from, "
    "p.status AS status, p.times_tested AS times_tested LIMIT 5000"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.environ.get("GRAPH_OWN", "orion_worldview"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from sqlalchemy import create_engine, text

    engine = create_engine(os.environ["POSTGRES_URI"])
    with engine.connect() as conn:
        offered = [dict(r) for r in conn.execute(text(OFFERED_SQL)).mappings().all()]

    reader = WorldviewReader(
        host=os.environ.get("GRAPH_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRAPH_PORT", "6379")),
        graph_name=args.graph,
    )
    priors = reader.query(PRIORS_CYPHER)

    card = score_hypotheses(offered, priors).as_dict()
    if args.json:
        print(json.dumps(card, indent=2))
        return 0
    for arm, s in card["arms"].items():
        print(
            f"{arm:8s} offered={s['offered']:4d} adopted={s['adopted']:4d} tested={s['tested']:4d} "
            f"supported={s['supported']} revised={s['revised']} refuted={s['refuted']} "
            f"adoption_rate={s['adoption_rate']} support_rate={s['support_rate']}"
        )
    print(f"unmatched priors: {card['unmatched_priors']}")
    print(f"verdict: {card['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
