#!/usr/bin/env python3
"""Read-only sample: what would Choice decide right now, for one want per real act family?

Body economy Patch C1 (SHADOW). Reads the LIVE posted prices (orion-durable-runs GET /physiology/state)
and the live act history (grammar_acts + act_downstream_use, last 30 days, read-only psql), then runs
the live Choice code (orion/physiology/choice.py) locally for one want per bidding family in
config/physiology/families.yaml, as if each producer bid now. Nothing is published, reserved on the
live service or written anywhere: the reservation book here is a throwaway, and the bids are decided
in sequence so later ones see earlier ones' reservations (as the live endpoint would).

Usage (from the repo root):
  .venv/bin/python scripts/analysis/sample_body_economy_bids.py [--state-url http://localhost:8124/physiology/state] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
import _body_economy_data as data_mod  # noqa: E402
from orion.physiology.choice import ReservationBook, build_context, decide  # noqa: E402
from orion.physiology.families import load_families  # noqa: E402
from orion.schemas.physiology import parse_physiology_state  # noqa: E402
from orion.schemas.physiology_want import WantV1  # noqa: E402
from replay_body_economy import acts_from_extract  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-url", default="http://localhost:8124/physiology/state")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    with urllib.request.urlopen(a.state_url, timeout=5) as r:   # noqa: S310 -- local debug route
        body = json.loads(r.read())
    if not body.get("ok"):
        print(f"no live physiology state: {body.get('reason')}")
        return 1
    state = parse_physiology_state(body["state"])
    fams, err = load_families(_HERE.parents[1] / "config" / "physiology" / "families.yaml")
    if err:
        raise SystemExit(err)
    ex = data_mod.fetch_acts((now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    acts, uses = acts_from_extract(ex)
    ctx = build_context(fams, acts, uses, state, now)
    book = ReservationBook()
    out = []
    for fam, cfg in sorted(fams.families.items(), key=lambda kv: (kv[1].tier, kv[0])):
        w = WantV1(want_id=f"sample:{fam}:{now:%H%M%S}", family=fam, producer="sample", verb="sample",
                   tier=cfg.tier if cfg.bids else 1)
        d = decide(w, state, ctx, book, now)
        out.append(d)
    if a.json:
        print(json.dumps([d.model_dump(mode="json") for d in out], indent=1))
        return 0
    age = (now - state.generated_at).total_seconds()
    print(f"Live prices from {state.tick_id} ({age:.0f} s old); strain: {state.strain.why if state.strain else 'n/a'}")
    print("| family | tier | outcome | reason | lane | intensity | rank (basis) | weight (basis) | charge | net | binding | warnings |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for d in out:
        warns = ", ".join(w for w in d.warnings if not w.startswith("unpriced:heat") and not w.startswith("unpriced:energy")
                          and not w.startswith("unpriced:claude"))
        print(f"| {d.family} | {d.tier} | {d.outcome} | {d.reason} | {d.lane or '-'} | {f(d.intensity)} | "
              f"{f(d.value_rank)} ({d.value_basis or '-'}) | {f(d.family_weight)} ({d.weight_basis or '-'}) | "
              f"{f(d.charge)} | {f(d.net)} | {d.binding_constraint or '-'} | {warns or '-'} |")
    print("\nEvery decision is shadow=true and lists heat, energy and claude as unpriced (per-act watt-hours and "
          "limit events are not measured yet).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
