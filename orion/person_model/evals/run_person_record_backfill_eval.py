#!/usr/bin/env python3
"""Dry run: who the person record would link to whom, from live data (2026-10-22).

Reads the named camera clusters and "different people" verdicts in a READ ONLY
transaction, plus config/person_model/people.yaml, and runs the real planner
(orion/person_model/record.py) as if no person rows existed yet. Writes nothing.

Prints one line per person (name, clusters, belongings, camera days), the
conflicts (same name, said to be different people), and the named things that
were ignored. Names are personal data: print to the terminal or a /tmp file,
never commit the output.

    python orion/person_model/evals/run_person_record_backfill_eval.py \\
        --live postgresql://postgres:...@localhost:55432/conjourney

Exit 1 if any planned link did not come from a typed name or people.yaml.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.person_model import record as pr  # noqa: E402
from orion.person_model.person import DEFAULT_PEOPLE_PATH, load_people  # noqa: E402


def run(live_dsn: str) -> dict:
    from sqlalchemy import create_engine, text

    engine = create_engine(live_dsn)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = conn.execute(text(
            "SELECT individual_id, kind, label, label_ask_id, distinct_days FROM vision_individual "
            "WHERE label IS NOT NULL AND btrim(label) <> ''")).fetchall()
        different = [(r[0], r[1]) for r in conn.execute(text(
            "SELECT individual_a, individual_b FROM vision_individual_pair_verdict WHERE verdict = 'different'"))]
        conn.rollback()
    engine.dispose()
    days = {r[0]: int(r[4] or 0) for r in rows}
    clusters = pr.clusters_from_rows([r[:4] for r in rows])
    plan = pr.plan(clusters, different, load_people(DEFAULT_PEOPLE_PATH), [], {})
    names = {p.person_id: p.display_name for p in plan.new_persons}
    per = defaultdict(lambda: {"clusters": 0, "belongings": 0, "days": 0})
    for link in plan.add_links:
        row = per[link.person_id]
        if link.source_kind == pr.SOURCE_INDIVIDUAL:
            row["clusters"] += 1
            row["days"] += days.get(link.source_ref, 0)
        else:
            row["belongings"] += 1
    linked = {l.source_ref for l in plan.add_links} | {c.individual_id for c in plan.conflicts}
    ignored = sorted({(c.kind, c.label) for c in clusters if c.individual_id not in linked})
    bad = [l for l in plan.add_links if l.decided_by != pr.DECIDED_CAMERA]
    return {"people": [{"name": names.get(pid, pid), **v} for pid, v in sorted(per.items(), key=lambda kv: -kv[1]["days"])],
            "unlinked_people": sorted(n for pid, n in names.items() if pid not in per),
            "conflicts": [(c.individual_id, c.name_key, c.reason) for c in plan.conflicts],
            "ignored_named_things": ignored, "links": len(plan.add_links), "bad_links": len(bad)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", required=True, help="live DSN (opened READ ONLY)")
    report = run(ap.parse_args().live)
    print(f"{len(report['people'])} people from {report['links']} links")
    for p in report["people"]:
        print(f"  {p['name']}: {p['clusters']} camera clusters, {p['belongings']} belongings, {p['days']} camera days")
    print(f"people with no camera link: {', '.join(report['unlinked_people']) or 'none'}")
    print(f"conflicts (same name, said to be different people): {report['conflicts'] or 'none'}")
    print(f"named things ignored (not a person, no matching person): {report['ignored_named_things'] or 'none'}")
    if report["bad_links"]:
        print(f"FAIL: {report['bad_links']} link(s) not from a typed name", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
