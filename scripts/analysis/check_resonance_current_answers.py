#!/usr/bin/env python3
"""Live check for #3396: were the lived answers offered to reading resonance CURRENT?

Read-only. For every `reading_resonance` packet candidate with ref_kind=lived_answer
created after --since, compare its answer with the current answer for that question
AT THE PACKET'S built_at (point in time, not now): the newest answer with evidence,
else the newest of any kind -- the rule in
orion/curiosity/self_inquiry.py:newest_answer_per_question(require_evidence=True).
Also counts which candidates the stored verdict cited (result_json.raw citations).

  python3 scripts/analysis/check_resonance_current_answers.py --since 2026-11-17
Baseline 11-11..11-14: 32 candidates, 24 not current, 3 cited (2 of those not current).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import json

SQL = ("select r.request_json->'brief'->'packet'->>'built_at', c->>'ref_id', "
       "coalesce(r.result_json->>'raw','') like '%%'||(c->>'ref_id')||'%%' "
       "from reading_resonance r, jsonb_array_elements(r.request_json->'brief'->'packet'->'candidates') c "
       "where c->>'ref_kind'='lived_answer' and r.created_at >= '{since}'")
CYPHER = ("MATCH (a:LivedAnswer) RETURN a.question_id + '|' + a.run_id + '|' + toString(coalesce(a.written_at, 0)) "
          "+ '|' + CASE WHEN a.evidence IS NULL OR a.evidence = [] OR a.evidence = '' OR a.evidence = '[]' "
          "THEN '[]' ELSE 'x' END")


def has_evidence(raw: str) -> bool:
    """Truthiness of orion/curiosity/self_inquiry.py:_evidence_list on a redis-cli value
    (a list prints as "[a, b]" or "[]"); kept dependency-free so this runs on bare python3."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            return bool([x for x in json.loads(raw) if str(x).strip()])
        except ValueError:
            return raw.strip("[] ") != ""
    return bool(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="created_at lower bound, e.g. 2026-11-17")
    args = ap.parse_args()
    dt.date.fromisoformat(args.since)  # refuse anything that is not a date
    out = subprocess.run(["docker", "exec", "orion-athena-falkordb", "redis-cli", "--csv", "GRAPH.RO_QUERY",
                          "orion_worldview", CYPHER], capture_output=True, text=True, check=True).stdout
    by_q: dict[str, list[tuple[int, str, int]]] = {}
    for cell in next(csv.reader([out.replace("\n", ",")]), []):
        parts = cell.split("|", 3)
        if len(parts) != 4:
            continue
        try:
            written = int(float(parts[2]))
        except ValueError:
            continue
        # Same evidence parse as the code under test (a list may come back as "[a, b]" or "[]").
        by_q.setdefault(parts[0], []).append((written, parts[1], int(has_evidence(parts[3]))))
    rows = subprocess.run(["docker", "exec", "orion-athena-sql-db", "sh", "-c",
                           'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F"\t" -c "$0"',
                           SQL.format(since=args.since)], capture_output=True, text=True, check=True).stdout
    total = stale = cited = cited_stale = 0
    for line in filter(None, rows.splitlines()):
        built, ref, was_cited = line.split("\t")
        qid, run = ref.rsplit("@", 1)
        at = int(dt.datetime.fromisoformat(built.replace("Z", "+00:00")).timestamp() * 1000)
        held = sorted((a for a in by_q.get(qid, []) if a[0] <= at), key=lambda a: (a[0], a[1]), reverse=True)
        with_ev = [a for a in held if a[2]]
        current = (with_ev or held or [(0, "", 0)])[0][1]
        total += 1
        stale += run != current
        if was_cited == "t":
            cited += 1
            cited_stale += run != current
    print(f"lived_answer candidates={total} not_current={stale} cited={cited} cited_not_current={cited_stale}")
    print("PASS" if total and not stale else ("NO DATA" if not total else "FAIL"))
    return 0 if not stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
