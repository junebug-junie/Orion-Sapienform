#!/usr/bin/env python3
"""#3393 baseline: do curiosity targets come back on a second day? Read-only.

Over `substrate_endogenous_curiosity_candidates` for the last --days days, every
focal node a stored candidate pointed at is a "target". For each target, count
the distinct days it appeared, in Juniper's local day (America/Denver), so an
evening that crosses midnight UTC is still one day. Prints one line per measure:

  targets            distinct focal node ids
  targets_2plus_days targets seen on 2 or more distinct days
  one_day_pct        share of targets seen on one day only
  world_targets      targets that are world-reading concepts (sub-concept-wp-read-*)
  world_2plus_days   world-reading targets seen on 2 or more days  <- the #3393 number
  subject_refs       candidate subject_ref values and their counts

--repeats also prints each target seen on 2+ days, with its dates.

  python3 scripts/analysis/curiosity_target_return_days.py --days 30 [--repeats]
"""
from __future__ import annotations

import argparse
import subprocess

WORLD_PREFIX = "sub-concept-wp-read-"

LOCAL_TZ = "America/Denver"

TARGET_DAYS = """
with t as (
  select f as target, (c.created_at at time zone '{tz}')::date as d
  from substrate_endogenous_curiosity_candidates c,
       jsonb_array_elements(c.candidates_json) s,
       jsonb_array_elements_text(coalesce(s->'focal_node_refs', '[]'::jsonb)) f
  where c.created_at > now() - interval '{days} days'
)"""

SQL = TARGET_DAYS + """, per as (select target, count(distinct d) n_days from t group by target)
select 'targets', count(*)::text from per
union all select 'targets_2plus_days', count(*) filter (where n_days >= 2)::text from per
union all select 'one_day_pct', round(100.0 * count(*) filter (where n_days = 1) / nullif(count(*), 0), 1)::text from per
union all select 'world_targets', count(*) filter (where target like '{world}%')::text from per
union all select 'world_2plus_days', count(*) filter (where target like '{world}%' and n_days >= 2)::text from per
union all select 'subject_refs', coalesce(string_agg(sr || '=' || n, ' '), '') from (
  select coalesce(s->>'subject_ref', '(none)') sr, count(*)::text n
  from substrate_endogenous_curiosity_candidates c, jsonb_array_elements(c.candidates_json) s
  where c.created_at > now() - interval '{days} days' group by 1 order by count(*) desc) x
"""

REPEATS_SQL = TARGET_DAYS + """
select 'repeat ' || target, string_agg(d::text, ',' order by d)
from (select distinct target, d from t) x
group by target having count(*) >= 2
order by count(*) desc, target
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--repeats", action="store_true",
                    help="also print each target seen on 2+ local days, with its dates")
    args = ap.parse_args()
    if not 1 <= args.days <= 365:
        raise SystemExit("--days must be 1..365")
    sql = SQL.format(days=args.days, world=WORLD_PREFIX, tz=LOCAL_TZ)
    if args.repeats:
        sql += ";\n" + REPEATS_SQL.format(days=args.days, tz=LOCAL_TZ)
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "sh", "-c",
         'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F"\t" -c "$0"', sql],
        capture_output=True, text=True, check=True).stdout
    for line in filter(None, out.splitlines()):
        key, _, value = line.partition("\t")
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
