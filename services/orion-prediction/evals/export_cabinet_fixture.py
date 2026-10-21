"""Export the cabinet replay fixture from production Postgres (read-only), 5-min bucket means.

    python services/orion-prediction/evals/export_cabinet_fixture.py --until 2026-10-15T22:00:00Z

Writes evals/fixtures/cabinet_5min.csv.gz: bucket_start, cabinet_temp_c, chassis_watts, cooling_watts,
circe_watts, hecate_watts (fleet table; hecate empty before it ran). Outdoor weather is deliberately not
exported: an hourly temperature series from a fixed spot narrows down where it is, and this repo is public.
from 2026-09-26 (the AC install, the registry's train_from) to --until. Runs psql inside the DB container.
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "cabinet_5min.csv.gz"
SQL = """
COPY (
  WITH b AS (
    SELECT to_timestamp(floor(extract(epoch FROM timestamp::timestamptz) / 300) * 300) AS bucket,
           avg((measurements->>'cabinet_temp_c')::float) AS cabinet_temp_c,
           avg((measurements->>'chassis_watts')::float) AS chassis_watts
    FROM orion_biometrics_summary
    WHERE node = 'athena' AND timestamp >= '2026-09-26' AND timestamp::timestamptz < '{until}'::timestamptz
    GROUP BY 1),
  c AS (
    SELECT to_timestamp(floor(extract(epoch FROM ts) / 300) * 300) AS bucket, avg(cooling_watts) AS cooling_watts
    FROM home_cooling_sample WHERE role = 'cabinet_cooling' AND ts >= '2026-09-26' AND ts < '{until}'
    GROUP BY 1),
  f AS (
    SELECT to_timestamp(floor(extract(epoch FROM observed_at) / 300) * 300) AS bucket,
           avg((measurements_by_node->'circe'->>'chassis_watts')::float) AS circe_watts,
           avg((measurements_by_node->'hecate'->>'chassis_watts')::float) AS hecate_watts
    FROM orion_biometrics_cluster WHERE observed_at >= '2026-09-26' AND observed_at < '{until}' GROUP BY 1)
  SELECT to_char(b.bucket AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS bucket_start,
         round(b.cabinet_temp_c::numeric, 3) AS cabinet_temp_c, round(b.chassis_watts::numeric, 1) AS chassis_watts,
         round(c.cooling_watts::numeric, 1) AS cooling_watts, round(f.circe_watts::numeric, 1) AS circe_watts,
         round(f.hecate_watts::numeric, 1) AS hecate_watts
  FROM b LEFT JOIN c USING (bucket) LEFT JOIN f USING (bucket) ORDER BY b.bucket
) TO STDOUT WITH CSV HEADER
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True)
    ap.add_argument("--container", default="orion-athena-sql-db")
    args = ap.parse_args()
    out = subprocess.run(["docker", "exec", "-i", args.container, "psql", "-U", "postgres", "-d", "conjourney",
                          "-v", "ON_ERROR_STOP=1", "-c", SQL.format(until=args.until)],
                         capture_output=True, text=True, check=True).stdout
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", compresslevel=9) as fh:
        fh.write(out)
    print(f"wrote {OUT} ({len(out.splitlines()) - 1} rows)")


if __name__ == "__main__":
    main()
