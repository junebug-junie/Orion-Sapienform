#!/usr/bin/env python3
"""What has each autonomous action actually been worth?

Reads the action-outcome ledger (services/orion-sql-db/
manual_migration_action_outcome_ledger.sql) and reports, per
(dispatch_kind, target_id, signal_id):

  n            how many scored observations
  nats/action  mean Bayesian surprise -- the information one run of this
               action buys, in nats. Converges toward 0 for an action whose
               effect is already known, which is the whole point: a tic
               earns nothing without anyone writing a rule against tics.
  total nats   n * mean -- the cumulative value of the whole habit
  |err|        mean absolute prediction error, in the signal's own units.
               Directly auditable against baseline/observed_after, unlike
               the nats.
  posterior    current belief about the mean effect, and its sample count
  sole%        share of observations where NO other candidate in the same
               tick claimed the same signal. The field delta is frame-wide,
               so a low sole% means the attribution is shared and the row
               should be read with that in mind. Reported rather than
               hidden: phase 1 scores both and shows the split.

Read-only. Safe to run any time.

    python3 scripts/analysis/report_action_value.py --days 7
"""

from __future__ import annotations

import argparse
import os
import sys

QUERY = """
SELECT dispatch_kind,
       target_id,
       signal_id,
       direction,
       count(*)                                              AS n,
       avg(surprise_nats)                                    AS mean_nats,
       sum(surprise_nats)                                    AS total_nats,
       avg(abs(prediction_error))                            AS mean_abs_err,
       avg(observed_delta)                                   AS mean_delta,
       max(posterior_mean)     FILTER (WHERE rn = 1)         AS posterior_mean,
       max(posterior_n)        FILTER (WHERE rn = 1)         AS posterior_n,
       count(*) FILTER (WHERE co_predictors = 0)             AS sole_n,
       avg(latency_ms)                                       AS mean_latency_ms
  FROM (
        SELECT *,
               row_number() OVER (
                   PARTITION BY dispatch_kind, target_id, signal_id
                   ORDER BY observed_at DESC, id DESC
               ) AS rn
          FROM substrate_action_outcomes
         WHERE observed_at > now() - make_interval(days => %(days)s)
       ) s
 GROUP BY 1, 2, 3, 4
 ORDER BY total_nats DESC
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument(
        "--dsn",
        default=os.environ.get(
            "ORION_SQL_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
        ),
    )
    args = ap.parse_args()

    import psycopg2
    import psycopg2.extras

    with psycopg2.connect(args.dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(QUERY, {"days": args.days})
            rows = cur.fetchall()

    if not rows:
        print(
            f"No scored action outcomes in the last {args.days} day(s).\n"
            "That is a real answer, not an empty report: either nothing has been\n"
            "dispatched with a declared expected_effect yet, or the feedback\n"
            "runtime has not caught up. Check with:\n"
            "  SELECT count(*) FROM substrate_action_outcomes;"
        )
        return 0

    header = (
        f"{'kind':<10} {'target':<42} {'signal':<21} {'dir':<10} "
        f"{'n':>6} {'nats/act':>9} {'total':>10} {'|err|':>7} "
        f"{'post_mu':>8} {'post_n':>7} {'sole%':>6}"
    )
    print(header)
    print("-" * len(header))
    total_nats = 0.0
    total_n = 0
    for r in rows:
        total_nats += float(r["total_nats"] or 0.0)
        total_n += int(r["n"])
        sole_pct = 100.0 * int(r["sole_n"]) / int(r["n"])
        print(
            f"{r['dispatch_kind']:<10} {r['target_id'][:42]:<42} {r['signal_id']:<21} "
            f"{r['direction']:<10} {int(r['n']):>6} "
            f"{float(r['mean_nats']):>9.5f} {float(r['total_nats']):>10.3f} "
            f"{float(r['mean_abs_err']):>7.4f} "
            f"{float(r['posterior_mean'] or 0.0):>8.4f} {int(r['posterior_n'] or 0):>7} "
            f"{sole_pct:>5.0f}%"
        )
    print("-" * len(header))
    print(
        f"{total_n} scored actions over {args.days} day(s), "
        f"{total_nats:.3f} nats total, {total_nats / total_n:.5f} nats/action mean."
    )
    print(
        "\nActions near 0 nats/action are actions whose effect is already known.\n"
        "Repeating them buys nothing -- that is the measurement, not a bug."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
