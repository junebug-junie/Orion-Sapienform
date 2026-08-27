#!/usr/bin/env python3
"""Read-only report on the contested-scarcity quota budget.

THIS SCRIPT IS THE GATE, NOT A DASHBOARD.

Per `docs/superpowers/specs/2026-08-27-claude-quota-contested-scarcity-design.md`,
`orion/autonomy/quota_budget.py` ships read-only first and is wired into no
allocator. The decision on whether to wire it in rests on one question this
script exists to answer:

    Over a week of REAL spend, at a given allowance and a given ask cost,
    would this budget ever actually have refused anything?

If the answer is no, do not wire it in. The allowance is set too high or the
action is too cheap to matter, and shipping it would be the same ornamental
scarcity the 2026-07-07 internal-economy spec was correctly refused for.

    # where the current window stands
    python3 scripts/report_quota_budget.py --allowance-usd 40

    # the gate: replay real history, count refusals
    python3 scripts/report_quota_budget.py --allowance-usd 40 --ask-usd 2.50 --replay-days 7

Read-only. Issues a single SELECT and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.autonomy.quota_budget import (  # noqa: E402
    LedgerTick,
    quota_state,
    sum_window,
    window_elapsed_fraction,
)

SELECT_TICKS = """
    SELECT observed_at, total_tokens, total_estimated_cost_usd, unpriced_session_count
    FROM dev_economics_ledger_log
    WHERE observed_at > %s
    ORDER BY observed_at
"""


def load_ticks(dsn: str, since: datetime) -> list[LedgerTick]:
    import psycopg2

    with psycopg2.connect(dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_TICKS, (since,))
            return [
                LedgerTick(
                    observed_at=row[0],
                    total_tokens=int(row[1] or 0),
                    cost_usd=float(row[2]) if row[2] is not None else None,
                    unpriced_session_count=int(row[3] or 0),
                )
                for row in cur.fetchall()
            ]


def fmt(value: float | None, spec: str = ".2f", none: str = "unknown") -> str:
    return none if value is None else format(value, spec)


def report_current(ticks: list[LedgerTick], allowance_usd: float, window_hours: float) -> None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=window_hours)
    window = [t for t in ticks if t.observed_at >= start]
    spend = sum_window(window)

    state = quota_state(
        allowance_usd=allowance_usd,
        spend=spend,
        # A trailing window is always fully elapsed -- it is the last N hours,
        # not a window that started at a fixed boundary and is filling up.
        window_elapsed_fraction=1.0,
    )

    print(f"\n=== current {window_hours:g}h window (trailing, to {now:%Y-%m-%d %H:%M} UTC) ===")
    if state is None:
        print("  allowance:        NOT CONFIGURED -- no budget (distinct from a budget with nothing left)")
        print(f"  observed spend:   ${spend.spend_usd:.2f} over {spend.tick_count} ticks")
        return

    print(f"  allowance:        ${state.allowance_usd:.2f}  (CALIBRATED, not read -- see module docstring)")
    print(f"  ticks:            {spend.tick_count} ({spend.active_tick_count} active)")
    if not state.spend_known:
        print("  spend:            UNKNOWN -- producer wrote nothing in this window.")
        print("                    Not the same as $0.00. would_refuse() fails closed.")
        return
    floor = "  (FLOOR -- undercount disclosed)" if spend.is_floor else ""
    print(f"  spend:            ${spend.spend_usd:.2f}{floor}")
    print(f"  tokens:           {spend.tokens:,}")
    print(f"  remaining:        ${state.remaining_usd:.2f}   ({fmt(state.fraction_remaining, '.1%')})")
    print(f"  exhausted:        {state.exhausted}")
    print(f"  mode:             {state.mode}  (nothing enforces this; Hub holds the docker socket)")


def replay(
    ticks: list[LedgerTick], allowance_usd: float, ask_usd: float, window_hours: float
) -> None:
    """Walk every tick boundary as a decision point and count refusals.

    Each tick is treated as a moment Orion might have wanted to spend `ask_usd`.
    The question asked at each is exactly the one the allocator would ask.
    """
    if not ticks:
        print("\nno ticks in range -- nothing to replay")
        return

    refused = 0
    refused_unknown = 0
    considered = 0
    exhausted_moments = 0
    peak_spend = 0.0

    for i, at in enumerate(ticks):
        start = at.observed_at - timedelta(hours=window_hours)
        window = [t for t in ticks[: i + 1] if t.observed_at >= start]
        spend = sum_window(window)
        state = quota_state(
            allowance_usd=allowance_usd, spend=spend, window_elapsed_fraction=1.0
        )
        if state is None:
            continue
        considered += 1
        peak_spend = max(peak_spend, spend.spend_usd)
        if state.exhausted:
            exhausted_moments += 1
        if state.would_refuse(ask_usd):
            refused += 1
            if not state.spend_known:
                refused_unknown += 1

    first = ticks[0].observed_at
    last = ticks[-1].observed_at
    print(f"\n=== replay: {first:%Y-%m-%d} -> {last:%Y-%m-%d}, {window_hours:g}h rolling window ===")
    print(f"  allowance:              ${allowance_usd:.2f} per window")
    print(f"  hypothetical ask:       ${ask_usd:.2f}")
    print(f"  decision points:        {considered}")
    print(f"  peak window spend:      ${peak_spend:.2f}")
    print(f"  windows exhausted:      {exhausted_moments}")
    print(f"  WOULD HAVE REFUSED:     {refused}  ({refused / considered:.1%})" if considered else "  n/a")
    print(f"    of which on unknown:  {refused_unknown}  (producer silent -- failed closed)")

    real_refusals = refused - refused_unknown
    print()
    if real_refusals == 0:
        print("  GATE: FAILED. This budget never refused a single ask on observed spend.")
        print("  Do NOT wire it into the allocator. Either the allowance is too high or")
        print("  the ask is too cheap to matter -- lower --allowance-usd or raise --ask-usd")
        print("  until it bites, and decide whether that setting is one you would defend.")
    else:
        print(f"  GATE: PASSED. {real_refusals} refusals on observed spend -- this budget binds.")
        print("  Wiring it in is defensible. Keep it advisory (nothing can enforce it).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allowance-usd", type=float, required=True,
                    help="CALIBRATED window allowance. Not readable from any API -- discovered by "
                         "hitting the limit once and recording cumulative spend at that moment.")
    ap.add_argument("--window-hours", type=float, default=5.0)
    ap.add_argument("--ask-usd", type=float, default=2.50,
                    help="Cost of one hypothetical ask-Claude turn, for the replay gate.")
    ap.add_argument("--replay-days", type=float, default=0.0,
                    help="Replay this many days of real history and count refusals. 0 = current window only.")
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL"))
    args = ap.parse_args()

    if not args.dsn:
        print("no DSN: set POSTGRES_URI or DATABASE_URL, or pass --dsn", file=sys.stderr)
        return 2

    lookback_days = max(args.replay_days, args.window_hours / 24.0)
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days + 1)
    try:
        ticks = load_ticks(args.dsn, since)
    except Exception as exc:  # noqa: BLE001 -- report, do not traceback at an operator
        print(f"could not read dev_economics_ledger_log: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {len(ticks)} ticks since {since:%Y-%m-%d %H:%M} UTC")
    report_current(ticks, args.allowance_usd, args.window_hours)
    if args.replay_days > 0:
        replay(ticks, args.allowance_usd, args.ask_usd, args.window_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
