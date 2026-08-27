#!/usr/bin/env python3
"""Read-only report on Claude spend, and the harness that judged its denominator.

WHAT THIS ALREADY DECIDED
-------------------------
This script was written as the gate for the contested-scarcity design
(`docs/superpowers/specs/2026-08-27-claude-quota-contested-scarcity-design.md`):
replay real spend, and if a dollar budget would never have refused anything, do
not wire it into the allocator.

It was run, and the answer came back worse than "never refuses". Measured
against 15 real rate-limit events, the limit fires anywhere between $85.39 and
$289.76 of trailing 5h spend, while the largest window ever observed ($420.07)
did not trip it -- so **no allowance separates limited from not-limited**. See
`docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md`.

That is why the replay below reports counts but does NOT render a pass/fail
verdict. On this axis a refusal count is not evidence of a working budget: it
is evidence that the allowance was set low enough to produce refusals, which is
a knob, not a finding. Reporting "PASSED" here would recommend exactly the
action the calibration finding forbids.

Kept because the machinery is denominator-agnostic and a better signal can be
dropped straight into it.

    # where the current window stands (a spend report, not a quota gauge)
    python3 scripts/report_quota_budget.py --allowance-usd 150

    # replay real history and count what a given allowance would have stopped
    python3 scripts/report_quota_budget.py --allowance-usd 150 --ask-usd 2.50 --replay-days 7

Read-only. Issues a single SELECT and writes nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.autonomy.quota_budget import (  # noqa: E402
    LedgerTick,
    QuotaState,
    WindowSpend,
    quota_state,
    sum_window,
)

SELECT_TICKS = """
    SELECT observed_at, total_tokens, total_estimated_cost_usd, unpriced_session_count
    FROM dev_economics_ledger_log
    WHERE observed_at > %s
    ORDER BY observed_at
"""

# A trailing N-hour window is by definition fully elapsed, so every QuotaState
# built here passes 1.0. Note the consequence: under 1.0, `pace` is identically
# `spent_fraction` and `projected_window_usd` is identically `spend_usd`. Both
# are therefore degenerate in this script and are deliberately never printed --
# they exist for a fixed-boundary window (a real 5h session window) that this
# trailing view does not model.
TRAILING_WINDOW_ELAPSED = 1.0


def load_ticks(dsn: str, since: datetime) -> list[LedgerTick]:
    import psycopg2

    # `with psycopg2.connect(...)` manages the transaction, not the socket.
    with contextlib.closing(psycopg2.connect(dsn, connect_timeout=5)) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(SELECT_TICKS, (since,))
            return [
                LedgerTick(
                    observed_at=row[0],
                    total_tokens=int(row[1] or 0),
                    # `is not None` rather than `or None`: a genuine $0.00 tick
                    # is priced, and must not be laundered into "unpriced".
                    cost_usd=float(row[2]) if row[2] is not None else None,
                    unpriced_session_count=int(row[3] or 0),
                )
                for row in cur.fetchall()
            ]


def window_at(ticks: list[LedgerTick], at: datetime, window_hours: float) -> WindowSpend:
    start = at - timedelta(hours=window_hours)
    return sum_window([t for t in ticks if start <= t.observed_at <= at])


def state_at(
    ticks: list[LedgerTick], at: datetime, window_hours: float, allowance_usd: float
) -> QuotaState | None:
    return quota_state(
        allowance_usd=allowance_usd,
        spend=window_at(ticks, at, window_hours),
        window_elapsed_fraction=TRAILING_WINDOW_ELAPSED,
    )


def report_current(ticks: list[LedgerTick], allowance_usd: float, window_hours: float) -> None:
    now = datetime.now(timezone.utc)
    spend = window_at(ticks, now, window_hours)
    state = quota_state(
        allowance_usd=allowance_usd,
        spend=spend,
        window_elapsed_fraction=TRAILING_WINDOW_ELAPSED,
    )

    print(f"\n=== current {window_hours:g}h window (trailing, to {now:%Y-%m-%d %H:%M} UTC) ===")
    if state is None:
        print("  allowance:        NOT CONFIGURED -- no budget")
        print("                    (distinct from a budget with nothing left)")
        if spend.observed:
            print(f"  observed spend:   ${spend.spend_usd:.2f} over {spend.tick_count} ticks")
        else:
            print("  spend:            UNKNOWN -- producer wrote nothing in this window")
        return

    print(f"  allowance:        ${state.allowance_usd:.2f}")
    print("                    NOT calibratable on this axis -- see module docstring.")
    print(f"  ticks:            {spend.tick_count} ({spend.active_tick_count} active)")

    if not state.spend_known:
        print("  spend:            UNKNOWN -- producer wrote nothing in this window.")
        print("                    Not the same as $0.00. would_refuse() fails closed.")
        return

    floor = state.spend_is_floor
    mark = "  [FLOOR]" if floor else ""
    print(f"  spend:            ${spend.spend_usd:.2f}{mark}")
    print(f"  tokens:           {spend.tokens:,}")
    frac = state.fraction_remaining
    print(f"  remaining:        ${state.remaining_usd:.2f}{mark}"
          f"   ({frac:.1%})" if frac is not None else "  remaining:        unknown")
    print(f"  exhausted:        {state.exhausted}")
    print(f"  mode:             {state.mode}  (nothing enforces this; Hub holds the docker socket)")
    if floor:
        print(f"  [FLOOR] spend is a known UNDERCOUNT "
              f"({spend.unpriced_session_count} unpriced sessions, "
              f"{spend.unpriced_active_tick_count} unpriced ticks).")
        print("          Every derived number above is a bound, not a value, and errs")
        print("          toward letting spending through.")


@dataclass(frozen=True)
class ReplayResult:
    """Pure result of a replay. Printing is separate so this stays testable."""

    considered: int
    refused: int
    refused_unknown: int
    exhausted_moments: int
    floor_moments: int
    peak_spend_usd: float
    first: datetime | None
    last: datetime | None

    @property
    def refused_on_observed(self) -> int:
        return self.refused - self.refused_unknown


def replay(
    ticks: list[LedgerTick],
    *,
    allowance_usd: float,
    ask_usd: float,
    window_hours: float,
    step_minutes: float = 15.0,
) -> ReplayResult:
    """Walk a FIXED CLOCK GRID and ask the allocator's question at each point.

    Decision points are generated on a clock, not on tick arrivals, for two
    reasons that an earlier tick-driven version got wrong:

    1. A tick-driven grid can only sample moments where a tick exists, so it
       structurally cannot land in a producer outage -- which made the
       unknown-spend disclosure dead code that always printed zero. Live tick
       counts run 45-95 against a theoretical 96/day, so multi-hour gaps are
       routine and worth sampling.
    2. The first `window_hours` of any range have windows truncated at the
       query cutoff, so they under-count spend and under-refuse. The grid
       starts one full window after the first tick, excluding that warm-up
       instead of quietly biasing the result.
    """
    if not ticks:
        return ReplayResult(0, 0, 0, 0, 0, 0.0, None, None)

    start = ticks[0].observed_at + timedelta(hours=window_hours)
    end = ticks[-1].observed_at
    step = timedelta(minutes=step_minutes)
    if step <= timedelta(0) or start > end:
        return ReplayResult(0, 0, 0, 0, 0, 0.0, None, None)

    considered = refused = refused_unknown = exhausted = floors = 0
    peak = 0.0
    first_point: datetime | None = None
    last_point: datetime | None = None

    at = start
    while at <= end:
        state = state_at(ticks, at, window_hours, allowance_usd)
        if state is None:
            at += step
            continue
        considered += 1
        first_point = first_point or at
        last_point = at
        peak = max(peak, state.spend.spend_usd)
        if state.spend_is_floor:
            floors += 1
        if state.exhausted:
            exhausted += 1
        if state.would_refuse(ask_usd):
            refused += 1
            if not state.spend_known:
                refused_unknown += 1
        at += step

    return ReplayResult(
        considered=considered,
        refused=refused,
        refused_unknown=refused_unknown,
        exhausted_moments=exhausted,
        floor_moments=floors,
        peak_spend_usd=peak,
        first=first_point,
        last=last_point,
    )


def print_replay(r: ReplayResult, *, allowance_usd: float, ask_usd: float, window_hours: float) -> None:
    if r.considered == 0:
        print("\n=== replay ===")
        print("  NO DECISION POINTS. Nothing was replayed, so nothing was concluded.")
        print("  (Too little history for one full window, or no allowance configured.)")
        return

    print(f"\n=== replay: {r.first:%Y-%m-%d %H:%M} -> {r.last:%Y-%m-%d %H:%M} UTC, "
          f"{window_hours:g}h rolling window ===")
    print(f"  allowance:              ${allowance_usd:.2f} per window")
    print(f"  hypothetical ask:       ${ask_usd:.2f}")
    print(f"  decision points:        {r.considered}  (fixed clock grid, warm-up excluded)")
    print(f"  peak window spend:      ${r.peak_spend_usd:.2f}")
    print(f"  windows exhausted:      {r.exhausted_moments}")
    print(f"  windows on a FLOOR:     {r.floor_moments}  (spend a known undercount)")
    print(f"  would have refused:     {r.refused}  ({r.refused / r.considered:.1%})")
    print(f"    on observed spend:    {r.refused_on_observed}")
    print(f"    on unknown spend:     {r.refused_unknown}  (producer silent -- failed closed)")

    print()
    print("  NO VERDICT IS RENDERED, DELIBERATELY.")
    print("  The dollar denominator was measured against 15 real rate-limit events and")
    print("  refuted: the limit fired between $85.39 and $289.76 of 5h spend, and a")
    print("  $420.07 window did not trip it. No allowance separates limited from not.")
    print("  A refusal count here therefore measures how low the allowance was set,")
    print("  not whether the budget binds. Do not wire this into the allocator on the")
    print("  dollar axis, and do not tune --allowance-usd until it produces refusals.")
    print("  See docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--allowance-usd", type=float, required=True,
                    help="Window allowance. NOT calibratable on this axis -- see the "
                         "calibration finding. Treat any fraction of it as illustrative.")
    ap.add_argument("--window-hours", type=float, default=5.0)
    ap.add_argument("--ask-usd", type=float, default=2.50,
                    help="Cost of one hypothetical ask-Claude turn, for the replay.")
    ap.add_argument("--replay-days", type=float, default=0.0,
                    help="Replay this many days of decision points. Warm-up is loaded "
                         "but excluded, so the replayed range is exactly this long.")
    ap.add_argument("--step-minutes", type=float, default=15.0,
                    help="Decision-point cadence on the replay clock grid.")
    ap.add_argument("--dsn", default=os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL"))
    args = ap.parse_args()

    if not args.dsn:
        print("no DSN: set POSTGRES_URI or DATABASE_URL, or pass --dsn", file=sys.stderr)
        return 2

    # Load one extra window of warm-up so the first replayed point has a full
    # window behind it. `replay` excludes it rather than counting it.
    lookback = timedelta(days=max(args.replay_days, 0.0)) + timedelta(hours=args.window_hours)
    since = datetime.now(timezone.utc) - lookback
    try:
        ticks = load_ticks(args.dsn, since)
    except Exception as exc:  # noqa: BLE001 -- report, do not traceback at an operator
        print(f"could not read dev_economics_ledger_log: {exc}", file=sys.stderr)
        return 1

    print(f"loaded {len(ticks)} ticks since {since:%Y-%m-%d %H:%M} UTC")
    report_current(ticks, args.allowance_usd, args.window_hours)
    if args.replay_days > 0:
        result = replay(
            ticks,
            allowance_usd=args.allowance_usd,
            ask_usd=args.ask_usd,
            window_hours=args.window_hours,
            step_minutes=args.step_minutes,
        )
        print_replay(result, allowance_usd=args.allowance_usd, ask_usd=args.ask_usd,
                     window_hours=args.window_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
