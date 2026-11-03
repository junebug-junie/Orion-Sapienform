"""Print Orion's live faculty track record (read-only).

    POSTGRES_URI=postgresql://... python -P -m orion.self_monitor.report [--days 7] [--json]

Reads each bet source for the window, adapts every row, reduces per faculty, and prints one line per
faculty. A source that cannot be read is named under ``source_errors``, never skipped silently.
This is the patch-1 surface of the organ: it changes nothing anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1, SettledBetV1
from orion.self_monitor import adapters
from orion.self_monitor.catalog import UNGRADEABLE
from orion.self_monitor.record import WINDOW_DAYS, reduce_bets

# (source name, SQL with :since, adapter). Each query is bounded by the window.
SOURCES: tuple[tuple[str, str, Callable[[dict], Optional[SettledBetV1]]], ...] = (
    ("grammar_acts",
     "SELECT act_id, verb, object_ref, predicted, opened_at, closed_at, outcome_value, outcome_source, "
     "outcome_note FROM grammar_acts WHERE closed_at >= :since",
     adapters.grammar_act),
    ("prediction_turn_self_bet",
     "SELECT bet_id, faculty, claim_kind, placed_at, settled_at, status, void_reason, stated_p, coverage "
     "FROM prediction_turn_self_bet WHERE settled_at >= :since",
     adapters.turn_self_bet),
    ("prediction_action_bet",
     "SELECT bet_id, domain, skill, p, placed_at, settled_at, result FROM prediction_action_bet "
     "WHERE settled_at >= :since",
     adapters.action_bet),
    ("prediction_forecast",
     "SELECT forecast_id, series_id, role, issued_at, scored_at, in_80 FROM prediction_forecast "
     "WHERE scored_at >= :since AND role = 'champion'",
     adapters.forecast),
    ("vision_percept_expectation",
     "SELECT expectation_id, emitted_at, scored_at, status, confidence FROM vision_percept_expectation "
     "WHERE scored_at >= :since",
     adapters.percept_expectation),
)


def collect(postgres_url: str, *, now: datetime, window_days: float) -> SelfMonitorStateV1:
    from sqlalchemy import create_engine, text

    engine = create_engine(
        postgres_url, pool_size=1, max_overflow=0,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=60000 -c default_transaction_read_only=on"},
    )
    since = now - timedelta(days=window_days)
    bets: list[SettledBetV1] = []
    errors: list[str] = []
    for name, sql, adapt in SOURCES:
        try:
            with engine.connect() as conn:
                rows = [dict(r._mapping) for r in conn.execute(text(sql), {"since": since})]
        except Exception as exc:  # named, never silent
            errors.append(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
            continue
        for row in rows:
            try:
                bet = adapt(row)
            except Exception as exc:
                errors.append(f"{name}: adapter {type(exc).__name__} on one row: {exc}"[:200])
                continue
            if bet is not None:
                bets.append(bet)
    engine.dispose()
    faculties = reduce_bets(bets, now=now, window_days=window_days, ungradeable=UNGRADEABLE)
    return SelfMonitorStateV1(computed_at=now, window_days=window_days, faculties=faculties, source_errors=errors)


def _pct(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{value:5.0%}"


def render(state: SelfMonitorStateV1) -> str:
    order = {"broken": 0, "unreliable": 1, "miscalibrated": 2, "reliable": 3, "unknown": 4}
    rows: list[FacultyRecordV1] = sorted(state.faculties, key=lambda r: (order[r.status], -r.n_bets, r.faculty))
    lines = [f"Orion's faculties, {state.window_days:g} days to {state.computed_at:%Y-%m-%d %H:%MZ}",
             f"{'status':<13} {'faculty':<46} {'bets':>5} {'graded':>6} {'works':>5} {'stated':>6} {'noexec':>6}  why"]
    for r in rows:
        lines.append(
            f"{r.status:<13} {r.faculty[:46]:<46} {r.n_bets:>5} {r.n_graded:>6} {_pct(r.reliability)} "
            f"{_pct(r.stated_mean):>6} {_pct(r.execution_failure):>6}  {'; '.join(r.reasons)}"
            + (f" -> {r.unlock}" if r.unlock and r.gradeability != "graded" and r.n_bets == 0 else "")
        )
    for err in state.source_errors:
        lines.append(f"SOURCE ERROR {err}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=float, default=WINDOW_DAYS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    url = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if not url:
        print("POSTGRES_URI or DATABASE_URL required", file=sys.stderr)
        return 2
    state = collect(url, now=datetime.now(timezone.utc), window_days=args.days)
    print(state.model_dump_json(indent=2) if args.json else render(state))
    return 1 if state.source_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
