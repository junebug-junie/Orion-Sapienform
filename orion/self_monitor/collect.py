"""The organ's one read-and-adapt path, shared by the live host and the report.

``SOURCES`` names every bet source: its SQL (bounded by ``:since``, the window start) and its pure
adapter. The caller runs the SQL with whatever driver it has (the report: SQLAlchemy; orion-durable-runs:
its psycopg pool, ``psycopg_sql``) and hands the rows, or the error, to ``build_state``. So there is one
query text and one adapt rule, never two copies that drift.

A source that cannot be read, or rows an adapter cannot shape, are named in ``source_errors``, never
skipped silently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from orion.schemas.self_monitor import SelfMonitorStateV1, SettledBetV1
from orion.self_monitor import adapters
from orion.self_monitor.catalog import UNGRADEABLE
from orion.self_monitor.record import reduce_bets

Adapter = Callable[[dict], Optional[SettledBetV1]]

# (source name, SQL with :since, adapter). Each query is bounded by the window.
SOURCES: tuple[tuple[str, str, Adapter], ...] = (
    ("grammar_acts",
     "SELECT act_id, actor, verb, object_ref, signal_id, predicted, opened_at, closed_at, outcome_value, outcome_source, "
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

# The host keeps a missing faculty's held status only when one of its sources erred (it cannot tell
# "no bets in the window" from "unread").
SOURCE_NAMES: tuple[str, ...] = tuple(name for name, _, _ in SOURCES)


def psycopg_sql(sql: str) -> str:
    """The same text in psycopg's parameter style (the only parameter is ``:since``)."""
    return sql.replace(":since", "%(since)s")


def first_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0][:160]}"


def adapt_rows(name: str, rows: Iterable[Mapping], adapt: Adapter) -> tuple[list[SettledBetV1], Optional[str]]:
    """Adapt one source's rows. Returns the bets and an error line when any row failed."""
    bets: list[SettledBetV1] = []
    bad, total, example = 0, 0, ""
    for row in rows:
        total += 1
        try:
            bet = adapt(dict(row))
        except Exception as exc:  # noqa: BLE001 -- counted and named below
            bad += 1
            example = example or first_line(exc)
            continue
        if bet is not None:
            bets.append(bet)
    return bets, (f"{name}: adapter failed on {bad}/{total} rows, e.g. {example}" if bad else None)


SourceResult = Union[Sequence[Mapping], BaseException]


def build_state(results: Mapping[str, SourceResult], *, now: datetime, window_days: float,
                extra_errors: Sequence[str] = ()) -> SelfMonitorStateV1:
    """Rows (or the read error) per source name -> the window's candidate state, before hysteresis.
    A source missing from ``results`` is named as unread."""
    bets: list[SettledBetV1] = []
    errors: list[str] = list(extra_errors)
    for name, _, adapt in SOURCES:
        result = results.get(name)
        if result is None:
            errors.append(f"{name}: not read")
            continue
        if isinstance(result, BaseException):
            errors.append(f"{name}: {first_line(result)}")
            continue
        got, err = adapt_rows(name, result, adapt)
        bets.extend(got)
        if err:
            errors.append(err)
    faculties = reduce_bets(bets, now=now, window_days=window_days, ungradeable=UNGRADEABLE)
    return SelfMonitorStateV1(computed_at=now, window_days=window_days, faculties=faculties, source_errors=errors)


def erred_sources(state: SelfMonitorStateV1) -> set[str]:
    """Source names named in ``source_errors`` (a ``engine:`` error means every source)."""
    out: set[str] = set()
    for err in state.source_errors:
        head = err.split(":", 1)[0]
        if head in SOURCE_NAMES:
            out.add(head)
        else:
            out.update(SOURCE_NAMES)
    return out


__all__ = ["SOURCES", "SOURCE_NAMES", "adapt_rows", "build_state", "erred_sources", "first_line", "psycopg_sql"]
