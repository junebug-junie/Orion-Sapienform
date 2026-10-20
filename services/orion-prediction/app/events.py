"""Change events -> the effective training rules for one series.

A change event is a known moment the world changed under a series (a GPU swap, the cabinet opened, a
sensor moved). Two effects, each with a concrete consequence:

- ``exclude`` [started_at, ended_at): masked out of training AND of backtest actuals, like the
  registry's ``exclude`` windows -- a disturbance that belongs to neither regime.
- ``break`` at started_at: the series' behaviour changed. Every model trains only on data after the
  latest break once ``MIN_POST_BREAK_DAYS`` of it exist (enough for the backtest: 40 windows x 3 h +
  a day of horizon + 2 days of minimum training). Until then the break is ``pending``: training keeps
  the older data and Prophet gets a level-shift regressor at the break instead (``SeriesSpec.breaks``).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Iterable

from app.series import SeriesSpec

MIN_POST_BREAK_DAYS = 10   # floor; Engine passes the age its own backtest settings need (adopt_after)


def applies(event: dict, series_id: str) -> bool:
    return event["scope"] in ("*", series_id)


def adopt_after(backtest_windows: int, backtest_step_min: int, spec: SeriesSpec) -> timedelta:
    """How old a break must be before it can be the training start without starving the backtest:
    every window's origin needs MIN_TRAIN days of post-break training behind it."""
    from app.backtest import MIN_TRAIN_STEPS_DAYS

    need = timedelta(minutes=backtest_windows * backtest_step_min + max(spec.horizons_min or (0,)),
                     days=MIN_TRAIN_STEPS_DAYS + 1)
    return max(timedelta(days=MIN_POST_BREAK_DAYS), need)


def with_events(spec: SeriesSpec, events: Iterable[dict], now: datetime,
                min_age: timedelta = timedelta(days=MIN_POST_BREAK_DAYS)) -> tuple[SeriesSpec, dict]:
    """``spec`` with the events folded in, plus a summary for /health (incl. a fingerprint the
    selection records, so any change to the applied events forces a reselect)."""
    mine = [e for e in events if applies(e, spec.series_id) and e["started_at"] <= now]
    excludes = tuple(sorted((e["started_at"], e["ended_at"]) for e in mine
                            if e["effect"] == "exclude" and e["ended_at"] is not None))
    breaks = sorted(e["started_at"] for e in mine if e["effect"] == "break")
    adopted = [b for b in breaks if now - b >= min_age]
    train_from = spec.train_from
    if adopted and (train_from is None or adopted[-1] > train_from):
        train_from = adopted[-1]
    pending = tuple(b for b in breaks if train_from is None or b > train_from)
    out = dataclasses.replace(spec, exclude=tuple(spec.exclude) + excludes, train_from=train_from, breaks=pending)
    summary = {
        "train_from": train_from.isoformat() if train_from else None,
        "break_adopted": train_from.isoformat() if adopted and train_from == adopted[-1] else None,
        "breaks_pending": [{"at": b.isoformat(), "adopts_at": (b + min_age).isoformat()} for b in pending],
        "excluded_windows": [[a.isoformat(), b.isoformat()] for a, b in excludes],
    }
    summary["fingerprint"] = "|".join([summary["train_from"] or "-", ",".join(x["at"] for x in summary["breaks_pending"]),
                                       ",".join(f"{a}/{b}" for a, b in summary["excluded_windows"])])
    return out, summary


def breaks_between(events: Iterable[dict], series_id: str, start: datetime, end: datetime) -> list[datetime]:
    return sorted(e["started_at"] for e in events
                  if applies(e, series_id) and e["effect"] == "break" and start < e["started_at"] <= end)
