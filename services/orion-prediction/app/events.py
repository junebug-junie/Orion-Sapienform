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

MIN_POST_BREAK_DAYS = 10


def applies(event: dict, series_id: str) -> bool:
    return event["scope"] in ("*", series_id)


def with_events(spec: SeriesSpec, events: Iterable[dict], now: datetime) -> tuple[SeriesSpec, dict]:
    """``spec`` with the events folded in, plus a summary for /health."""
    mine = [e for e in events if applies(e, spec.series_id) and e["started_at"] <= now]
    excludes = tuple(sorted((e["started_at"], e["ended_at"]) for e in mine
                            if e["effect"] == "exclude" and e["ended_at"] is not None))
    breaks = sorted(e["started_at"] for e in mine if e["effect"] == "break")
    adopted = [b for b in breaks if now - b >= timedelta(days=MIN_POST_BREAK_DAYS)]
    train_from = spec.train_from
    if adopted and (train_from is None or adopted[-1] > train_from):
        train_from = adopted[-1]
    pending = tuple(b for b in breaks if train_from is None or b > train_from)
    out = dataclasses.replace(spec, exclude=tuple(spec.exclude) + excludes, train_from=train_from, breaks=pending)
    summary = {
        "train_from": train_from.isoformat() if train_from else None,
        "breaks_adopted": [b.isoformat() for b in adopted],
        "breaks_pending": [{"at": b.isoformat(),
                            "adopts_at": (b + timedelta(days=MIN_POST_BREAK_DAYS)).isoformat()} for b in pending],
        "excluded_windows": [[a.isoformat(), b.isoformat()] for a, b in excludes],
    }
    return out, summary
