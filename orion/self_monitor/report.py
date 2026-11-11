"""Print Orion's live faculty track record (read-only).

    POSTGRES_URI=postgresql://... python -P -m orion.self_monitor.report [--days 7] [--json]

Reads each bet source for the window, adapts every row, reduces per faculty, and prints one line per
faculty. A source that cannot be read is named under ``source_errors``, never skipped silently.
This is the patch-1 surface of the organ: it changes nothing anywhere.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from orion.schemas.self_monitor import FacultyRecordV1, SelfMonitorStateV1
from orion.self_monitor.collect import SOURCES, SourceResult, build_state, first_line
from orion.self_monitor.record import WINDOW_DAYS

_ENGINE_CACHE_MAX = 4
_ENGINE_CACHE: dict[str, Any | None] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


_first_line = first_line


def _engine(postgres_url: str) -> Any:
    """Return one read-only pool per configured URL without racing first use.

    A process is expected to use one database URL. The small hard cap catches
    accidental URL churn without evicting and disposing an engine another
    thread may still be using.
    """
    from sqlalchemy import create_engine

    with _ENGINE_CACHE_LOCK:
        if postgres_url not in _ENGINE_CACHE:
            if len(_ENGINE_CACHE) >= _ENGINE_CACHE_MAX:
                raise RuntimeError(
                    f"self-monitor engine cache is full ({_ENGINE_CACHE_MAX} database URLs)"
                )
            _ENGINE_CACHE[postgres_url] = None
        if _ENGINE_CACHE[postgres_url] is None:
            try:
                _ENGINE_CACHE[postgres_url] = create_engine(
                    postgres_url,
                    pool_pre_ping=True,
                    pool_size=1,
                    max_overflow=0,
                    connect_args={
                        "connect_timeout": 5,
                        "options": "-c statement_timeout=60000 -c default_transaction_read_only=on",
                    },
                )
            except Exception:
                del _ENGINE_CACHE[postgres_url]
                raise
        return _ENGINE_CACHE[postgres_url]


def _dispose_cached_engines() -> None:
    """Dispose every cached pool when no callers are active (tests/shutdown)."""
    with _ENGINE_CACHE_LOCK:
        engines = [engine for engine in _ENGINE_CACHE.values() if engine is not None]
        _ENGINE_CACHE.clear()
    for engine in engines:
        engine.dispose()


atexit.register(_dispose_cached_engines)


def collect(postgres_url: str, *, now: datetime, window_days: float) -> SelfMonitorStateV1:
    """Read every source with SQLAlchemy (read-only, 60 s per statement) and build the state through
    the same path the live host uses (``orion.self_monitor.collect.build_state``)."""
    from sqlalchemy import text

    since = now - timedelta(days=window_days)
    try:
        engine = _engine(postgres_url)
    except Exception as exc:
        return build_state({}, now=now, window_days=window_days, extra_errors=[f"engine: {first_line(exc)}"])
    results: dict[str, SourceResult] = {}
    for name, sql, _ in SOURCES:
        try:
            with engine.connect() as conn:
                results[name] = [dict(r._mapping) for r in conn.execute(text(sql), {"since": since})]
        except Exception as exc:  # named by build_state, never silent
            results[name] = exc
    return build_state(results, now=now, window_days=window_days)


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
