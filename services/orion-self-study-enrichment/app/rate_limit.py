"""Authoritative daily-ceiling backstop for real `claude -p` enrichment runs.

The hook side (`scripts/self_study_enrichment_hook.py`) has its own cheap
belt-check before publishing so a burst of commits doesn't flood the channel,
but that check only counts *publishes*, not *executed runs* -- this module is
the suspenders: the service is the only thing that knows how many real
subprocess spawns it actually made today, so it is the source of truth for
"have we hit the ceiling."

State is a small JSON file (date + count), not Redis/Postgres -- this is a
single-process consumer with no need for a shared counter, and a plain file
is trivially inspectable/resettable by an operator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RateLimitState:
    date: str
    count: int


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def read_state(path: str | Path) -> RateLimitState:
    p = Path(path)
    today = _today()
    if not p.exists():
        return RateLimitState(date=today, count=0)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return RateLimitState(date=today, count=0)
    date = data.get("date")
    count = int(data.get("count", 0) or 0)
    if date != today:
        return RateLimitState(date=today, count=0)
    return RateLimitState(date=today, count=count)


def write_state(path: str | Path, state: RateLimitState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": state.date, "count": state.count}), encoding="utf-8")


def allow_and_record(path: str | Path, *, max_per_day: int) -> bool:
    """Returns True (and records the run) if under the ceiling for today,
    False (no state change) if the ceiling is already hit. `max_per_day <= 0`
    always denies -- an explicit "disabled" state, not a bug."""
    if max_per_day <= 0:
        return False
    state = read_state(path)
    if state.count >= max_per_day:
        return False
    write_state(path, RateLimitState(date=state.date, count=state.count + 1))
    return True
