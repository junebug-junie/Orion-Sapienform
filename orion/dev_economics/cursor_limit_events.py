"""Cursor contested-budget observer — fail closed until a real meter exists.

WHY THIS EXISTS
---------------
Hiring a Cursor peer spends against a shared, contested pool (same class of
constraint as Claude Code session/weekly limits). An autonomous hire must not
proceed on an unread or guessed meter.

v1 live sources (plan Patch 1 Task 10 — env/file counter until a first-party
Cursor usage API exists):

1. ``fixture=`` — tests / dry-runs only
2. ``CURIOSITY_PEER_CURSOR_BUDGET_FILE`` — JSON or plain-text reading
3. ``CURIOSITY_PEER_CURSOR_BUDGET_STATE`` — operator-asserted ``clear|limited|unknown``

Unset / unreadable / invalid → **unobserved** and ``decide_cursor_budget``
refuses. An unread meter and a full tank must not authorise the same hire.

SEMANTICS (same as Claude LimitObservation for the budget gate)
---------------------------------------------------------------
`observed` / `state` in {unknown, clear, limited} / `staleness_sec`.
`decide_cursor_budget` copies `ask_claude_trigger._budget_refusal`: only a
fresh, observed `clear` returns None (allow). Missing, unobserved, limited,
unknown, and incoherent (observed clear with no staleness) all refuse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

LimitState = Literal["clear", "limited", "unknown"]

CursorBudgetRefusal = Literal[
    "budget_limited",
    "budget_unknown",
    "budget_unobserved",
    "budget_observation_missing",
    "budget_observation_incoherent",
]

ENV_STATE = "CURIOSITY_PEER_CURSOR_BUDGET_STATE"
ENV_FILE = "CURIOSITY_PEER_CURSOR_BUDGET_FILE"
_VALID_STATES = frozenset({"clear", "limited", "unknown"})


@dataclass(frozen=True)
class CursorLimitObservation:
    """What we know about Cursor pool contention right now.

    Field meanings match `orion.dev_economics.rate_limit_events.LimitObservation`
    for the budget-gate surface: `observed`, `state`, `staleness_sec`.
    """

    observed: bool
    state: LimitState
    staleness_sec: float | None = None


def _unobserved() -> CursorLimitObservation:
    return CursorLimitObservation(
        observed=False,
        state="unknown",
        staleness_sec=None,
    )


def _parse_state(raw: object) -> LimitState | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in _VALID_STATES:
        return text  # type: ignore[return-value]
    return None


def _staleness_from_iso(raw: object, *, now: datetime) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        ts = datetime.fromisoformat(text)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts.astimezone(timezone.utc)).total_seconds())
    except ValueError:
        return None


def _from_file(path: str, *, now: datetime) -> CursorLimitObservation | None:
    """Read a file meter. Returns None if path empty/missing/unparseable."""
    if not path or not str(path).strip():
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        body = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not body:
        return None

    state: LimitState | None = None
    staleness: float | None = None

    if body.startswith("{"):
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        state = _parse_state(payload.get("state"))
        staleness = _staleness_from_iso(payload.get("observed_at"), now=now)
        if staleness is None:
            # Fall back to file mtime age when observed_at absent.
            try:
                staleness = max(0.0, now.timestamp() - p.stat().st_mtime)
            except OSError:
                staleness = 0.0
    else:
        # Plain text: first token is the state.
        state = _parse_state(body.split()[0] if body.split() else "")
        try:
            staleness = max(0.0, now.timestamp() - p.stat().st_mtime)
        except OSError:
            staleness = 0.0

    if state is None:
        return None
    return CursorLimitObservation(
        observed=True,
        state=state,
        staleness_sec=staleness if staleness is not None else 0.0,
    )


def _from_env_state(raw: str | None) -> CursorLimitObservation | None:
    """Operator-asserted state. Empty / unset → no reading."""
    if raw is None:
        return None
    state = _parse_state(raw)
    if state is None:
        return None
    # Operator asserts "right now" — staleness 0.0 keeps decide_cursor_budget
    # from treating a clear reading as incoherent.
    return CursorLimitObservation(
        observed=True,
        state=state,
        staleness_sec=0.0,
    )


def observe_cursor_limit(
    *,
    fixture: CursorLimitObservation | None = None,
    state: str | None = None,
    file_path: str | None = None,
    now: datetime | None = None,
) -> CursorLimitObservation:
    """Return a Cursor limit reading.

    Precedence: fixture → file → env/state arg → env CURIOSITY_PEER_* →
    unobserved (fail closed).
    """
    if fixture is not None:
        return fixture

    clock = now or datetime.now(timezone.utc)

    path = file_path if file_path is not None else os.environ.get(ENV_FILE)
    from_file = _from_file(str(path) if path else "", now=clock)
    if from_file is not None:
        return from_file

    raw_state = state if state is not None else os.environ.get(ENV_STATE)
    from_env = _from_env_state(raw_state)
    if from_env is not None:
        return from_env

    return _unobserved()


def decide_cursor_budget(
    limit: CursorLimitObservation | None,
) -> Optional[CursorBudgetRefusal]:
    """Fail CLOSED on anything short of a fresh, observed `clear`.

    Same precedence as `orion.autonomy.ask_claude_trigger._budget_refusal`.
    `unknown` is NOT permission. An unread meter and a full tank must not
    authorise the same hire.
    """
    if limit is None:
        return "budget_observation_missing"
    if not limit.observed:
        return "budget_unobserved"
    if limit.state == "limited":
        return "budget_limited"
    if limit.state != "clear":
        return "budget_unknown"
    if limit.staleness_sec is None:
        # Observed clear but no freshest-timestamp age is a producer
        # contradiction, not a fresh reading — refuse rather than trust a
        # self-inconsistent meter. This is NOT a staleness threshold.
        return "budget_observation_incoherent"
    return None
