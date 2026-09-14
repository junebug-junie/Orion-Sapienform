"""Cursor contested-budget observer — fail closed until a real meter exists.

WHY THIS EXISTS
---------------
Hiring a Cursor peer spends against a shared, contested pool (same class of
constraint as Claude Code session/weekly limits). An autonomous hire must not
proceed on an unread or guessed meter.

Live Cursor meter wiring is required before enabling the contractor-peer flag
in production. Until a first-party Cursor usage signal exists (SDK usage API,
env/file counter, or equivalent), the default observation is **unobserved**
and `decide_cursor_budget` refuses. Dry-run / unit tests inject readings via
`observe_cursor_limit(fixture=...)`.

SEMANTICS (same as Claude LimitObservation for the budget gate)
---------------------------------------------------------------
`observed` / `state` in {unknown, clear, limited} / `staleness_sec`.
`decide_cursor_budget` copies `ask_claude_trigger._budget_refusal`: only a
fresh, observed `clear` returns None (allow). Missing, unobserved, limited,
unknown, and incoherent (observed clear with no staleness) all refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

LimitState = Literal["clear", "limited", "unknown"]

CursorBudgetRefusal = Literal[
    "budget_limited",
    "budget_unknown",
    "budget_unobserved",
    "budget_observation_missing",
    "budget_observation_incoherent",
]


@dataclass(frozen=True)
class CursorLimitObservation:
    """What we know about Cursor pool contention right now.

    Field meanings match `orion.dev_economics.rate_limit_events.LimitObservation`
    for the budget-gate surface: `observed`, `state`, `staleness_sec`.
    """

    observed: bool
    state: LimitState
    staleness_sec: float | None = None


def observe_cursor_limit(
    *,
    fixture: CursorLimitObservation | None = None,
) -> CursorLimitObservation:
    """Return a Cursor limit reading.

    With no live meter, the default is unobserved (fail closed). Tests and
    dry-runs pass `fixture=` to inject a known observation.
    """
    if fixture is not None:
        return fixture
    return CursorLimitObservation(
        observed=False,
        state="unknown",
        staleness_sec=None,
    )


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
