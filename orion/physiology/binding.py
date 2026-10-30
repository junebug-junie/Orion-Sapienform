"""The binding rule (body economy Patch D): one market decision -> go now, or hold and bid again later.

Pure. Shared by every producer whose bid binds, so they all obey the same rule:

* orion-durable-runs admission (``app.run_bids``: curiosity investigate, #3010), before a run asks the
  GPU pool for a hold;
* orion-dream (``app.physiology_bind``: the dream cycle), before a sleep opens its act.

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Patch D row. The rule shape
and the live evidence behind it: docs/superpowers/pr-reports/2026-10-30-patch-d-curiosity-binding-pr.md.

Each producer computes its own ``end`` (when a held want stops waiting and goes anyway: its family's
deferrable window, and any floor of its own). This module never invents one: no ``end`` means "do not
hold" (an unbounded hold could silence the producer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

REBID_MIN_SEC = 120.0           # one physiology tick: a sooner re-bid reads the same prices
REBID_MAX_SEC = 3600.0          # one lane window: a lane's measured use is a trailing hour
REBID_DEFAULT_SEC = 600.0       # the market named no relief time (a stuck price, sated)
# The producers whose bids are OBEYED (want_ledger's ``shadow`` is always true, so the producer is the record).
# The market applies Prime's regulate charge only to these: a charge on a shadow bid changes nothing, and
# grading Prime on it would credit Prime with holds that never happened (a rollback to shadow, a forced run).
BINDING_PRODUCERS = frozenset({
    "orion-durable-runs.admission.binding",     # curiosity investigate (#3010), services/orion-durable-runs/app/run_bids.py
    "orion-dream.dream_cycle.binding",          # the dream cycle, services/orion-dream/app/physiology_bind.py
})
# The market could not decide: these are "no answer", not deferrals.
NO_ANSWER_REASONS = frozenset({"physiology_unavailable", "unknown_family", "family_never_bids"})


@dataclass(frozen=True)
class BindVerdict:
    """What a binding bid tells its producer. ``admit``: go now. Otherwise hold until ``retry_at``."""
    admit: bool
    why: str                         # funded | queued_at_pool | no_answer:<reason> | window_end | deferred:<reason>
    retry_at: Optional[datetime] = None
    detail: dict = field(default_factory=dict)


def decision_detail(decision: Any) -> dict:
    """The market's own reason, as a log line or event carries it: a refusal must be readable."""
    if decision is None:
        return {}
    keys = ("decision_id", "outcome", "reason", "binding_constraint", "lane", "net", "charge", "cost_frac",
            "retry_after_sec", "why", "state_tick_id")
    out = {}
    for k in keys:
        v = getattr(decision, k, None)
        if v is not None:
            out[k] = v
    return out


def bind_verdict(decision: Any, now: datetime, end: Optional[datetime], *,
                 capacity_queues_at_pool: bool = True) -> BindVerdict:
    """Pure: a binding family's decision -> admit now, or hold until the next bid.

    * no decision, or a NO_ANSWER reason -> admit (today's path; no other gate is removed);
    * funded -> admit;
    * ``capacity_reserved`` with a POSITIVE net -> admit (``queued_at_pool``). The lane is full, but
      the want is worth its price: the GPU pool's own queue is the queue (it holds the work until a
      real slot frees). A second queue in front of it, judged on a trailing hour of lane use, held
      43 of 46 real curiosity runs for hours in the 7-day replay (scripts/analysis/
      body_economy_binding_replay.py, ``capacity_queues_at_pool=False``). A capacity deferral is
      queueing, never refusal;
    * any other deferral -- ``price`` (charge beats value; Prime's slow charge counts here), ``sated``
      (no hunger), ``capacity_reserved`` with net <= 0, ``no_acceptable_lane`` -> HOLD: the next bid is
      at ``retry_after_sec`` clamped to [REBID_MIN_SEC, REBID_MAX_SEC] (REBID_DEFAULT_SEC when the
      market named none), never later than ``end``; at ``end`` the want goes anyway (``window_end``:
      held, never dropped);
    * no ``end`` (nothing bounds the hold) -> admit: an unbounded hold could silence the producer.
    """
    detail = decision_detail(decision)
    if decision is None:
        return BindVerdict(True, "no_answer:no_decision", detail=detail)
    reason = str(getattr(decision, "reason", "") or "")
    if getattr(decision, "outcome", None) == "funded":
        return BindVerdict(True, "funded", detail=detail)
    if reason in NO_ANSWER_REASONS:
        return BindVerdict(True, f"no_answer:{reason}", detail=detail)
    net = getattr(decision, "net", None)
    worth_it = isinstance(net, (int, float)) and math.isfinite(net) and net > 0
    if capacity_queues_at_pool and reason == "capacity_reserved" and worth_it:
        return BindVerdict(True, "queued_at_pool", detail=detail)
    if end is None:
        return BindVerdict(True, "no_answer:unbounded_hold", detail=detail)
    if now >= end:
        return BindVerdict(True, "window_end", detail=detail)
    ra = getattr(decision, "retry_after_sec", None)
    wait = float(ra) if isinstance(ra, (int, float)) and math.isfinite(ra) else REBID_DEFAULT_SEC
    wait = min(REBID_MAX_SEC, max(REBID_MIN_SEC, wait))
    return BindVerdict(False, f"deferred:{reason}", retry_at=min(end, now + timedelta(seconds=wait)), detail=detail)


__all__ = ["BINDING_PRODUCERS", "BindVerdict", "NO_ANSWER_REASONS", "REBID_DEFAULT_SEC", "REBID_MAX_SEC", "REBID_MIN_SEC",
           "bind_verdict", "decision_detail"]
