"""The binding rule (body economy Patch D): one market decision -> go now, or hold and bid again later.

Pure. Shared by every producer whose bid binds, so they all obey the same rule:

* orion-durable-runs admission (``app.run_bids``: curiosity investigate, #3010), before a run asks the
  GPU pool for a hold;
* orion-dream (``app.physiology_bind``: the dream cycle), before a sleep opens its act;
* orion-hub outreach (``scripts.outreach_bind``: hub.outreach), before a door composes a message;
* orion-sql-writer camera asks (``app.ask_bind``: vision.ask, attention patch 4b), before a card opens. A held
  ask keeps bidding with no deadline (issue #3285), so it passes a rolling ``end`` that never arrives;
* orion-dream's compaction applier (``app.compaction_consent``, attention patch 3d), with the hold ending now:
  it is one-shot, so only consent decides; a consent block is never admitted;
* orion-durable-runs' cooling world act (``app.cooling_act``, via ``orion.autonomy.shed_act.shed_verdict``),
  which is stricter: only ``funded`` sheds, and "no answer" or the window's end never does.

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
    "orion-durable-runs.admission.reading.binding",   # reading stage 1 (attention 3c), same file, flag ORION_PHYSIOLOGY_BID_READING
    "orion-dream.dream_cycle.binding",          # the dream cycle, services/orion-dream/app/physiology_bind.py
    "orion-durable-runs.cooling_act.binding",   # the cooling world act, services/orion-durable-runs/app/cooling_act.py
    "orion-hub.outreach.binding",               # Hub outreach (hub.outreach), services/orion-hub/scripts/outreach_bind.py
    "orion-sql-writer.ask_events.binding",      # camera asks (vision.ask, attention 4b), services/orion-sql-writer/app/ask_bind.py
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
    consent = getattr(decision, "consent", None)
    if consent is not None:
        out["consent_verdict"] = getattr(consent, "verdict", None)
        out["consent_rule"] = getattr(consent, "rule_id", None)
    return out


def consent_block(decision: Any) -> Optional[str]:
    """Why a decision may not run on consent grounds (attention patch 3d), or None.

    Blocked: reason ``consent`` (the gate said refuse or needs_consent), a verdict other than ``allow``, or a
    want that asked for consent but came back with no verdict (a market that dropped the gate). Fails closed."""
    verdict = getattr(decision, "consent", None)
    if str(getattr(decision, "reason", "") or "") == "consent":
        return str(getattr(verdict, "verdict", None) or "refuse")
    if verdict is not None:
        v = str(getattr(verdict, "verdict", "") or "")
        return None if v == "allow" else (v or "refuse")
    want = getattr(decision, "want", None)
    if want is not None and getattr(want, "consent_request", None) is not None:
        return "no_verdict"
    return None


def bind_verdict(decision: Any, now: datetime, end: Optional[datetime], *,
                 capacity_queues_at_pool: bool = True) -> BindVerdict:
    """Pure: a binding family's decision -> admit now, or hold until the next bid.

    * a consent block (``consent_block``: reason ``consent``, a non-allow verdict, or a consent request with
      no verdict) -> HOLD, always: never admitted, even at ``end`` or with no ``end`` (attention patch 3d).
      The next bid is REBID_MAX_SEC on;
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
    blocked = consent_block(decision)
    if blocked is not None:
        # Never admitted: not at the window's end, not with no end. Waiting does not grant consent; the
        # next bid (an hour on) only helps once a standing consent is merged.
        return BindVerdict(False, f"deferred:consent:{blocked}", retry_at=now + timedelta(seconds=REBID_MAX_SEC),
                           detail=detail)
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


__all__ = ["BINDING_PRODUCERS", "BindVerdict", "consent_block", "NO_ANSWER_REASONS", "REBID_DEFAULT_SEC", "REBID_MAX_SEC", "REBID_MIN_SEC",
           "bind_verdict", "decision_detail"]
