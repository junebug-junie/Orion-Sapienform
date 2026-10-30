"""Prime's acts on the act ledger (rung 2), the holdback draw, and how a nomination settles.

Three acts, all ``actor = "prime"`` (the body-economy family ``prime``, so Prime's own history
prices its next bid):

* **thought** -- verb ``infer``. Opens when a funded thought starts, closes when it ends.
  ``artifact_produced`` = the model returned an answer that parsed (a preempted, shed, timed-out
  or unparseable thought is 0). Horizon: the LLM timeout plus a margin, because the runtime closes
  the act itself in the same coroutine.
* **nomination** -- verb ``ask`` (a call to another organ, design section 5). One question for
  curiosity's next regular investigate run. ``test_before = "holdback:<arm>"`` records the draw:
  ``treated`` queues it in Hub's focus slot (orion/curiosity/focus.py); ``control`` is the narrator
  arm (design section 9): logged, never sent.

* **concern** -- verb ``predict`` (Prime step 1, 2026-10-30). Holding a concern is a prediction:
  "this is worth acting on, and acting on it will produce something". ``predict`` is the honest verb:
  the act makes no call (not ``ask``), does not just hold still (``wait`` would grade patience, and a
  concern is graded on whether Prime followed through), and is checked against a stated outcome by a
  horizon. Opened when the runtime first sees the concern; ``artifact_produced`` = settled: a TREATED
  question citing the concern (``goal_ref prime_concern:<id>``) settled produced (its run wrote
  something) within ``CONCERN_HORIZON_SEC`` of the act opening. A control-arm question was never sent,
  so it cannot settle a concern. 0 when Prime drops the concern unresolved, lets it fade (not kept for
  a day), or holds it past the horizon unsettled ("lapsed"). Settling or lapsing closes the act but
  does not remove the concern: Prime decides whether to keep holding it. If orion-mind is down past
  the horizon, the ledger closes the act first as ``expired`` (outcome_source expired, not graded in
  orion/physiology/appetite.py) and Prime's later "lapsed" close is ignored; run_prime_report.py counts
  those apart as ``expired``.

How a nomination closes (both arms on the same rule, so the arms are comparable):

* the run it is graded on is, for ``treated``, the investigate run that took the focus; for
  ``control``, the first investigate run that started after the act opened (the run that would
  have been shown it);
* ``artifact_produced`` = that run wrote something: its ``outcome_kind`` (orion/curiosity/run_story.py)
  is ``finished``, ``reached_out_sent`` or ``reached_out_blocked``. ``wrote_nothing``, ``died`` and
  ``cancelled`` are 0. Measured 2026-10-30 over 30 days of investigate runs (n=195): 89 wrote
  something, 83 wrote nothing, 19 died -- not saturated, so the two arms can differ;
* a treated question Hub refused to queue closes 0 at once; one no run took before the focus TTL
  closes 0.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from orion.curiosity.focus import FOCUS_TTL, ORIGIN_SELF, focus_id_for
from orion.curiosity.run_story import (
    OUTCOME_CANCELLED,
    OUTCOME_DIED,
    OUTCOME_EMPTY,
    OUTCOME_FINISHED,
    OUTCOME_REACH_BLOCKED,
    OUTCOME_SENT,
)
from orion.grammar.acts import build_act
from orion.prime.state import Arm, PrimeStateV1, close_concern, note_change
from orion.schemas.grammar import ActV1

ACTOR = "prime"
SOURCE_SERVICE = "orion-mind"
THOUGHT_COMPONENT = "prime.thought"
NOMINATION_COMPONENT = "prime.nomination"

# The run's own act horizon (services/orion-durable-runs/app/run_acts.py, curiosity_investigate:
# 86400 s from its first GPU grant) on top of the longest a focus waits for a run (FOCUS_TTL, 48 h).
RUN_HORIZON_SEC = 86400.0
NOMINATION_HORIZON_SEC = FOCUS_TTL.total_seconds() + RUN_HORIZON_SEC
NOMINATION_BASIS = (
    "artifact_produced = the investigate run that took this focus (treated) or the first investigate run "
    "started after it (control) wrote something (outcome_kind finished/reached_out_*); horizon = focus TTL "
    "48h (orion/curiosity/focus.py) + the run act's own 86400s (services/orion-durable-runs/app/run_acts.py)"
)
# A concern's horizon: two question horizons. Only one question is open at a time, so a fresh concern
# may first wait out another concern's open question (up to NOMINATION_HORIZON_SEC, 72 h: the 48 h focus
# TTL, measured from the investigate line's longest gap of 21.1 h over 30 days, plus the run act's 24 h)
# and then its own (another 72 h). 144 h. An unkept concern fades after CONCERN_TTL (24 h) long before
# this, so the horizon only binds concerns Prime keeps holding. Live 2026-10-30 there were 2 Prime
# questions in total, too few to measure a settle time from Prime's own record; re-derive once there are.
CONCERN_HORIZON_SEC = 2 * NOMINATION_HORIZON_SEC
CONCERN_COMPONENT = "prime.concern"
CONCERN_BASIS = (
    "artifact_produced = settled: a treated Prime question citing this concern (goal_ref prime_concern:<id>) "
    "settled produced (its investigate run wrote something) within the horizon; 0 when dropped unresolved, "
    "faded (not kept for 24h) or lapsed. horizon = 2 x the question horizon (one question open at a time: "
    "another concern's question may be open first), 2 x 259200s"
)
THOUGHT_MARGIN_SEC = 120.0
THOUGHT_BASIS = (
    "artifact_produced = the model returned a Prime answer that parsed; horizon = the LLM call timeout "
    "+ 120s, the runtime closes the act in the same coroutine"
)

PRODUCED_OUTCOMES = frozenset({OUTCOME_FINISHED, OUTCOME_SENT, OUTCOME_REACH_BLOCKED})
EMPTY_OUTCOMES = frozenset({OUTCOME_EMPTY, OUTCOME_DIED, OUTCOME_CANCELLED})
INVESTIGATE_LINE = "investigate"


def thought_act(thought_id: str, *, llm_timeout_sec: float) -> ActV1:
    return build_act(
        actor=ACTOR, verb="infer", key=thought_id, signal_id="artifact_produced",
        horizon_sec=float(llm_timeout_sec) + THOUGHT_MARGIN_SEC, basis=THOUGHT_BASIS,
        object_ref=f"prime_thought:{thought_id}",
    )


def prime_focus_id(*, question: str, why: str, thought_id: str) -> str:
    """The focus row id Hub will write for this question (deterministic, orion/curiosity/focus.py),
    so the act can name its object before Hub answers -- and a control arm names one never written."""
    return str(focus_id_for(question, why, ORIGIN_SELF, chat_correlation_id(thought_id)))


def chat_correlation_id(thought_id: str) -> str:
    """What Hub stores as the focus row's ``chat_correlation_id``: marks the row as Prime's."""
    return f"prime:{thought_id}"


def draw_arm(focus_id: str, holdback_fraction: float) -> Arm:
    """Deterministic per question, so a retry draws the same arm."""
    u = int(hashlib.sha256(f"prime-holdback|{focus_id}".encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
    return "control" if u < max(0.0, min(1.0, float(holdback_fraction))) else "treated"


def nomination_act(
    *, focus_id: str, arm: Arm, thought_act_id: str, concern_id: Optional[str],
) -> ActV1:
    return build_act(
        actor=ACTOR, verb="ask", key=f"{focus_id}", signal_id="artifact_produced",
        horizon_sec=NOMINATION_HORIZON_SEC, basis=NOMINATION_BASIS,
        object_ref=f"curiosity_focus:{focus_id}", test_before=f"holdback:{arm}",
        parent_act_id=thought_act_id, goal_ref=f"prime_concern:{concern_id}" if concern_id else None,
    )


def concern_act(concern_id: str) -> ActV1:
    """Stable per concern (stable_act_id keyed on concern_id): a restart re-opens or re-closes the same act."""
    return build_act(
        actor=ACTOR, verb="predict", key=concern_id, signal_id="artifact_produced",
        horizon_sec=CONCERN_HORIZON_SEC, basis=CONCERN_BASIS, object_ref=f"prime_concern:{concern_id}",
    )


def lapse_concerns(state: PrimeStateV1, now: datetime) -> PrimeStateV1:
    """Concerns held past their act's horizon with no settling question: the act closes 0 ("lapsed").
    The concern itself stays; Prime decides whether to keep holding it."""
    concerns = []
    for c in state.concerns:
        if (c.graded is None and c.act_opened_at is not None
                and (now - c.act_opened_at).total_seconds() > CONCERN_HORIZON_SEC):
            state = close_concern(state, c, end="lapsed", now=now,
                                  note=f"lapsed: no question about it settled within {CONCERN_HORIZON_SEC / 3600:.0f} h")
            state = note_change(state, f"concern lapsed (held {CONCERN_HORIZON_SEC / 3600:.0f} h, no question "
                                       f"about it settled): {c.text[:120]!r}")
            c = c.model_copy(update={"graded": "lapsed"})
        concerns.append(c)
    return state.model_copy(update={"concerns": concerns})


def settle_concern(state: PrimeStateV1, concern_id: Optional[str], *, arm: Arm, produced: bool, now: datetime,
                   note: str) -> tuple[PrimeStateV1, Optional[str]]:
    """A question citing ``concern_id`` settled. A treated, produced question settles a held concern whose
    act is still open; returns (state, a plain line for Prime) or (state, None) when nothing changed."""
    if not concern_id or arm != "treated" or not produced:
        return state, None
    concern = next((c for c in state.concerns if c.concern_id == concern_id), None)
    if concern is None or concern.graded is not None:
        return state, None
    state = close_concern(state, concern, end="settled", now=now, note=f"settled: {note}")
    concerns = [c.model_copy(update={"graded": "settled"}) if c.concern_id == concern_id else c for c in state.concerns]
    return state.model_copy(update={"concerns": concerns}), f"concern settled: {concern.text[:120]!r}"


@dataclass(frozen=True)
class Settlement:
    produced: bool
    note: str
    run_id: Optional[str] = None


def _started(run: dict[str, Any]) -> Optional[datetime]:
    iso = run.get("started_at_iso")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None


def _graded(run: dict[str, Any], arm: Arm, disposition: Optional[str]) -> Optional[Settlement]:
    kind = str(run.get("outcome_kind") or "")
    rid = str(run.get("run_id") or "")
    tail = f" disposition={disposition}" if disposition else ""
    if kind in PRODUCED_OUTCOMES:
        return Settlement(True, f"{arm} run {rid} {kind}{tail}", rid)
    if kind in EMPTY_OUTCOMES:
        return Settlement(False, f"{arm} run {rid} {kind}{tail}", rid)
    return None  # running, or the stores disagree ("unknown"): wait


# Consecutive settle reads (PRIME_SETTLE_INTERVAL_SEC apart) that must agree a treated focus is gone
# before it is closed: one read can land between a run taking the focus and the run being listed.
GONE_READS_TO_SETTLE = 2


def treated_focus_gone(focus_id: str, curiosity: Optional[dict[str, Any]]) -> Optional[bool]:
    """True when Hub's pending slot does not hold this focus AND no listed run took it: replaced by a
    chat focus, cleared by Juniper, expired, or never written (an RPC with no answer). None = unknown."""
    if not isinstance(curiosity, dict) or not isinstance(curiosity.get("runs"), list):
        return None
    pending = ((curiosity.get("schedule") or {}).get("pending_focus") or {})
    if not pending.get("available"):
        return None
    if ((pending.get("focus") or {}).get("focus_id")) == focus_id:
        return False
    return not any((r.get("focus") or {}).get("focus_id") == focus_id for r in curiosity["runs"])


def settle_nomination(
    *, arm: Arm, focus_id: str, opened_at: datetime, queued: Optional[bool], queue_error: Optional[str],
    runs: Iterable[dict[str, Any]], now: datetime,
) -> Optional[Settlement]:
    """None while the outcome is not known yet. ``runs`` is Hub's GET /curiosity/api/runs list."""
    if arm == "treated" and queued is False:
        return Settlement(False, f"treated not queued: {queue_error or 'refused'}")
    runs = [r for r in runs if r.get("line") == INVESTIGATE_LINE]
    if arm == "treated":
        for run in runs:
            focus = run.get("focus") or {}
            if focus.get("focus_id") == focus_id:
                return _graded(run, arm, focus.get("disposition"))
        if now - opened_at > FOCUS_TTL + timedelta(hours=1):
            return Settlement(False, "treated focus not taken by any run before it expired")
        return None
    later = [r for r in runs if (_started(r) or opened_at) > opened_at]
    if not later:
        return None
    first = min(later, key=lambda r: _started(r) or now)
    return _graded(first, arm, None)
