"""The cooling world act (``shed_background_gpu``), decided by the want market. Pure: no I/O, no clock.

Spec: docs/superpowers/specs/2026-11-04-retire-proposal-policy-dispatch-rungs.md, "PR 5" and "Decided"
item 1 (Juniper, 2026-11-05: the cooling act BINDS in the market from day one; the 50% control group
stays; the pool's thermal reflex stays separate). Until this PR the act's only trigger was execution
dispatch (an attention-workspace winner on the cabinet/biometrics node). Now
``services/orion-durable-runs/app/cooling_act.py`` runs it on its own clock:

1. **Eligibility** -- unchanged: ``orion.autonomy.self_shed.evaluate_shed_eligibility`` (cabinet
   elevated/hot/unknown and not critical, hardware-watch idle, background work present, nothing in
   flight), then the arm-symmetric pool admission (``admission_refusal``, moved here from dispatch).
2. **The event want** -- "the cabinet is warming" is an unusual body moment, so the loop posts an event
   want (family ``self_regulate.shed``, ``event_ref`` = the cabinet reading it is about, ``salience`` =
   ``cabinet_salience``).
3. **The binding rule** (``shed_verdict``) -- funded: proceed. A deferral is "not now": the want stays
   open and bids again, until its event expires. No answer (timeout, error, stale prices): do NOT shed;
   the pool's reflex is the gate that remains. "No answer" is never "funded".
4. **The experiment, exactly as before** -- the same 0.5 treated/control draw on the same hash rule
   (``stable_draw``), the same precommitted ``substrate_world_action_episodes`` row with the eligibility
   snapshot and expected effect (``episode_row``) before the shed RPC, the same settlement
   (``orion/autonomy/shed_settlement.py``). The scorer in orion-sql-writer
   (``orion/autonomy/world_settlement_store.py``) reads the same table and is unchanged.

The event and its salience (metric quality gate in the PR report, docs/superpowers/pr-reports/2026-11-05-
cooling-act-market-pr.md). The event is a WARMING moment (``is_warming``): the cabinet is in Orion's band and
rising by at least 0.5 C in 15 minutes -- the same rise (``cabinet_rise_c``) and threshold that made the
cabinet node win the attention workspace, which was the old trigger. Its magnitude is that rise (C). The
source's history is every warming minute of the last 30 days (``warming_minutes``); salience = the share of
them with a strictly smaller rise (``orion.physiology.event_wants.source_salience``). Not
``cabinet_heat_pressure``: that is the act's OUTCOME (scored as its level change), and the cabinet-heat
module keeps trigger and score different numbers. Rest point: 0 at the smallest usual rise (just at the
threshold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.autonomy.cabinet_heat import (CABINET_HEAT_SIGNAL, CABINET_NODE, DEFAULT_RISE_THRESHOLD_C, MINUTE_SEC,
                                         RISE_WINDOW_SEC)
from orion.autonomy import world_episodes
from orion.autonomy.contrast import pooled_treated_mean
from orion.autonomy.prediction import EffectPosterior
from orion.autonomy.self_shed import SHED_TEMPLATE
from orion.autonomy.thermal_gate import ThermalState, thermal_state
from orion.hardware_watch.rules import TempPoint, cabinet_rise_c
from orion.physiology.binding import bind_verdict
from orion.physiology.event_wants import SOURCE_HISTORY, event_want, source_salience
from orion.schemas.action_prediction import ExpectedEffectV1
from orion.schemas.physiology_want import WantV1

# --- the want ----------------------------------------------------------------------------------------
FAMILY = "self_regulate.shed"
VERB = "regulate"
TIER = 1                      # a PROPOSAL: Juniper signs off on the tier in PR review (families.yaml)
# Its own producer, so want_ledger (whose ``shadow`` is always true) tells this obeyed decision from a
# shadow one (orion.physiology.binding.BINDING_PRODUCERS).
PRODUCER = "orion-durable-runs.cooling_act.binding"
SOURCE = "cabinet_warm"
EVENT_PREFIX = "cabinet_reading:"
# How long one warm moment stays the same event: the shed's own TTL and the 15-minute rise window. Past
# it the cabinet has moved on; the loop posts a new event if the cabinet is still warm and eligible.
EVENT_WINDOW_SEC = 900.0
WARM_STATES: frozenset[str] = frozenset({"elevated", "hot"})

# --- the act (the same names the old dispatch path wrote, so one ledger holds both eras) --------------
TEMPLATE = SHED_TEMPLATE
DISPATCH_KIND = "self_regulate"
TARGET_ID = "pool:background_gpu"
EXPECTED_DIRECTION = "decrease"
HOLDBACK_FRACTION = 0.5
# t0 -> t0 + TTL + 5 min: both arms score on this clock (attend-to-act design, "Settle rule").
SCORING_TAIL_SEC = 300.0
# The pool refusals that mean "the pool will refuse the next one too" for an hour.
POOL_REFUSING = ("disabled", "lever_disabled", "ledger_unavailable")


def _utc(v: datetime) -> datetime:
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


# --- the event and its salience ------------------------------------------------------------------------
def is_warming(cabinet: Any) -> bool:
    """The event: the cabinet is in Orion's band (the thermal gate reads elevated or hot) AND rising by at least
    its rise threshold within 15 minutes (``cabinet_rise_c``, the reflex's own definition of "rising"). Unknown is
    never warming (no reading, no rise), and critical is refused by eligibility (the reflex covers both)."""
    rise = getattr(cabinet, "rise_c", None)
    return (getattr(cabinet, "thermal_state", None) in WARM_STATES and rise is not None
            and rise >= float(getattr(cabinet, "rise_threshold_c", DEFAULT_RISE_THRESHOLD_C)))


def warming_minutes(points: Sequence[TempPoint], *, rise_threshold_c: float = DEFAULT_RISE_THRESHOLD_C
                    ) -> list[tuple[datetime, float]]:
    """The source's own history: (minute, rise C) for every minute whose last reading was a warming moment --
    the thermal gate, folded with its hysteresis over every reading (ascending), reads elevated or hot, and
    ``cabinet_rise_c`` over the 15 minutes before that reading is at least the threshold. Raw readings, the
    same ones the live loop's rise is computed on, so history and the live magnitude are one measure."""
    pts = sorted((p for p in points if math.isfinite(p.value)), key=lambda p: _utc(p.ts))
    out: list[tuple[datetime, float]] = []
    state: ThermalState = "normal"
    j = 0
    for i, p in enumerate(pts):
        state = thermal_state(temp_c=p.value, age_sec=0.0, previous_state=state).state
        ts = _utc(p.ts)
        minute = ts.replace(second=0, microsecond=0)
        if i + 1 < len(pts) and _utc(pts[i + 1].ts).replace(second=0, microsecond=0) == minute:
            continue                              # emit once, at the minute's last reading
        while (ts - _utc(pts[j].ts)).total_seconds() > RISE_WINDOW_SEC:
            j += 1
        if state not in WARM_STATES:
            continue
        rise = cabinet_rise_c(pts[j:i + 1], ts, RISE_WINDOW_SEC)
        if rise is not None and rise >= rise_threshold_c:
            out.append((minute, float(rise)))
    return out


def cabinet_salience(rise_c: Optional[float], history: Sequence[tuple[datetime, float]], now: datetime) -> Optional[float]:
    """Share of the warming minutes of the 30 days before ``now`` with a strictly smaller rise than ``rise_c``.
    None: no rise, or fewer than MIN_SOURCE_HISTORY warming minutes (no reference: post nothing)."""
    if rise_c is None:
        return None
    since = now - SOURCE_HISTORY
    return source_salience(float(rise_c), [r for at, r in history if since <= at < now - timedelta(seconds=MINUTE_SEC)])


def event_ref_for(reading_at: datetime) -> str:
    """The cabinet reading the event is about: ``cabinet_reading:<node>:<UTC second>``."""
    return f"{EVENT_PREFIX}{CABINET_NODE}:{_utc(reading_at).strftime('%Y-%m-%dT%H:%M:%SZ')}"


def cooling_want(*, reading_at: datetime, salience: float, opened_at: datetime) -> WantV1:
    """The event want for one warm moment. Deterministic id per reading (a restart re-finds it)."""
    ref = event_ref_for(reading_at)
    return event_want(source=SOURCE, key=ref, family=FAMILY, producer=PRODUCER, verb=VERB, tier=TIER,
                      event_ref=ref, salience=salience, opened_at=opened_at,
                      latest=opened_at + timedelta(seconds=EVENT_WINDOW_SEC), object_ref=TARGET_ID)


# --- the binding rule ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ShedVerdict:
    """``proceed``: draw the arm now. ``close``: the event is over (no shed for it). Otherwise wait until
    ``retry_at`` and bid the same want again."""
    proceed: bool
    close: bool
    why: str
    retry_at: Optional[datetime] = None
    detail: Mapping[str, Any] | None = None


# A failed bid is retried after one physiology tick: a sooner bid reads the same prices.
NO_ANSWER_RETRY_SEC = 120.0


def shed_verdict(decision: Any, now: datetime, end: datetime) -> ShedVerdict:
    """Juniper's rule (2026-11-05) on top of the shared ``bind_verdict``: ONLY ``funded`` proceeds.

    * funded -> proceed;
    * a deferral (price, sated, capacity) -> hold until the market's retry time (``bind_verdict``'s clamp),
      never past the event's end;
    * the event's end reached -> close: a warm moment that has passed is not shed late;
    * no answer (no decision, ``physiology_unavailable``, unknown family, an unbounded hold) -> do NOT
      shed; retry after one physiology tick while the event lasts. Unlike curiosity's rule, "no answer"
      never goes ahead: the pool's reflex is the gate that remains.
    * ``queued_at_pool`` (a capacity deferral with a positive net) does not apply: the act holds no lane.
      It is treated as a deferral, never as funded."""
    v = bind_verdict(decision, now, end)
    # The event's end comes FIRST: the market still funds a want whose window has passed (it only warns
    # window_passed), and a funded bid on a passed moment would shed late (review 2026-11-05).
    if now >= end or v.why == "window_end":
        return ShedVerdict(False, True, "event_expired", detail=v.detail)
    if v.why == "funded":
        return ShedVerdict(True, False, "funded", detail=v.detail)
    if v.why.startswith("no_answer"):
        return ShedVerdict(False, False, v.why, retry_at=min(end, now + timedelta(seconds=NO_ANSWER_RETRY_SEC)),
                           detail=v.detail)
    retry = v.retry_at or min(end, now + timedelta(seconds=NO_ANSWER_RETRY_SEC))
    why = v.why if v.why.startswith("deferred") else f"deferred:{v.why}"
    return ShedVerdict(False, False, why, retry_at=min(end, retry), detail=v.detail)


# --- the experiment (moved unchanged from services/orion-execution-dispatch-runtime/app/worker.py) -----
def stable_draw(episode_id: str) -> float:
    """The ledger's one draw rule (``orion.autonomy.world_episodes.stable_draw``): a replay keeps its arm."""
    return world_episodes.stable_draw(episode_id)


def arm_for(episode_id: str, holdback_fraction: float = HOLDBACK_FRACTION) -> str:
    return "control" if holdback_fraction > 0.0 and stable_draw(episode_id) < holdback_fraction else "treated"


def episode_id_for(want_id: str) -> str:
    """The episode (and the shed RPC's dispatch_id) for one funded cooling want."""
    return f"cooling_act:{want_id}"


def admission_refusal(*, in_flight: Sequence[str], treated: Iterable[Mapping[str, Any]], now: datetime,
                      ttl_sec: float, min_gap_sec: float, max_sec_per_day: float) -> Optional[str]:
    """Arm-symmetric admission, BEFORE the draw: anything the pool would refuse, and anything still in
    flight, is neither treated nor control. ``treated``: the treated episodes of the last 24 h."""
    if in_flight:
        return "world_action_in_flight"
    used, last_end, last_refusal = 0.0, None, None
    for row in treated:
        state = str(row.get("settlement_state") or "")
        if state.startswith("refused"):
            last_refusal = (row["decided_at"], state)
            continue
        settlement = row.get("settlement") or {}
        check = settlement.get("manipulation_check") or {}
        start = datetime.fromisoformat(check["started_at"]) if check.get("started_at") else row["decided_at"]
        end = datetime.fromisoformat(check["ended_at"]) if check.get("ended_at") else start + timedelta(seconds=ttl_sec)
        used += max(0.0, (min(end, now) - start).total_seconds())
        last_end = end if last_end is None else max(last_end, end)
    if last_refusal and (now - last_refusal[0]).total_seconds() < 3600 and last_refusal[1].split(":", 1)[-1] in POOL_REFUSING:
        return f"pool_refusing:{last_refusal[1].split(':', 1)[-1]}"
    if last_end is not None and (now - last_end).total_seconds() < min_gap_sec:
        return "pool_gap"
    if max_sec_per_day - used < min(ttl_sec, 300.0):
        return "pool_daily_cap"
    return None


def expected_effect(posteriors: Mapping[tuple, EffectPosterior] | None) -> ExpectedEffectV1:
    """The claim, precommitted on the episode: the pooled treated posterior of this act on the cabinet
    (``pooled_treated_mean``, the same rule dispatch's builder used). Cold prior when there is none."""
    posterior = pooled_treated_mean(dict(posteriors or {}), DISPATCH_KIND, TARGET_ID, CABINET_HEAT_SIGNAL)
    cold = posterior is None or posterior.n == 0
    posterior = posterior or EffectPosterior.cold()
    return ExpectedEffectV1(signal_id=CABINET_HEAT_SIGNAL, direction=EXPECTED_DIRECTION,  # type: ignore[arg-type]
                            predicted_delta=posterior.mean, predictor_variance=posterior.variance,
                            predictor_n=posterior.n, cold_start=cold)


def episode_row(*, episode_id: str, arm: str, now: datetime, eligibility: Mapping[str, Any],
                effect: Optional[ExpectedEffectV1], decision_id: Optional[str], ttl_sec: float,
                settlement: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The precommit row (``orion.autonomy.world_episodes.insert_decision``). No attention loop, proposal or
    dispatch frame any more: ``decision_id`` is the market's decision, and the eligibility snapshot carries
    the want (``eligibility["want"]``)."""
    return {
        "episode_id": episode_id, "template": TEMPLATE, "dispatch_kind": DISPATCH_KIND, "target_id": TARGET_ID,
        "arm": arm, "decided_at": now, "open_loop_id": None, "broadcast_log_id": None, "node_id": None,
        "proposal_id": None, "decision_id": decision_id, "dispatch_frame_id": None,
        "eligibility": dict(eligibility), "expected_effect": effect.model_dump(mode="json") if effect else None,
        "settlement": {"ttl_sec": ttl_sec, **dict(settlement or {})},
        "scoring_due_at": now + timedelta(seconds=ttl_sec + SCORING_TAIL_SEC),
    }


__all__ = ["DISPATCH_KIND", "EVENT_PREFIX", "EVENT_WINDOW_SEC", "FAMILY", "HOLDBACK_FRACTION", "NO_ANSWER_RETRY_SEC",
           "PRODUCER", "SCORING_TAIL_SEC", "SOURCE", "ShedVerdict", "TARGET_ID", "TEMPLATE", "TIER", "VERB",
           "admission_refusal", "arm_for", "cabinet_salience", "cooling_want", "episode_id_for", "episode_row",
           "event_ref_for", "expected_effect", "is_warming", "shed_verdict", "stable_draw", "warming_minutes"]
