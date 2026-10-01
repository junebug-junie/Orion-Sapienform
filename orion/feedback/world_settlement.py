"""Score a world-action episode at SETTLE time, on the world sensor, on one clock for both arms.

Spec: docs/superpowers/specs/2026-09-29-attend-to-act-loop-design.md -- D3 and "Amendment
2026-09-29" ("Settle rule", "Clean learning"). Pure: the feedback runtime supplies the episode row,
the cabinet readings, the hardware-watch incidents that opened in the window, and the current priors;
it persists whatever comes back.

- Clock: before = cabinet minute-mean at ``t0`` (decision time); after = minute-mean at
  ``t0 + TTL + 5 min`` (20 min). Intention-to-treat: what the action does in the world, including
  when running background work keeps going.
- Outcome: ``cabinet_heat_pressure`` level change (orion/autonomy/cabinet_heat.py). Never the
  warming signal that triggered the action -- trigger and score are different numbers.
- Overlap tags: ``overlap:reflex`` (a cooling incident opened in the window) -> EXCLUDED from the
  ledger and the posterior in BOTH arms (an AC failure is caused by the AC, so dropping it does not
  select on the outcome). ``overlap:heat_incident`` (CPU/GPU heat) and ``overlap:render_gate``
  (cabinet reached hot) -> KEPT and reported (both can be moved by the treatment; dropping them
  would select on the outcome).
- Treated: only ``expired`` updates the posterior. ``cancelled``/``preempted_by_reflex``/
  ``refused:*``/``settlement_timeout`` are recorded on the episode, never scored.
- Control (randomized holdback): a ledger row with ``arm=randomized_holdback``, no posterior update,
  and one observation into the (signal, randomized_holdback, bin) control cell.
- Loop verdict: Orion writes ONLY the non-final ``acted`` (when the shed actually started), never
  ``resolved``/``dismissed`` -- acting must not silence its own attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from orion.autonomy.cabinet_heat import CABINET_HEAT_SIGNAL, cabinet_heat_pressure, minute_mean
from orion.autonomy.contrast import ControlCell, baseline_bin
from orion.autonomy.prediction import EffectPosterior, score_observation
from orion.autonomy.thermal_gate import DEFAULT_HOT_C
from orion.feedback.outcome_resolution import NO_CHANGE_EPSILON, claim_upheld
from orion.hardware_watch.rules import TempPoint
from orion.schemas.action_prediction import ActionOutcomeRecordV1

SETTLE_TAIL_SEC = 300.0
DEFAULT_TTL_SEC = 900.0
# A treated episode whose dispatch never settled is scored "unsettled" this long after it was due.
UNSETTLED_GRACE_SEC = 600.0
LOOP_VERDICT = "acted"
LOOP_ACTOR = "orion"
CONTROL_ARM = "randomized_holdback"


def _ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            v = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


def overlap_tags(*, incidents: Sequence[Mapping[str, Any]], points: Sequence[TempPoint], start: datetime,
                 end: datetime) -> list[str]:
    tags: set[str] = set()
    for inc in incidents:
        opened = _ts(inc.get("opened_at"))
        if opened is None or not (start <= opened <= end):
            continue
        tags.add("overlap:reflex" if inc.get("rule") == "cooling" else "overlap:heat_incident")
    if any(start <= p.ts <= end and p.value >= DEFAULT_HOT_C for p in points):
        tags.add("overlap:render_gate")
    return sorted(tags)


@dataclass(frozen=True)
class WorldScore:
    outcome: dict[str, Any]
    record: ActionOutcomeRecordV1 | None
    control_cell: tuple[tuple[str, str, int], ControlCell] | None
    loop_outcome: dict[str, Any] | None


def score_world_episode(
    *,
    episode: Mapping[str, Any],
    points: Sequence[TempPoint],
    incidents: Sequence[Mapping[str, Any]],
    prior: EffectPosterior | None,
    control_prior: ControlCell | None,
    now: datetime,
) -> WorldScore | None:
    """None while the episode is not ready (window open, or treated and not yet settled)."""
    t0 = _ts(episode.get("decided_at"))
    if t0 is None:
        return None
    settlement = dict(episode.get("settlement") or {})
    ttl = float(settlement.get("ttl_sec") or DEFAULT_TTL_SEC)
    after_at = t0 + timedelta(seconds=ttl + SETTLE_TAIL_SEC)
    if now < after_at:
        return None
    arm = str(episode.get("arm") or "")
    terminal = str(episode.get("settlement_state") or settlement.get("terminal") or "")
    treated = arm == "treated"
    if treated and terminal in ("", "active", "rpc_error"):
        if now < after_at + timedelta(seconds=UNSETTLED_GRACE_SEC):
            return None
        terminal = "unsettled"

    before_c, after_c = minute_mean(points, t0), minute_mean(points, after_at)
    tags = overlap_tags(incidents=incidents, points=points, start=t0, end=after_at)
    manip = dict(settlement.get("manipulation_check") or {})
    outcome: dict[str, Any] = {
        "signal_id": CABINET_HEAT_SIGNAL, "arm": arm, "terminal": terminal or None,
        "t0": t0.isoformat(), "after_at": after_at.isoformat(),
        "before_temp_c": None if before_c is None else round(before_c, 3),
        "after_temp_c": None if after_c is None else round(after_c, 3),
        "overlap": tags, "manipulation_check": manip, "excluded_reason": None,
    }
    excluded = None
    if before_c is None or after_c is None:
        excluded = "missing_reading:" + ("before" if before_c is None else "after")
    elif "overlap:reflex" in tags:
        excluded = "overlap:reflex"
    elif treated and terminal != "expired":
        excluded = f"terminal:{terminal}"
    acted = treated and bool(manip.get("started_at") or settlement.get("pool_state") == "active")
    loop_outcome = None
    if acted and episode.get("open_loop_id"):
        loop_outcome = {
            "loop_id": str(episode["open_loop_id"]), "verdict": LOOP_VERDICT, "actor": LOOP_ACTOR,
            "note": f"shed_background_gpu {terminal}"[:500],
            "features_at_close": {"episode_id": episode.get("episode_id"), "terminal": terminal,
                                  "overlap": tags, "arm": arm},
        }
    if excluded is not None:
        outcome["excluded_reason"] = excluded
        if loop_outcome is not None:
            loop_outcome["features_at_close"]["excluded_reason"] = excluded
        return WorldScore(outcome=outcome, record=None, control_cell=None, loop_outcome=loop_outcome)

    baseline = float(cabinet_heat_pressure(before_c))
    observed_after = float(cabinet_heat_pressure(after_c))
    delta = observed_after - baseline
    bin_index = baseline_bin(baseline)
    expected = dict(episode.get("expected_effect") or {})
    predicted = float(expected.get("predicted_delta") or 0.0)
    direction = str(expected.get("direction") or "decrease")
    prior = prior or EffectPosterior.cold()
    control_cell = None
    if treated:
        posterior, surprise, _ = score_observation(prior, delta)
        record_arm = "dispatched"
    else:
        posterior, surprise, record_arm = prior, 0.0, CONTROL_ARM
        cell = control_prior or ControlCell(EffectPosterior.cold())
        new_post, _n, _r = score_observation(cell.posterior, delta)
        control_cell = ((CABINET_HEAT_SIGNAL, CONTROL_ARM, bin_index),
                        cell.observe(new_post, moved=abs(delta) >= NO_CHANGE_EPSILON))
    record = ActionOutcomeRecordV1(
        dispatch_id=str(episode["episode_id"]),
        dispatch_frame_id=str(episode.get("dispatch_frame_id") or ""),
        feedback_frame_id=f"world_settle:{episode['episode_id']}",
        dispatch_kind=str(episode.get("dispatch_kind") or "self_regulate"),
        target_id=str(episode.get("target_id") or ""),
        signal_id=CABINET_HEAT_SIGNAL,
        direction=direction,  # type: ignore[arg-type]
        arm=record_arm,  # type: ignore[arg-type]
        baseline_bin=bin_index,
        frame_dispatch_count=1 if treated else 0,
        observed_at=after_at,
        baseline=baseline,
        observed_after=observed_after,
        observed_delta=delta,
        predicted_delta=predicted,
        prediction_error=delta - predicted,
        surprise_nats=max(0.0, float(surprise)),
        posterior_mean=posterior.mean,
        posterior_variance=posterior.variance,
        posterior_n=posterior.n,
        claim_upheld=claim_upheld(direction, delta),
        co_predictors=0,
        latency_ms=None,
    )
    outcome.update({"baseline": round(baseline, 4), "observed_after": round(observed_after, 4),
                    "observed_delta": round(delta, 4), "predicted_delta": predicted,
                    "surprise_nats": round(float(surprise), 6), "baseline_bin": bin_index,
                    "posterior_updated": treated})
    if loop_outcome is not None:
        loop_outcome["features_at_close"]["observed_delta"] = round(delta, 4)
    return WorldScore(outcome=outcome, record=record, control_cell=control_cell, loop_outcome=loop_outcome)


__all__ = ["CONTROL_ARM", "LOOP_ACTOR", "LOOP_VERDICT", "WorldScore", "overlap_tags", "score_world_episode"]
