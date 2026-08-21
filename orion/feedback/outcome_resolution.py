"""Score the predictions an action made, once its field window has closed.

This is the half of the loop that did not exist before 2026-08-21. The
dispatch runtime writes ~5,400 real actions a day; the feedback runtime
writes ~32,000 outcome observations a day; and NOTHING read either one back
into what Orion chooses to do next (verified by grep: neither
`orion/proposals/` nor `orion/execution_dispatch/` referenced a feedback
frame or an outcome score). The ledger this module produces is the first
artifact in the autonomy path that can answer "did that action do anything".

Phase 2 (2026-08-21, same day) adds the half that makes the answer mean
anything: a control arm. `resolve_action_outcomes` now emits, alongside the
treated records, an untreated observation for every signal on every tick
where NOTHING was dispatched. See orion.autonomy.contrast for why that is
the only honest control population available and why the design spec's
capacity-blocked arm is not one.

Pure functions only -- no I/O. The caller supplies the field snapshots and
the prior posteriors and persists whatever comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.autonomy.contrast import (
    MOVEMENT_EPSILON,
    ControlCell,
    ControlCellKey,
    TreatedCellKey,
    baseline_bin,
)
from orion.autonomy.prediction import EffectPosterior, score_observation
from orion.field.pressure import field_pressures
from orion.schemas.action_prediction import ActionOutcomeRecordV1
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
from orion.schemas.field_state import FieldStateV1

# Below this, a delta is not a movement. Same 1e-6 dead band
# orion.feedback.extractors.classify_pressure_deltas already applies to these
# exact channels -- reused rather than invented so "moved" means one thing
# across the feedback path, not two.
NO_CHANGE_EPSILON = 1e-6

# A blocked candidate counts as a lost capacity race only for this reason.
# `requires_operator_review` and deferral blocks are excluded on purpose:
# those are blocked for reasons correlated with the action's own content,
# which is a worse confounder than the one being removed.
CAPACITY_BLOCK_PREFIX = "max_dispatch_candidates:"


def claim_upheld(direction: str, observed_delta: float) -> bool | None:
    """Did the signal do what the action said it would?

    Returns None only for a directional claim whose observed delta landed
    inside the dead band -- genuinely undecidable, not a soft pass. A
    `no_change` claim is decidable in both directions and never returns None.

    Computed for control-arm records too, and that is the point: if a claim
    is "upheld" just as often on ticks where the action did NOT run, the
    claim is describing the weather, not the action.
    """
    moved = abs(observed_delta) >= NO_CHANGE_EPSILON
    if direction == "no_change":
        return not moved
    if not moved:
        return None
    return observed_delta > 0 if direction == "increase" else observed_delta < 0


@dataclass(frozen=True)
class ControlObservation:
    """One untreated reading of a signal, from a tick where nothing ran.

    Not a ledger row. Emitting one row per (tick, signal) would add ~128k
    rows/day for a quantity that is only ever read in aggregate; these fold
    straight into the control cell posteriors instead, which is O(1) writes
    and keeps the ledger growing at the real action rate.
    """

    signal_id: str
    arm: str
    baseline_bin: int
    baseline: float
    observed_after: float
    observed_delta: float
    observed_at: datetime


@dataclass(frozen=True)
class OutcomeResolution:
    """Records that could be scored, plus an honest account of the rest."""

    records: list[ActionOutcomeRecordV1]
    # Treated-arm cells, keyed (dispatch_kind, target_id, signal_id, bin).
    posteriors: dict[TreatedCellKey, EffectPosterior]
    # Control-arm cells, keyed (signal_id, arm, bin).
    control_posteriors: dict[ControlCellKey, ControlCell]
    control_observations: list[ControlObservation]
    # dispatch_id -> why it produced no record. Never silently dropped:
    # "no action was scorable this tick" and "this tick had no actions" are
    # different facts and used to be indistinguishable.
    skipped: dict[str, str]


def posterior_key(
    record_kind: str, target_id: str, signal_id: str, bin_index: int
) -> TreatedCellKey:
    return (record_kind, target_id, signal_id, bin_index)


def control_key(signal_id: str, arm: str, bin_index: int) -> ControlCellKey:
    return (signal_id, arm, bin_index)


def _present_pressures(field: FieldStateV1 | None) -> dict[str, float]:
    """Read the field's pressures WITHOUT filling absent keys with 0.0.

    Deliberately not orion.feedback.extractors.extract_field_pressure_snapshot:
    that helper returns 0.0 for any channel the field did not produce, which
    for a scored prediction is not graceful degradation -- it manufactures an
    observed delta of exactly 0.0 and would confirm a `no_change` claim that
    was never actually measured. `sustained_load_pressure` is the live case:
    it is throttled and present only on the ticks that genuinely recomputed
    it (see orion/field/pressure.py), so most ticks would score it as a
    perfect, fabricated confirmation.
    """
    if field is None:
        return {}
    return dict(field_pressures(field))


def _is_capacity_blocked(candidate: ExecutionDispatchCandidateV1) -> bool:
    return any(str(b).startswith(CAPACITY_BLOCK_PREFIX) for b in candidate.blocked_by)


def resolve_action_outcomes(
    *,
    dispatch_frame: ExecutionDispatchFrameV1,
    feedback_frame_id: str,
    field_before: FieldStateV1 | None,
    field_after: FieldStateV1 | None,
    priors: dict[TreatedCellKey, EffectPosterior] | None = None,
    control_priors: dict[ControlCellKey, ControlCell] | None = None,
    latency_by_dispatch_id: dict[str, float] | None = None,
    now: datetime | None = None,
) -> OutcomeResolution:
    observed_at = now or datetime.now(timezone.utc)
    working: dict[TreatedCellKey, EffectPosterior] = dict(priors or {})
    control_working: dict[ControlCellKey, ControlCell] = dict(control_priors or {})
    records: list[ActionOutcomeRecordV1] = []
    control_observations: list[ControlObservation] = []
    skipped: dict[str, str] = {}
    touched_control: set[ControlCellKey] = set()

    before = _present_pressures(field_before)
    after = _present_pressures(field_after)
    have_window = field_before is not None and field_after is not None

    dispatched = list(dispatch_frame.dispatched_candidates)
    blocked = [c for c in dispatch_frame.blocked_candidates if _is_capacity_blocked(c)]
    frame_dispatch_count = len(dispatched)

    # The control condition is "nothing ran this tick", not "nothing claiming
    # this signal ran". 5 of 16 live templates declare no signal at all and
    # account for 72% of dispatch volume -- an undeclared action still acts,
    # so a tick containing one is not untreated for any signal.
    if have_window and frame_dispatch_count == 0:
        for signal, baseline in before.items():
            if signal not in after:
                continue
            observed_after = float(after[signal])
            baseline = float(baseline)
            bin_index = baseline_bin(baseline)
            key = control_key(signal, "no_action", bin_index)
            cell = control_working.get(key) or ControlCell(EffectPosterior.cold(), 0)
            delta = observed_after - baseline
            posterior, _nats, _residual = score_observation(cell.posterior, delta)
            # Movement is counted, not inferred. A Normal-Normal posterior
            # with fixed observation variance shrinks as 1/n whether the
            # data varies or is one constant repeated, so a frozen channel
            # would otherwise produce the most confident cell in the table.
            control_working[key] = ControlCell(
                posterior=posterior,
                moved_n=cell.moved_n + (1 if abs(delta) >= MOVEMENT_EPSILON else 0),
            )
            touched_control.add(key)
            control_observations.append(
                ControlObservation(
                    signal_id=signal,
                    arm="no_action",
                    baseline_bin=bin_index,
                    baseline=baseline,
                    observed_after=observed_after,
                    observed_delta=observed_after - baseline,
                    observed_at=observed_at,
                )
            )

    scorable = [(c, "dispatched") for c in dispatched] + [
        (c, "capacity_blocked") for c in blocked
    ]
    if not scorable:
        return OutcomeResolution(
            records=[],
            posteriors={},
            control_posteriors={k: control_working[k] for k in touched_control},
            control_observations=control_observations,
            skipped={},
        )

    # Attribution bookkeeping: the field delta is frame-wide, so when N
    # dispatched candidates in one tick claim the same signal, none of them
    # alone produced it. Counted over the DISPATCHED arm only -- a blocked
    # candidate did not act, so it is not a co-cause of anything. Counted
    # once up front so every record in the frame agrees on the number.
    claim_counts: dict[str, int] = {}
    for candidate in dispatched:
        effect = candidate.expected_effect
        if effect is not None:
            claim_counts[effect.signal_id] = claim_counts.get(effect.signal_id, 0) + 1

    for candidate, arm in scorable:
        effect = candidate.expected_effect
        if effect is None:
            skipped[candidate.dispatch_id] = "no_declared_signal"
            continue
        signal = effect.signal_id
        if not have_window:
            skipped[candidate.dispatch_id] = "missing_field_window"
            continue
        if signal not in before or signal not in after:
            # Real absence, not a zero. See _present_pressures.
            skipped[candidate.dispatch_id] = f"signal_absent_from_field:{signal}"
            continue

        baseline = float(before[signal])
        observed_after = float(after[signal])
        observed_delta = observed_after - baseline
        bin_index = baseline_bin(baseline)

        # Only the treated arm advances the belief about what the ACTION
        # does. A capacity-blocked candidate never ran; folding its reading
        # into the action's own posterior would be recording the weather as
        # the action's effect, which is the entire defect this arm exists to
        # expose. Its row is written, its cell is not.
        if arm == "dispatched":
            key = posterior_key(
                candidate.dispatch_kind, candidate.target_id, signal, bin_index
            )
            prior = working.get(key) or EffectPosterior.cold()
            posterior, surprise, _residual_vs_prior = score_observation(
                prior, observed_delta
            )
            working[key] = posterior
        else:
            posterior = working.get(
                posterior_key(
                    candidate.dispatch_kind, candidate.target_id, signal, bin_index
                )
            ) or EffectPosterior.cold()
            surprise = 0.0

        # Finding 6 (review, 2026-08-21): this used to store
        # score_observation's residual, which is measured against
        # `prior.mean` -- the belief at SCORING time. `predicted_delta` on
        # the row is the claim recorded at DISPATCH time. Whenever the
        # posterior moved in between (routinely), the two disagreed, and a
        # reader recomputing `observed_delta - predicted_delta` from the row
        # got a different number, sometimes with the opposite sign. The
        # error of a prediction has to be measured against that prediction.
        error = observed_delta - effect.predicted_delta

        records.append(
            ActionOutcomeRecordV1(
                dispatch_id=candidate.dispatch_id,
                dispatch_frame_id=dispatch_frame.frame_id,
                feedback_frame_id=feedback_frame_id,
                dispatch_kind=candidate.dispatch_kind,
                target_id=candidate.target_id,
                signal_id=signal,
                direction=effect.direction,
                arm=arm,  # type: ignore[arg-type]
                baseline_bin=bin_index,
                frame_dispatch_count=frame_dispatch_count,
                observed_at=observed_at,
                baseline=baseline,
                observed_after=observed_after,
                observed_delta=observed_delta,
                # The prediction as it was RECORDED on the candidate before
                # the action ran -- not prior.mean, which can differ if an
                # earlier candidate in this same frame already advanced the
                # posterior for this key. Scoring against anything other
                # than what was actually claimed is not scoring a prediction.
                predicted_delta=effect.predicted_delta,
                prediction_error=error,
                surprise_nats=surprise,
                posterior_mean=posterior.mean,
                posterior_variance=posterior.variance,
                posterior_n=posterior.n,
                claim_upheld=claim_upheld(effect.direction, observed_delta),
                co_predictors=max(claim_counts.get(signal, 0) - 1, 0)
                if arm == "dispatched"
                else claim_counts.get(signal, 0),
                latency_ms=(latency_by_dispatch_id or {}).get(candidate.dispatch_id),
            )
        )

    return OutcomeResolution(
        records=records,
        posteriors={
            posterior_key(
                r.dispatch_kind, r.target_id, r.signal_id, r.baseline_bin
            ): EffectPosterior(
                mean=r.posterior_mean, variance=r.posterior_variance, n=r.posterior_n
            )
            for r in records
            if r.arm == "dispatched"
        },
        control_posteriors={k: control_working[k] for k in touched_control},
        control_observations=control_observations,
        skipped=skipped,
    )
