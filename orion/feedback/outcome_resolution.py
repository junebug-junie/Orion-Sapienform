"""Score the predictions an action made, once its field window has closed.

This is the half of the loop that did not exist before 2026-08-21. The
dispatch runtime writes ~5,400 real actions a day; the feedback runtime
writes ~32,000 outcome observations a day; and NOTHING read either one back
into what Orion chooses to do next (verified by grep: neither
`orion/proposals/` nor `orion/execution_dispatch/` referenced a feedback
frame or an outcome score). The ledger this module produces is the first
artifact in the autonomy path that can answer "did that action do anything".

Pure functions only -- no I/O. The caller supplies the field snapshots and
the prior posteriors and persists whatever comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.autonomy.prediction import EffectPosterior, score_observation
from orion.field.pressure import field_pressures
from orion.schemas.action_prediction import ActionOutcomeRecordV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.field_state import FieldStateV1

PosteriorKey = tuple[str, str, str]

# Below this, a delta is not a movement. Same 1e-6 dead band
# orion.feedback.extractors.classify_pressure_deltas already applies to these
# exact channels -- reused rather than invented so "moved" means one thing
# across the feedback path, not two.
NO_CHANGE_EPSILON = 1e-6


def claim_upheld(direction: str, observed_delta: float) -> bool | None:
    """Did the action do what it said it would?

    Returns None only for a directional claim whose observed delta landed
    inside the dead band -- genuinely undecidable, not a soft pass. A
    `no_change` claim is decidable in both directions and never returns None.
    """
    moved = abs(observed_delta) >= NO_CHANGE_EPSILON
    if direction == "no_change":
        return not moved
    if not moved:
        return None
    return observed_delta > 0 if direction == "increase" else observed_delta < 0


@dataclass(frozen=True)
class OutcomeResolution:
    """Records that could be scored, plus an honest account of the rest."""

    records: list[ActionOutcomeRecordV1]
    posteriors: dict[PosteriorKey, EffectPosterior]
    # dispatch_id -> why it produced no record. Never silently dropped:
    # "no action was scorable this tick" and "this tick had no actions" are
    # different facts and used to be indistinguishable.
    skipped: dict[str, str]


def posterior_key(record_kind: str, target_id: str, signal_id: str) -> PosteriorKey:
    return (record_kind, target_id, signal_id)


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


def resolve_action_outcomes(
    *,
    dispatch_frame: ExecutionDispatchFrameV1,
    feedback_frame_id: str,
    field_before: FieldStateV1 | None,
    field_after: FieldStateV1 | None,
    priors: dict[PosteriorKey, EffectPosterior] | None = None,
    latency_by_dispatch_id: dict[str, float] | None = None,
    now: datetime | None = None,
) -> OutcomeResolution:
    observed_at = now or datetime.now(timezone.utc)
    working: dict[PosteriorKey, EffectPosterior] = dict(priors or {})
    records: list[ActionOutcomeRecordV1] = []
    skipped: dict[str, str] = {}

    candidates = list(dispatch_frame.dispatched_candidates)
    if not candidates:
        return OutcomeResolution(records=[], posteriors={}, skipped={})

    before = _present_pressures(field_before)
    after = _present_pressures(field_after)

    # Attribution bookkeeping: the field delta is frame-wide, so when N
    # candidates in one tick claim the same signal, none of them alone
    # produced it. Counted once up front over the whole frame so every
    # record in the frame agrees on the number.
    claim_counts: dict[str, int] = {}
    for candidate in candidates:
        effect = candidate.expected_effect
        if effect is not None:
            claim_counts[effect.signal_id] = claim_counts.get(effect.signal_id, 0) + 1

    for candidate in candidates:
        effect = candidate.expected_effect
        if effect is None:
            skipped[candidate.dispatch_id] = "no_declared_signal"
            continue
        signal = effect.signal_id
        if field_before is None or field_after is None:
            skipped[candidate.dispatch_id] = "missing_field_window"
            continue
        if signal not in before or signal not in after:
            # Real absence, not a zero. See _present_pressures.
            skipped[candidate.dispatch_id] = f"signal_absent_from_field:{signal}"
            continue

        baseline = float(before[signal])
        observed_after = float(after[signal])
        observed_delta = observed_after - baseline

        key = posterior_key(candidate.dispatch_kind, candidate.target_id, signal)
        prior = working.get(key) or EffectPosterior.cold()
        posterior, surprise, _residual_vs_prior = score_observation(prior, observed_delta)
        working[key] = posterior

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
                co_predictors=max(claim_counts.get(signal, 1) - 1, 0),
                latency_ms=(latency_by_dispatch_id or {}).get(candidate.dispatch_id),
            )
        )

    return OutcomeResolution(
        records=records,
        posteriors={
            posterior_key(r.dispatch_kind, r.target_id, r.signal_id): EffectPosterior(
                mean=r.posterior_mean, variance=r.posterior_variance, n=r.posterior_n
            )
            for r in records
        },
        skipped=skipped,
    )
