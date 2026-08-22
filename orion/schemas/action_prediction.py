"""Predictions attached to autonomous actions, and their scored outcomes.

An action that does not say what it expects to change cannot be wrong, so it
cannot be learned from, so it repeats forever. These two models are the
contract that makes an Orion dispatch falsifiable:

  ExpectedEffectV1     -- written BEFORE the action runs, onto the dispatch
                          candidate. A claim.
  ActionOutcomeRecordV1 -- written AFTER the action's field window closes.
                          The claim, scored.

See orion/autonomy/prediction.py for the theory and the arithmetic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.autonomy.contrast import ActionArm

# The signals an action is allowed to make a claim about. These are exactly
# the field pressure channels the feedback runtime already snapshots before
# and after every dispatch (config/feedback/feedback_policy.v1.yaml's
# `pressure_channels`, plus the two derived scalars carried on FieldStateV1
# itself). Deliberately closed: a claim about a signal nothing measures is
# not a claim, and a free-text signal_id would let one back in.
PredictableSignal = Literal[
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "deviation_pressure",
    "sustained_load_pressure",
]

EffectDirection = Literal["increase", "decrease", "no_change"]


class ExpectedEffectV1(BaseModel):
    """What an action claims it will do, recorded before it does it.

    `predicted_delta` is NOT hand-authored. It is the current posterior mean
    for this (dispatch_kind, target_id, signal_id) triple -- i.e. what the
    action has actually done to this signal so far. A template author
    declares only `signal_id` and `direction`: which signal this action
    touches, and which way they believe it moves. That is the part a human
    genuinely knows and code cannot infer. Magnitude is measured, never
    typed -- typing it would repeat the exact defect this patch exists to
    fix (risk_score is five hand-written constants in a YAML file, 67% of
    them identical, and it is the reason nothing can be ranked).

    `direction` being declarable as "no_change" is load-bearing, not a
    placeholder. A read-only inspect that claims it moves nothing is making
    a real, falsifiable prediction; confirming it repeatedly drives its
    Bayesian surprise to ~0, which is the evidence that the action is a tic.
    """

    model_config = ConfigDict(extra="forbid")

    signal_id: PredictableSignal
    direction: EffectDirection

    predicted_delta: float
    predictor_variance: float = Field(ge=0.0)
    predictor_n: int = Field(ge=0)

    # True when this prediction came from a cold prior rather than from any
    # real observed history. Kept explicit so an analysis can separate
    # "we predicted 0.0 because we have learned it does nothing" from
    # "we predicted 0.0 because we have never seen it run" -- identical
    # floats, opposite meanings.
    cold_start: bool = False


class ActionOutcomeRecordV1(BaseModel):
    """One scored prediction. The unit of the action-outcome ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["action.outcome.record.v1"] = "action.outcome.record.v1"

    dispatch_id: str
    dispatch_frame_id: str
    feedback_frame_id: str

    dispatch_kind: str
    target_id: str
    signal_id: PredictableSignal
    direction: EffectDirection

    # Which arm this observation belongs to. `dispatched` is the treated arm.
    # `no_action` is the control arm: a tick in which NOTHING claiming this
    # signal ran, which is the only untreated population that actually
    # exists (see orion.autonomy.contrast for why capacity-blocked ticks are
    # not one). `capacity_blocked` is recorded but is not admissible as a
    # control; `randomized_holdback` is the experimental arm, off by default.
    arm: ActionArm = "dispatched"

    # Fixed-width decile of `baseline`, stamped by the writer so a reader can
    # never re-derive it differently. Matching happens within a bin.
    baseline_bin: int = Field(ge=0, le=9, default=0)

    # How many candidates were dispatched in the source tick at all, and
    # therefore how contaminated THIS (treated) row is. The field delta is
    # measured frame-wide, so a row from a tick where four other actions
    # also ran is not a clean reading of this one. `frame_dispatch_count ==
    # 1` is the sole-actor subset. Consumed by
    # scripts/analysis/report_action_value.py's `alone%` column.
    #
    # The first version of this comment claimed it existed to let an
    # analysis filter contaminated CONTROL observations. That was impossible
    # and was caught in review: control observations are emitted only when
    # this count is 0 (orion/feedback/outcome_resolution.py), and they get no
    # ledger row at all, so the field can never describe one. It is always
    # >= 1 on every row that exists.
    frame_dispatch_count: int = Field(ge=0, default=0)

    observed_at: datetime

    baseline: float
    observed_after: float
    observed_delta: float

    predicted_delta: float

    # KNOWN INCONSISTENCY, recorded rather than papered over (review finding
    # 10). This is `observed_delta - predicted_delta`, where predicted_delta
    # is pooled ACROSS baseline bins, while `surprise_nats` below is scored
    # against the per-bin prior. They are two different notions of "how wrong
    # were we" on one record. Live, the same action's treated mean ranges
    # from +0.162 (bin 1) to -0.376 (bin 7) -- a 0.54 spread -- so this
    # residual is dominated by WHICH BIN the tick landed in, not by
    # prediction quality. Do not read it as a quality measure; `surprise_nats`
    # is the bin-matched quantity.
    #
    # The fix would be to bin-match predicted_delta at dispatch time, which
    # needs the field, which the dispatch builder does not have (it is handed
    # a field TICK ID). Resolving it there is one more DB read on the tick
    # path, and the daily-risk-cap read on that same path is already 49.8% of
    # this database's entire buffer traffic. Deferred deliberately, not
    # forgotten.
    prediction_error: float

    surprise_nats: float = Field(ge=0.0)

    # Finding 3 (review, 2026-08-21): `direction` is the ONLY field a template
    # author hand-writes, and it had a schema, a producer and a persister but
    # no consumer -- a keyword cathedral by this repo's own definition, and it
    # made the patch's central claim false: with direction unscored, an action
    # that declared `decrease` and produced `+0.4` earned exactly the same
    # nats as one that declared `increase`. This is the consumer. None only
    # when the delta sits inside the dead band and the claim was directional
    # (i.e. genuinely undecidable), never as a shrug.
    claim_upheld: bool | None = None

    posterior_mean: float
    posterior_variance: float = Field(ge=0.0)
    posterior_n: int = Field(ge=0)

    # Attribution honesty. The field delta is measured frame-wide, so when
    # several candidates in the same tick claim the same signal, none of
    # them individually caused the whole delta. This records how many did,
    # so an analysis can split sole-attribution samples (co_predictors == 0,
    # clean) from shared ones rather than the ambiguity being invisible in
    # the stored row. Phase 1 updates the posterior on both and reports the
    # split; it does not silently drop shared samples.
    co_predictors: int = Field(ge=0, default=0)

    # Real measured cost of the action, when the cortex result carried one.
    # None means the result did not report it -- never coerced to 0.0, which
    # would read as "free".
    latency_ms: float | None = None
