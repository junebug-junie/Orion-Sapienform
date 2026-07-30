from __future__ import annotations

from typing import Literal

from orion.attention.field_attention.candidate_precision_weighted import (
    PrecisionWeightedSalienceResult,
    normalize_across_targets,
    precision_weighted_salience,
)
from orion.attention.field_attention.candidate_society_of_mind import novelty_scorer
from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01, target_had_real_prior_entry
from orion.field.pressure import (
    RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    RECENT_PERTURBATION_ZSCORE_SATURATION,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

ObservationMode = Literal["watch", "inspect", "summarize", "ignore"]

# 2026-07-30: the only real theory-grounded attention targets. Maps each
# prediction-error-native node_id (Feldman & Friston 2010 precision-weighting,
# `candidate_precision_weighted.py`) to the reducer_key its own real historical
# series is filed under in `substrate_reduction_receipts`. Confirmed live
# against `services/orion-substrate-runtime/app/worker.py`'s exact
# `_prediction_error_receipt(reducer_key=..., node_id=...)` call sites -- a
# real, checked 1:1 correspondence, not assumed.
#
# `node:substrate.transport` is deliberately NOT here (code review caught an
# earlier draft claiming it was, incorrectly): its prediction_error write was
# permanently removed 2026-07-26 (docs/superpowers/specs/2026-07-26-transport-
# domain-retirement-bus-synaptic-successor-design.md, worker.py's own comment
# at the old call site) because transport_prediction_error() is a narrow,
# non-representative 2-Redis-Stream census, not real bus traffic -- the exact
# same "fake signal winning real budget slots" disease this whole patch
# exists to stop, already killed once at the source. Adding it back here
# would silently resurrect a signal that was correctly retired elsewhere,
# not fill a real gap. `bus_synaptic` is its real successor and is included.
PREDICTION_ERROR_NATIVE_TARGETS: dict[str, str] = {
    "node:substrate.biometrics": "node_biometrics",
    "node:substrate.execution": "execution_trajectory",
    "node:substrate.chat": "chat_session",
    "node:substrate.route": "route_arbitration",
    "node:substrate.bus_synaptic": "bus_synaptic",
}

# Same discipline as scripts/analysis/measure_precision_weighted_salience_probe.py's
# QUALIFYING_MIN_ROWS: below this many real samples, a variance/precision
# estimate isn't trustworthy. Reused here as confidence_score's denominator
# too -- confidence in a precision estimate is a real function of how much
# data backs it (more real samples = a more trustworthy variance estimate),
# not a hand-picked constant.
QUALIFYING_MIN_ROWS: int = 20


# 2026-07-30 fix (code review Finding 1, CRITICAL): these are the only real
# "higher is better" channels in the node/capability vectors -- confirmed by
# reading services/orion-field-digester/app/tensor/channels.py directly
# (DEFAULT_NODE_VECTOR/DEFAULT_CAPABILITY_VECTOR), not assumed from names.
# Every other channel in NODE_CHANNELS/CAPABILITY_CHANNELS is a
# "*_pressure"/"*_load"/deficit-style channel that rises with real distress
# and defaults to 0.0. Without inverting these before taking max(),
# _current_pressure_proxy() would report ~1.0 for a fully calm target just
# because e.g. `availability` sits at its healthy default of 1.0 -- confirmed
# by hand this session: a severely-overloaded vector and a fully-calm vector
# produced near-identical proxy output as long as these channels stayed near
# default, which is the normal case. This is a direction correction on the
# existing order statistic, not a resurrection of compute_salience()'s
# weighted blend -- no weights, no calibration, still just max().
_HIGHER_IS_BETTER_CHANNELS: frozenset[str] = frozenset(
    {
        "availability",  # node
        "delivery_confidence",  # node (node:athena only, single-observer)
        "stream_backlog_health",  # node (node:athena only, single-observer)
        "confidence",  # capability
        "available_capacity",  # capability
    }
)


def _current_pressure_proxy(vector: dict[str, float]) -> float:
    """The single most-elevated real channel reading in a raw pressure
    vector -- `max()`, an order statistic with zero free parameters, not a
    weighted combination of the vector's channels.

    2026-07-30: `novelty_scorer()` (Candidate B) needs a `current_salience`
    value per target to diff against that target's *prior* frame -- for
    `node:substrate.*` targets this doesn't apply (Candidate A's precision-
    weighting already IS the theory of "how surprising is this now," a
    second novelty layer on top would double-count); for physical host
    nodes and capabilities, no such theory exists yet, and nothing in this
    vector was ever the deleted `compute_salience()`'s output -- there is
    no real "current salience" left to reuse. `max()` is offered here as
    the honest minimum: unlike `compute_salience()`'s hand-picked 0.45/
    0.20/0.25/0.10 blend of ALL channels, this makes no claim about which
    channels matter more than others or how to combine them -- it only
    reports which single real channel is most elevated right now.

    `_HIGHER_IS_BETTER_CHANNELS` entries are inverted (`1 - value`) before
    the max comparison -- a real drop in e.g. `availability` is itself real
    pressure worth surfacing, not noise to discard, so inversion (not
    exclusion) was chosen: a genuinely unhealthy target should still be able
    to win the max() via one of these channels alone. Malformed/non-numeric
    vector entries are dropped defensively, never crash a tick.
    """
    values: list[float] = []
    for channel, v in vector.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv or fv in (float("inf"), float("-inf")):  # NaN/inf guard
            continue
        fv = clamp01(fv)
        if channel in _HIGHER_IS_BETTER_CHANNELS:
            fv = 1.0 - fv
        values.append(fv)
    return max(values) if values else 0.0


def _novelty_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
    *,
    vectors: dict[str, dict[str, float]],
    target_kind: str,
) -> list[FieldAttentionTargetV1]:
    """Shared implementation for host and capability targets: Candidate B's
    `novelty_scorer()` (Global Workspace/Society-of-Mind, Baars 1988 /
    Dehaene 2014) alone -- NOT the full three-scorer Borda aggregation.
    `magnitude_scorer()` doesn't apply (no real prediction-error history for
    these targets, the same reason Candidate A excludes them); `dwell_scorer()`
    is deliberately deferred, not silently dropped -- confirmed live
    2026-07-21 that `attended_node_ids` is the empty list in 2837/2840
    (99.9%) of real `substrate_coalition_dwell_log` rows over a 24h window,
    so it would almost never contribute a real vote. Building the cross-
    service wiring for a signal that's silent 99.9% of the time is not a
    good first cut; named as a real, disclosed follow-up, not solved here.

    With exactly one real scorer, Borda rank-aggregation has nothing to
    aggregate -- the novelty ranking IS the ranking. `novelty_for_target()`
    already returns a real `[0,1]` value (no separate normalization needed,
    unlike Candidate A's unbounded precision-weighted salience).

    Targets with zero real prior context (`previous_frame is None`, i.e.
    the very first tick, or a target absent from the prior frame) get
    `novelty_score=0.0` -- correctly "no observed change," per `novelty_
    for_target()`'s own contract -- not excluded, since "this vector exists
    right now" is itself real evidence, unlike Candidate A's targets where
    zero history means zero evidence at all.

    2026-07-30 fix (code review Finding 2, HIGH): `confidence_score` must
    reflect whether THIS SPECIFIC target_id had a real entry in the
    previous frame, not just whether any previous frame existed at all.
    `novelty_for_target()`/`prior_salience_for_target()` return the same
    0.0 novelty for "no previous frame" and "target absent from an
    existing previous frame" -- but claiming `confidence_score=1.0` for a
    target that is brand-new to this tick (e.g. a capability that just
    started reporting) would be dishonest: there is no real prior
    observation backing that confidence, identical to the no-frame-at-all
    case.

    2026-07-30 second fix (code review verification pass, same day): the
    first version of this fix checked presence in only `previous_frame.
    node_targets`/`.capability_targets` -- the two "active" buckets -- but
    `prior_salience_for_target()` (what `novelty_scorer()` actually uses to
    compute the novelty diff above) searches five buckets, including
    `suppressed_targets`. A target scored below `suppress_below` last tick
    lands ONLY in `suppressed_targets` (`build_attention_frame()`), a real
    prior observation, not an absent one -- the two-bucket check
    under-reported confidence=0.0 for such a target even though a real
    prior value was actually used to compute its novelty. Now uses
    `target_had_real_prior_entry()` (`scoring.py`), the same five-bucket
    search `prior_salience_for_target()` uses, so confidence and novelty
    are always answered from the same real fact.

    **One-time transition artifact, disclosed (2026-07-30):** `novelty_for_
    target()` diffs this tick's `_current_pressure_proxy()` value against
    whatever `salience_score` the *previous persisted frame* recorded for
    the same target_id, regardless of what formula produced that prior
    value. Confirmed live: the currently-deployed frame (pre-this-patch)
    has real `compute_salience()`-blend scores for `node:athena`/`atlas`
    (0.295/0.110) -- a different scale and formula than this patch's
    `_current_pressure_proxy()`. The FIRST real tick after this patch
    deploys will diff the new proxy against that old-formula value,
    producing an artificially large "novelty" reading that reflects the
    formula changeover, not a real event. Self-resolving after exactly one
    tick (every subsequent frame diffs new-formula against new-formula) --
    not fixed here, since there is no principled way to retroactively
    reinterpret an old frame's score under a formula it was never computed
    with; disclosed so a post-deploy operator doesn't misread tick one as a
    real signal.
    """
    target_ids = sorted(vectors.keys())
    if not target_ids:
        return []
    current_salience = {tid: _current_pressure_proxy(vectors[tid]) for tid in target_ids}
    novelty_scores = novelty_scorer(target_ids, current_salience, previous_frame)

    targets: list[FieldAttentionTargetV1] = []
    for target_id in target_ids:
        novelty = novelty_scores.get(target_id, 0.0)
        confidence = 1.0 if target_had_real_prior_entry(target_id, previous_frame) else 0.0
        targets.append(
            FieldAttentionTargetV1(
                target_id=target_id,
                target_kind=target_kind,
                salience_score=novelty,
                pressure_score=current_salience[target_id],
                novelty_score=novelty,
                urgency_score=0.0,
                confidence_score=confidence,
                dominant_channels={},
                reasons=[
                    f"Candidate B novelty-only salience: current-tick pressure proxy "
                    f"{current_salience[target_id]:.4f} vs. prior frame "
                    f"(novelty={novelty:.4f}); magnitude/dwell scorers not applied "
                    "(no real data for this target universe / near-always-empty "
                    "coalition, respectively -- see selector docstring)"
                ],
                evidence_refs=[f"field:{field.tick_id}"],
                suggested_observation_mode=observation_mode_for(novelty, policy),
            )
        )
    return targets


def observation_mode_for(salience: float, policy: FieldAttentionPolicyV1) -> ObservationMode:
    modes = policy.observation_modes
    if salience >= modes.inspect_threshold:
        return "inspect"
    if salience >= modes.summarize_threshold:
        return "summarize"
    if salience >= modes.watch_threshold:
        return "watch"
    return "ignore"


def select_node_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    prediction_error_histories: dict[str, list[float]],
) -> list[FieldAttentionTargetV1]:
    """Real, precision-weighted node targets only (Candidate A -- Feldman &
    Friston 2010, "Attention, Uncertainty, and Free-Energy":
    salience = precision x |prediction_error|, precision = 1/variance of the
    target's own real historical error series).

    2026-07-30: replaces the old hand-weighted pressure/novelty/urgency/
    confidence linear blend (`compute_salience()`, deleted from
    `scoring.py`) across ALL of `field.node_vectors` with a real theory for
    the specific nodes that have one, and NOTHING for the rest -- no
    fallback formula, per "kill means kill, no fallback to the thing being
    killed" (CLAUDE.md §0A, previously applied to retiring dependencies and
    metrics; applied here to retiring an entire scoring approach). Physical
    host nodes (`node:athena`/`atlas`/`circe`/`prometheus`/`rpc_timeout`)
    have no real historical prediction-error series of their own -- their
    `prediction_error` vector entry is a hardcoded `0.0` placeholder, not a
    tracked signal -- so they simply do not appear as attention targets
    until something builds real grounding for them, rather than continuing
    to report a salience score computed from hand-picked channel weights.
    Same for all capability targets (`select_capability_targets`, below).

    `prediction_error_histories`: {node_id: real ASC-by-time error history},
    caller-fetched (`AttentionRuntimeStore.load_prediction_error_history`)
    so this stays a pure function -- see that store method's own docstring
    for the query shape and the ~30-minute rolling-retention caveat
    (`substrate_reduction_receipts` only retains recent success receipts).
    A target with zero real history (`n_samples == 0`) is excluded entirely,
    not scored 0.0 -- "no data" and "confidently calm" are different claims.
    """
    results: dict[str, PrecisionWeightedSalienceResult] = {}
    raw_scores: dict[str, float] = {}
    for target_id, reducer_key in PREDICTION_ERROR_NATIVE_TARGETS.items():
        history = prediction_error_histories.get(target_id, [])
        result = precision_weighted_salience(history)
        if result.n_samples == 0:
            continue
        results[target_id] = result
        raw_scores[target_id] = result.salience

    normalized = normalize_across_targets(raw_scores)

    targets: list[FieldAttentionTargetV1] = []
    for target_id, result in results.items():
        salience = normalized.get(target_id, 0.0)
        confidence = clamp01(result.n_samples / QUALIFYING_MIN_ROWS)
        reasons = [
            f"precision-weighted prediction-error salience (current error "
            f"{result.current_error:.4f}, precision {result.precision:.2f}, n={result.n_samples})"
        ]
        if result.variance_floored:
            reasons.append("variance-floor instability: near-constant recent error history")
        targets.append(
            FieldAttentionTargetV1(
                target_id=target_id,
                target_kind="node",
                salience_score=salience,
                pressure_score=clamp01(abs(result.current_error)),
                novelty_score=0.0,
                urgency_score=clamp01(abs(result.current_error)),
                confidence_score=confidence,
                dominant_channels={"prediction_error": result.current_error},
                reasons=reasons,
                evidence_refs=[f"field:{field.tick_id}"],
                suggested_observation_mode=observation_mode_for(salience, policy),
            )
        )
    return targets


def select_host_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    """Physical host nodes (`node:athena`/`atlas`/`circe`/`prometheus`/
    `rpc_timeout`, or any `field.node_vectors` key not in `PREDICTION_ERROR_
    NATIVE_TARGETS`) -- excluded by `select_node_targets` (no real
    prediction-error history of their own), now real-theory-grounded via
    Candidate B's `novelty_scorer()` instead. See `_novelty_targets()`'s own
    docstring for the full scoring contract and its disclosed scope
    (novelty only, magnitude/dwell not applied here).
    """
    host_vectors = {
        tid: v for tid, v in field.node_vectors.items() if tid not in PREDICTION_ERROR_NATIVE_TARGETS
    }
    return _novelty_targets(
        field, policy, previous_frame, vectors=host_vectors, target_kind="node"
    )


def select_capability_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    """2026-07-30: was killed outright (always `[]`) when Candidate A's
    live-wiring shipped -- no capability target had a real prediction-error
    series to ground a precision-weighted salience in, and "kill means
    kill" ruled out reviving the deleted hand-weighted blend as a fallback.
    Now real-theory-grounded via Candidate B's `novelty_scorer()` instead
    (same signal Candidate A can't offer capabilities, since there's no
    real per-capability prediction-error history either way). See
    `_novelty_targets()`'s own docstring for the full scoring contract.
    """
    return _novelty_targets(
        field, policy, previous_frame, vectors=dict(field.capability_vectors), target_kind="capability"
    )


def select_system_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
) -> list[FieldAttentionTargetV1]:
    """Surfaces ``field:recent_perturbations`` as a system attention target
    when recent perturbation activity is elevated relative to this mesh's
    own recent baseline.

    2026-07-28: was ``min(1.0, count / 10.0)`` -- live-confirmed permanently
    saturated at 1.0 (real steady-state count is ~100-118 in the 60s window,
    5-10x past the old cap), so this target used to report maximum salience
    on essentially every tick regardless of whether anything was actually
    unusual. Now scores ``field.recent_perturbation_zscore`` (EWMA baseline
    maintained by apply_perturbations() -- see field_state.py's
    recent_perturbation_ewma* docstring), same fix shape as
    bus_synaptic_prediction_error's calm-floor correction: reused, not
    reinvented. A below-baseline dip (negative zscore) is clamped to 0 --
    "quieter than usual" isn't a reason to attend here, only "busier than
    usual" is.

    Not migrated to Candidate A in this patch (2026-07-30): this is a single
    aggregate count, not a per-target error series with its own variance to
    compute real precision from -- a different shape of signal than the six
    `node:substrate.*` targets above. Still has one hand-picked constant
    (`RECENT_PERTURBATION_ZSCORE_SATURATION`), disclosed as an open gap, not
    silently left unexamined.
    """
    count = len(field.recent_perturbations)
    if count == 0:
        return []
    zscore = field.recent_perturbation_zscore
    if zscore is None or field.recent_perturbation_ewma_n < RECENT_PERTURBATION_EWMA_MIN_SAMPLES:
        return []
    salience = min(1.0, max(0.0, zscore) / RECENT_PERTURBATION_ZSCORE_SATURATION)
    if salience < policy.thresholds.min_salience:
        return []
    return [
        FieldAttentionTargetV1(
            target_id="field:recent_perturbations",
            target_kind="system",
            salience_score=salience,
            pressure_score=salience,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
            dominant_channels={},
            reasons=[
                f"recent field perturbation count is {count} (z={zscore:.2f} vs baseline)"
            ],
            evidence_refs=[f"field:{field.tick_id}"],
            suggested_observation_mode=observation_mode_for(salience, policy),
        )
    ]
