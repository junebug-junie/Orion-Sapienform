from __future__ import annotations

from typing import Literal

from orion.attention.field_attention.candidate_precision_weighted import (
    PrecisionWeightedSalienceResult,
    normalize_across_targets,
    precision_weighted_salience,
)
from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01
from orion.field.pressure import (
    RECENT_PERTURBATION_EWMA_MIN_SAMPLES,
    RECENT_PERTURBATION_ZSCORE_SATURATION,
)
from orion.schemas.field_attention_frame import FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1

ObservationMode = Literal["watch", "inspect", "summarize", "ignore"]

# 2026-07-30: the only real theory-grounded attention targets. Maps each
# prediction-error-native node_id (Feldman & Friston 2010 precision-weighting,
# `candidate_precision_weighted.py`) to the reducer_key its own real historical
# series is filed under in `substrate_reduction_receipts`. Confirmed live
# against `services/orion-substrate-runtime/app/worker.py`'s exact
# `_prediction_error_receipt(reducer_key=..., node_id=...)` call sites -- a
# real, checked 1:1 correspondence, not assumed.
PREDICTION_ERROR_NATIVE_TARGETS: dict[str, str] = {
    "node:substrate.biometrics": "node_biometrics",
    "node:substrate.execution": "execution_trajectory",
    "node:substrate.chat": "chat_session",
    "node:substrate.route": "route_arbitration",
    "node:substrate.transport": "transport_bus",
    "node:substrate.bus_synaptic": "bus_synaptic",
}

# Same discipline as scripts/analysis/measure_precision_weighted_salience_probe.py's
# QUALIFYING_MIN_ROWS: below this many real samples, a variance/precision
# estimate isn't trustworthy. Reused here as confidence_score's denominator
# too -- confidence in a precision estimate is a real function of how much
# data backs it (more real samples = a more trustworthy variance estimate),
# not a hand-picked constant.
QUALIFYING_MIN_ROWS: int = 20


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


def select_capability_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
) -> list[FieldAttentionTargetV1]:
    """Killed, not replaced. No capability target has a real historical
    prediction-error series to ground a precision-weighted salience in
    (unlike the six `node:substrate.*` targets `select_node_targets` scores)
    -- returns `[]` rather than falling back to the deleted hand-weighted
    blend. "Kill means kill, no fallback to the thing being killed"
    (CLAUDE.md §0A). Real capability attention needs its own theory-grounded
    instrument built first, not a silent revival of the disease this patch
    removes.
    """
    del field, policy
    return []


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
