"""Self-revision rung: turn sustained self-model prediction error into governed
mutation *signals*.

This is the metacognitive close of the self-modeling loop. Rung 1 fed prediction
error into pressure (attention); this rung feeds *sustained* prediction error on
self-model dimensions into the existing mutation pipeline as ``MutationSignalV1``s.
Those signals flow through the unchanged governance path:

    signal -> PressureAccumulator (decay + activation threshold)
           -> ProposalFactory.from_pressure -> DRAFT proposal (never auto-applied)

We deliberately emit only *signals*, not proposals or applied changes: thresholding,
cooldown, draft generation, trials and rollback all remain owned by the existing
pipeline. A single surprising tick produces nothing; only error that persists past
the accumulator's activation threshold becomes a reviewable draft.

Mapping self-model dimensions to the existing cognitive mutation surfaces is
intentionally conservative — only dimensions with a clear, already-supported
cognitive surface are routed; everything else is ignored rather than invented.
"""

from __future__ import annotations

from orion.core.schemas.substrate_mutation import MutationSignalV1
from orion.schemas.self_state import SelfStateV1

# self-model dimension -> existing cognitive mutation surface (see
# orion/substrate/mutation_proposals.py SURFACE_TO_CLASS / SURFACE_TO_LANE).
# Only surfaces that the ProposalFactory already knows how to draft are listed.
_DIMENSION_TO_SURFACE: dict[str, str] = {
    "continuity_pressure": "cognitive_identity_continuity_adjustment",
    "introspection_pressure": "cognitive_identity_continuity_adjustment",
    "social_pressure": "cognitive_social_continuity_repair",
    "coherence": "cognitive_contradiction_reconciliation",
}

# Minimum standing prediction error before a self-dimension is allowed to emit a
# signal at all. Below this the loop stays silent — surprise must be real.
_DEFAULT_MIN_ERROR = 0.3


def prediction_error_mutation_signals(
    self_state: SelfStateV1,
    *,
    min_error: float = _DEFAULT_MIN_ERROR,
) -> list[MutationSignalV1]:
    """Map sustained self-model prediction error to governed mutation signals.

    Returns one ``MutationSignalV1`` per self-dimension whose standing prediction
    error exceeds ``min_error`` *and* maps to a supported cognitive surface. The
    signal ``strength`` is the prediction error itself (already 0..1), so the
    accumulator integrates worse-predicted dimensions to threshold faster.
    """
    signals: list[MutationSignalV1] = []
    for dim_id, surface in _DIMENSION_TO_SURFACE.items():
        error = float(self_state.prediction_error_scores.get(dim_id, 0.0) or 0.0)
        if error < min_error:
            continue
        strength = max(0.0, min(1.0, error))
        signals.append(
            MutationSignalV1(
                event_kind=f"self_model_drift:{dim_id}",
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_surface=surface,
                target_zone="concept_graph",
                strength=strength,
                evidence_refs=[
                    f"self_state:{self_state.self_state_id}",
                    f"self_dimension:{dim_id}",
                ],
                source_ref=f"self_state:{self_state.self_state_id}",
                metadata={
                    "source_kind": "self_model_prediction_error",
                    "self_dimension_id": dim_id,
                    "prediction_error": round(strength, 6),
                    "trajectory": round(float(self_state.dimension_trajectory.get(dim_id, 0.0) or 0.0), 6),
                },
            )
        )
    return signals
