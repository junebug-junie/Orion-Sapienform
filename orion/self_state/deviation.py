"""Per-dimension deviation instrumentation for SelfStateV1 (2026-07-12,
self-state/mesh substrate redesign Phase 2).

Measurement-only: this module logs how much each dimension deviates from its
own learned baseline this tick. It does not add a schema field, gate any
behavior, or change any score -- "measure before you name it" (see the
2026-07-12 brainstorm's recommended starting point). Once a live measurement
window shows what these values actually look like, a schema field can be
shaped around real data instead of guessed upfront.

Reuses orion.autonomy.deviation_gate.DeviationGate directly (the same EWMA
baseline + z-threshold mechanism already proven for chat/biometric tensions,
DEVIATION_EWMA_ALPHA=0.1/DEVIATION_Z_THRESHOLD=1.5/DEVIATION_SIGMA_FLOOR=0.02
live in services/orion-spark-concept-induction/.env -- identical to
DeviationGate's own dataclass defaults) rather than inventing a second
baseline-tracking mechanism.

The per-dimension "worse" direction lives in
config/self_state/self_state_policy.v1.yaml's `dimension_worse_direction`
(config, not a hardcoded table here) -- this repo already has an established
pattern for exactly this kind of fact (config/autonomy/signal_drive_map.yaml),
and a second, independently-maintained Python source of the same semantic
fact is exactly the kind of drift risk this redesign exists to eliminate.
"""

from __future__ import annotations

from orion.autonomy.deviation_gate import DeviationGate
from orion.schemas.self_state import SelfStateV1
from orion.self_state.policy import SelfStatePolicyV1

_SIGNAL_KIND = "self_state"


def observe_dimension_deviation(
    gate: DeviationGate,
    state: SelfStateV1,
    policy: SelfStatePolicyV1,
) -> dict[str, float]:
    """Feed every dimension present on `state` through `gate`, keyed by
    dimension id. Returns the resulting impulses (>=0; 0.0 during warmup or
    for a dimension whose |z| hasn't crossed the gate's threshold in its
    worse direction). Never raises -- DeviationGate.observe() already
    degrades to 0.0 on bad input for its `x` argument, and a dimension_id
    absent from policy.dimension_worse_direction (should not happen for a
    live SelfStateV1, but config and code can drift) is skipped rather than
    guessed. score/confidence are defensively cast to float, matching the
    existing orion/autonomy/signal_tension.py caller's pattern for this same
    shared gate -- DeviationGate.observe()'s own confidence coercion isn't
    guarded the way its score coercion is.
    """
    impulses: dict[str, float] = {}
    for dim_id, dim in state.dimensions.items():
        worse = policy.dimension_worse_direction.get(dim_id)
        if worse is None:
            continue
        try:
            confidence = float(dim.confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        impulses[dim_id] = gate.observe(
            _SIGNAL_KIND,
            dim_id,
            dim.score,
            confidence=confidence,
            worse=worse,
        )
    return impulses
