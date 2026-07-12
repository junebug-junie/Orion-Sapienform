"""Registry of every "what does Orion currently feel/perceive" signal in the
repo (2026-07-12, inner-state unification).

Context: five independent times in one session, a real, computed inner-state
signal turned out to either silently duplicate another one, or never reach a
cognition consumer at all (DriveEngine/AutonomyStateV2, CLUSTER_ROLE_WEIGHTS
x3, the phi/heuristic split, FieldAttentionFrameV1's discarded per-node
scores). Each was found by manual grep-archaeology. This registry is the
deterministic alternative: every inner-state-shaped schema gets one entry
naming its producer, cadence, composition status, and cognition consumer(s)
(or an explicit, dated reason it has none yet) -- so the next one is a
registry entry, not a sixth rediscovery.

See docs/superpowers/specs/2026-07-12-inner-state-unification-design.md for
the full per-signal tracing this registry is populated from.

This is deliberately NOT a merge into orion/schemas/registry.py: that file is
a general-purpose name->class lookup with hundreds of unrelated entries (bus
envelope resolution, dynamic dispatch). This registry is narrow and
purpose-built -- it imports classes from where they already live, it does
not redefine or duplicate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel

from orion.autonomy.models import AutonomyStateV2
from orion.core.schemas.drives import DriveStateV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateV1
from orion.schemas.telemetry.biometrics import BiometricsClusterV1
from orion.schemas.telemetry.phi_encoder import PhiIntrinsicRewardV1


class Cadence(str, Enum):
    """How often a signal is (re)computed."""

    PER_TICK = "per_tick"
    CHAT_TURN_GATED = "chat_turn_gated"
    OFFLINE_TRAINED = "offline_trained"
    EVENT_GATED = "event_gated"  # e.g. only on a turn_effect_alert
    MULTI_SERVICE = "multi_service"  # spans several per-tick services (the L7-L11 ladder)


class CompositionStatus(str, Enum):
    """Where a signal stands relative to SelfStateV1, the one schema every
    cognition-facing prompt-builder is expected to read from."""

    COMPOSED = "composed_into_self_state"
    SHADOW = "shadow_declared_not_composed"
    DUPLICATE = "unresolved_duplicate"
    REHEARSAL = "no_cognition_consumer"


@dataclass(frozen=True)
class InnerStateSignal:
    signal_id: str
    schema: Optional[Type[BaseModel]]
    producer_service: str
    cadence: Cadence
    composition_status: CompositionStatus
    cognition_consumers: tuple[str, ...] = ()
    duplicate_of: Optional[str] = None
    shadow_reason: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.composition_status is CompositionStatus.DUPLICATE and not self.duplicate_of:
            raise ValueError(f"{self.signal_id}: DUPLICATE status requires duplicate_of")
        if self.composition_status is CompositionStatus.SHADOW and not self.shadow_reason:
            raise ValueError(f"{self.signal_id}: SHADOW status requires shadow_reason")


REGISTRY: tuple[InnerStateSignal, ...] = (
    InnerStateSignal(
        signal_id="field_state.v1",
        schema=FieldStateV1,
        producer_service="orion-field-digester",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(),
        notes=(
            "The body. Per-node/per-capability raw channel vectors. Diffusion "
            "fixed 2026-07-12 (9d367d4f, 4dc965f2) after a permanent-saturation "
            "bug; not read by cognition directly, composed into self_state.v1."
        ),
    ),
    InnerStateSignal(
        signal_id="field_attention_frame.v1",
        schema=FieldAttentionFrameV1,
        producer_service="orion-attention-runtime",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(),
        notes=(
            "Phase 1 of the companion plan (2026-07-12): widened from SHADOW "
            "to COMPOSED. orion/self_state/builder.py previously kept only "
            "the top-5 target_id strings on SelfStateV1."
            "dominant_attention_targets, discarding pressure_score/"
            "dominant_channels/reasons -- real, non-theater engineering (see "
            "field_attention/scoring.py's weighted_pressure/urgency_score/"
            "confidence_from_vector) thrown away one hop downstream. Now "
            "structured per-target data (target_kind, pressure_score, top "
            "dominant_channel, top reason) survives on the additive "
            "SelfStateV1.dominant_attention_target_details field via "
            "AttentionTargetSummaryV1 (orion/schemas/self_state.py)."
        ),
    ),
    InnerStateSignal(
        signal_id="self_state.v1",
        schema=SelfStateV1,
        producer_service="orion-self-state-runtime",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(
            "services.orion-spark-introspector.app.inner_state:build_inner_state_features",
        ),
        notes=(
            "The mood. ~12 dimension scores + confidence + reasons, composed "
            "from field_state.v1 + field_attention_frame.v1 (partially -- see "
            "that entry). Feeds phi's InnerStateFeaturesV1 and the L7 ladder."
        ),
    ),
    InnerStateSignal(
        signal_id="drive_state.v1",
        schema=DriveStateV1,
        producer_service="orion.spark.concept_induction.drives",
        cadence=Cadence.EVENT_GATED,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="autonomy_state_v2",
        cognition_consumers=(),
        notes=(
            "Computed from config/autonomy/signal_drive_map.yaml, a CLOSED "
            "typed map over biometrics_state/mesh_health/spark_signal/"
            "failure_event/chat_social_hazard/chat_reasoning_quality -- NOT "
            "from self_state.v1.dimensions. Do not build a drives<->self-state "
            "crosswalk: traced and rejected in the design spec, they are "
            "siblings over disjoint evidence with exactly one narrow, "
            "event-gated overlap point (spark_signal.{coherence,valence,"
            "novelty}, itself only published by orion-spark-introspector on a "
            "turn_effect_alert, not continuously). Live: 363 samples/24h "
            "confirmed 2026-07-12, real variance "
            "(coherence~0.20, continuity~0.35, capability~0.47)."
        ),
    ),
    InnerStateSignal(
        signal_id="autonomy_state_v2",
        schema=AutonomyStateV2,
        producer_service="orion.autonomy.reducer",
        cadence=Cadence.CHAT_TURN_GATED,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="drive_state.v1",
        cognition_consumers=(
            "services.orion-cortex-exec.app.chat_stance:_run_autonomy_reducer",
        ),
        notes=(
            "Same 6-drive taxonomy (DRIVE_KEYS) as drive_state.v1, "
            "independently reduced, gated behind AUTONOMY_STATE_V2_REDUCER_ENABLED. "
            "9 samples/24h confirmed 2026-07-12, all zero -- too little traffic "
            "to compare against drive_state.v1 yet. Merge-or-keep-separate "
            "decision is Phase 4 of the mesh-substrate-redesign plan, already "
            "on record; NOT resolved by this registry."
        ),
    ),
    InnerStateSignal(
        signal_id="phi_heuristic.valence",
        schema=None,
        producer_service="orion-spark-introspector",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.SHADOW,
        cognition_consumers=(
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative",
        ),
        shadow_reason=(
            "_phi_from_self_state()'s valence formula is the only surviving "
            "slice of the pre-2026-07-12 untrained EKG heuristic. No trained "
            "phi-encoder latent dimension correlates with any hedonic-adjacent "
            "felt dimension per the active encoder's probes.json -- explicitly "
            "verified, not assumed. This is the model case for a correctly "
            "justified, documented SHADOW entry (see "
            "_golden_phi_overrides()'s docstring in "
            "services/orion-spark-introspector/app/worker.py)."
        ),
        notes="Not a schema -- a formula. Tracked here because it reaches cognition.",
    ),
    InnerStateSignal(
        signal_id="phi_intrinsic_reward.v1",
        schema=PhiIntrinsicRewardV1,
        producer_service="orion-spark-introspector",
        cadence=Cadence.OFFLINE_TRAINED,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative",
        ),
        notes=(
            "Golden phi. Trained MLP autoencoder over InnerStateFeaturesV1. "
            "Fixed + deployed 2026-07-12 (654a9803, 79a6d966) -- was dark "
            "(SQL sink + debug WebSocket EKG panel only) for weeks before that. "
            "Phase 2 (2026-07-12) adds dominant_node/dominant_node_reason, "
            "sourced from self_state.v1's dominant_attention_target_details "
            "(Phase 1), filtered to real hardware nodes only via "
            "orion-spark-introspector's _dominant_hardware_node() -- excludes "
            "target_kind!='node' (a system-kind entry like "
            "field:recent_perturbations was confirmed live to frequently win "
            "top salience) and the two synthetic pseudo-nodes."
        ),
    ),
    InnerStateSignal(
        signal_id="biometrics_cluster.v1",
        schema=BiometricsClusterV1,
        producer_service="orion-biometrics",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="field_state.v1",
        cognition_consumers=(
            "services.orion-cortex-exec.app.executor:_metacog_biometrics_cue",
        ),
        notes=(
            "Third independent reimplementation of 'how much should each "
            "node's health count' alongside field-topology edges and "
            "orion-hub's BIOMETRICS_ROLE_WEIGHTS_JSON fallback. Dark in "
            "orion-biometrics (BIOMETRICS_MODE=agent); the hub fallback is "
            "what's actually live. Resolution already recommended "
            "(docs/notes/2026-07-12-phase4-cluster-weighting-research.md), "
            "not this registry's job."
        ),
    ),
    InnerStateSignal(
        signal_id="l7_l11_ladder",
        schema=None,
        producer_service=(
            "orion-proposal-runtime, orion-policy-runtime, "
            "orion-execution-dispatch-runtime, orion-feedback-runtime, "
            "orion-consolidation-runtime"
        ),
        cadence=Cadence.MULTI_SERVICE,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes=(
            "Five schemas (ProposalFrameV1 -> ConsolidationV1), tracked as one "
            "row. Confirmed rehearsal: docs/notes/2026-07-12-phase5-research-"
            "findings.md -- EXECUTION_DISPATCH_MODE=dry_run live, every reader "
            "outside the ladder is a self-labeled Hub debug route, the one "
            "non-Hub reader (orion-thought reverie grounding) only appends an "
            "inert ID tag to an already-generated thought. Out of scope for "
            "resolution here."
        ),
    ),
)


def get(signal_id: str) -> InnerStateSignal:
    for entry in REGISTRY:
        if entry.signal_id == signal_id:
            return entry
    raise KeyError(f"No inner-state registry entry for {signal_id!r}")


def duplicates() -> tuple[InnerStateSignal, ...]:
    """All entries currently flagged as unresolved duplicates of another entry."""
    return tuple(e for e in REGISTRY if e.composition_status is CompositionStatus.DUPLICATE)


def shadows() -> tuple[InnerStateSignal, ...]:
    """All entries deliberately not composed into self_state.v1, with a reason."""
    return tuple(e for e in REGISTRY if e.composition_status is CompositionStatus.SHADOW)
