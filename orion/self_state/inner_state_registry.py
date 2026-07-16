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
from orion.schemas.telemetry.field_channel_corpus import FieldChannelCorpusRowV1
from orion.schemas.telemetry.mood_arc import MoodArcCorpusRowV1, MoodArcEncoderManifestV1
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
            "9 samples/24h confirmed 2026-07-12, all zero, vs. drive_state.v1's "
            "363 samples/24h with real variance. Merge decision resolved "
            "2026-07-16: drive_state.v1 wins as the live signal for chat stance "
            "and Mind (orion/autonomy/drives_and_autonomy_retrospective.md §8); "
            "this reducer's evidence-compiler pattern is kept as an async "
            "mapping layer feeding drive_state.v1 rather than as its own "
            "consumer-facing pressure output."
        ),
    ),
    InnerStateSignal(
        signal_id="phi_heuristic.valence",
        schema=None,
        producer_service="orion-spark-introspector",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative",
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_hint",
        ),
        notes=(
            "Not a schema -- a formula. Tracked here because it reaches cognition. "
            "Was SHADOW until 2026-07-13 on the claim that 'no trained phi-encoder "
            "latent dimension correlates with any hedonic-adjacent felt dimension "
            "per the active encoder's probes.json'. That claim was checked against "
            "the real numbers and was false: agency_readiness (50% of this "
            "formula's weight) is one of the encoder's 8 actual input_features "
            "and correlates with 6 of 8 latents at |r| up to 0.686. The 0.0 shown "
            "for coherence/field_intensity/social_pressure/reliability_pressure/"
            "continuity_pressure/introspection_pressure in probes.json is not a "
            "verified null result -- none of those six were ever encoder inputs, "
            "so the probe's zero-variance guard fires trivially for all of them; "
            "that was misread as 'checked, no relationship'. Fixed by "
            "_agency_valence_proxy() in worker.py: a probe-weighted linear "
            "readout of the trained latent space, tanh-squashed to [-1,1], "
            "applied via _golden_phi_overrides() whenever an encoder tick "
            "succeeds. Real remaining limitation, not theater: social_pressure "
            "was never an encoder input, so there is genuinely no trained analog "
            "for the social_ease half of the heuristic formula -- the fallback "
            "(_phi_from_self_state, encoder disabled/degraded ticks only) still "
            "uses agency_readiness + social_ease, with the previously-hardcoded "
            "policy_ease=1.0 dead constant deleted outright rather than kept. "
            "Also real, also not theater: the proxy is a post-hoc statistical "
            "readout (Pearson coefficients, not fitted regression weights), not "
            "a native encoder output like coherence/energy/novelty are -- and "
            "agency_readiness is itself one of the encoder's raw inputs, so this "
            "is a lossy reconstruction of an already-available value, not newly "
            "discovered structure. A code-review pass also found that swapping "
            "between the two formulas tick-to-tick produced spurious deltas "
            "reaching spark_phi_hint/spark_phi_narrative's live metacog prompts "
            "as unearned valence swings; fixed via _PHI_PREV_VALENCE_SOURCE "
            "tracking that suppresses turn_effect's valence delta specifically "
            "on a source-swap tick (worker.py), plus a valence_source field on "
            "SparkStateSnapshotV1.metadata for observability."
        ),
    ),
    InnerStateSignal(
        signal_id="phi_intrinsic_reward.v1",
        schema=PhiIntrinsicRewardV1,
        producer_service="orion-spark-introspector",
        cadence=Cadence.OFFLINE_TRAINED,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative",
            "services.orion-cortex-exec.app.spark_narrative:spark_embodiment_narrative",
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
            "top salience) and the two synthetic pseudo-nodes. Phase 3 "
            "(2026-07-12) threads dominant_node/dominant_node_reason through "
            "SparkStateSnapshotV1 (a separate relay schema, "
            "orion/schemas/telemetry/spark.py) into "
            "spark_embodiment_narrative, rendered into both metacog prompt "
            "templates alongside spark_phi_narrative -- confirmed live in "
            "production (node:atlas/node:circe alternating as "
            "capability:llm_inference's real GPU contention winner) before "
            "this phase was built."
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
    InnerStateSignal(
        signal_id="mood_arc_corpus.v1",
        schema=MoodArcCorpusRowV1,
        producer_service="orion-spark-introspector",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes=(
            "Item 1 of docs/superpowers/specs/2026-07-13-felt-state-arc-"
            "roadmap-spec.md -- an append-only JSONL corpus sink, not a bus "
            "signal. Gate is simply MOOD_ARC_CORPUS_PATH being configured "
            "(off/no-op by default). Earlier drafted as 'gated on "
            "valence_source being present' -- corrected by code review, "
            "2026-07-13: valence_source is a plain str defaulting to "
            "'heuristic', never None on this code path, so that was never "
            "a real filter, just an accurate-by-accident description. Not "
            "composed anywhere and has no cognition consumer by design: "
            "this is training data collection for a not-yet-built windowed "
            "sequence autoencoder (roadmap item 2), explicitly gated on "
            "accumulating real hours of data first. REHEARSAL is correct "
            "here, same precedent as l7_l11_ladder -- not a gap to close. "
            "Rotation/retention (CORPUS_SINK_MAX_BYTES/ROTATED_KEEP, "
            "settings.py) is a SHARED policy with phi_intrinsic_reward.v1's "
            "own corpus sink -- tuned against that sink's real size/cadence "
            "(~104MB/36.8k rows/5 days), not independently verified against "
            "this signal's different growth rate (~8-9MB/day per the "
            "roadmap spec's own estimate). Revisit if/when this corpus is "
            "actually used for training and the shared thresholds turn out "
            "wrong for its cadence (found by code review, 2026-07-13). "
            "Superseded by field_channel_corpus.v1 (below) as roadmap item "
            "2's actual target substrate -- this sink is still real, still "
            "running, not a gap to close by disabling it here."
        ),
    ),
    InnerStateSignal(
        signal_id="field_channel_corpus.v1",
        schema=FieldChannelCorpusRowV1,
        producer_service="orion-field-digester",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes=(
            "Item 1 v2 of docs/superpowers/specs/2026-07-13-felt-state-arc-"
            "roadmap-spec.md -- the corrected raw-substrate corpus "
            "collector, superseding mood_arc_corpus.v1 as roadmap item 2's "
            "intended input. Same session finding: a full pass at item 2 "
            "(a windowed autoencoder, feat/mood-arc-encoder-cli) found that "
            "any 'trajectory structure' detected in mood_arc_corpus.v1 was "
            "almost entirely explained by orion-field-digester's own "
            "apply_decay(0.92) leaky-integrator mechanism, not anything "
            "emergent -- because that corpus captured "
            "_phi_from_self_state()'s OUTPUT (four already-smoothed, "
            "already hand-weighted scalars: coherence/energy/novelty/"
            "valence), not raw substrate. This corpus instead captures "
            "collect_field_channel_pressures()'s flat channel-name-keyed "
            "dict (orion/self_state/scoring.py) -- the merged node_vectors/"
            "capability_vectors pressures BEFORE any dimension-level hand-"
            "weighting is applied. It still carries apply_decay(0.92) "
            "(baked in at the point FieldStateV1 itself is computed -- "
            "unavoidable without touching the digester's own decay "
            "mechanism, out of scope here), but it is NOT additionally "
            "hand-composited into 4 scalars, which is the corrected layer "
            "to test for genuine emergent structure. mood_arc_corpus.v1 "
            "keeps running, untouched, real data for what it is -- this is "
            "additive, not a replacement in the running-system sense. Off "
            "by default (FIELD_CHANNEL_CORPUS_PATH empty). Row width is "
            "NOT fixed -- channels is a variable-key dict, not four named "
            "float fields; a future rework of scripts/fit_mood_arc_"
            "encoder.py (not this patch) will need to handle that directly "
            "rather than assume MoodArcCorpusRowV1's fixed shape."
        ),
    ),
    InnerStateSignal(
        signal_id="mood_arc_encoder.v1",
        schema=MoodArcEncoderManifestV1,
        producer_service="orion-spark-introspector",
        cadence=Cadence.OFFLINE_TRAINED,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes=(
            "Item 2 of docs/superpowers/specs/2026-07-13-felt-state-arc-"
            "roadmap-spec.md -- the windowed felt-state-trajectory "
            "autoencoder trained by scripts/fit_mood_arc_encoder.py over "
            "mood_arc_corpus.v1 rows. A dark, disk-only training artifact "
            "(manifest.json/weights.npz/probes.json under --out): no bus "
            "publish, no service wiring, no cognition consumer by design -- "
            "same REHEARSAL precedent as mood_arc_corpus.v1 and "
            "l7_l11_ladder, not a gap to close. This entry is the exact "
            "follow-up the sibling schema PR "
            "(feat/mood-arc-encoder-manifest-schema, MoodArcEncoderManifestV1 "
            "registered in orion/schemas/registry.py) flagged and correctly "
            "declined to add itself, since it had no producer yet -- this "
            "patch is the producer, so the registry gap is now closed. "
            "Methodology note (2026-07-13, same session): the spec's "
            "original single shuffle-baseline gate and its "
            "hidden_dim=8/latent_dim=4 defaults both failed empirically -- "
            "vanilla-SGD/zero-init training never converged and scored "
            "worse than a trivial mean-repeat baseline. Fixed via "
            "mean-initialized decoder bias + Adam (converges in ~80-120 "
            "epochs) and hidden_dim=32/latent_dim=16 (latent_dim=4 lacked "
            "capacity). Separately, raw-signal ACF analysis traced the "
            "corpus's real autocorrelation to a known, deliberate "
            "leaky-integrator decay mechanism "
            "(BIOMETRICS_FIELD_DECAY_RATE=0.92, "
            "services/orion-field-digester/app/digestion/decay.py) -- so "
            "the original single shuffle gate could pass purely by learning "
            "that already-known filter, not anything Orion-specific. "
            "Addressed with a two-tier gate (shuffle floor, hard-gated per "
            "spec; AR(1)-surrogate ceiling, diagnostic-only, not yet "
            "calibrated) and a purged/embargoed temporal train/held-out "
            "split (naive random window sampling leaks given ~10-15 tick "
            "autocorrelation from 50%-overlapping windows) -- see "
            "MoodArcEncoderManifestV1's docstring for the full field-level "
            "rationale. None of this two-tier/purged-split methodology is "
            "in the original written spec doc; it is stricter than what "
            "was originally asked for."
        ),
    ),
    InnerStateSignal(
        signal_id="chat_stance_disposition",
        schema=None,
        producer_service="orion-hub",
        cadence=Cadence.CHAT_TURN_GATED,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes=(
            "Not a standalone schema -- a field group (stance_disposition, "
            "stance_disposition_reasons, stance_boundary_register) on "
            "ChatTurnStateV1 (orion/schemas/chat_projection.py), same "
            "schema-vs-formula distinction as phi_heuristic.valence. Captures "
            "the Thought stance decision (proceed/defer/refuse, reasons, "
            "boundary_register) every unified-turn chat turn resolves "
            "(orion/hub/turn_orchestrator.py::execute_unified_turn), via the "
            "chat_grammar reducer (orion/substrate/chat_loop/"
            "grammar_extract.py). Reaches active_chat_session (Postgres "
            "JSONB) only. compute_chat_pressure_hints does not read it, so it "
            "never reaches orion-field-digester's delta_to_perturbations, "
            "substrate_field_state, or SelfStateV1 -- traced, not assumed. "
            "The obvious composition target (map to social_pressure, the "
            "dimension repair_pressure/conversation_load already feed) was "
            "considered and rejected 2026-07-13: SEEDV4_THEATER_FELT "
            "(services/orion-spark-introspector/app/inner_state.py) already "
            "excludes social_pressure from phi's live seed-v4 trainable "
            "feature set (docs/superpowers/specs/2026-07-09-phi-seedv4-"
            "feature-set-design.md), so that route reaches no cognition "
            "consumer while mutating an existing tracked dimension's "
            "provenance the same day mood_arc_corpus.v1 started real "
            "corpus collection. Full trace and three candidate paths "
            "forward (own SelfStateV1 dimension, feed mood_arc_corpus.v1 "
            "directly, or defer to a seed-v5 feature-set redesign once real "
            "data exists) in docs/superpowers/specs/2026-07-13-stance-"
            "disposition-inner-state-path.md -- none chosen yet. REHEARSAL "
            "is the honest status: real, computed, no cognition consumer, "
            "not a gap to silently close."
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
