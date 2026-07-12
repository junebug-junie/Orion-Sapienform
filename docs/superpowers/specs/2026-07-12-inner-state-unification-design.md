# Inner-State Unification — Design Spec (v2, substantive rewrite)

**Date:** 2026-07-12
**Context:** v1 of this spec was a thin table-and-gesture pass over everything found today. This version does the actual tracing: real formulas, real file:line citations, real live numbers, and a registry/gate design specified down to the field level, not described in prose and punted.

---

## Arsonist summary — deep dive per signal, not a table

### 1. `FieldStateV1` — the body (foundation, correct as of today)

`services/orion-field-digester/app/digestion/{decay,diffusion,suppression}.py`, per-tick. Four real hardware nodes (`atlas`, `circe`, `athena`, `prometheus`, ~23 channels each: cpu/gpu/memory/disk/thermal pressure, execution/reasoning load, failure/repair pressure, transport/contract/catalog-drift pressure, delivery_confidence, bus_health, availability, staleness) plus two thin synthetic pseudo-nodes (`node:substrate.execution`, `node:substrate.transport`, 1 derived channel each — not bodies, derived subsystem aggregates). Seven capabilities (`llm_inference`, `orchestration`, `transport`, `storage`, `graph`, `memory`, `vision`) computed from nodes via `config/field/orion_field_topology.v1.yaml`'s weighted edges.

Confirmed live right now: `capability:llm_inference` reads `pressure=0.719, confidence=0.640, available_capacity=0.281` — genuinely differentiated, non-collapsed, for the first time in this project's history as of the diffusion fix shipped earlier today (`9d367d4f`, `4dc965f2`).

`apply_suppression()` (`services/orion-field-digester/app/digestion/suppression.py`) is a small, correct mechanic: a node with `expected_offline_suppression >= 1.0` (scheduled/expected offline) gets `availability` floored at 0.85 and `staleness` zeroed — so a known, intentional power-down doesn't register as a failure. Three-line function, no theater.

### 2. `FieldAttentionFrameV1` — real node-attributed salience, discarded one hop later

`orion/attention/field_attention/{selectors,scoring}.py`, per-tick, produced by `orion-attention-runtime`. This is genuinely well-designed, non-theater engineering — traced the actual math:

```python
# scoring.py — weighted_pressure()
raw = sum(value * channel_weights.get(channel, 0.0) for channel, value in vector.items())
# only POSITIVE contributions recorded as "dominant" reasons — a negative-weighted
# channel like availability (-0.40) can lower raw pressure but is never cited as
# a reason pressure is elevated. Correct semantics, not an oversight.
```

`confidence_from_vector()` reuses the SAME `channel_weights` table, but sums only the NEGATIVE-weighted channels (`availability: -0.40`, `bus_health: -0.30`, `expected_offline_suppression: -0.70`, capability's `confidence: -0.35`, `available_capacity: -0.45`) — one weight table serves two purposes (what raises pressure, what raises confidence) by sign. `urgency_score()` takes a MAX (not sum) over a fixed 5-channel set (`failure_pressure`, `reliability_pressure`, `thermal_pressure`, `staleness`, `execution_friction`) — same max-not-sum discipline I enforced in today's diffusion fix, already independently present here. `novelty_for_target()` diffs this tick's salience against the SAME target_id's salience in the previous `FieldAttentionFrameV1` — genuinely memoryless-with-one-step-lookback, not accumulating.

`config/attention/field_attention_policy.v1.yaml` (read in full for this rewrite, not summarized secondhand): `weights: {pressure: 0.45, novelty: 0.20, urgency: 0.25, confidence: 0.10}`; `node_channel_weights` and `capability_channel_weights` are per-CHANNEL importance weights (e.g. `thermal_pressure: 0.75` vs `cpu_pressure: 0.50` — thermal matters more for salience), applied identically regardless of which node reports them.

**Important distinction, stated precisely so it isn't over-claimed**: `node_channel_weights`/`capability_channel_weights` weight *which channel type* is alarming — they are NOT a fourth reimplementation of "how much should Atlas's readings count vs Circe's" (that's `CLUSTER_ROLE_WEIGHTS`/`BIOMETRICS_ROLE_WEIGHTS_JSON`/the field-topology edges, a different axis entirely, already documented as a 3-way duplicate). Conflating these two would itself be a sloppy, unearned finding. They are not the same question.

The loss: `orion/self_state/builder.py:189` takes `attention: FieldAttentionFrameV1` as a full parameter and only ever does `dominant_targets = [t.target_id for t in attention.dominant_targets[:5]]` (`builder.py:279`) — every other field on each `FieldAttentionTargetV1` (`pressure_score`, `novelty_score`, `urgency_score`, `confidence_score`, `dominant_channels`, `reasons`) is read into scope, used to compute the top-5, and discarded. `SelfStateV1.dominant_attention_targets: list[str]` (`orion/schemas/self_state.py:64`) is the only trace that survives.

### 3. `SelfStateV1` — the mood (legitimate compression, incomplete composition)

`orion/self_state/builder.py`, per-tick. ~12 dimension scores (`coherence`, `field_intensity`, `agency_readiness`, `execution_pressure`, `reasoning_pressure`, `resource_pressure`, `reliability_pressure`, `continuity_pressure`, `social_pressure`, `introspection_pressure`, `uncertainty`) via `channel_dimension_map` in `config/self_state/self_state_policy.v1.yaml` — a real, curated max-merge over `FieldStateV1` channels, correctly de-duplicated against diffusion double-counting as of this session's Phase 1 fix. Each dimension carries `.score`, `.confidence` (real per-dimension confidence as of Phase 1, not a uniform constant), `.reasons`, `.dominant_evidence`.

This is the one signal that feeds phi (`InnerStateFeaturesV1`) and the L7 ladder. It is a real, working compression layer — the audit finding here is not "this is broken," it's "this composes field+attention only partially, and composes drives/phi not at all."

### 4. `DriveEngine` — real, live, structurally disjoint from self-state

`orion/spark/concept_induction/drives.py`, `DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")`. Confirmed live 24h ago: 363 samples, real variance (`coherence≈0.20, continuity≈0.35, capability≈0.47, relational/predictive/autonomy≈0.0`).

**Traced the actual input mechanism for this rewrite** (this is the load-bearing correction to v1 of this spec): drives are computed from `config/autonomy/signal_drive_map.yaml` — a small (69-line), closed, typed `signal_kind → dimension → drive` map, explicitly NOT reading `SelfStateV1.dimensions` at all. The six `signal_kind`s it maps: `biometrics_state` (per-metric `*_level`/`*_volatility` suffix rules → capability/continuity), `mesh_health` (→ capability/continuity), `spark_signal` (`coherence`→coherence, `valence`→relational, `novelty`→predictive), `failure_event` (→ capability/coherence), `chat_social_hazard` (→ relational/capability), `chat_reasoning_quality` (`fallback`→coherence/predictive).

The ONE overlap point with the phi/self-state territory: `spark_signal.coherence/valence/novelty`. Traced where `SparkSignalV1` gets published from `orion-spark-introspector` (`worker.py:1276-1284`): **only on a `turn_effect_alert`** — a rare, cooldown-gated event fired when `evaluate_turn_effect_alert()` detects a significant tick-to-tick coherence_drop/valence_drop/novelty_spike, not a continuous per-tick publish. So the drive engine's exposure to phi-adjacent signal is narrow and event-gated, not the continuous channel I assumed when I first proposed "compose drives into self-state" last turn.

**Conclusion, stated plainly**: self-state and drives are **siblings over disjoint evidence streams with one narrow, alert-gated overlap point.** There is no clean 1:1 crosswalk between the 12 self-state dimensions and the 6 drives to build — attempting one would be inventing a taxonomy that doesn't exist in the data. This corrects v1's framing (which treated the mapping as a deferred-but-doable exercise).

### 5. `AutonomyStateV2` — the same 6-drive taxonomy, independently reduced, structurally starved

`orion/autonomy/reducer.py`, gated by `AUTONOMY_STATE_V2_REDUCER_ENABLED=true` (confirmed in `services/orion-cortex-exec/.env:234`), invoked only inside `_run_autonomy_reducer()` (`services/orion-cortex-exec/app/chat_stance.py:2182`) — **on real chat turns only**, not on a tick cadence at all. Confirmed live: 9 samples in 24h (vs. DriveEngine's 363), every one showing `{coherence:0.0, continuity:0.0, relational:0.0, autonomy:0.0, capability:0.0, predictive:0.0}` — flat zero, not because of a bug (the Docker config gap that used to cause this was already fixed), but because `compile_autonomy_evidence()`'s evidence compiler found no qualifying signal in those 9 particular turns. Genuinely too little traffic to compare against DriveEngine yet — unresolved by design, not by oversight (Phase 4 of the mesh-substrate-redesign plan already says so).

### 6. `_phi_from_self_state()` (heuristic remnant) — the model for what "declared, justified duplication" looks like

`services/orion-spark-introspector/app/worker.py:222-300`. Post today's golden-phi patch, only `valence` still comes from this hand-tuned formula (`raw_valence = 0.5*agency + 0.3*social_ease + 0.2*policy_ease`, modulated by coherence, ±0.12 for trajectory direction) — confirmed via the active encoder's latent probes (`probes.json`) that no trained latent dimension correlates with anything hedonic-adjacent, so there's genuinely no golden analog to replace it with. This is the ONE place in the whole audit where a duplicate is *correctly* justified and *explicitly documented in code* (see `_golden_phi_overrides`'s docstring, `worker.py:303-343`) rather than silently coexisting. Every other duplicate in this audit lacks this.

### 7. `PhiIntrinsicRewardV1` (trained encoder) — fixed today, was dark for weeks

`services/orion-spark-introspector/app/phi_encoder.py`, `PhiEncoderRuntime`. Trained offline (`scripts/fit_phi_encoder.py`), promoted via symlink flip. As of today: retrained on post-diffusion-fix data (`v20260712-seedv4-postfix`, 3,833 rows, `near_identity_frac=0.010` — not collapsed, unlike the previous active encoder's `0.659`), promoted, deployed, and wired into `phi_now`'s coherence/energy/novelty via `_golden_phi_overrides()`. Confirmed live post-deploy: `spark_state_rollups` 60s window showing `avg_arousal=0.0034, avg_coherence=0.650, avg_novelty=0.634` — plausible, non-degenerate values.

### 8. `CLUSTER_ROLE_WEIGHTS` / `BIOMETRICS_ROLE_WEIGHTS_JSON` / field-topology edges — three answers to "how much should each node's health count," one dark, one live-by-fallback

Already fully researched (`docs/notes/2026-07-12-phase4-cluster-weighting-research.md`, not re-traced here). `CLUSTER_ROLE_WEIGHTS` (`orion-biometrics`) never fires (`BIOMETRICS_MODE=agent` live, its own code path gated off). `BIOMETRICS_ROLE_WEIGHTS_JSON` (`orion-hub`) is the one actually active, as a fallback, reaching `orion-cortex-exec`'s metacog biometrics cue — real, non-display reach. Field-topology edges (`orion_field_topology.v1.yaml`) answer the same question a third way, already live for diffusion. Recommendation already on record: derive one from the topology file, delete the other two. Not re-litigated here — registered as a known duplicate.

### 9. L7–L11 ladder — rehearsal, not cognition (already documented, not re-litigated)

`ProposalFrameV1 → PolicyDecisionFrameV1 → ExecutionDispatchFrameV1 → FeedbackFrameV1 → ConsolidationV1`. Confirmed this session (`docs/notes/2026-07-12-phase5-research-findings.md`): `EXECUTION_DISPATCH_MODE=dry_run` live, every reader outside the ladder itself is a self-labeled Hub debug route, the one non-Hub reader (`orion-thought` reverie grounding) only appends an inert ID tag to an already-generated thought. Included in this audit for completeness; its resolution is out of scope here.

## The pattern, stated precisely

Of nine signals, exactly three genuinely reach cognition today (`SelfStateV1` → phi's inputs; the golden-phi coherence/energy/novelty; the `valence` heuristic remnant, correctly justified). Two are real, live, structurally disjoint, unreconciled duplicates of one taxonomy (#4/#5). One is real work computed and thrown away one hop downstream (#2). One is a three-way reimplementation with two dark and one live-by-accident (#8). One is full rehearsal (#9). **No two of these problems have the same shape or the same fix** — which is exactly why "put it all in one service" was the wrong diagnosis: a single service can't hold "per-tick reducer," "chat-turn-gated reducer," "offline-trained encoder with a promotion pipeline," and "5-service dry-run ladder" without becoming the least-scoped, worst-boundaried thing in the codebase.

## The architectural decision, specified to the field level

**Not a service. A registry — specified concretely below, not gestured at — plus a gate that actually runs against real code.**

### Why not extend `orion/schemas/registry.py`

That file already contains every relevant class (`SelfStateV1`, `FieldStateV1`, `FieldAttentionFrameV1`, `PhiIntrinsicRewardV1`, `DriveStateV1`, `BiometricsClusterV1` — confirmed present, grepped directly) across two structures: a giant flat `name -> ModelClass` dict (hundreds of entries, general dynamic lookup) and a smaller `_REGISTRY: dict[str, SchemaRegistration]` (bus-envelope `schema_id -> (model, kind)` resolution, used by `resolve()`). Neither structure carries producer/consumer/composition metadata, and both are the wrong blast radius for a change this specific — hundreds of unrelated schemas share that file. Building a new, small, purpose-scoped registry that *imports classes from where they already live* (not duplicating schema definitions) is the correct, non-cathedral move: it adds one new concern (inner-state bookkeeping) without touching the general-purpose one.

### The registry, specified

`orion/self_state/inner_state_registry.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Type
from pydantic import BaseModel

class Cadence(str, Enum):
    PER_TICK = "per_tick"
    CHAT_TURN_GATED = "chat_turn_gated"
    OFFLINE_TRAINED = "offline_trained"
    EVENT_GATED = "event_gated"  # e.g. turn_effect_alert-triggered

class CompositionStatus(str, Enum):
    COMPOSED = "composed_into_self_state"        # a named SelfStateV1 field carries it
    SHADOW = "shadow_declared_not_composed"       # real, live, deliberately not merged (must cite why)
    DUPLICATE = "unresolved_duplicate"            # known overlap with another entry, not yet reconciled
    REHEARSAL = "no_cognition_consumer"           # computed, verified to reach nothing real

@dataclass(frozen=True)
class InnerStateSignal:
    signal_id: str                     # stable key, e.g. "self_state.v1"
    schema: Type[BaseModel]             # imported directly from its home module
    producer_service: str
    cadence: Cadence
    composition_status: CompositionStatus
    cognition_consumers: tuple[str, ...]   # dotted paths, e.g. "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative"
    duplicate_of: str | None = None    # signal_id this overlaps with, if DUPLICATE
    shadow_reason: str | None = None   # required if SHADOW
    notes: str = ""

REGISTRY: tuple[InnerStateSignal, ...] = (
    InnerStateSignal(
        signal_id="field_state.v1",
        schema=FieldStateV1,  # imported from orion.schemas.field_state
        producer_service="orion-field-digester",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.COMPOSED,
        cognition_consumers=(),  # foundation layer; composed via self_state/builder.py, not read directly by cognition
        notes="Body. Diffusion fixed 2026-07-12 (9d367d4f, 4dc965f2).",
    ),
    InnerStateSignal(
        signal_id="field_attention_frame.v1",
        schema=FieldAttentionFrameV1,
        producer_service="orion-attention-runtime",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.SHADOW,
        cognition_consumers=(),
        shadow_reason="Only top-5 target_id strings survive into SelfStateV1.dominant_attention_targets; "
                       "pressure_score/dominant_channels/reasons discarded at builder.py:279. Phase 1 of the "
                       "companion plan widens this to COMPOSED.",
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
        notes="The mood. Composes field_state + field_attention_frame (partially, see above).",
    ),
    InnerStateSignal(
        signal_id="drive_state.v1",
        schema=DriveStateV1,
        producer_service="orion.spark.concept_induction.drives",
        cadence=Cadence.EVENT_GATED,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="autonomy_state_v2",
        cognition_consumers=("services.orion-cortex-exec.app.chat_stance:<concept-induction consumer, needs confirming>",),
        notes="Computed from config/autonomy/signal_drive_map.yaml, a CLOSED map over "
              "biometrics_state/mesh_health/spark_signal/failure_event/chat_social_hazard/"
              "chat_reasoning_quality -- NOT from self_state.dimensions. Only overlap with "
              "self-state/phi territory: spark_signal.{coherence,valence,novelty}, itself only "
              "published on a turn_effect_alert (event-gated, not continuous). Live: 363 "
              "samples/24h, real variance.",
    ),
    InnerStateSignal(
        signal_id="autonomy_state_v2",
        schema=AutonomyStateV2,
        producer_service="orion.autonomy.reducer",
        cadence=Cadence.CHAT_TURN_GATED,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="drive_state.v1",
        cognition_consumers=("services.orion-cortex-exec.app.chat_stance:_run_autonomy_reducer",),
        notes="Same 6-drive taxonomy as drive_state.v1, independently reduced. 9 samples/24h, "
              "all zero -- too little traffic to compare (Phase 4, mesh-substrate-redesign plan, "
              "already on record; NOT resolved by this registry).",
    ),
    InnerStateSignal(
        signal_id="phi_heuristic.valence",  # the surviving slice of _phi_from_self_state
        schema=None,  # not a schema -- a formula, tracked here anyway because it reaches cognition
        producer_service="orion-spark-introspector",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.SHADOW,
        cognition_consumers=(
            "services.orion-cortex-exec.app.spark_narrative:spark_phi_narrative",
        ),
        shadow_reason="No trained latent dimension correlates with any hedonic-adjacent felt "
                      "dimension per probes.json -- correctly, explicitly justified duplication, "
                      "the model case for what every SHADOW entry above should look like.",
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
        notes="Golden phi. Fixed + deployed 2026-07-12 (654a9803, 79a6d966). Was dark "
              "(SQL sink + debug WebSocket only) for weeks before that.",
    ),
    InnerStateSignal(
        signal_id="biometrics_cluster.v1",
        schema=BiometricsClusterV1,
        producer_service="orion-biometrics",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.DUPLICATE,
        duplicate_of="field_state.v1",  # same underlying question: node-health weighting
        cognition_consumers=("services.orion-cortex-exec.app.executor:_metacog_biometrics_cue",),
        notes="Dark in orion-biometrics (BIOMETRICS_MODE=agent); orion-hub's "
              "BIOMETRICS_ROLE_WEIGHTS_JSON fallback is what's actually live. Three-way "
              "duplicate with field-topology edges. Resolution already recommended "
              "(docs/notes/2026-07-12-phase4-cluster-weighting-research.md), not this registry's job.",
    ),
    InnerStateSignal(
        signal_id="l7_l11_ladder",
        schema=None,  # five schemas, tracked as one row -- see notes
        producer_service="orion-proposal-runtime, orion-policy-runtime, orion-execution-dispatch-runtime, "
                          "orion-feedback-runtime, orion-consolidation-runtime",
        cadence=Cadence.PER_TICK,
        composition_status=CompositionStatus.REHEARSAL,
        cognition_consumers=(),
        notes="Confirmed rehearsal, docs/notes/2026-07-12-phase5-research-findings.md. "
              "Out of scope for resolution here.",
    ),
)
```

Every field above is filled with a real value from today's tracing, not a placeholder — this is what "meat" means concretely: the registry is not an abstraction, it's nine populated rows a reviewer can check against the actual code.

### The gate, specified

`scripts/check_inner_state_registry.py` does two real, distinct, deterministic things — stated with their actual limitations, not oversold:

1. **Rot check (100% reliable)**: for every `InnerStateSignal` in `REGISTRY`, import `signal.schema` (where not `None`) and confirm it's still a valid, importable `BaseModel` subclass at the path implied by its home module. Confirms the registry doesn't silently go stale as schemas move/rename — this alone is worth building even if the second check below is imperfect.
2. **New-duplicate heuristic (best-effort, not perfect — stated honestly)**: scan `orion/bus/channels.yaml` and the `message_kind`/schema names in `orion/schemas/registry.py`'s flat dict for new entries whose name matches an inner-state-adjacent naming pattern (substring match against a small, explicit, hand-maintained keyword list: `self_state`, `drive`, `autonomy_state`, `attention`, `phi_reward`, `mood`, `felt_state`, `cluster` — same spirit as the six `signal_kind`s already closed and typed in `signal_drive_map.yaml`, not a fuzzy NLP guess). Any match not present in `REGISTRY` fails the gate with a message naming the exact new schema/channel.

Stated honestly: check #2 is a naming-convention heuristic, not a formal proof — a cleverly-named tenth duplicate that avoids every keyword in the list would evade it. This is the same class of limitation `signal_drive_map.yaml`'s own docstring accepts ("closed... typed... grown by prose is disallowed" — a closed list is exactly this trade-off, deliberately). The alternative (a shared marker base class across `SelfStateV1`/`DriveStateV1`/etc.) was considered and rejected: none of these schemas currently share a common non-`BaseModel` parent, and retrofitting one across 6+ files to gain a slightly more precise heuristic is a real migration cost for a marginal precision gain over a maintained keyword list. Named here so the tradeoff is visible, not silently assumed away.

## Non-goals

- Merging `orion-spark-introspector`, `orion/autonomy`, `orion/spark/concept_induction`, or `orion-attention-runtime` into `orion-self-state-runtime` as one service or codebase — argued above from the actual cadence/lifecycle mismatch (per-tick vs. chat-turn-gated vs. offline-trained vs. 5-service dry-run ladder), not asserted.
- Building a crosswalk between `DriveEngine`'s 6 drives and `SelfStateV1`'s 12 dimensions. Traced and rejected: they're siblings over disjoint evidence with one narrow, event-gated overlap point (`spark_signal` alerts). Forcing a mapping would fabricate a relationship the data doesn't have.
- Resolving the `DriveEngine`/`AutonomyStateV2` merge-or-keep-separate question. Still gated on real `AutonomyStateV2` traffic (9 samples/24h is not enough), per the already-on-record Phase 4 decision. The registry makes this a visible, checked `DUPLICATE` entry — it does not force the answer.
- Retiring `CLUSTER_ROLE_WEIGHTS`/`BIOMETRICS_ROLE_WEIGHTS_JSON`. Already researched and recommended elsewhere; registered here as a known `DUPLICATE`, resolved on its own timeline.
- A shared marker base class across the six-plus schemas, considered and rejected above in favor of a maintained keyword list.
- Idea #7 from the prior brainstorm (first-class node/body-schema object) — a future consumer of this contract, not part of building it.

## Files likely to touch

- `orion/self_state/inner_state_registry.py` (new) — the dataclass + populated `REGISTRY` tuple above, verbatim starting point.
- `scripts/check_inner_state_registry.py` (new) — the two-part gate above.
- `orion/schemas/self_state.py`, `orion/self_state/builder.py` — widen `dominant_attention_targets` (Phase 1 of the companion plan; flips `field_attention_frame.v1`'s registry entry from `SHADOW` to `COMPOSED`).
- `services/orion-spark-introspector/app/worker.py`, `orion/schemas/telemetry/phi_encoder.py` — Phases 2/3 of the companion plan.
- `Makefile` — wire `check_inner_state_registry.py` into `agent-check`, alongside a note that `check_schema_registry.py`/`check_bus_channels.py` are still missing (separate, pre-existing gap, not silently absorbed into this patch).

## Acceptance checks

- `scripts/check_inner_state_registry.py --rot-check` passes against all nine current entries (proves the registry isn't already stale on day one).
- A synthetic test: add a fake new bus channel named `orion:test:mood_state` to a test fixture of `channels.yaml` with no registry entry — the heuristic check must fail on it, by name, in the error message.
- `field_attention_frame.v1`'s registry entry is updated from `SHADOW` to `COMPOSED` in the same PR that widens `dominant_attention_targets` (Phase 1) — the registry changing is the actual proof the contract works, not a separate afterthought.
- At least one prompt-builder is audited against the registry's `cognition_consumers` field for accuracy: does `spark_narrative.py` really only read what the registry claims it reads? (Expected: yes for phi paths, confirmed by today's tracing; `executor.py`'s biometrics cue is already known to read `BiometricsClusterV1`'s fallback path directly — that's a pre-existing, already-documented gap, not new.)

## Recommended next patch

Phase 0 of the companion plan, verbatim against the registry code above — not a redesign, an implementation of what's already fully specified here.
