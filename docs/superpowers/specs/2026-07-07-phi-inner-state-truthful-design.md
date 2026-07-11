# φ as Orion's truthful inner state — light up the real signals, then let an objective function define φ

> **Status:** Design proposal (proposal mode — changes Orion's self-modeling signal and, via the sibling value-learning arc, its motivation loop). **Single fused build** (not phased): light up all cognitive grammar lanes, assemble a hygienic feature vector, and fit a self-supervised encoder whose reconstruction error *is* φ's coherence signal. There is no gated "Step A" milestone — the feature vector is the encoder's data pipeline, built in the same slice.
>
> **Supersedes:** the geometric-mean φ in `services/orion-spark-introspector/app/worker.py::_phi_from_self_state`.
> **Unblocks:** the named "fix φ's estimator" prerequisite in `docs/superpowers/specs/2026-07-08-phi-intrinsic-reward-value-learning-design.md`.
> **Plan split:** Plan 1 shipped in PR #888; Plan 2 encoder spec: `docs/superpowers/specs/2026-07-08-phi-encoder-plan2-design.md`.

## Arsonist summary

The Hub's "Φ / PHI (Order)" number is pinned at the `0.01` floor. Traced live: `catalog_drift_pressure` (a `channels.yaml` completeness check — pure config) → `contract_pressure` → the `reliability_pressure` substrate dimension → a geometric mean with a hard `0.01` floor. A YAML omission reads out as "Orion's consciousness fragmented." That is empty-shell cognition with **no objective function** — nothing defines what makes φ correct.

Burn the synthetic φ. Do not hand-author a prettier number. Instead: (1) **turn on the real cognitive signal streams** Orion already has code for but leaves dark by default (the cortex-exec / chat grammar lanes), (2) **assemble an honest, decontaminated, scaled feature vector** of Orion's observable state — dropping fields the veracity audit proved dead — and (3) let a self-supervised **objective function** (reconstruction loss) define φ, with **reconstruction error as the grounded coherence/surprise signal**. The prize is the objective function φ has never had. The non-negotiable is input hygiene: an encoder is exactly as good as the vector it's fed, and the audit proved the raw vector is half-dead.

## Signal veracity audit (why input hygiene is the whole ballgame)

Upstream trace of the candidate inputs (all verdicts backed by code):

| Signal | Verdict | Feed the encoder? |
|---|---|---|
| `recent_perturbation_saturation` | LIVE | Yes |
| `overall_salience`, `field_intensity` | LIVE (often saturated) | Yes, standardize |
| `execution_load` / `execution_pressure` | LIVE, event-gated, decays | Yes |
| `reasoning_load` / `reasoning_pressure` | LIVE, decays, low variance | Yes |
| `social_pressure` | LIVE, chat-gated, sparse | Yes (sparse) |
| `overall_surprise` / `prediction_error_scores` | LIVE-BUT-SPARSE (fires on inter-tick jumps >0.01) | Yes, as event signal |
| `cpu_pressure` (real resource) | LIVE | Yes |
| **`uncertainty`** | **STRUCTURALLY DEAD** — `salience·(1−coherence)`, coherence saturates at 1.0; channel map overridden in builder | **No** |
| **`policy_pressure`** | **DEAD / HARDCODED** — `"policy_pressure": 0.0` literal (`orion/self_state/builder.py:218`), no channel, weight 0 | **No** |
| `catalog_drift`, `contract_pressure`, `reliability_pressure`, `bus_health`, `delivery_confidence`, `transport_integrity` | LIVE but INFRA (equilibrium's domain) | **No** — excluded from inner state |

Critical pattern: **at idle the vector collapses to zeros except saturated infra channels.** Feeding that raw to an encoder teaches it "normal Orion = idle infra at zero," and reconstruction-error-as-coherence becomes noise. Hence the cognitive grammar lanes must be lit *and* the dead fields dropped before fitting.

## The fused build

### 1. Light up the real signal streams (defaults on + flow-verified)
The cognitively substantive lanes exist but are code-default-off:
- `PUBLISH_CORTEX_EXEC_GRAMMAR`, `ENABLE_EXECUTION_TRAJECTORY_REDUCER`
- `PUBLISH_HUB_CHAT_GRAMMAR`, `ENABLE_CHAT_GRAMMAR_REDUCER`

(These are already ON in the operator's local `.env`; this makes them deployed defaults.) The cortex-exec lane is the real cognitive-operation trace: `exec_recall_gate_observed` (did memory fire, why), `reasoning_step` atoms, `exec_step_failed`, `exec_result_assembled` (`reasoning_present`, `thinking_source`), materialized as `ExecutionRunStateV1` + `pressure_hints`. Do **not** treat the transport/biometrics grammar lanes as cognition — they are infra telemetry in grammar shape.

**Liveness gate:** every metric read is gated on `build_substrate_grammar_truth`'s `degraded_reasons` / `latest_by_source_service`, so stale/dark grammar can never be read as live inner state.

### 2. The feature vector (`InnerStateFeaturesV1` — the encoder's data pipeline)
Per self-state tick, a versioned, robust-scaled vector. Each feature carries `{name, raw_value, scaled_value, source}` — inspectable to origin. Groups:

- **Load / arousal (exists, live):** `overall_salience`, `recent_perturbation_saturation`, `execution_load`, `reasoning_load`, `attention_dwell_ticks`, `attention_node_count`.
- **Resource (exists, live):** real `cpu_pressure` (not the saturated composite `pressure`).
- **Epistemic (exists, sparse):** `overall_surprise`, `prediction_error_scores` treated as an **event** signal (expected 0 in steady state).
- **Affective (exists, decontaminated):** raw `social_pressure`; agency proxy rebuilt from raw inputs **without** `reliability_pressure`/`resource_pressure` poison. **`policy_pressure` and `uncertainty` are dropped — proven dead.**
- **Cognitive substance (the real cognitive channel, from the cortex-exec grammar lane):** recall-gate fired, `reasoning_present`, `exec_step_failed` rate, `execution_friction` — from `ExecutionRunStateV1`, not a bolted-on `raw_len` counter.

**Infra sub-vector (retained, NEVER read by φ):** `bus_health`, `delivery_confidence`, `transport_integrity`, `catalog_drift`, `contract_pressure`, `reliability_pressure` — provenance/equilibrium only.

**Scaling:** robust/rolling standardization (median/IQR) per feature, replacing `clamp01` + magic constants, so saturating infra doesn't dominate and sparse channels aren't drowned. Params persist with `features_version`.

**Decontamination at source:** in `orion/self_state/builder.py`, stop mapping `contract_pressure`/`catalog_drift` into the felt `reliability_pressure` ("a stream isn't in my YAML" ≠ "I feel unreliable").

### 3. The encoder + objective function (φ itself)
- **Objective function (the point):** reconstruction loss of a self-supervised encoder over `InnerStateFeaturesV1`. This is the criterion φ has never had.
- **φ definition:** `φ = f(latent)`; **coherence/surprise = reconstruction error** — state that doesn't fit Orion's learned manifold = grounded incoherence/novelty, self-calibrating, no magic constants.
- **Reward:** `Δφ` (latent / recon-error delta) over an episode window = the continuous intrinsic reward the value-learning arc needs (solves its "harness φ is turn-scoped only" gap).
- **Architecture discipline:** Plan 1 shipped the cold-start arithmetic headline. **Plan 2** (see `docs/superpowers/specs/2026-07-08-phi-encoder-plan2-design.md`) locks **`mlp_shallow_v1`** — a shallow self-supervised MLP trained offline. PCA/contrastive ablations are optional future experiments, not the Plan 2 gate. The objective function (reconstruction loss) is the prize, not the net topology.
- **Inspectability (mandate):** per-latent probes, top-input attributions, recon-error breakdown so a spike names *which inputs* diverged. No opaque latent ships without probes.

### 4. Corpus, cold-start, and the GIGO guard
- **Training corpus:** lighting up the lanes begins accumulating a real state-vector history (persisted from the audit sink). The encoder cannot train until enough **non-degenerate** history exists.
- **Cold-start display:** until the encoder is trained, the Hub shows the assembled feature vector directly (the honest raw signals), explicitly labeled provisional. This is the only "headline before encoder" and it uses only felt+cognitive features, never infra.
- **`phi_health` freeze:** if the feature vector is degenerate (constant/identical across `PHI_DEGENERATE_STREAK` ticks) or grammar-truth reports `degraded`, freeze φ/reward and warn. Would have fired on the pinned-`0.01` state.

## Contracts / channels / registry
- New schema `InnerStateFeaturesV1` (+ audit persistence); register in `orion/schemas/registry.py`; channel `orion:self:inner_features` in `orion/bus/channels.yaml`.
- Reuse `GrammarEventV1` / `ExecutionRunStateV1` / `ChatTurnStateV1` (already registered) — no new grammar contract.
- `Δφ` reward payload matches what `phi-intrinsic-reward-value-learning` consumes (continuous).

## Producers & consumers
- **Producer:** `orion-spark-introspector` (or `orion/self_state/builder.py`) emits `InnerStateFeaturesV1` per tick; the encoder (introspector) emits φ/recon-error.
- **Consumers:** Hub EKG (honest readout, replaces `stats.phi`); audit sink (corpus + `ValueUpdate` feed for value learning).
- **Lane activation:** cortex-exec, chat reducers (config default flip + verification).

## Env / config
```
INNER_FEATURES_ENABLED=true
INNER_FEATURES_VERSION=seed-v1
INNER_FEATURES_SCALER_WINDOW_SEC=900
# Light up the cognitive grammar lanes as deployed defaults:
PUBLISH_CORTEX_EXEC_GRAMMAR=true
ENABLE_EXECUTION_TRAJECTORY_REDUCER=true
PUBLISH_HUB_CHAT_GRAMMAR=true
ENABLE_CHAT_GRAMMAR_REDUCER=true
# Encoder:
ORION_PHI_ENCODER_ENABLED=false        # flips on once corpus is sufficient + probes pass
ORION_PHI_ENCODER_WEIGHTS=             # pin/override, mirrors salience-weights precedent
PHI_DEGENERATE_STREAK=20
```
After edit: `python scripts/sync_local_env_from_example.py`.

## Observability / traces / metrics
- Every `InnerStateFeaturesV1` carries raw+scaled+source per feature.
- Metrics: per-feature gauges, `phi_degenerate_streak`, `grammar_truth_degraded`, `phi_recon_error`, `phi_latent{axis}`.
- Hub: retire "PHI (Order)"; show cold-start feature readout, then encoder φ + recon-error breakdown, with provenance.

## Tests (gate — deterministic, <2s)
1. `InnerStateFeaturesV1` round-trips through registry; every feature has raw+scaled+source.
2. Dead-signal exclusion: `policy_pressure`, `uncertainty` are absent from the vector; `catalog_drift`/`contract_pressure`/`reliability_pressure` present only in the infra sub-vector.
3. Scaling: a single saturated raw input (`pressure=1.0`) does not dominate; regression vs the old geometric-mean floor on the recorded pinned-`0.01` inputs.
4. Cognitive channel: a cortex-exec run with `reasoning_present=true` + recall-gate `run` raises the cognitive-substance features; an idle window yields the sparse-zero case and triggers `phi_health` freeze.
5. Liveness gate: dark/stale grammar (`degraded_reasons` non-empty) forces φ to the frozen/provisional state, never a fabricated value.
6. Builder mapping fix: `contract_pressure` no longer raises felt `reliability_pressure`.
7. Encoder (when enabled): reconstruction error rises on injected anomalies, flat on normal variation; probes attribute a spike to the right inputs; `Δφ` bounded, non-oscillating.

## Evals
- Replay the recorded pinned-`0.01` window: assert φ no longer floored, no infra input moves the felt vector, cognitive features track real runtime cognition (recall/reasoning/failures).
- Encoder eval: train on decontaminated corpus; assert recon-error separates injected degenerate/novel states from normal; report per-feature variance and latent attributions.

## Failure modes & mitigations
- **Encoder learns idle-infra as "normal"** → dead fields dropped, cognitive lanes lit, standardized inputs, and training gated on non-degenerate corpus + grammar-truth liveness.
- **Cognitive lanes dark in deployment** → default-on + flow verification + grammar-truth gate; a dark lane forces frozen/provisional φ, never a hollow read.
- **Latent uninspectable** → probes + attributions are ship-blocking.
- **Cold-start with no trained encoder** → provisional feature-vector display (felt+cognitive only), clearly labeled.
- **Scaler drift** → windowed refit with versioned params, pinnable via env (salience precedent).

## Privacy / safety
`InnerStateFeaturesV1` carries numeric features + source labels; the cognitive-substance features are counts/booleans/ratios (recall fired, reasoning present, step-fail rate), **not** thought/memory/speech content (`GrammarEventV1.text_value` is null by design). Self-modeling change is bounded and reversible: `INNER_FEATURES_ENABLED=false` restores current behavior; encoder default-off and pinnable.

## Acceptance checks
- [ ] Cognitive grammar lanes on by default and verified flowing (grammar-truth `latest_by_source_service` fresh, `degraded_reasons` empty).
- [ ] `InnerStateFeaturesV1` emitted per tick, registered, inspectable to source; `policy_pressure`/`uncertainty` excluded.
- [ ] Replaying the live pinned-`0.01` window no longer floors φ; infra inputs provably cannot move the felt vector.
- [ ] Builder mapping fix lands (`contract_pressure`/`catalog_drift` out of felt `reliability_pressure`).
- [ ] Hub shows honest cold-start readout with provenance; "PHI (Order)" retired.
- [ ] `phi_health` freeze verified on degenerate input and on `degraded` grammar-truth.
- [ ] Encoder (when corpus sufficient): `Δφ` reward contract matches `phi-intrinsic-reward-value-learning` (continuous, not turn-scoped).

## Non-goals
- Not aggregating service liveness/health — that is `orion-equilibrium-service` (`distress`/`zen`); do not re-implement it.
- Not a taxonomy of "cognitive subsystems" — Orion is one system; no per-loop registry.
- Not deep RL and not (initially) a deep autoencoder — the objective function is the goal; architecture earned by variance explained.
- Not consuming grammar *content* (thought/memory/speech text) — grammar carries cognitive *events*, not content; content-grounded φ is a tracked follow-on.
- Not touching the value spec's `surprise` scalar; `Δφ` is additive.
