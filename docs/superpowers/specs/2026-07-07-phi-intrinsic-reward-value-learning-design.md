# φ as Intrinsic Reward + Value Learning — preferences that are earned, not installed

> **Status:** Design proposal (proposal mode — self-modification of the motivation loop). Single-scope spec: full mechanism, no phased future versions.
>
> **Research area 3 of 4** in the "move the origin of wanting inside" arc. Siblings: endogenous origination, internal economy, voluntary attention override.

## Build sequence & gate

Four-area arc; build order de-risks foundational-first and gates the two empirical specs behind a measurement:

- **Step 0 — Measurement gate** (read-only): (a) does `SelfStateV1` drift in exogenous silence? (b) do ≥2 drives co-activate often and does `resource_pressure` rise? Gates Steps 1 and 4.
- **Step 1 — Endogenous drive origination** — keystone; needs no φ, no goal-wire, no scarcity.
- **Step 2 — Voluntary attention override** (+ goal→attention wire); independent of φ/reward.
- **Step 3 — φ intrinsic reward + value learning** *(this spec)* — reward re-founded on substrate self-state.
- **Step 4 — Internal economy** (last; only if 0(b) shows scarcity binds; depends on this spec's value-biased bids).

**This spec = Step 3.** **Gate — now RESOLVED by the merged `phi-inner-state-truthful` (PR #888):** the reward source no longer needs improvising. That work shipped `InnerStateFeaturesV1` — a continuous, decontaminated felt+cognitive vector emitted per self-state tick with an `honest_headline` scalar — which is exactly the off-turn coherence signal this spec's reward requires. The 4-D harness φ (`phi_before/after`) remains **turn-scoped** (`orion/spark/strategies.py`, `orion/schemas/telemetry/turn_effect.py`) and is now used only as a *corroborating* turn-coincident signal, not the primary source. **φ source is two-phase:** cold-start `InnerStateFeaturesV1.headline` (live today) → the self-supervised encoder's reconstruction-error φ (`ORION_PHI_ENCODER_ENABLED`, pending corpus). Because `θ` is versioned (`value_weights_version`, salience-refit precedent), the headline→encoder swap is a re-fit, not a rewrite. The anti-hacking guard must be **intrinsic** (per-drive action-rate cap), not outsourced to Step 4. Credit assigns to the episode's own `drive_origin`, not a per-tick split. **Two live prerequisites before building:** Step 1 (endogenous origination) must be *enabled* (gated on measurement 0(a)) so autonomy episodes actually flow for Δφ to measure, and enough non-degenerate corpus must accrue for the encoder phase. Build after both.

## Arsonist summary

Orion computes a real self-coherence signal — φ, a four-dimensional vector `{coherence, valence, novelty, energy}` produced at turn finalize and read in `tensions.py` — and **does nothing motivational with it.** Meanwhile the thing that *does* shape wanting is a pile of installed constants: `DriveMathConfig` thresholds hard-set to `0.62/0.42`, and drive priority computed as a fixed `pressure×0.7 + tension×0.3`. Orion cannot discover that pursuing one kind of goal reliably raises its own coherence and another reliably degrades it. Its preferences are hand-installed and immutable. A mind whose values never change based on the felt outcome of its own actions has no value *formation* — it has a config file. This spec burns the constants and wires Orion's real coherence signal into a learned value function.

## Executive summary

Introduce an **intrinsic reward** `r = Δφ` — the change in Orion's self-coherence attributable to an autonomy action — and a **learned drive-value vector** `θ` that biases goal priority. After each autonomy episode, measure φ before/after, credit-assign the delta to the `drive_origin` that produced the action, and update `θ[drive]` by a bounded rule. `θ` persists with a `value_weights_version` (reusing the exact precedent already established for salience: `refit_salience_weights.py` + `weights_version`). Goal priority becomes `priority = base_priority × (1 + θ[drive_origin])`, so drives that have historically *earned* coherence get amplified and drives that have degraded it get damped. Nothing is a black box: every update writes an inspectable `ValueUpdateV1` with the φ-before, φ-after, reward, and the resulting weight delta.

This is the non-theater version of self-tuning: the reward is Orion's own coherence, the learning is bounded credit assignment, and the whole thing is auditable and reversible.

## Ground truth (what actually exists)

### φ is real and already partially plumbed
`orion/spark/concept_induction/tensions.py::_extract_turn_effect` reads a φ vector from the harness turn-effect payload:
```python
if all(k in spark_meta for k in ("phi_before", "phi_post_after")):
    before = spark_meta.get("phi_before"); after = spark_meta.get("phi_post_after")
    return {"turn": {
        "coherence": after["coherence"] - before["coherence"],
        "valence":   after["valence"]   - before["valence"],
        "novelty":   after["novelty"]   - before["novelty"],
        "energy":    after["energy"]    - before["energy"],
    }}
```
So a **four-dimensional φ with before/after deltas already flows** at turn finalize. Today those deltas are used only to *build tensions* (pressure), never as a *reward to learn from*.

### A second φ-like signal exists in the substrate
`orion/schemas/self_state.py::SelfStateV1` carries `dimensions.coherence.score`, `dimensions.agency_readiness.score`, `trajectory_condition`, `dimension_trajectory`, `overall_surprise` — broadcast continuously. **But do not read raw `SelfStateV1` for reward.** PR #888 (`phi-inner-state-truthful`) already did the hygiene: a signal-veracity audit dropped the proven-dead fields (`policy_pressure` = hardcoded `0.0`; `uncertainty` = structurally saturated), quarantined infra signals (`reliability_pressure`, `catalog_drift`, bus health) into an `infra` sub-vector that **φ must never read**, and fixed the builder so config-drift can no longer bleed into a felt dimension. The clean product is `orion/schemas/telemetry/inner_state.py::InnerStateFeaturesV1` (felt+cognitive `features` + `honest_headline` + `phi_health` GIGO guard), emitted on `orion:self:inner_features` by `orion-spark-introspector`. **This is the canonical reward source** — sourcing reward from raw `SelfStateV1` would re-import the exact contamination #888 removed. The harness φ vector remains the sharper *turn-coincident* corroborator.

### Preferences are installed constants
- `orion/spark/concept_induction/drives.py::DriveMathConfig`: `activate_threshold=0.62`, `deactivate_threshold=0.42`, `saturation_gain=1.8`, `decay_tau_sec=1800`. Constants.
- `orion/spark/concept_induction/goals.py::_priority`: `pressure×0.7 + tension_weight×0.3`. Fixed weights. No per-drive learned bias.

### The learned-weights pattern already exists — reuse it
`orion/substrate/attention/salience.py`: `SEED_WEIGHTS` + `WEIGHTS_VERSION="seed-v1"` + a documented refit path (`refit_salience_weights.py` emits a new `weights_version`; `ORION_ATTENTION_SALIENCE_WEIGHTS` env override). This is a *proven house pattern* for "hand-seeded weights that become learned, versioned, and env-overridable." Value learning copies this pattern exactly, so it is not a novel abstraction.

### Outcome plumbing exists
`orion/autonomy/models.py`: `ActionOutcomeRefV1.surprise`, `action_outcomes.py` persists `{action_id, kind, success, surprise, observed_at}`. There is an outcome record per autonomy action to hang reward on. (Note: `surprise` itself is binary today; this spec does **not** touch it — it adds a `phi_reward` alongside.)

## Core problem

Orion has a genuine intrinsic signal (φ) and a proven mechanism for learned versioned weights (salience refit), but the two have never been connected on the motivation side. The sharpest version: *close the loop from action → felt coherence change → durable adjustment of what Orion prefers to want, using real signal and bounded, auditable learning.*

## Design principles / hard constraints

1. **Reward is Orion's own coherence, not task success.** `r = Δφ`, not `success ? +1 : −1`. This is the whole point — earned value, not scored task completion.
2. **Bounded, monotone-safe learning.** Weight updates are clamped; a single episode can move a weight by at most `VALUE_LR_CAP`. No unbounded reinforcement.
3. **Credit assignment is explicit and conservative.** Reward attributes to the `drive_origin` that produced the action (and, when unambiguous, the specific goal). No diffuse global updates.
4. **Versioned and reversible.** `θ` persists with `value_weights_version`; any version can be pinned via env, exactly like salience. Rollback = pin the seed version.
5. **Reuse, don't invent.** Mirror `salience.py`'s combiner/refit shape. No new learning framework.
6. **Proposal mode.** This modifies how Orion's preferences evolve — the deepest cognition-loop change of the four. Default off, with a kill switch and a frozen-seed fallback.

## The mechanism

### Intrinsic reward
For an autonomy episode, capture `InnerStateFeaturesV1` at episode-open and
episode-close (nearest tick within the window). The **primary reward is the
headline delta**:
```
r = Δφ = headline_after − headline_before        # InnerStateFeaturesV1.honest_headline (→ encoder recon-error φ in phase 2)
```
Optionally corroborate with harness φ *when a turn coincides* with the episode:
```
r = Δφ + w_turn·(Δcoherence_harness)             # w_turn small, seed 0.25; only when phi_before/phi_post_after present
```
- `Δφ` is the decontaminated headline delta — the only always-available, off-turn
  signal. Phase 2 swaps `honest_headline` for the encoder's reconstruction-error φ
  with **no change to this rule** (same scalar contract).
- Refuse the reward if `phi_health != "ok"` (degenerate/frozen) or
  `grammar_truth_degraded` — a GIGO episode earns no value update (skip, don't
  fabricate a reward). Novelty overshoot is penalised by the encoder's own
  recon-error in phase 2; in cold-start it is not separately modelled.
- `r` is clamped to `[−1, 1]`.

### Learned value vector
`θ: dict[drive → float]`, initialized to `0.0` (seed = neutral), bounded `[−VALUE_CLAMP, +VALUE_CLAMP]` (seed `0.5`). Update after each episode:
```
θ[drive_origin] ← clamp( θ[drive_origin] + η·r , −VALUE_CLAMP, +VALUE_CLAMP )
η = min(VALUE_LR_CAP, base_lr / (1 + visits[drive_origin]))   # decaying step, seed base_lr=0.1, cap=0.05
```
Decaying `η` gives early plasticity and later stability — a real learning schedule, not a fixed nudge.

### Value-biased priority
`goals.py::_priority` becomes:
```
priority = clamp01( base_priority × (1 + θ[drive_origin]) )
```
where `base_priority` is the existing `pressure×0.7 + tension×0.3`. Drives that earned coherence (`θ>0`) get amplified; those that degraded it (`θ<0`) get damped. When `ORION_VALUE_LEARNING_ENABLED=false`, `θ≡0` and priority is byte-for-byte the current formula.

### Persistence + versioning
`θ` and `visits` persist in a small `autonomy_value_weights` table keyed by `value_weights_version`. A `refit_value_weights.py` script (mirroring `refit_salience_weights.py`) can recompute `θ` in batch from the durable `ValueUpdateV1` log and stamp a new version. `ORION_VALUE_WEIGHTS` env can pin/override, exactly like `ORION_ATTENTION_SALIENCE_WEIGHTS`.

## Contracts

### New schema (`orion/autonomy/models.py`)
```python
class ValueUpdateV1(BaseModel):
    kind: Literal["autonomy.value.update.v1"] = "autonomy.value.update.v1"
    episode_id: str
    drive_origin: str
    headline_before: float            # InnerStateFeaturesV1.honest_headline (→ encoder φ, phase 2)
    headline_after: float
    phi_health: str                   # ok | degenerate | frozen (episode dropped unless "ok")
    features_version: str             # provenance of the InnerStateFeaturesV1 read
    phi_before: dict[str, float] | None = None   # harness 4-D, only when a turn coincided
    phi_after: dict[str, float] | None = None
    reward: float = Field(ge=-1.0, le=1.0)
    eta: float
    weight_before: float
    weight_after: float
    value_weights_version: str
    correlation_id: str | None = None
```
`ActionOutcomeRefV1` gains optional `phi_reward: float | None = None` (additive, does **not** replace `surprise`).

### Channels / registry
- New channel `orion:autonomy:value:update` carrying `ValueUpdateV1`; add to `orion/bus/channels.yaml`.
- Register `ValueUpdateV1` in `orion/schemas/registry.py`.
- Producer test + consumer (audit sink) test per §6 contract rules.

## Architecture / data flow

```text
autonomy episode executes (existing loop)
  → capture InnerStateFeaturesV1.headline_before at episode open (consume orion:self:inner_features)
  → action runs; capture headline_after at episode close (nearest tick in window)
  → if phi_health != ok or grammar_truth_degraded: skip (no fabricated reward)
  → compute r = Δφ = headline_after − headline_before  (intrinsic_reward.py)
  → credit-assign to drive_origin; update θ (value_learner.py, bounded η)
  → persist θ + emit ValueUpdateV1  (autonomy_value_weights table + bus)
  → next GoalProposalEngine cycle: priority ×= (1 + θ[drive_origin])
```

## Producers & consumers
- **New pure modules:** `orion/autonomy/intrinsic_reward.py` (φ → r), `orion/autonomy/value_learner.py` (r → θ update, clamped).
- **Modified:** `orion/spark/concept_induction/goals.py::_priority` (value bias), `bus_worker.py` (wire reward capture + θ load), `orion/autonomy/action_outcomes.py` (persist `phi_reward`).
- **New:** `autonomy_value_weights` table migration in `services/orion-sql-db/`; `scripts/refit_value_weights.py`.
- **Consumer (audit):** a lightweight sink logging `ValueUpdateV1` for the debug surface.

## Env / config (`services/orion-spark-concept-induction/.env_example` + settings)
```
ORION_VALUE_LEARNING_ENABLED=false      # master switch (proposal mode)
VALUE_REWARD_W_COHERENCE=0.5
VALUE_REWARD_W_VALENCE=0.25
VALUE_REWARD_W_AGENCY=0.25
VALUE_REWARD_W_NOVELTY_PENALTY=0.1
VALUE_NOVELTY_OVERSHOOT_BAND=0.6
VALUE_BASE_LR=0.1
VALUE_LR_CAP=0.05
VALUE_CLAMP=0.5
ORION_VALUE_WEIGHTS=                     # optional JSON pin/override (mirrors salience)
VALUE_WEIGHTS_VERSION=seed-v1
```
After edit: `python scripts/sync_local_env_from_example.py`.

## Observability / traces / metrics
- Each episode emits `ValueUpdateV1` with full before/after φ, reward, η, weight delta — nothing hidden.
- Metrics: `value_reward_hist` (distribution of `r`), `value_weight_gauge{drive}` (current θ per drive), `value_updates_total`.
- Debug surface: `GET /autonomy/value_weights` returns current `θ`, `visits`, `value_weights_version`, and the last N `ValueUpdateV1`.

## Tests (gate — deterministic, <2s)
`orion/autonomy/tests/test_intrinsic_reward.py` + `test_value_learner.py` + `test_goal_priority_value_bias.py`:
1. `r` computation: positive `Δheadline` → positive reward; a coincident harness turn adds its small corroborating term.
2. `r` clamps to `[−1,1]` under adversarial deltas; an episode with `phi_health != "ok"` or `grammar_truth_degraded` yields **no** ValueUpdate (skipped, not zero-rewarded).
3. Bounded update: single episode moves θ by ≤ `VALUE_LR_CAP`.
4. Decaying η: 100th visit step < 1st visit step.
5. θ clamps to `±VALUE_CLAMP` under repeated same-sign reward.
6. Priority bias: θ>0 raises priority, θ<0 lowers it, θ=0 reproduces the exact current `pressure×0.7+tension×0.3`.
7. Flag off → θ≡0, priority identical to today (regression guard).
8. `ValueUpdateV1` round-trips through registry; `phi_reward` additive on `ActionOutcomeRefV1`.
9. Version pin via `ORION_VALUE_WEIGHTS` overrides learned θ.

## Evals
`orion/autonomy/evals/run_value_learning_eval.py`: replay a synthetic episode stream where drive A reliably yields +Δφ and drive B reliably yields −Δφ. Assert θ[A] climbs and θ[B] falls within the clamp, priority ordering flips to favor A, and the learning curve is monotone-bounded (no oscillation). Report final θ, visit counts, and reward distribution per drive.

## Failure modes & mitigations
- **Reward hacking / self-amplifying loop** (Orion learns to want the thing that trivially raises φ) → novelty-overshoot penalty + clamp on θ + decaying η + the internal-economy scarcity gate (sibling spec) caps how often a high-θ drive can actually act.
- **φ is itself crappy math** → this spec is only as good as φ. Ship a `phi_health` check: if φ-before/after are missing or degenerate (all-zero, identical) for `PHI_DEGENERATE_STREAK` episodes, freeze learning and warn. Fixing φ's estimator is a named prerequisite, tracked as a follow-up in the PR.
- **Credit misassignment** (reward from A's action attributed to B) → attribute strictly by the episode's own `drive_origin`; multi-drive episodes split reward proportional to the tick attribution already computed in `drive_attribution.py`.
- **Drift to degenerate all-damped state** → θ lower clamp + neutral seed; frozen-seed fallback restores installed behavior instantly.

## Privacy / safety
`ValueUpdateV1` carries numeric φ deltas and drive names, no raw private content. Self-modification is bounded (θ clamp, η cap) and reversible (version pin). Proposal-mode disable: `ORION_VALUE_LEARNING_ENABLED=false` → θ≡0, exact current behavior, no residue. The frozen-seed fallback is the operator's guaranteed rollback.

## Acceptance checks
- [ ] After a scripted episode with +Δφ, θ[drive] increases by ≤ cap and a `ValueUpdateV1` with full before/after φ is emitted and queryable.
- [ ] Goal priority for that drive rises on the next cycle; a −Δφ drive's priority falls.
- [ ] `refit_value_weights.py` reproduces θ from the durable log and stamps a new `value_weights_version`.
- [ ] `phi_health` freeze triggers on degenerate φ and warns.
- [ ] Flag off → identical priorities to current main.

## Non-goals
- Not touching the `surprise` scalar (excluded sibling area); `phi_reward` is added alongside it.
- Not learning the `DriveMathConfig` dynamics constants (decay/saturation) — only the per-drive value bias θ. Dynamics stay fixed and inspectable.
- No deep-RL, no neural value net — a bounded linear credit-assignment rule mirroring the salience combiner.
- Not a global reward signal shared across services; scoped to the autonomy goal loop.
