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

**This spec = Step 3.** **Gate:** reward must be re-founded on the continuous `SelfStateV1` coherence/agency delta — the 4-D harness φ (`phi_before/after`) is **turn-scoped** (`orion/spark/strategies.py`, `orion/schemas/telemetry/turn_effect.py`) and has **no source for off-turn autonomy episodes**; harness φ is used only when a turn coincides. The anti-hacking guard must be **intrinsic** (per-drive action-rate cap), not outsourced to Step 4. Credit assigns to the episode's own `drive_origin`, not a per-tick split. Benefits from Step 1's richer episode stream, so build after it.

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
`orion/schemas/self_state.py::SelfStateV1` carries `dimensions.coherence.score`, `dimensions.agency_readiness.score`, `trajectory_condition ∈ {improving,degrading,stable}`, `dimension_trajectory`, and `overall_surprise`. This is a substrate-side coherence/agency signal broadcast continuously. Either signal can source reward; the harness φ vector is the sharper, action-aligned one.

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
For an autonomy episode with a captured φ-before and φ-after:
```
r = w_c·Δcoherence + w_v·Δvalence + w_a·Δagency_readiness − w_n·|Δnovelty_overshoot|
    (seed: w_c=0.5, w_v=0.25, w_a=0.25, w_n=0.1)
```
- `Δcoherence, Δvalence` from the harness φ vector (`phi_post_after − phi_before`).
- `Δagency_readiness` from `SelfStateV1` before/after the episode window (substrate-side).
- Novelty is rewarded up to a point then penalized past an overshoot band (curiosity that shatters coherence is not free). `r` is clamped to `[−1, 1]`.

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
    phi_before: dict[str, float]      # {coherence,valence,novelty,energy}
    phi_after: dict[str, float]
    agency_before: float
    agency_after: float
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
  → capture φ_before (harness turn-effect / self_state) at episode open
  → action runs; capture φ_after at episode close
  → compute r = Δφ  (intrinsic_reward.py)
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
1. `r` computation: positive Δcoherence → positive reward; novelty past overshoot band → penalized.
2. `r` clamps to `[−1,1]` under adversarial deltas.
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
