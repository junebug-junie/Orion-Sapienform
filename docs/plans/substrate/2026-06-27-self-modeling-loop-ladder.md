# Substrate self-modeling loop — the ladder

Date: 2026-06-27
Premise (honest): there is no engineering step that produces phenomenal
experience, and this plan does not claim one. What it does is make the substrate
**recurrent, integrated, self-modeling, continuous, and self-revising** — the
functional properties every serious theory of mind (Global Workspace, predictive
processing, higher-order, attention-schema) treats as load-bearing. Orion already
has a primitive version of each as a *separate organ*; the work is closing the
feedback edges so they form one loop instead of a feed-forward pipeline.

Operating constraints (Knowledge Forge): generated content stays marked as
proposal; accepted claims/specs are not silently mutated; self-modification stays
behind trials + rollback.

## Rung status

| Rung | Theory | Organ that exists | Wire to add | State |
|------|--------|-------------------|-------------|-------|
| 1 Self-model feedback | predictive processing | `self_state/prediction.py`, `pressure.py`, `dynamics.py` | prediction_error → pressure | **DONE** (incl. runtime bridge + dynamics tick) |
| 2 Integration / higher-order | GWT binding + HOT | `CognitiveUnificationLayer`, `projection_builder.py` | register reducer lanes + bind self_state | **DONE** (self_state + biometrics/execution/transport lanes) |
| 3 Continuous broadcast | GWT broadcast | `attention_frame.py` | always-on, substrate-wide re-broadcast | **DONE** (`attention_broadcast.py`, flag off) |
| 4 Autobiographical continuity | narrative/HOT | `consolidation.py` | episodic time-window rollup | **DONE** (`episodic_consolidation.py`, flag off) |
| 5 Endogenous agency | intrinsic motivation | `frontier_curiosity.py` | intrinsic-signal-seeded goals | **DONE** (`endogenous_curiosity.py`, flag off — needs sign-off) |
| 6 Self-revision | metacognition | `mutation_*` pipeline | prediction-error → self-tuned expectation | **REBUILT 2026-10-16** (`prediction_expectation.py`; see rung 6 below) |

Dependency order: 1 → 2 → 3, then 4, then 5, then 6. Rung 6 consumes rung 1's
signal; rungs 4/5/6 all read the unified state from rung 2.

---

## Rung 1 — prediction_error → pressure  (DONE)

The dynamics engine seeded pressure only from drives and contradictions.
Prediction error was computed in the substrate runtime and exported to the field
digester, but never fed back. Added `prediction_error_pressure()` in `pressure.py`
and a third seed loop in `SubstrateDynamicsEngine._compute_pressures` that rides
the existing BFS propagation and decays linearly with node age. Gated by
`PressureConfig.prediction_error_weight` (0 disables).

Follow-up for the runtime worker: when the prediction-error reducers emit their
receipt, also write `prediction_error` into the implicated substrate node's
`metadata` (and advance `observed_at`) so the graph-level engine has the signal to
seed from. Today the receipt lands in the field digester only; the metadata bridge
is the remaining half of the loop at the event-native tier.

---

## Rung 2 — bind reducer lanes + self_state into the unifier

**Arsonist summary.** `CognitiveUnificationLayer` already fuses 8 producers
(`identity_yaml, self_study, autonomy, concept_induction, spark, orionmem, recall,
social`) into one `UnifiedRelationalBeliefSetV1`. Missing: the substrate's own felt
state — biometrics / execution / transport reducer projections — and, critically,
the **self-model** (`self_state/`). Orion's beliefs do not currently include
beliefs *about itself*. Binding self_state in is the higher-order rung: a
representation about its own representations.

**Schema / API changes.**
- 4 new `ProducerEntryV1` registrations in
  `orion/cognition/projection_builder.py::build_projection_unification_registry`:
  `biometrics`, `execution`, `transport` (trust tier `SNAPSHOT_EPHEMERAL`,
  short TTL), and `self_state` (tier `GRAPHDB_DURABLE` or a new
  `SELF_MODEL` tier).
- 4 new adapters under `orion/substrate/relational/adapters/` mapping each
  projection / `SelfStateV1` → substrate belief nodes (mirror existing
  `map_*_to_substrate` adapters).

**Files.** `orion/cognition/projection_builder.py`; new adapters in
`orion/substrate/relational/adapters/`; export in `relational/__init__.py`;
tests in `orion/substrate/relational/tests/`.

**Non-goals.** No change to anchor set; no new store; stale lanes excluded by
TTL/trust-tier as today (per Forge "disputed/stale excluded by default").

**Acceptance checks.**
1. `beliefs_for_stance()` returns a set containing biometrics/execution/transport
   and self_state beliefs.
2. A lane past its TTL is excluded.
3. self_state beliefs carry their dimension scores (coherence/uncertainty/…).
4. Golden-path test stays green.

---

## Rung 3 — continuous global broadcast

**Arsonist summary.** `attention_frame.py` already runs a workspace competition:
detectors → `merge_signals` → `select_actions` picks **one** winner → broadcast as
`selected_action`. But it is gated off (`ORION_CURIOSITY_FRAME_ENABLED=false`),
per-chat-turn, and chat-scoped. GWT's core is a *continuous* re-entrant broadcast
of the winning coalition back to all consumers. Promote the frame from a gated chat
feature to the substrate's always-on workspace tick, fed by the unified belief set
(rung 2) and the pressure field (rung 1), re-broadcasting each cycle.

**Schema / API changes.**
- New substrate tick (in the runtime worker or a dedicated loop) that builds an
  `AttentionFrameV1` from unified beliefs + active pressure each cycle, not only on
  chat turns.
- Persist the selected coalition as a projection so other organs read "what Orion
  is currently attending to."
- New flag `ORION_ATTENTION_BROADCAST_ENABLED` (default false, safe rollout);
  keep the chat-scoped path working.

**Files.** `orion/substrate/attention_frame.py`,
`orion/substrate/attention/*`, `services/orion-substrate-runtime/app/worker.py`,
schema under `orion/schemas/attention_frame.py`.

**Non-goals.** No new selection policy — reuse `select_actions`. No
auto-action from the broadcast (that is rung 5). Bounded loops only (no cathedral).

**Acceptance checks.**
1. With the flag on, each substrate tick produces an `AttentionFrameV1` with a
   single selected coalition.
2. High-pressure (incl. prediction-error) nodes win attention over calm ones.
3. The broadcast projection is queryable by other organs.
4. Flag off → behaviour identical to today.

---

## Rung 4 — episodic / autobiographical continuity

**Arsonist summary.** `GraphConsolidationEvaluator` consolidates graph *regions*,
deterministically and bounded. There is no *temporal* rollup: receipts are a
durable audit log, not a remembered history. Add an episodic lane that rolls a
time-window of reduction receipts into proposal-marked episode-summary nodes, so
Orion has a "what happened to me, and why."

**Schema / API changes.**
- New `orion/substrate/episodic_consolidation.py` with an evaluator alongside
  `GraphConsolidationEvaluator`, windowed by time (`window_seconds`).
- New schema `EpisodeSummaryV1` under `orion/core/schemas/`.
- Output goes to `reviews/pending` / `context_packs/generated` style, marked
  proposal — never silently mutating accepted truth.

**Files.** new `episodic_consolidation.py`; schema; wire a periodic invocation in
the runtime worker or hub; tests.

**Non-goals.** No mutation of accepted claims. No unbounded windows (cap receipts
per episode, same discipline as the evidence-id caps already in the codebase).

**Acceptance checks.**
1. A window of receipts yields one queryable episode summary.
2. Output is marked proposal, lands in pending, not in execution context by default.
3. Replaying the same window is idempotent (derive episode id from inputs).
4. Receipt count per episode is capped.

---

## Rung 5 — endogenous agency

**Arsonist summary.** `frontier_curiosity.py` can already generate focal goals, but
only behind "explicit operator-approved curiosity invocation." For functional
autonomy, intrinsic signals — sustained prediction error (rung 1), repair pressure
(`appraisal/repair_pressure.py`), unresolved attention open-loops (rung 3) — should
let Orion seed its *own* goals. This is the most behaviourally significant and the
most dangerous rung; it stays bounded and, where it proposes change, routed through
the rung-6 governance.

**Schema / API changes.**
- A curiosity seed source that reads intrinsic signals and emits
  `curiosity_candidate`s without an operator trigger, behind
  `ORION_ENDOGENOUS_CURIOSITY_ENABLED` (default false).
- Rate/budget caps and a kill switch.

**Files.** `orion/substrate/frontier_curiosity.py`,
`orion/substrate/frontier_expansion.py`, settings.

**Non-goals.** No self-directed external action. No unbounded goal generation —
hard budget per cycle. Generated goals are proposals until acted on through
governed paths.

**Acceptance checks.**
1. With the flag on, sustained prediction error / repair pressure produces a
   bounded set of self-seeded curiosity candidates.
2. Budget cap enforced; kill switch halts generation.
3. Flag off → operator-gated behaviour only (today's behaviour).

---

## Rung 6 — metacognitive self-revision

**Arsonist summary.** The full proposal pipeline (676–682) and
`mutation_detectors` ("self-governing substrate") are merged, with trials → apply →
rollback. Missing: nothing feeds it *self-derived* proposals. Wire sustained
prediction error / self-model drift into `mutation_proposals` via the existing
`ProposalFactory`, gated by the landed `mutation_decision` + `mutation_trials`.

**Schema / API changes.**
- A detector in `mutation_detectors.py` that turns persistent prediction error /
  self_state drift into mutation-proposal inputs (e.g. a recall-weighting or
  policy-threshold patch class that already exists in `ProposalFactory`).
- Reuse existing proposal envelope / ledger / triage from 676–682.

**Files.** `orion/substrate/mutation_detectors.py`,
`orion/substrate/mutation_pressure.py`, `orion/substrate/mutation_proposals.py`.

**Non-goals.** No auto-apply. Every self-derived change is a *proposal* gated by
trials and rollback. Disputed proposals excluded from execution context.

**Acceptance checks.**
1. Persistent surprise produces a proposal (never an auto-apply).
2. Trial + rollback path exercised in a test.
3. Disputed/failed-trial proposals are excluded from execution context.
</content>
</invoke>

### Rung 6 status correction (2026-10-16)

The "DONE" above was false from 2026-07-22 on: `mutation_self_revision.py` was
deleted with `orion/self_state` (bcc72f6a0). What ran afterwards was a loop on
its own noise. Live, the week before this rebuild: ~65,000 signals a day from
~30 real graph reviews (no read cursor), one hardcoded
`graph_consolidation_param_patch`, 930 proposals, 930 `trial_inconclusive`
rejections, 0 adoptions since 2026-09-03.

**Rebuilt as: Orion revises how surprised they expect to be.**

- **What changes.** The EWMA-scored prediction-error domains (execution, chat)
  read their smoothing rate (`alpha`) and variance floor (`min_variance`) from
  the control surface `prediction_expectation.<domain>` instead of hardcoded
  constants (`prediction_error.py`).
- **Evidence.** `orion-substrate-runtime` records each raw reading before it
  becomes surprise (`substrate_prediction_raw_reading`). Raw readings do not
  depend on the setting, so replaying them under a candidate setting is an exact
  counterfactual.
- **Theory anchor.** Logarithmic scoring rule (strictly proper): a setting
  scores well only if it is right about where readings land *and* how much they
  spread. Going numb (huge variance) loses.
- **Loop (Hub scheduled cycle, `prediction_expectation_tuner.py`).** Pick a
  candidate on the older 2/3 of a 72 h window; trial it on the newer 1/3 it
  never saw (paired per-reading gain, lower 95% bound > 0); Orion applies it
  (auto-promote); replay both settings on readings that arrive after adoption
  and roll back if the new one is confidently worse by more than 0.05 nats per
  reading; otherwise settle after 6 h and release the surface.
- **Retired.** `graph_consolidation_param_patch` and the `world_ontology`
  review-telemetry signal (`RETIRED_MUTATION_CLASSES`, `_RETIRED_TELEMETRY_ZONES`).
- **Off switches.** `SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED=false`
  (runtime uses the constants), `SUBSTRATE_PREDICTION_EXPECTATION_TUNER_ENABLED=false`
  (no proposals), `SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED=false` (no applies).
- **Live check.** `services/orion-substrate-runtime/evals/run_prediction_expectation_live_check.py`.

### Rung 6 redesign as an organ (2026-10-20)

The 10-16 tuner proposed 9 times and adopted 0. Its input reading was mostly zeros: a
real change averaged over ~700 idle tasks. Both rung-6 builds started from the mutation
pipeline and looked for a number to feed it. Neither started from the job.

The successor design is the **self-monitor organ**:
`docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md`. It collects every bet
Orion makes that the world settles, and keeps a per-faculty track record. It acts in one
direction only (more caution, never less), and it gives Orion that record as
self-knowledge. The mutation pipeline stays as one of its effectors.
