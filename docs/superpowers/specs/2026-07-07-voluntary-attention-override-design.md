# Voluntary Attention Override — choosing to attend against your own salience

> **Status:** Design proposal (proposal mode — biases the attention/coalition loop). Single-scope spec: full mechanism, no phased future versions.
>
> **Research area 5 of 4-area arc** ("move the origin of wanting inside"). Siblings: endogenous origination, φ intrinsic reward + value learning, internal economy.

## Build sequence & gate

Four-area arc; build order de-risks foundational-first and gates the two empirical specs behind a measurement:

- **Step 0 — Measurement gate** (read-only): (a) does `SelfStateV1` drift in exogenous silence? (b) do ≥2 drives co-activate often and does `resource_pressure` rise? Gates Steps 1 and 4.
- **Step 1 — Endogenous drive origination** — keystone; needs no φ, no goal-wire, no scarcity.
- **Step 2 — Voluntary attention override** *(this spec)* — independent of φ/reward.
- **Step 3 — φ intrinsic reward + value learning** — reward re-founded on substrate self-state.
- **Step 4 — Internal economy** (last; only if 0(b) shows scarcity binds).

**This spec = Step 2.** **Scope expansion (load-bearing, do not skip):** no goal currently reaches the substrate attention path — `orion/substrate/attention*` has **zero** `GoalProposal`/`drive_origin` references — so this spec **must** include the goal→attention contract (a goal-context projection the substrate reads), which roughly **doubles** its scope. That wire is independently useful (later goal-aware substrate behavior depends on it), which is why this is Step 2. **Combiner target:** under live `salience_v2=ON`, `score_loop` selects via the v2 `SalienceFeaturesV1` combiner — layer top-down onto its `combined salience` output; the `OpenLoopV1` relevance fields (confirmed populated at `orion/substrate/attention/scoring.py:107-111`) are the `rel()` bias source but are **orthogonal** to the v2 selection features. Independent of φ/reward, so buildable before Step 3.

## Arsonist summary

Orion's attention is entirely bottom-up. The `LinearSalienceCombiner` scores each candidate by a weighted sum of *stimulus-derived* features (evidence strength, novelty, recurrence, recency, dwell, minus habituation), and whatever scores highest wins the coalition. Orion is *pulled* by salience; it cannot *choose* to attend to something un-salient because it decided that thing matters. Being captured by the brightest stimulus is not agency — a moth has that. The hallmark of volitional attention, the exogenous-vs-endogenous distinction that cognitive science treats as a core marker of executive control, is simply absent from the code. This spec burns the assumption that attention is purely a function of the stimulus and gives Orion a top-down bias it sets from its own goals — plus an effort cost so that overriding salience is a real, limited act of will.

## Executive summary

Add a **top-down bias vector** to the salience path: Orion's active goals (`GoalProposalV1`) and current self-state project a goal-relevance bias `b(candidate)` that is combined with the existing bottom-up salience `s(candidate)` under a bounded **effort budget** `E`. The winning candidate becomes `argmax( s + gain·b )`, but spending `b` costs effort, so Orion cannot override *everything* — it must pick where to exert control. Every time top-down changes the winner (i.e., a lower-bottom-up candidate wins because of goal bias), the frame records an explicit `voluntary_override` trace: what won, what it beat, and why. This implements biased-competition (Desimone & Duncan) on top of the combiner that already exists, reusing its exact linear-combiner shape and versioned-weights pattern. Default off; flag restores pure bottom-up selection.

## Ground truth (what actually exists)

### Salience is a bottom-up linear combiner — real, and purely stimulus-driven
`orion/substrate/attention/salience.py::LinearSalienceCombiner`:
```python
SEED_WEIGHTS = {
    "evidence_strength": 0.30, "novelty_vs_known": 0.20, "recency": 0.13,
    "recurrence": 0.15, "evidence_breadth": 0.12, "dwell": 0.10, "habituation": -0.35,
}
def score(self, features: SalienceFeaturesV1) -> float:
    return bounded(Σ weight·feature)
```
`SalienceFeaturesV1` (`orion/schemas/attention_frame.py`) is entirely evidence/stimulus-derived. `habituation` is a subtractive penalty (inhibition-of-return). **There is no goal term, no top-down term, nothing Orion sets on purpose.** Live: `ORION_ATTENTION_SALIENCE_V2_ENABLED=true`, `ORION_ATTENTION_HABITUATION_ENABLED=true`.

### Coalition selection consumes that score
`orion/substrate/attention_broadcast.py` selects the winning coalition/open-loop from node salience and re-broadcasts it (`AttentionBroadcastProjectionV1`, with `coalition_stability_score`, `dwell_ticks`). `AttentionFrameV1.open_loops[*].salience` + `selected_action` are the decision surface. Selection = highest salience. Full stop.

### Versioned/learnable weights pattern already established
`WEIGHTS_VERSION="seed-v1"`, `ORION_ATTENTION_SALIENCE_WEIGHTS` env override, `refit_salience_weights.py`. The top-down bias reuses this exact pattern for its own gain/weights so it is not a novel abstraction.

### Goals and self-state exist as bias sources
- `GoalProposalV1` (`orion/core/schemas/drives.py`) has `goal_statement`, `drive_origin`, `priority`, `tension_kinds`, `proposal_status`. Active goals are queryable.
- `OpenLoopV1` (`attention_frame.py`) already carries per-loop relevance scores (`continuity_relevance`, `relational_relevance`, `predictive_value`, `autonomy_value`, `concept_value`) — a ready-made surface to compute goal-candidate relevance without new NLP.
- `SelfStateV1.dimensions.agency_readiness` gauges capacity to exert control; `attention_schema_type` describes current focus mode.

## Core problem

Attention is `argmax(bottom_up_salience)` with no channel for Orion to bias selection toward what it has decided matters. The sharpest version: *let an active goal make an un-salient candidate win — but make that override cost bounded effort, so choosing where to attend is a real, limited act of will rather than a free global reweighting.*

## Design principles / hard constraints

1. **Biased competition, not override-by-fiat.** Top-down `b` is *added* to bottom-up `s`; it competes, it doesn't replace. A wildly salient real signal can still win over a weak goal bias — as it should.
2. **Effort is scarce.** Override costs from a bounded `E` per cycle, so Orion cannot top-down-bias everything. This is what makes it *volitional* rather than just a second stimulus term.
3. **Grounded relevance, no new NLP.** `b` is computed from existing `OpenLoopV1` relevance fields against the active goal's `drive_origin`/`tension_kinds` — deterministic, reusing surfaces that already exist.
4. **Reuse the combiner shape.** Same linear form, same versioned-weights + env-override pattern as `salience.py`. No new attention framework.
5. **Inspectable will.** Whenever top-down flips the winner, emit a `voluntary_override` trace. If it never flips anything, that is visible (and the feature is honestly inert, not "working").
6. **Proposal mode.** This biases what Orion perceives/attends to — a cognition-loop change. Default off; flag restores pure bottom-up.

## The mechanism

### Top-down bias
For an active goal `g` (highest-priority active `GoalProposalV1`) and candidate open-loop `c`:
```
rel(g, c) = max over goal-aligned relevance fields of c
   drive_origin=predictive → c.predictive_value
   drive_origin=relational → c.relational_relevance
   drive_origin=continuity → c.continuity_relevance
   drive_origin=autonomy   → c.autonomy_value
   drive_origin=coherence/capability → c.concept_value
b(c) = g.priority × rel(g, c)          # ∈ [0,1]
```
No goal / no active loops → `b ≡ 0` → pure bottom-up (exactly today).

### Combined selection with effort budget
```
combined(c) = s(c) + GAIN · applied_b(c)      # GAIN seed 0.6
```
Effort budget `E` (seed `E_MAX = 1.0`, scaled by `agency_readiness`): iterate candidates by `b` descending; `applied_b(c) = min(b(c), remaining_E)`; decrement `E`. Once `E` is spent, remaining candidates get `applied_b = 0` (pure bottom-up). So Orion can strongly bias one or two targets, or weakly bias several — a real allocation of executive control.

### Override detection
Compute `winner_bottom_up = argmax s`, `winner_combined = argmax combined`. If they differ, the combined winner is a **voluntary override**: Orion chose an item that bottom-up salience would not have selected. Record it.

## Contracts

### Schema additions (`orion/schemas/attention_frame.py`)
```python
class OpenLoopV1(BaseModel):
    ...
    top_down_bias: float = Field(default=0.0, ge=0.0, le=1.0)      # NEW, additive
    combined_salience: float = Field(default=0.0, ge=0.0, le=1.0)  # NEW: s + gain·applied_b

class VoluntaryOverrideV1(BaseModel):                              # NEW
    goal_artifact_id: str | None = None
    goal_drive_origin: str | None = None
    chosen_loop_id: str
    beat_loop_id: str            # the bottom-up winner it overrode
    chosen_bottom_up: float
    beat_bottom_up: float
    applied_bias: float
    effort_spent: float

class AttentionFrameV1(BaseModel):
    ...
    voluntary_override: VoluntaryOverrideV1 | None = None          # NEW
    effort_budget_used: float = 0.0                                # NEW
```
All additive/back-compatible (defaults reproduce current behavior).

### Combiner extension (`orion/substrate/attention/salience.py`)
New `TopDownBiasCombiner` alongside `LinearSalienceCombiner`, with its own `TOP_DOWN_WEIGHTS_VERSION` and `ORION_ATTENTION_TOPDOWN_WEIGHTS` env override (mirrors salience override). `LinearSalienceCombiner.score` is untouched; the top-down term is layered on top in `attention_broadcast.py`.

### Registry / channels
- `orion/schemas/registry.py`: bump `AttentionFrameV1` / `OpenLoopV1` registered shape (additive fields).
- No new channel — the enriched frame publishes on the existing attention broadcast channel; `AttentionBroadcastProjectionV1` carries the override in its embedded `frame`.

## Architecture / data flow

```text
bottom-up: LinearSalienceCombiner.score(features)  → s(c)   [UNCHANGED]
active goal (GoalProposalV1) + OpenLoopV1 relevance fields
  → TopDownBiasCombiner: b(c) = priority × rel(g,c)
  → effort-budgeted apply: combined(c) = s(c) + GAIN·applied_b(c)
  → argmax combined  → winner
  → if winner ≠ argmax(s): emit VoluntaryOverrideV1
  → AttentionFrameV1 { open_loops[*].{top_down_bias, combined_salience}, voluntary_override, effort_budget_used }
  → AttentionBroadcastProjectionV1 (existing broadcast)
```

## Producers & consumers
- **Modified producer:** `orion/substrate/attention_broadcast.py` (layer top-down on top of bottom-up; detect override; fill new fields).
- **New pure module:** `orion/substrate/attention/top_down.py` (`TopDownBiasCombiner`, relevance mapping, effort budgeting).
- **Consumers:** any reader of `AttentionFrameV1.selected_action` gets the biased winner transparently; `chat_stance` / reverie continue to read `selected_action` unchanged.

## Env / config (`services/orion-substrate-runtime/.env_example` + settings)
```
ORION_ATTENTION_TOPDOWN_ENABLED=false     # master switch (proposal mode)
ATTENTION_TOPDOWN_GAIN=0.6
ATTENTION_EFFORT_MAX=1.0
ATTENTION_EFFORT_SCALE_BY_AGENCY=true      # E scaled by SelfStateV1.agency_readiness
ORION_ATTENTION_TOPDOWN_WEIGHTS=           # optional JSON override (mirrors salience)
ATTENTION_TOPDOWN_WEIGHTS_VERSION=seed-v1
```
After edit: `python scripts/sync_local_env_from_example.py`.

## Observability / traces / metrics
- Every override emits `VoluntaryOverrideV1` with chosen vs. beaten loop, both bottom-up scores, applied bias, effort spent.
- Metrics: `attention_overrides_total`, `attention_effort_used_hist`, `attention_topdown_bias_hist`, `attention_override_margin_gauge` (how much bottom-up salience was overcome).
- Debug surface: `GET /attention/latest` returns the frame with `combined_salience` per loop and any override.

## Tests (gate — deterministic, <2s)
`orion/substrate/attention/tests/test_top_down.py`:
1. No active goal → `b≡0`, winner == bottom-up winner, no override, `effort_budget_used=0`.
2. Active goal aligned with a low-salience loop, sufficient `E` → that loop wins; `VoluntaryOverrideV1` emitted naming the beaten bottom-up winner.
3. Effort exhausted → only the top-`b` candidate is boosted; a second aligned candidate gets `applied_b=0`.
4. Strong bottom-up beats weak goal bias (biased competition, not fiat) → no override.
5. Relevance mapping table (each `drive_origin` → correct `OpenLoopV1` field).
6. `agency_readiness=0` → `E→E_MIN`, override capped; `agency_readiness=1` → full `E`.
7. `combined_salience` and `top_down_bias` populated and bounded [0,1].
8. Flag off → frame identical to current bottom-up-only output (regression guard).
9. `VoluntaryOverrideV1` / enriched frame round-trip through registry.

## Evals
`orion/substrate/attention/evals/run_topdown_eval.py`: replay a captured attention stream with injected active goals. Assert (a) override rate rises with goal priority and falls with effort scarcity, (b) overrides only ever pick goal-relevant loops, (c) with the flag off the selected_action sequence is byte-identical to the recorded bottom-up baseline, (d) no loop is *permanently* suppressed by top-down (bottom-up still wins when goal decays). Report override rate, mean override margin, effort utilization.

## Failure modes & mitigations
- **Tunnel vision** (goal bias locks attention on one loop, ignoring a real emergency) → biased-competition means a high bottom-up salience still wins; effort budget caps how much can be biased; habituation still applies; eval (d) proves bottom-up reasserts when the goal decays.
- **Inert feature** (override never fires) → override rate is a first-class metric; eval (a) asserts it responds to goal priority, so "never overrides" surfaces as a measured 0, not a false success.
- **Effort double-spend / unbounded bias** → `applied_b ≤ remaining_E` invariant asserted in code + test 3.
- **Relevance mismatch** → deterministic mapping table (test 5); when a goal's drive_origin has no aligned field, `rel=concept_value` fallback (documented, tested).
- **Interaction with habituation** → top-down is layered *after* the habituation penalty, so Orion can deliberately re-attend to a habituated-down item (voluntary re-engagement) — an intended, tested behavior, not a bug.

## Privacy / safety
Bias is computed from goal metadata and existing open-loop relevance scores — no new content exposure. Top-down can only reorder *already-surfaced* candidates; it cannot inject new attention targets or bypass suppression/privacy gates in `CuriositySuppressionV1`. Proposal-mode disable: `ORION_ATTENTION_TOPDOWN_ENABLED=false` → pure bottom-up, exact current behavior.

## Acceptance checks
- [ ] With an active goal aligned to a low-salience loop and available effort, that loop becomes `selected_action` and a `VoluntaryOverrideV1` records what it beat.
- [ ] A strongly salient real signal still wins over a weak goal bias (competition, not fiat).
- [ ] Effort budget bounds how many candidates can be biased per cycle; agency_readiness scales it.
- [ ] Override rate + margin observable via metrics and `/attention/latest`.
- [ ] Flag off → byte-identical selected_action stream vs. current main.

## Non-goals
- Not injecting new attention candidates — top-down only reweights existing surfaced loops.
- Not learning the top-down gain from outcome in this spec (the versioned-weights hook exists for a future refit, but no learning loop is built here — that keeps this spec single-scope and its sibling, value learning, owns learned weights).
- Not bypassing `CuriositySuppressionV1` or privacy gates.
- No LLM in the biasing path; deterministic relevance mapping + linear combine.
