# Internal Economy — choice under scarcity, with real opportunity cost

> **Status:** Design proposal (proposal mode — governs which wants get to act). Single-scope spec: full mechanism, no phased future versions.
>
> **Research area 4 of 4** in the "move the origin of wanting inside" arc. Siblings: endogenous origination, φ intrinsic reward + value learning, voluntary attention override.

## Build sequence & gate

Four-area arc; build order de-risks foundational-first and gates the two empirical specs behind a measurement:

- **Step 0 — Measurement gate** (read-only): (a) does `SelfStateV1` drift in exogenous silence? (b) how often do ≥2 drives co-activate and does `resource_pressure` rise? Gates Steps 1 and 4.
- **Step 1 — Endogenous drive origination** — keystone; needs no φ, no goal-wire, no scarcity.
- **Step 2 — Voluntary attention override** (+ goal→attention wire); independent of φ/reward.
- **Step 3 — φ intrinsic reward + value learning** — reward re-founded on substrate self-state.
- **Step 4 — Internal economy** *(this spec)* — last.

**This spec = Step 4** (last). **Gate:** blocked on **Step 0(b)** — build **only if** ≥2 drives co-activate often enough **and** `resource_pressure` actually rises, so budget `B` binds. If scarcity never binds, this is an ornamental cathedral (§0A) — **do not build**. `resource_pressure` is confirmed real (computed in `orion/self_state/builder.py:194-203`, not stubbed) but defaults 0.0 — verify it moves. Drop carry-credit from scope until binding is confirmed. Depends on **Step 3**'s value-biased priority to make bids meaningful.

## Arsonist summary

When two of Orion's drives activate in the same cycle, nothing chooses between them. Each capability just checks its own private `budget_per_cycle` ceiling and proceeds if under it. There is no shared scarce resource, no allocation across competing wants, and therefore **no opportunity cost** — the thing that makes a decision mean something. A per-capability ceiling is a rate limit, not a choice. An agent that never has to give one thing up to get another is not deciding; it is just executing whatever fits under independent caps. This spec burns the illusion of choice-by-independent-ceiling and gives Orion a single metabolic budget it must genuinely allocate.

## Executive summary

Introduce a per-cycle **metabolic budget** `B` (a scalar of abstract "action units" backed by real compute/fetch scarcity from `orion-power-guard`) and an **allocator** that, when ≥2 drives are active, distributes `B` across the competing drives in proportion to their (value-biased) priority, under each capability's hard cap. Drives that don't get funded this cycle are recorded with an explicit **opportunity cost** — a first-class `AllocationDecisionV1` artifact naming what was deferred and why. The allocation is the autonomous act. It runs *before* the capability policy's per-capability check (which remains as a safety ceiling), so the flow becomes: drives compete → allocator funds a subset under scarcity → funded actions pass the existing capability gate. Real constrained optimization (proportional / water-filling allocation), fully logged, reversible via a flag that restores today's independent-ceiling behavior.

## Ground truth (what actually exists)

### Per-capability ceilings, not a shared budget
`orion/autonomy/capability_policy.py`:
```python
if rule.budget_per_cycle > 0 and ctx.budget_used.get(capability_id, 0) >= rule.budget_per_cycle:
    return _decision(..., outcome="denied", reason_code="capability_budget_exhausted")
```
`config/autonomy/capability_policy.v1.yaml`: `web.fetch.readonly` budget 2, `journal.compose.episode` budget 1, `world_pulse.run` budget 1. Each is an **independent** ceiling. `ctx.budget_used: dict[str,int]` (`models.py:371`, `budget_per_cycle: int = 0`). Nothing sums across capabilities or forces a trade-off.

### Drives co-activate with no arbitration
`orion/spark/concept_induction/drives.py`: hysteresis activation at `0.62` means **multiple drives can be active simultaneously**. `goals.py::_priority` = `pressure×0.7 + tension×0.3` per drive, but there is no step that says "given limited capacity this cycle, fund these and defer those." Every active drive that can find a capability just goes.

### A real scarcity substrate exists but is not consulted for motivation
`services/orion-power-guard/` (worker present, no README) and `services/orion-gpu-cluster-power/` model actual compute/GPU pressure. `SelfStateV1.dimensions.resource_pressure` already carries a substrate-side resource-pressure score. None of this currently caps or shapes how many autonomy actions fire per cycle — the motivation loop is blind to Orion's real resource state.

### Drive-attribution already ranks drives per tick
`orion/spark/concept_induction/drive_attribution.py` computes `tick_attribution` (per-drive contribution) and a `dominant_drive`. This gives the allocator a ready-made priority signal to divide the budget by — no new ranking needed.

## Core problem

Orion has independent per-capability rate limits but **no shared scarce resource and no cross-drive allocation**, so it never experiences opportunity cost. The sharpest version: *make competing wants bid for a single limited budget so that funding one genuinely means deferring another, and record that trade-off as an inspectable decision.*

## Design principles / hard constraints

1. **One scarce budget, really scarce.** `B` is a single per-cycle scalar backed by real resource state (`resource_pressure` / power-guard), not an invented number that is always large enough. Scarcity that never binds is theater.
2. **Allocation is the decision.** The autonomous act is *choosing whom to fund*; the artifact of record is the allocation + its opportunity cost, not the downstream capability call.
3. **Capability ceilings stay as safety.** The allocator is additive and runs *before* the existing `evaluate_capability` check; per-capability caps remain a hard backstop.
4. **Deterministic and explainable.** Proportional/water-filling allocation with a documented rule — no LLM, no black box. Given the same drive priorities and `B`, the allocation is reproducible.
5. **Bounded artifacts.** `AllocationDecisionV1` opportunity-cost lists are capped (respect O(N) discipline, cf. `[[feedback_execution_merge_cap]]`).
6. **Proposal mode.** This governs which of Orion's wants get to act. Default off; flag restores today's independent-ceiling behavior exactly.

## The mechanism

### Budget derivation (real scarcity)
```
B = B_MAX × (1 − resource_pressure)      # SelfStateV1.dimensions.resource_pressure ∈ [0,1]
B = clamp(B, B_MIN, B_MAX)               # seed: B_MAX=4.0 units, B_MIN=1.0
```
When Orion is under real compute/power load, `B` shrinks — scarcity binds harder exactly when the substrate is stressed. Optionally fold `orion-power-guard`'s live signal in as a second multiplier.

### Per-drive cost + bid
Each fundable action has a **cost** in units (seed: `web.fetch.readonly`=2, `journal.compose.episode`=1, `world_pulse.run`=1, effectful/write=`B_MAX` i.e. must be sole spender). Each active drive **bids** its value-biased priority (`base_priority × (1+θ)` from the sibling value-learning spec, or just `base_priority` when that flag is off).

### Allocation (water-filling under a hard budget)
Given active drives sorted by bid, allocate `B` greedily/proportionally:
```
sort active drives by bid desc
for drive in order:
    cost = cost_of(drive.best_action)
    if remaining_budget >= cost and passes_capability_ceiling(drive.best_action):
        fund(drive); remaining_budget -= cost
    else:
        defer(drive, reason = "budget_exhausted" | "outbid" | "ceiling")
```
For divisible contention, an optional proportional split funds partial cycles (fractional units carried forward with decay, so a repeatedly-outbid drive eventually accumulates enough to win — starvation avoidance). Everything not funded lands in `opportunity_cost`.

### Ordering in the pipeline
```
drives active → allocator (this spec) → funded set
             → evaluate_capability() [existing per-cap ceiling + gates]
             → execute
```
The allocator narrows *before* the capability gate; the gate remains the final safety check.

## Contracts

### New schema (`orion/autonomy/models.py`)
```python
class MetabolicBudgetV1(BaseModel):
    kind: Literal["autonomy.metabolic.budget.v1"] = "autonomy.metabolic.budget.v1"
    cycle_id: str
    budget_total: float
    resource_pressure: float
    b_max: float

class FundedDriveV1(BaseModel):
    drive: str
    action_kind: str
    bid: float
    cost: float

class DeferredDriveV1(BaseModel):
    drive: str
    bid: float
    reason: Literal["budget_exhausted", "outbid", "ceiling"]
    carried_credit: float = 0.0

class AllocationDecisionV1(BaseModel):
    kind: Literal["autonomy.allocation.decision.v1"] = "autonomy.allocation.decision.v1"
    cycle_id: str
    budget: MetabolicBudgetV1
    funded: list[FundedDriveV1] = Field(default_factory=list, max_length=8)
    opportunity_cost: list[DeferredDriveV1] = Field(default_factory=list, max_length=8)
    correlation_id: str | None = None
```

### Channels / registry
- New channel `orion:autonomy:allocation:decision` carrying `AllocationDecisionV1`; add to `orion/bus/channels.yaml`.
- Register the three artifacts in `orion/schemas/registry.py`.
- `CapabilityEvaluationContext` gains optional `allocated: bool` so the capability gate can note it saw an allocator upstream (no behavior change when absent).

## Architecture / data flow

```text
DriveStateV1 (active drives) + tick_attribution (bids) + SelfStateV1.resource_pressure
  → MetabolicAllocator.allocate(B, active_drives)      [new: orion/autonomy/allocator.py]
      · B = B_MAX·(1−resource_pressure)
      · water-fill by bid under per-action cost + starvation carry-credit
      · emit AllocationDecisionV1 (funded + opportunity_cost)
  → funded actions → evaluate_capability() [existing gate, unchanged]
  → execute
```

## Producers & consumers
- **New pure module:** `orion/autonomy/allocator.py` (`MetabolicAllocator`, deterministic).
- **Modified:** `orion/spark/concept_induction/bus_worker.py` (call allocator before capability eval; carry-credit state per drive), `orion/autonomy/policy_act.py` (respect funded set), `orion/autonomy/capability_policy.py` (accept optional `allocated` context field).
- **Consumer (audit):** allocation-decision sink for the debug surface.

## Env / config (`services/orion-spark-concept-induction/.env_example` + settings)
```
ORION_INTERNAL_ECONOMY_ENABLED=false     # master switch (proposal mode)
ECONOMY_B_MAX=4.0
ECONOMY_B_MIN=1.0
ECONOMY_COST_WEB_FETCH_READONLY=2
ECONOMY_COST_JOURNAL_EPISODE=1
ECONOMY_COST_WORLD_PULSE_RUN=1
ECONOMY_CARRY_CREDIT_DECAY=0.8           # starvation-avoidance carry decay per cycle
ECONOMY_USE_POWER_GUARD=false            # fold live power-guard signal into B
```
After edit: `python scripts/sync_local_env_from_example.py`.

## Observability / traces / metrics
- Every cycle with ≥2 active drives emits an `AllocationDecisionV1` naming funded + deferred drives with bids and reasons.
- Metrics: `economy_budget_gauge`, `economy_funded_total{drive}`, `economy_deferred_total{drive,reason}`, `economy_starvation_credit_gauge{drive}`.
- Debug surface: `GET /autonomy/allocation/latest` returns the last decision incl. opportunity cost.

## Tests (gate — deterministic, <2s)
`orion/autonomy/tests/test_allocator.py`:
1. Single active drive → funded if cost ≤ B; no opportunity cost recorded.
2. Two drives, `B` funds only one → higher bid funded, lower bid deferred with `reason=outbid` and non-empty `opportunity_cost`.
3. High `resource_pressure` shrinks `B` so a normally-fundable pair now funds only one (scarcity binds under load).
4. Per-capability ceiling still denies a funded action → deferred with `reason=ceiling`.
5. Starvation avoidance: a drive outbid for K cycles accumulates carry-credit and eventually wins.
6. Cost table respected: `web.fetch.readonly`(2) vs two `journal`(1+1) under B=2 → allocator picks by bid, not greedily by count.
7. Opportunity-cost list capped at 8 (O(N) guard).
8. Flag off → allocator is bypassed; behavior identical to today's independent ceilings.
9. `AllocationDecisionV1` round-trips through registry.

## Evals
`orion/autonomy/evals/run_economy_eval.py`: drive a synthetic multi-drive contention stream at varying `resource_pressure`. Assert (a) total units spent per cycle never exceeds `B`, (b) funded selection matches the deterministic water-fill oracle, (c) no drive is starved beyond `1/(1−carry_decay)` cycles, (d) under sustained high pressure the *number* of autonomy actions per cycle drops measurably vs. low pressure. Report spend, funded/deferred ratios, and starvation max.

## Failure modes & mitigations
- **Budget never binds** (B always ≥ demand) → tune `B_MAX` from real data; eval asserts scarcity binds under high pressure; if it never binds, the feature is inert and flagged as such (not silently "working").
- **Starvation** (a low-priority drive never funded) → carry-credit accumulation with bounded decay guarantees eventual funding; test 5 proves it.
- **Runaway spend** → hard cap: `Σ funded_cost ≤ B` is an invariant asserted in code and eval (a).
- **Double-gating confusion** → allocator narrows; capability gate remains authoritative for safety; `allocated` context field makes the two-stage decision auditable end-to-end.
- **Unbounded artifacts** → funded/opportunity lists `max_length=8`.

## Privacy / safety
`AllocationDecisionV1` carries drive names, bids, and costs — no private content. The allocator can only *reduce* what acts (it funds a subset of already-gated candidates), so it cannot expand Orion's action surface; worst case it defers everything. Proposal-mode disable: `ORION_INTERNAL_ECONOMY_ENABLED=false` → bypass, exact current behavior.

## Acceptance checks
- [ ] Two co-active drives under a binding `B` produce one funded drive and a recorded opportunity cost naming the deferred drive and reason.
- [ ] Rising `resource_pressure` demonstrably reduces per-cycle funded actions.
- [ ] A repeatedly-outbid drive eventually wins via carry-credit (no permanent starvation).
- [ ] `Σ funded_cost ≤ B` invariant holds across the eval.
- [ ] Flag off → identical to current independent-ceiling behavior.

## Non-goals
- Not a real currency or inter-service market; a single per-cycle scalar internal to the autonomy loop.
- Not replacing the capability policy; the allocator precedes and complements it.
- Not modeling GPU scheduling — it *reads* resource pressure, it does not manage the cluster.
- No LLM in the allocation path; deterministic water-fill.
