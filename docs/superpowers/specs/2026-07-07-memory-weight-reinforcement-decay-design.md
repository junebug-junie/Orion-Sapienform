# Memory as weight + reinforcement + decay (not novelty + approval)

Design spec — 2026-07-07

## Arsonist summary

Crystallization memory is a filing cabinet with a human bouncer. A crystallization's
weight (`salience`) is computed **once at creation** and then frozen forever: it never
rises when the same thing recurs or gets recalled, and it never decays when ignored.
The only thing standing between "noticed" and "permanent canonical memory" is a human
clicking **approve**. Removing that click (the original ask: "auto-approve low-stakes")
does not reduce swamp — it just moves swamp from the `proposed` table into the `active`
table, turning canonical memory into an append-only log.

Meanwhile Orion's `substrate/` layer already models memory the way a mind does:
`activation.py` weights a node by `recency + salience + pressure` and **decays it on a
half-life**; `consolidation.py` runs review cycles that `reinforce` / `damp` / `retire`
regions by whether they are still active. The crystallization track is simply not wired
into any of it.

This spec closes that gap. Memory should form cheaply and weakly, **strengthen on
recurrence and recall** (repetition is a memory signal, not noise to dedupe), and
**decay and retire on disuse** (forgetting is the default). Weight + reinforcement +
decay replace the novelty-check and the approval button for routine memory. The human
gate shrinks to what actually deserves review: identity-level claims, contradictions,
and sensitive material.

## Current architecture

- **Formation:** `services/orion-memory-crystallizer/app/worker.py` — high-salience
  memory card → `proposer.propose()` → `insert_crystallization()` at `status=proposed`,
  `requires_manual_review=True`. No dedup, no reinforcement.
- **Weight:** `orion/memory/crystallization/salience.py` — `score_salience()` = kind
  base + evidence strength + confidence + planning boost. Computed once in
  `apply_salience()`; never updated afterward.
- **Promotion + projection:** hub `crystallization_routes.py:242` — human `approve` →
  `governor.approve()` → `project_crystallization()` (Graphiti/Chroma/cards) when
  `CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE=true`.
- **Dedup:** `detection.detect_duplicates()` (Jaccard) exists but only surfaces a
  warning on the manual `validate_proposal` endpoint; it never merges or reinforces.
- **The model that already works, but is disconnected:**
  - `orion/substrate/activation.py` — `seed_activation` (recency+salience+pressure),
    `decay_activation` (half-life decay to a floor).
  - `orion/substrate/consolidation.py` — outcomes `reinforce`/`damp`/`retire`/
    `keep_provisional`/`maintain_priority`/`requeue_review`.

Gap: a crystallization has no dynamic weight, no reinforcement counter, no recency of
reinforcement/recall, no decay, and no self-retirement. `salience` is a frozen prior.

## Missing questions (answered / to confirm)

- What is "importance" vs "current strength"? → keep `salience` as the intrinsic prior
  (from kind/evidence), add `activation` as the dynamic, moving weight. Mirrors substrate
  `node.signals.salience` vs `node.signals.activation`.
- Does reinforcement mutate canonical text? → No. Reinforcement bumps weight/recency and
  may append evidence refs; it does not rewrite `summary`/`claims`.
- What retires? → low `activation` + no recent reinforcement → `status=deprecated`
  (existing enum), de-projected from recall. Never hard-deletes canonical artifact.
- Privacy floor? → `intimate` sensitivity and high-stakes kinds stay human-gated
  regardless of weight. Decay/reinforcement still apply, but formation/promotion do not
  auto-bypass review for those.

## Proposed schema / API changes

Additive, defaulted, back-compat. New sub-model on `MemoryCrystallizationV1`:

```python
class CrystallizationDynamicsV1(BaseModel):
    activation: float = 0.0          # current dynamic weight (0..1)
    reinforcement_count: int = 0
    formed_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    last_recalled_at: datetime | None = None
    decay_half_life_days: float = 30.0
    retired_at: datetime | None = None
```

- `MemoryCrystallizationV1.dynamics: CrystallizationDynamicsV1 = Field(default_factory=...)`
- Persisted as a new `dynamics jsonb` column (`ADD COLUMN IF NOT EXISTS`), round-tripped
  in `repository.py`.
- Bus: add a `reinforced` lifecycle kind (Phase 2) alongside existing proposed/approved/
  project/etc.

Pure functions in new `orion/memory/crystallization/dynamics.py` (deterministic, no LLM,
reuse `orion.substrate.activation.decay_activation`):

- `seed_dynamics(crys, now)` — initial encoding: `activation = salience`, `formed_at=now`.
- `reinforce(crys, now, boost=0.2)` — `activation += (1-activation)*boost`, count+1,
  `last_reinforced_at=now`.
- `decay(crys, now)` — half-life decay from `last_reinforced_at or formed_at` to `now`.
- `recall_boost(crys, now, boost=0.08)` — smaller bump, `last_recalled_at=now`.
- `should_retire(crys, now, floor=0.05)` — decayed activation below floor.

## Files likely to touch

- `orion/memory/crystallization/schemas.py` — add `CrystallizationDynamicsV1` + field. (Phase 1)
- `orion/memory/crystallization/dynamics.py` — new pure functions. (Phase 1)
- `orion/core/storage/sql/memory_crystallizations.sql` — `dynamics jsonb` column. (Phase 1)
- `orion/memory/crystallization/repository.py` — round-trip `dynamics`. (Phase 1)
- `tests/test_memory_crystallization_dynamics.py` — unit tests. (Phase 1)
- `services/orion-memory-crystallizer/app/worker.py` — reinforce-on-recurrence. (Phase 2)
- `orion/memory/crystallization/bus_emit.py` + `orion/bus/channels.yaml` — `reinforced`. (Phase 2)
- `proposer.py` / formation path — cheap weak formation for low-stakes. (Phase 3)
- `orion-dream` / consolidation tick — periodic decay + retire reaper. (Phase 4)
- `retriever.py` / recall path — recall reinforces. (Phase 5)

## Non-goals

- No LLM anywhere in the weight path (deterministic, §4).
- No new taxonomy/labels without runtime behavior (no keyword cathedral).
- Not removing the governor/approve path — it stays for high-stakes/intimate.
- Not touching `intimate` gating or blocked-material privacy boundaries.
- No hard deletion of canonical artifacts — retirement is status + de-projection only.

## Acceptance checks

- Recurrence of the same subject/summary → **1 row**, `activation` increased,
  `reinforcement_count` incremented (no duplicate row). (Phase 2)
- An untouched low-stakes crystallization decays below floor after N half-lives →
  retired (`deprecated`) and dropped from the active recall packet. (Phase 4)
- Recall of a crystallization boosts its `activation` and sets `last_recalled_at`. (Phase 5)
- High-stakes / intimate still require human approval regardless of weight. (all phases)
- Phase 1: schema round-trips `dynamics`; `reinforce`/`decay`/`recall_boost`/`should_retire`
  behave per unit tests; **no live behavior changes** (nothing calls them yet).

## Phased plan

1. **Phase 1 (this patch, inert):** schema `dynamics` block + pure functions + SQL column
   + repository round-trip + unit tests. Nothing in the live path computes or reads
   dynamics yet. Fully additive and reversible.
2. **Phase 2:** reinforce-on-recurrence in the crystallizer worker (dedup → reinforce
   existing instead of insert new); `reinforced` lifecycle event.
3. **Phase 3:** cheap weak formation — low-stakes novel crystallizations form directly as
   `active` but low `activation`; high-stakes/intimate still gated. (This is where "stop
   making me click" lands safely, because decay bounds it.)
4. **Phase 4:** periodic decay + retire reaper (dream/consolidation tick).
5. **Phase 5:** recall reinforces (retriever bumps activation on packet inclusion).

## Recommended next patch

Phase 1 as scoped above. Behavior-changing phases (2+) require explicit go, since they
mutate what reaches canonical memory and recall.
