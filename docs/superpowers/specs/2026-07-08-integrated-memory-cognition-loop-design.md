# Integrated Memory Cognition Loop — Design

**Date:** 2026-07-08  
**Status:** Approved for implementation planning (operator review gate)  
**Authors:** Juniper + agent (brainstorming session)  
**Supersedes as north star:** Treats write gate, PCR, and weight/decay as **one program**, not three isolated patches.

**Program partners (existing specs, absorbed here):**

- [`2026-07-07-consolidation-crystallization-gate-design.md`](./2026-07-07-consolidation-crystallization-gate-design.md) — encode gate
- [`2026-07-07-purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md) — retrieve by purpose
- [`2026-07-07-memory-weight-reinforcement-decay-design.md`](./2026-07-07-memory-weight-reinforcement-decay-design.md) — lifecycle
- [`2026-07-07-unified-turn-self-grounding-design.md`](./2026-07-07-unified-turn-self-grounding-design.md) — orion unified lane carrier

---

## Arsonist summary

Orion built crystallizations, PCR, active-packet, and dynamics math — then chat still feels like **surface timeline SQL**. The loop is open at both ends: encoding stalls at `proposed` awaiting human clicks, and retrieval defaults to `continuity_only` with an empty belief store.

This spec closes one integrated cognition loop for **brain mode** and **orion unified mode**:

```text
ENCODE (cheap, weak) → STORE (canonical belief) → PROJECT (cards/chroma)
                              ↓ reinforce / recall_boost       ↓
                         LIFECYCLE (activation/decay)  ←── RETRIEVE (PCR by purpose)
```

Human gates stay for graph/RDF structural edits and high-stakes crystallizations (identity, intimate, contradiction, gated kinds). Routine memory forms automatically, strengthens on use, and fades on disuse.

---

## Executive summary

| Gap | Symptom today | Fix in this spec |
|-----|---------------|------------------|
| **Encoding** | Proposals pile up in `proposed`; nothing becomes active without operator click | `formation_policy` auto-activates low-stakes kinds at weak activation |
| **Storage** | Canonical beliefs thin; cards/chroma disconnected from live encode path | Auto-project on `auto_activate`; recall eligibility uses `activation` floor |
| **Retrieval** | PCR shipped but brain defaults to timeline SQL; unified gets Phase 3 only | Full PCR 0→1→stance→3 on both lanes; intent rule fetches beliefs when store has signal |
| **Lifecycle** | `dynamics.py` exists, nothing calls it | Wire reinforce/recall_boost/decay reaper |

**v1 lane scope:** Hub `brain` (`chat_general`) and `orion` unified turn (`stance_react`). Quick, grounded_small, agent, council unchanged.

---

## Current architecture

### Write path (shipped, blocked at governor)

```text
sql-writer → memory.turn.persisted → orion-memory-consolidation
  → turn_change_appraisal + window close
  → consolidation_memory_gate (skip | propose)
  → MemoryCrystallizationV1 status=proposed, requires_manual_review=True
  → Hub governor approve → project (cards/chroma/graphiti/rdf)
```

**Choke point:** `orion/memory/crystallization/governor.py::can_activate` requires `approved_by` when `requires_manual_review=True`. All consolidation intake sets that flag.

### Read path (shipped, starved)

```text
chat_general: PCR phase0+1 (sql_chat continuity) → stance → phase3 (often skipped)
stance_react: stance → phase3 only via grounding_capsule (no pre-stance continuity)
active_packet: list_crystallizations(status=active) — empty when store is proposed-only
```

**Choke points:**

- `orion/memory/retrieval_intent.py` — default `continuity_only` on most turns
- `orion/recall/profiles/chat.continuity.v1.yaml` — sql_chat only, 2hr window
- Hub `brain.recall.v1` — legacy score-soup profile (sql_chat + sql_timeline + rdf) when PCR off

### Lifecycle (inert)

- `orion/memory/crystallization/dynamics.py` — pure functions, unit-tested
- `salience` frozen at creation; no reinforce-on-recurrence in consolidation intake
- `detection.detect_duplicates()` — warning on Hub validate only

---

## Target architecture — the closed loop

```text
                    ENCODE                    STORE                 RETRIEVE              LIFECYCLE
                      │                         │                      │                     │
Turn persisted ──► consolidation gate ──► crystallization ──► projections ──► PCR by intent ──► reinforce/decay
                      │                         │                 (cards/chroma)              │
                      │                    canonical belief          │                     │
                      └──────────────── skip (low signal) ──────────┴─────────────────────┘
```

### Deterministic choke points

| Phase | Owner | Function / module |
|-------|-------|-------------------|
| Encode | `orion-memory-consolidation` | `consolidation_memory_gate` → `resolve_formation_policy` |
| Store | `orion/memory/crystallization/formation_executor.py` (new) | `auto_activate` or `governor_queue` + `seed_dynamics` |
| Project | `orion-hub` / crystallization projector | cards + chroma on auto_activate (not/rdf/graphiti) |
| Retrieve | `orion-cortex-exec` + `orion-recall` | `pcr_chat_memory` + `retrieval_intent` + `active_packet` |
| Lifecycle | `dynamics.py` + reaper tick | reinforce, recall_boost, decay, retire |

---

## Section 1 — Encoding

### When to encode

No change to consolidation gate inputs:

- `turn_change_appraisal` (novelty, shift_kind)
- `memory_significance_score`
- `is_low_info_social` skip
- Read-only grammar event refs (repair signal)

Gate output remains `skip` or `propose`. Formation policy decides what happens **after** propose.

### Granularity (one belief per closed window)

Kind from dominant shift in window (existing proposer logic):

| Signal | Kind | Summary shape |
|--------|------|-----------------|
| TOPIC | `semantic` | Subject + 1–3 sentences |
| STANCE / relational | `stance` or `episode` | Relational frame or bounded episode |
| REPAIR | `open_loop` | Unresolved thread |
| Planning markers in stance | `procedure` | Steps / decision path |
| Contradiction seed in attention | `contradiction` | Always gated |

### Formation policy (new)

**File:** `orion/memory/crystallization/formation_policy.py`

```python
AUTO_ACTIVE_KINDS = frozenset({"semantic", "episode", "open_loop", "procedure"})
GATED_KINDS = frozenset({"stance", "decision", "contradiction", "attractor", "failure_mode"})
IDENTITY_SCOPE_PREFIX = "identity:"  # scope tags requiring governor
```

| Condition | Policy | Result |
|-----------|--------|--------|
| Duplicate (Jaccard ≥ threshold) | `reinforce_existing` | `reinforce()` on existing row; append evidence ref |
| `kind in AUTO_ACTIVE_KINDS` AND `sensitivity != intimate` AND no identity scope tag | `auto_activate` | `status=active`, `seed_dynamics`, weak activation |
| `kind in GATED_KINDS` OR `sensitivity == intimate` OR identity scope | `governor_queue` | `status=proposed`, `requires_manual_review=True` (current behavior) |
| Validation fails on auto path | `governor_queue` | Downgrade; emit `formation_policy_downgrade` trace |

**Initial activation on auto_activate:**

```text
activation = clamp(salience * AUTO_ENCODE_ACTIVATION_RATIO, 0.05, 0.35)
# AUTO_ENCODE_ACTIVATION_RATIO default 0.4
```

Beliefs enter cognition weakly and must earn strength through recurrence and recall.

### Recurrence (dedup → reinforce)

Wire `detection.detect_duplicates()` into consolidation intake **before** insert:

1. Match found → `reinforce(existing, boost=0.2)` + append window evidence
2. Emit `memory.crystallization.reinforced.v1` (new lifecycle kind)
3. No new row

---

## Section 2 — Storage and projections

### Canonical

`memory_crystallizations` Postgres — source of truth. Add / complete `dynamics jsonb` column per weight/decay spec.

**Schema addition (if not already merged):**

```python
class CrystallizationDynamicsV1(BaseModel):
    activation: float = 0.0
    reinforcement_count: int = 0
    formed_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    last_recalled_at: datetime | None = None
    decay_half_life_days: float = 30.0
    retired_at: datetime | None = None
```

### Recall eligibility

```text
eligible_for_recall(crys) :=
  crys.status == "active"
  AND crys.dynamics.activation >= ACTIVATION_RECALL_FLOOR  # default 0.08
  AND crys.status not in (deprecated, archived, rejected)
```

`list_crystallizations` and `active_packet` must filter on activation, not status alone.

### Projections on auto_activate

| Projection | Auto? | Notes |
|------------|-------|-------|
| Memory card | Yes | `build_memory_card_projection()` — chat injection surface |
| Chroma | Yes | crystallization vector collection |
| RDF memory_graph | No | Human graph editor (existing C) |
| Graphiti | No | Human / explicit activation spec |

### auto_activate executor (new)

**File:** `orion/memory/crystallization/formation_executor.py`

Parallel to `governor.approve()`:

```python
def auto_activate(crystallization, *, actor: str = "system:formation_policy") -> tuple[MemoryCrystallizationV1, dict]:
    # validate_proposal must pass
    # status → active
    # governance.approval_mode → auto_policy
    # governance.requires_manual_review → False
    # governance.approved_by → actor
    # seed_dynamics()
    # trigger project_crystallization(cards + chroma only)
```

Governor `approve()` unchanged for gated items.

---

## Section 3 — Retrieval (PCR, breadth and depth)

### Lane wiring (v1 scope)

| Hub mode | Cortex verb | PCR phases | Change |
|----------|-------------|------------|--------|
| `brain` | `chat_general` | 0 → 1 → stance → 3 | Hub stops defaulting `brain.recall.v1`; PCR only |
| `orion` | `stance_react` | 0 → 1 → stance → 3 | Add phase 0+1 before stance (today: phase 3 only in `grounding_capsule`) |
| `quick`, `grounded_small` | `chat_quick`, etc. | unchanged | Out of scope v1 |

### Hub recall default change

`services/orion-hub/scripts/cortex_request_builder.py`:

- Brain mode with `use_recall=true` and no explicit profile → omit legacy `brain.recall.v1`
- Cortex-exec PCR path owns profile selection via phase/intent
- Explicit `recall_profile` overrides preserved

Deprecate `brain.recall.v1` in docs; remove as default in M4 (profile file may remain for explicit override during transition).

### Intent rule fix

**File:** `orion/memory/retrieval_intent.py`

Add rule **before** `continuity_only` fallback:

```text
IF hub_chat_lane IN ("brain", "orion")
AND phase0 did not skip
AND session has eligible crystallizations (activation >= floor)
AND no higher-priority rule fired
THEN return ("semantic", "brain_lane_belief_default")
```

Higher-priority rules unchanged: `open_loop`, `relational`, `contradiction`, `procedural`, topic shift, entity query.

`hub_chat_lane` must be passed from Hub surface_context into cortex ctx (verify wire; add if missing for brain/orion modes).

### Depth per intent (existing profiles, activation-ranked packet)

| Intent | Backends | max items | render budget |
|--------|----------|-----------|---------------|
| `continuity` | sql_chat | 6 | 96 tokens |
| `semantic` | active_packet, cards, rdf_chat | 8 | 128 tokens |
| `relational` | active_packet(stance), cards, rdf | 6 | 128 tokens |
| `open_loop` | active_packet(open_loops), rdf_chat | 6 | 128 tokens |
| `procedural` | active_packet(procedures), cards | 6 | 128 tokens |
| `contradiction` | active_packet, cards, graphiti (if enabled) | 6 | 128 tokens |

**active_packet ranking:** `score = dynamics.activation * salience` (replace salience-only sort in `retriever.py`).

### Unified grounding capsule

`assemble_stance_grounding` runs full PCR (phase 0+1 before stance slice parse, phase 3 after). `GroundingCapsuleV1` carries `continuity_digest` + `belief_digest` (parity with brain mode per unified-turn self-grounding spec).

### CHAT_PCR_QUICK_PHASE3

Unchanged for quick lane. Brain lane is not quick — phase 3 always runs when intent warrants.

---

## Section 4 — Lifecycle

Wire `orion/memory/crystallization/dynamics.py` (deterministic, no LLM):

| Event | Function | Trigger |
|-------|----------|---------|
| Formation | `seed_dynamics()` | `auto_activate` or `governor.approve` |
| Recurrence | `reinforce(boost=0.2)` | dedup match on encode |
| Recall | `recall_boost(boost=0.08)` | crystallization included in active_packet response |
| Decay | `decay()` | reaper tick |
| Retire | `should_retire(floor=0.05)` | activation below floor after decay |

**Retire behavior:**

- `status → deprecated`
- `dynamics.retired_at → now`
- De-project from cards/chroma (not hard delete)
- Emit `memory.crystallization.retired.v1`

**Reaper:** New function in `orion-memory-consolidation` worker or `orion-dream` tick (every 6h, env `MEMORY_COGNITION_REAPER_ENABLED`). Batch scan `status=active`, apply decay, retire eligible, log counts.

**Privacy:** Intimate and gated kinds never `auto_activate`. Dynamics apply only after human approval for those.

---

## Section 5 — Error handling and degradation

| Failure | Behavior |
|---------|----------|
| PCR RPC timeout | Turn continues; continuity-only or identity-only capsule |
| active_packet empty | Fallback: cards + sql_chat continuity; metric `belief_store_thin` |
| auto_activate validation fail | Downgrade to `proposed`; trace `formation_policy_downgrade` |
| Chroma project fail | Belief remains in Postgres; recall uses cards/sql |
| reaper row error | Skip row, increment error count; never hard-delete canonical |

**Telemetry** (extend existing `recall.decision.v1` / cortex debug):

- `formation_policy`, `activation_floor`, `belief_count`, `continuity_count`
- `retrieval_intent`, `rule_id`, `pcr_phase`
- `recall_boost_applied` (list of crystallization_ids)

---

## Proposed schema / bus changes

### Additive only

| Change | Location |
|--------|----------|
| `CrystallizationDynamicsV1` + `dynamics` field | `orion/memory/crystallization/schemas.py` |
| `dynamics jsonb` column | `orion/core/storage/sql/memory_crystallizations.sql` |
| `memory.crystallization.reinforced.v1` | `orion/bus/channels.yaml`, registry |
| `memory.crystallization.retired.v1` | `orion/bus/channels.yaml`, registry |
| `memory.crystallization.auto_activated.v1` | optional lifecycle event for debug UI |
| Env keys | See below |

### Env keys (new)

```text
# Formation
MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=false  # flip true after M1 smoke passes
MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO=0.4
MEMORY_FORMATION_IDENTITY_SCOPE_PREFIX=identity:

# Recall
ACTIVATION_RECALL_FLOOR=0.08
MEMORY_COGNITION_BRAIN_BELIEF_DEFAULT=true

# Lifecycle
MEMORY_COGNITION_REAPER_ENABLED=true
MEMORY_COGNITION_REAPER_INTERVAL_HOURS=6
MEMORY_COGNITION_RETIRE_ACTIVATION_FLOOR=0.05
```

Update `.env_example` for: `orion-memory-consolidation`, `orion-cortex-exec`, `orion-recall`, `orion-hub` as touched.

---

## Files likely to touch

### Encode + store

- `orion/memory/crystallization/formation_policy.py` (new)
- `orion/memory/crystallization/formation_executor.py` (new)
- `orion/memory/crystallization/intake_consolidation_window.py`
- `orion/memory/crystallization/detection.py` — wire dedup into intake
- `orion/memory/crystallization/schemas.py`, `repository.py`
- `services/orion-memory-consolidation/app/worker.py`
- `services/orion-hub/scripts/crystallization_routes.py` — project on auto_activate

### Retrieve

- `orion/memory/retrieval_intent.py`
- `services/orion-cortex-exec/app/pcr_chat_memory.py`
- `services/orion-cortex-exec/app/router.py` — stance_react phase 0+1
- `services/orion-cortex-exec/app/grounding_capsule.py`
- `orion/memory/crystallization/retriever.py`, `active_packet.py`
- `services/orion-recall/app/collectors/active_packet.py`
- `services/orion-hub/scripts/cortex_request_builder.py`

### Lifecycle

- `orion/memory/crystallization/dynamics.py` (existing)
- `services/orion-recall/app/worker.py` — recall_boost after fetch
- `services/orion-memory-consolidation/app/reaper.py` (new) or dream tick

### Tests + smoke

- `tests/test_formation_policy_auto_vs_gated.py` (new)
- `tests/test_encode_reinforce_not_duplicate.py` (new)
- `services/orion-cortex-exec/tests/test_pcr_chat_memory.py` — brain belief default
- `services/orion-cortex-exec/tests/test_unified_pcr_phase01.py` (new)
- `services/orion-recall/tests/test_active_packet_activation_floor.py` (new)
- `scripts/smoke_memory_cognition_loop_e2e.sh` (new)

---

## Non-goals

- No new memory stores, ontology nodes, or keyword cathedrals
- No vector recall re-enablement for chat turns (Chroma for crystallization neighborhood only)
- No auto RDF / Graphiti projection
- No grammar substrate schema changes
- Quick / grounded_small / agent / council lane changes in v1
- No LLM in formation policy, weight path, or retrieval intent
- No hard deletion of canonical crystallizations

---

## Acceptance checks

### Encoding

- [ ] Topic-shift window with `semantic` kind → `status=active` without Hub click when `MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true`
- [ ] `contradiction` or `intimate` → `status=proposed`, governor queue unchanged
- [ ] Duplicate subject/summary → one row, `reinforcement_count` incremented, no second insert

### Storage

- [ ] Auto-activated belief has `dynamics.activation` in `[0.05, 0.35]` range
- [ ] Card + Chroma projections exist after auto_activate
- [ ] No RDF/Graphiti projection on auto_activate

### Retrieval (brain mode)

- [ ] Brain turn with active beliefs → `retrieval_intent=semantic`, `belief_digest` non-empty
- [ ] Greeting / low-info → phase 0 skip, empty belief_digest
- [ ] `brain.recall.v1` not used as Hub default

### Retrieval (orion unified)

- [ ] `stance_react` turn runs phase 0+1 before stance
- [ ] `GroundingCapsuleV1` has non-empty `belief_digest` when store has eligible beliefs

### Lifecycle

- [ ] Recall of crystallization sets `last_recalled_at` and bumps activation
- [ ] Simulated 90-day decay on untouched low-stakes belief → `deprecated`, excluded from active_packet
- [ ] Reaper emits `memory.crystallization.retired.v1`

### Smoke

- [ ] `scripts/smoke_memory_cognition_loop_e2e.sh` PASS on live stack

---

## Phased rollout

| Milestone | Ships | Operator-visible |
|-----------|-------|------------------|
| **M1 — Activate the store** | `formation_policy`, `auto_activate`, dynamics schema, reinforce-on-dedup, auto-project cards/chroma | Beliefs appear without clicks |
| **M2 — Unify read path** | PCR lane fixes, intent brain default, unified phase 0+1, activation-ranked packet | Chat stops being timeline-only |
| **M3 — Lifecycle live** | recall_boost, decay reaper, retire + de-project | Repetition strengthens; disuse fades |
| **M4 — Deprecate legacy** | Hub default recall change, `brain.recall.v1` docs marked deprecated | Brain fully on PCR |

**Recommended ship:** M1 + M2 together for maximum impact on brain and orion unified lanes.

---

## Risks and mitigations

| Severity | Risk | Mitigation |
|----------|------|------------|
| Medium | Auto-activate produces low-quality beliefs | Weak initial activation + decay retires noise; gated kinds unchanged |
| Medium | Belief default rule over-fetches | Activation floor + render budgets; intent priority rules unchanged |
| Low | Chroma projection lag | Postgres + cards fallback; metric alerts |
| Low | Reaper retires still-useful belief | Floor tunable via env; reinforce on recall |

---

## Recommended next patch

**M1 Task 1:** `formation_policy.py` + unit tests + wire into `intake_consolidation_window.py` behind `MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED` (default false until smoke passes, then flip true).

Invoke implementation plan via `writing-plans` after operator approves this spec file.
