# Substrate-fed motivation → governed action (v1)

**Date:** 2026-07-06  
**Status:** Approved for implementation planning  
**Authors:** Operator + agent (brainstorming session)  
**Related:** self-modeling loop ladder (rungs 4–5), autonomy goals v2, world-pulse reflective journal, fcc-claude MCP, memory crystallizer, brainstorming appendix Idea 3 (action outcome feedback)

---

## Executive summary

Orion has rich substrate-runtime grammar, signal bridge, pressure field, and curiosity machinery — but autonomy (drives, goals, planner execution) was built **before** that substrate and still runs largely in parallel. Motivation does not reliably metabolize from substrate signals into the six canonical drives, and governed external action (Tier B) has no unified decision gate.

This spec defines **v1**: a thin metabolism adapter that feeds existing autonomy primitives, a **C-on-A+B** capability policy layer, and a **golden-path acceptance check** (world-pulse GPU coverage gap → curiosity → read-only web fetch → episode journal → sleep crystallization). It explicitly defers Tier A (internal-only auto), general MCP routing (EXP-C), and a full episode orchestrator.

**Approach:** Adapter retrofit (not a new motivation organ, not a hardcoded workflow pipe).

---

## Problem statement

### Observed gap

| Layer | Exists | Gap |
|-------|--------|-----|
| Substrate metabolism | Grammar events, signal bridge, pressure field, attention broadcast | Does not feed `DriveStateV1` / `GoalProposalEngine` |
| World pulse | Section rollups, `hardware_compute_gpu`, coverage status | Sparse section does not raise `predictive` drive or curiosity |
| Curiosity | `endogenous_curiosity.py`, `frontier_curiosity.py` | `ontology_sparse_region` is graph-native; no `world_coverage_gap` |
| Goals | `GoalProposalV1`, lifecycle to `planned` / `executing` | Mostly `proposal_only`; no episode lineage from spawn event |
| Action outcomes | `ActionOutcomeRefV1` in reducer | `no_action_outcome_history` flagged every cycle — never written |
| External fetch | fcc-claude + firecrawl MCP (planned) | Not wired to drive/goal/policy path |
| Sleep memory | episodic consolidation, memory crystallizer | No `autonomy_episode` source kind |

### Architectural mistake to avoid

Building a fixed workflow pipe:

```text
if gpu_section.empty → web_search → journal   # FORBIDDEN
```

Target: **episode-native, substrate-driven** transitions with grammar traces and causal parents.

---

## Goals

1. Substrate signals (world pulse, molecules, grammar events) reliably update the **six canonical drives** via existing `DriveEngine`.
2. Coverage gaps (e.g. empty `hardware_compute_gpu` digest) seed **curiosity** and **goal proposals** without operator trigger (flag-gated).
3. **Capability policy** (Layer C) decides auto vs promote-required using Layer A thresholds + Layer B goal state.
4. Tier B golden path: read-only web fetch → episode-closure journal → sleep crystallization, all with `spawned_correlation_id` from world-pulse `run_id`.
5. Close **action outcome feedback** loop (appendix Idea 3) for episode actions.

## Non-goals (v1)

- Keyword triggers on digest text (`if "GPU" in summary`)
- New drive taxonomy beyond coherence, continuity, capability, relational, predictive, autonomy
- Auto-promote to `planned` for write or external side-effect capabilities
- Replacement of chat `proposal_only` stance behavior
- Full `CognitiveEpisodeV1` orchestrator (minimal lineage fields only)
- Graphiti rail changes
- Tier A internal-only autonomy (deferred — see Future explorations)
- General fcc-claude MCP capability router (deferred)

---

## Tier model (D with explicit deferrals)

| Tier | Surface | v1 scope |
|------|---------|----------|
| **A** | Internal cognition auto (curiosity passes, concept induction, substrate review) | **Deferred** — EXP-A |
| **B** | Planner / mesh execution, read-only web fetch, episode journal | **v1 depth** |
| **C** | fcc-claude MCP as general capability dispatcher | **Deferred** — EXP-C |
| **D** | Escalation ladder (readonly auto → promote for writes) | **Authority model for v1** |

Future-you should explore EXP-A, EXP-B-ext, EXP-C, EXP-EP in separate design cycles (documented below).

---

## Authority model: C on A + B

Three layers, one decision. **C does not replace A or B** — it reads them.

### Layer A — Substrate imperative

*Should Orion feel a pull to act?*

- `predictive_pressure >= ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE` (default `0.55`)
- `curiosity_signal.strength >= ORION_METABOLISM_MIN_CURIOSITY_STRENGTH` (default `0.5`)
- Endogenous curiosity master flag + kill switch respected

### Layer B — Goal artifact

*What is the structured intent?*

- `GoalProposalV1` with `drive_origin=predictive`, `proposal_status=proposed`
- `ArtifactProvenance.spawned_correlation_id` = world-pulse `run_id`
- Existing `GoalProposalEngine` + cooldown/signature semantics unchanged

### Layer C — Capability policy table

*Given pull + intent, may this capability run?*

Checked-in policy file: `config/autonomy/capability_policy.v1.yaml`

Each rule:

| Field | Purpose |
|-------|---------|
| `capability_id` | e.g. `web.fetch.readonly`, `journal.compose.episode` |
| `side_effect_class` | `readonly` \| `write` \| `external` |
| `auto_execute` | whether policy may allow without operator |
| `requires_goal_status` | minimum goal status (`proposed`, `planned`, …) |
| `required_drive_origins` | e.g. `[predictive]` |
| `required_signal_kinds` | e.g. `[world_coverage_gap]` |
| `budget_per_cycle` | hard cap per substrate tick / episode |

Evaluator: `orion/autonomy/capability_policy.py` → `CapabilityDecisionV1` with `allowed | denied | requires_promote` and reason codes for grammar traces.

### v1 policy examples

| capability_id | side_effect | auto_execute | requires_goal_status |
|---------------|-------------|--------------|----------------------|
| `web.fetch.readonly` | readonly | true (if A passes) | proposed |
| `journal.compose.episode` | write | true (after fetch phase) | proposed |
| `world_pulse.run` | readonly | true (scheduler) | none |
| `web.fetch.write` | external | false | planned + operator |

---

## Architecture

### Current vs target

```text
AS-IS (parallel):
  bus events → concept-induction → drives/goals → RDF (proposal_only)
  substrate-runtime → grammar/signals/molecules (does not close loop)

TARGET (v1):
  bus events + substrate signals
    → substrate_metabolism_adapter (NEW)
    → DriveEngine + GoalProposalEngine (EXISTING)
    → capability_policy.evaluate() (NEW)
    → planner verb / journal / fcc-claude MCP (EXISTING, wired)
    → ActionOutcomeRefV1 (EXISTING schema, NEW writers)
    → episodic consolidation + crystallizer (EXISTING)
```

### Choke points

| File | Role |
|------|------|
| `orion/autonomy/substrate_metabolism.py` | **NEW** — `metabolize_substrate_signals()` |
| `orion/autonomy/capability_policy.py` | **NEW** — Layer C evaluator |
| `config/autonomy/capability_policy.v1.yaml` | **NEW** — policy table |
| `orion/spark/concept_induction/bus_worker.py` | Call metabolism before `DriveEngine` / `GoalProposalEngine` |
| `orion/signals/adapters/world_pulse.py` | Section-level gap hints in signal notes/dimensions |
| `orion/substrate/endogenous_curiosity.py` | Accept `world_coverage_gap` signal type |
| `orion/core/schemas/drives.py` | `spawned_correlation_id` on provenance |
| `orion/journaler/worker.py` | `autonomy_episode` trigger + narrative contract |
| `orion/autonomy/reducer.py` | Consume `ActionOutcomeRefV1` from episode actions |
| `services/orion-planner-react/app/api.py` | Episode-aware goal execute context |
| `orion/memory/crystallization/` | `source_kind=autonomy_episode` intake |

---

## Section 1 — Substrate metabolism adapter

### New module: `orion/autonomy/substrate_metabolism.py`

```python
def metabolize_substrate_signals(
    *,
    signals: Sequence[OrionSignalV1],
    molecules: Sequence[SubstrateMoleculeV1] | None = None,
    world_pulse_result: WorldPulseRunResultV1 | None = None,
) -> MetabolismResultV1:
    ...
```

**`MetabolismResultV1` outputs:**

| Field | Maps to |
|-------|---------|
| `drive_deltas: dict[str, float]` | Six canonical drive keys → pressure delta |
| `tensions: list[TensionEventV1]` | Optional, when drive spread exceeds threshold |
| `curiosity_signals: list[FrontierInvocationSignalV1]` | Including `world_coverage_gap` |

### Inputs

| Source | Consumed how |
|--------|--------------|
| `OrionSignalV1` (`world_pulse`) | Coverage level, section rollups in notes |
| `WorldPulseRunResultV1` (direct, when envelope is run result) | Per-section `digest_item_count == 0` → gap |
| `SubstrateMoleculeV1` | Gradient salience → capability/continuity deltas |
| `GrammarEventV1` (v1.1 optional) | `action_candidate` atoms |

### New signal type: `world_coverage_gap`

Distinct from `ontology_sparse_region` (substrate graph sparsity).

| Signal type | Meaning |
|-------------|---------|
| `ontology_sparse_region` | Substrate concept graph has sparse region |
| `world_coverage_gap` | World-pulse section had zero digest items this run |

Emission rule (deterministic):

```text
for section_rollup in digest.section_rollups:
  if rollup.digest_item_count == 0 and rollup.status in ("sparse", "empty"):
    emit world_coverage_gap(section=rollup.section, run_id=run.run_id, strength=f(section, status))
```

Default strength: `0.65` for tracked sections in `config/world_pulse/sources.yaml` `recommended_sections`; `0.45` otherwise.

Drive mapping:

- `world_coverage_gap` → `predictive` pressure +`0.15` (clamped)
- sustained gap across N runs (future) → tension event; v1 uses single-run only

### Wiring

In `ConceptWorker.handle_envelope()` (or substrate-runtime tick hook when envelope is world-pulse result):

```text
1. metabolize_substrate_signals(...)
2. DriveEngine.update(existing_state, deltas=result.drive_deltas)
3. merge curiosity_signals into endogenous/frontier path
4. GoalProposalEngine.propose(...)  # unchanged contract
```

Flag: `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED` (default `false`).

---

## Section 2 — Curiosity bridge

Extend `endogenous_curiosity.py` to accept `world_coverage_gap` alongside prediction error, repair pressure, and attention open-loops. Candidates still:

- Respect `ORION_ENDOGENOUS_CURIOSITY_ENABLED` and kill switch
- Respect per-cycle budget (`ORION_ENDOGENOUS_CURIOSITY_BUDGET`)
- Target `concept_graph` zone only (existing guardrail)
- Emit signals only — no direct external action

`FrontierCuriosityEvaluator.evaluate()` already accepts `endogenous_signals` — no new invocation authority.

Goal text: existing `GoalProposalEngine` + `goal_generator.py` with `drive_origin=predictive` and evidence from section rollup in `evidence_items`.

---

## Section 3 — Golden-path episode flow (acceptance check)

Reference scenario — **not** implementation shape. Each step must be provable via fixtures and grammar traces.

```text
1. PERCEIVE
   Scheduler → world-pulse run (dry_run=false)
   → orion:world_pulse:run:result
   → WorldPulseAdapter (level=0.4 sparse)
   → metabolism: hardware_compute_gpu digest_item_count=0
   → predictive pressure delta + world_coverage_gap signal
   → GrammarEventV1 action_candidate "coverage_gap_detected" (parent: run_id)

2. CURIOSITY
   endogenous + frontier decision → should_invoke=true
   → GoalProposalV1 with spawned_correlation_id=run_id
   → GrammarEventV1 action_candidate "curiosity_goal_proposed"

3. DECIDE (C on A+B)
   capability_policy.evaluate("web.fetch.readonly", ctx) → allowed
   → GrammarEventV1 action_candidate "capability_decision_allowed"

4. ACT (Tier B)
   Planner verb or mesh fcc-claude turn with firecrawl MCP
   → fetch ≤2 articles (budget from policy)
   → GrammarEventV1 per fetch, derived_from run_id
   → ActionOutcomeRefV1 (success/failure, surprise)

5. EPISODE CLOSURE
   journal.compose:
     trigger_kind = "autonomy_episode" (new)
     source_ref = goal artifact_id
     spawned_correlation_id = run_id
     narrative contract: digest → gap → curiosity → fetch → learnings → next steps
   → GrammarEventV1 memory_claim / spoken_output atoms

6. SLEEP CRYSTALLIZATION
   episodic_consolidation → EpisodeSummaryV1 (day window)
   → MemoryCrystallizationV1 proposed:
       source_kind = "autonomy_episode"
       source_grammar_event_ids = [...]
       spawned_correlation_id = run_id
   → governor approve → Chroma / Graphiti projection (existing paths)
```

### Minimal lineage (no full episode orchestrator)

Extend existing provenance — do not introduce `CognitiveEpisodeV1` in v1.

| Artifact | New fields |
|----------|------------|
| `ArtifactProvenance` | `spawned_correlation_id: str \| None`, `episode_id: str \| None` |
| `JournalEntryWriteV1` / draft | `episode_id`, `causal_parent_refs: list[ArtifactEventRef]` |
| `MemoryCrystallizationV1` | `source_grammar_event_ids`, `spawned_correlation_id` in governance/provenance |

`episode_id` derived deterministically from `spawned_correlation_id` + goal `artifact_id` when journal closes (same discipline as `derive_episode_id` in episodic consolidation).

---

## Section 4 — Error handling and safety

| Condition | Behavior |
|-----------|----------|
| `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=false` | No metabolism; existing autonomy path unchanged |
| `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=false` | All external capabilities require promote |
| Policy deny | Goal stays `proposed`; Hub/metadata shows reason code |
| Budget exhausted | Skip auto-execute; grammar event `capability_budget_exhausted` |
| Fetch failure | `ActionOutcomeRefV1` with `success=false`, `surprise=1.0`; episode journal includes honest failure |
| Kill switch | `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH=true` halts all endogenous seeds |

Invariants:

- No auto-apply substrate mutations
- Write/external capabilities require `planned` + operator (or internal service token where already authorized)
- Every auto-executed capability emits at least one `GrammarEventV1` with causal parent

---

## Section 5 — Env / config

| Key | Default | Effect |
|-----|---------|--------|
| `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED` | `false` | Metabolism adapter |
| `ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE` | `0.55` | Layer A threshold |
| `ORION_METABOLISM_MIN_CURIOSITY_STRENGTH` | `0.5` | Layer A threshold |
| `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED` | `false` | Layer C auto for readonly rules |
| `ORION_ENDOGENOUS_CURIOSITY_ENABLED` | `false` | Existing; extended for world_coverage_gap |
| `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH` | `false` | Existing |
| `ACTIONS_WORLD_PULSE_JOURNAL_ENABLED` | `false` | Existing; needed for step 1 journal seed |

Policy file path: `config/autonomy/capability_policy.v1.yaml` (versioned, loaded at boot).

---

## Section 6 — Testing and acceptance

### Gate tests (deterministic, <2s each)

1. `test_metabolism_sparse_gpu_section_raises_predictive` — fixture `WorldPulseRunResultV1` with empty `hardware_compute_gpu` rollup → `predictive` delta > 0
2. `test_world_coverage_gap_signal_emitted` — signal type and `run_id` in focal refs
3. `test_capability_policy_allows_readonly_when_goal_proposed` — A+B satisfied → `allowed`
4. `test_capability_policy_denies_without_goal` — A satisfied, no goal → `denied`
5. `test_episode_journal_carries_spawned_correlation_id` — journal payload lineage
6. `test_action_outcome_clears_reducer_flag` — reducer no longer emits `no_action_outcome_history` when outcome present

### Integration smoke (optional, flag on)

```text
world-pulse run (fixture or live) → metabolism → goal proposed → policy allow
→ mock web.fetch → journal.compose → crystallization propose
```

### Evals (periodic)

- Curiosity-to-action conversion rate (proposed goals → executed readonly fetches)
- Budget adherence (denials when cap hit)
- Episode journal completeness (narrative sections present vs contract)

---

## Future explorations (not v1)

Document for subsequent design sessions:

| ID | Topic | Notes |
|----|-------|-------|
| **EXP-A** | Tier A internal cognition auto | Curiosity passes, concept induction, substrate review without external fetch |
| **EXP-B-ext** | Planner mesh auto-promote | Broader `autonomy.goal.execute.v1` verb classes beyond readonly research |
| **EXP-C** | MCP capability router | fcc-claude as general dispatcher from drive state |
| **EXP-EP** | `CognitiveEpisodeV1` orchestrator | If minimal lineage fields prove insufficient for operator debug / replay |

---

## Schema / bus changes (v1)

### Added

- `MetabolismResultV1`, `CapabilityDecisionV1`, `CapabilityPolicyRuleV1` (or equivalent in `orion/autonomy/models.py`)
- `world_coverage_gap` in `FrontierInvocationSignalV1` signal_type union (`orion/core/schemas/frontier_curiosity.py`)
- `JournalTriggerKind`: `autonomy_episode`
- `spawned_correlation_id` on `ArtifactProvenance`
- Grammar event kinds / atom labels for capability decisions (reuse `action_candidate` where possible)

### Behavior changed

- `ConceptWorker` optionally calls metabolism before drive/goal engines (flag-gated)
- `endogenous_curiosity` accepts world-pulse gap signals
- Reducer consumes episode `ActionOutcomeRefV1`

### Unchanged

- Six canonical drive keys
- Goal lifecycle states and operator promote path for write/external
- Chat stance `proposal_only` semantics
- RDF autonomy materialization contract

---

## Recommended implementation phases

| Phase | Ships | PR boundary |
|-------|-------|-------------|
| **0** | Metabolism adapter + tests; flag off | Substrate → drive deltas only |
| **1** | `world_coverage_gap` + endogenous bridge + goal with lineage | Curiosity + goal proposed |
| **2** | Capability policy + readonly auto (flag off) | Decision gate |
| **3** | Tier B act: planner/fcc-claude fetch + `ActionOutcomeRefV1` | External read loop |
| **4** | Episode journal + sleep crystallization intake | Full golden path |

Each phase has tests before the next starts.

---

## Spec self-review (2026-07-06)

- [x] No TBD / placeholder sections
- [x] Architecture consistent with C-on-A+B and Approach 1
- [x] Scope bounded to Tier B; EXP-A/C/EP explicitly deferred
- [x] `world_coverage_gap` vs `ontology_sparse_region` disambiguated
- [x] No keyword cathedral / digest text triggers
- [x] Choke points named with file paths
- [x] Acceptance checks are structural (fixtures), not prompt substring asserts
