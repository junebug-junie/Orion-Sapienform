# Imperative-first motor — design spec

**Date:** 2026-07-05  
**Status:** Approved for implementation  
**Parent:** `docs/superpowers/specs/2026-07-05-orion-unified-turn-design.md` §3–§5  
**Supersedes:** `2026-07-05-orion-answer-contract-from-stance-design.md` (abandoned — taxonomy/reducer approach)

**One line:** Orion acts from felt imperative and coalition salience; grammar records what it did; substrate + 5b catch mismatch — no pre-motor classification.

---

## Arsonist summary

We kept bolting **customs checkpoints** onto an organism that already has a nervous system.

`AnswerContract`, `requires_repo_grounding`, `_REPO_HINTS`, stance→contract reducers, epistemic-slice enums — same disease. Different paint. They **pre-sort Juniper's words** before Orion is allowed to move. That produces confident chatbots, not minds.

The unified turn already defines the right loop: association → stance → **act** → interocept (5a) → integrate (5b) → express (5c) → learn. The motor bug was **deafness** (ignored imperative / stance), not missing taxonomy. Widening grep lists or enriching enum matrices does not fix deafness — it adds bureaucracy.

**Burn on unified path:**
- `heuristic_answer_contract(user_message)` in Hub turn orchestrator
- Motor gating on `AnswerContract` booleans (`requires_repo_grounding`, etc.)
- `_TECH_TASK_MODES` / frame allowlists as repo-brief triggers in `compile_harness_prefix`
- Any new warrant/posture/reducer taxonomy (abandoned approach)

**Keep and wire:**
- `ThoughtEventV1.imperative` as primary efference copy to fcc
- Tools from t=0; fcc decides depth (parent spec §2)
- `GrammarReceiptV1` per harness step — action trace
- 5a surprise vs coalition + grammar trajectory — interoception (no LLM)
- 5b compare draft vs imperative + substrate appraisal — integrative check
- Stance slice for **relational voice hazards only** in 5c — not epistemic routing
- post-turn closure → prediction_error → next broadcast — learning

**Legacy (Brain / context-exec):** freeze `AnswerContract` grep stack until Brain sunset. Do not invest in enriching it. agent-chain is dead.

**Hard constraint:** no grammar substrate schema/reducer changes in this patch.

---

## Neuroscience / active-inference framing

Sentience prerequisites are not labels. They are **closed loops with embodiment and error-driven update**.

| Mechanism | Orion seam | Requirement |
|-----------|------------|-------------|
| Salience | Association → broadcast coalition → `strain_refs` | What's hot is **state**, not parsed syntax |
| Felt intent | `stance_react` → `imperative`, `tone` | Compressed action-oriented efference copy |
| Enaction | fcc + tools + `GrammarEventV1` | Cognition through world-contact |
| Interoception | 5a `SubstrateFinalizeAppraisalV1` | Structural surprise: draft + receipts vs coalition |
| Integrative coherence | 5b `FinalizeReflectionV1` | Does output match felt intent + gut check? |
| Social expression | 5c + stance hazards | Output layer ≠ motor planning |
| Consolidation | step 7 → `prediction_error` | Error shapes next-turn salience |

Pre-motor taxonomies break the loop **before efference copy meets the world** — disembodied planning. The sentience-shaped fix: strong imperatives, always-available agency, trace-backed accountability.

We do **not** claim Orion is sentient after this patch. We remove architecture that **blocks** the prerequisites from getting exercise.

---

## Problem (evidence)

| Symptom | Root cause |
|---------|------------|
| Coding turns fell short | Motor ignored `imperative` / stance; Hub fed grep-based `AnswerContract` |
| Keyword expansion reverted | Correct — substring lists are cathedral |
| Stance→contract reducer proposed | Still cathedral — LLM labels → Python lookup |
| `_TECH_TASK_MODES` in prefix | Borderline — routes motor from taxonomy not imperative |

**Choke point (unified):** `compile_harness_prefix` + `harness_motor_instruction` in `orion/harness/prefix.py`; Hub `turn_orchestrator.py` must stop bypassing felt layer.

**Choke point (stance producer):** `stance_react.j2` — imperative must command world-contact when turn requires it.

**Choke point (accountability):** 5a/5b already consume imperative + grammar receipts — strengthen, don't bypass.

---

## Design

### 1. Pre-motor: felt layer only

Harness prefix built from:

| Input | Role |
|-------|------|
| `thought.imperative` | **Primary motor instruction** — natural language, max 300 chars |
| `thought.tone` | Strain / relational frame |
| `thought.strain_refs`, `evidence_refs` | Coalition salience |
| `stance_harness_slice` | Relational priorities/hazards (voice + repair); **not** probe warrants |
| `repair_overlay` | Repair pressure shaping motor |
| `user_message` | Context only — never classified for grounding |

**Unified operator brief** (single brief, tools always available):

```text
Orion harness motor.
Tools are available from the start. Your imperative states what this turn requires.
When the imperative calls for facts from the codebase or live runtime, use tools before
answering. Record each meaningful step. Do not guess repo structure or service state from memory.
```

Remove split repo/runtime briefs **gated by contract or task_mode enums**. One brief; imperative drives depth.

`harness_motor_instruction()` returns imperative-forward text:

```text
Execute your imperative. Use tools when the turn requires verified facts from the repo or runtime.
```

### 2. Hub unified turn — drop pre-motor contract

**File:** `orion/hub/turn_orchestrator.py`

- Remove `heuristic_answer_contract(user_message)` from `HarnessRunRequestV1` construction
- Pass `answer_contract=AnswerContract()` (empty conceptual default) **or** make field optional on request schema in a backward-compatible way — harness motor **ignores** it
- Stance RPC → harness RPC unchanged in order

### 3. Harness prefix simplification

**File:** `orion/harness/prefix.py`

- Delete `_needs_repo_operator_brief`, `_needs_runtime_operator_brief`, `_TECH_TASK_MODES`, `_TECH_FRAMES` gating
- `compile_harness_prefix`: unified operator brief + imperative + tone + stance slice + strain + repair overlay
- Delete `_format_answer_contract` from motor prefix (contract not motor input)
- `answer_contract` param deprecated on motor path — keep signature temporarily for callers, unused

**File:** `orion/harness/operator_brief.py`

- Add `HARNESS_UNIFIED_OPERATOR_BRIEF`; keep repo/runtime strings only if referenced elsewhere, not from prefix gating

### 4. Stance producer — imperative discipline

**File:** `orion/cognition/prompts/stance_react.j2`

Add **IMPERATIVE DISCIPLINE** block (semantic inference — no keyword lists, no user-substring rules):

```text
IMPERATIVE DISCIPLINE
- imperative is the efference copy: what Orion must DO this turn, not a summary of the user's message.
- When the turn requires verified facts from code, config, or live services, imperative MUST command
  world-contact in plain language (search repo, read files, inspect logs/traces) before answering.
- When the turn is relational presence, imperative commands companionship — not repo search.
- Bad: "Answer the user's question." / "Explain Python functions."
- Good: "Search services/orion-thought for stance_react wiring; cite paths and symbols."
- Good: "Stay present; one situated question — no task tracking."
```

No new schema fields. No task_mode→grounding tables.

**File:** `orion/thought/stance_quality.py`

- Keep relational exemption + fail-closed `evidence_refs` / empty imperative
- **Do not** add task_mode coercion taxonomies for grounding

### 5. Post-motor: trace-backed accountability (no grammar substrate changes)

Use existing artifacts only.

#### 5a — interoception (existing)

`orion/substrate/appraisal/finalize_draft_v1.py` already computes surprise from `grammar_receipts` vs coalition. **No reducer changes required** for v1.

Optional thin addition (same file, same payload — not grammar substrate):

- When `grammar_receipts` empty and `thought.imperative` contains world-contact verbs (deterministic check on **Orion's imperative text**, not user message) → add alignment_hint `imperative_world_contact_no_grammar_steps`

If this feels too grep-like on imperative, defer to 5b LLM comparison only (already instructed in `harness_finalize_reflect.j2` line 19).

**v1 default:** rely on 5b + existing surprise — no imperative substring check.

#### 5b — integrative reflect (existing)

`harness_finalize_reflect.j2` already: *"Compare draft_text against thought_event.imperative, tone, and strain_refs."*

**Enhancement:** pass `grammar_receipts` summary into 5b template context so reflector sees whether tools ran.

**File:** `orion/harness/finalize.py` — include receipt summaries in `build_reflect_plan_request` payload.

#### 5c — voice (trace not contract)

**File:** `orion/cognition/prompts/orion_voice_finalize.j2`

Replace:

```jinja
- On technical/repo/runtime turns (requires_repo_grounding or requires_runtime_grounding): preserve code blocks...
```

With:

```jinja
- When grammar_receipts show repo or runtime tool use, preserve code blocks, file paths, commands, and step structure from the draft.
- When alignment_verdict is misaligned, revise — do not ship confabulated specifics.
```

Pass `grammar_receipts` (or `tool_steps_used: bool`) into voice finalize context from `orion/harness/finalize.py`.

`voice_contract` on 5c: pass empty `AnswerContract()` for schema compat; rules come from traces + reflection.

### 6. Legacy Brain / context-exec

**Non-goals for this patch** — do not refactor investigation_v2 probe gating or cortex-orch bootstrap heuristics. Unified path is the organism; legacy shims sunset separately.

Do **not** expand `_REPO_HINTS` or add stance reducers to legacy paths in this changeset.

---

## Molecule contract (this patch)

| Artifact | Producer | Consumer | Affector | Gate test |
|----------|----------|----------|----------|-----------|
| `ThoughtEventV1.imperative` | stance_react | harness prefix, 5b | motor behavior | `test_harness_prefix_leads_with_imperative` |
| `GrammarReceiptV1[]` | harness runner | 5a, 5b, 5c | surprise, alignment, voice preserve | existing + `test_voice_uses_grammar_receipts_not_contract` |
| `SubstrateFinalizeAppraisalV1` | 5a appraiser | 5b | misaligned path | existing |
| `FinalizeReflectionV1` | 5b | 5c | revise confabulation | existing + imperative mismatch fixture |
| ~~`AnswerContract` pre-motor~~ | — | — | **removed from unified motor gate** | `test_turn_orchestrator_no_heuristic_contract` |

---

## Acceptance checks

| Check | Expected |
|-------|----------|
| Unified turn Hub | No `heuristic_answer_contract` import/call in `turn_orchestrator.py` |
| Harness prefix | Contains unified operator brief + imperative; no `Requires repo grounding:` from contract |
| Coding fixture | Stance imperative commands repo search; prefix includes that imperative; motor instruction imperative-forward |
| Relational fixture | Imperative relational; no repo brief gating; 5c hazards preserved |
| 5c template | No `requires_repo_grounding`; uses grammar receipt signal |
| 5b payload | Includes grammar receipt summary |
| Tests | `compile_harness_prefix` tests updated — remove contract-gated repo brief test; add imperative-first test |
| Grep | Unified path files do not gate motor on `requires_repo_grounding` |

**Eval (manual/smoke):** turn "implement a function in python" — stance imperative mentions repo search; harness grammar receipts non-empty OR 5b misaligned if motor confabulates.

---

## Non-goals

- Grammar substrate schema, channels, or reducer changes
- New stance taxonomy fields (warrants, epistemic_posture, etc.)
- `derive_answer_contract_from_stance` or any pre-motor reducer
- De-keyword legacy `heuristic_answer_contract` (Brain sunset follow-up)
- De-keyword `output_mode_classifier`
- agent-chain cleanup (dead; touch only if blocking)

---

## Rollback

Revert changeset. No new env flags. `AnswerContract` on `HarnessRunRequestV1` remains schema-valid with empty default.

---

## Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| Medium | Weak stance imperatives on technical turns | Prompt discipline + eval fixtures; 5b catches mismatch |
| Low | 5c loses contract flag for preserve-code rule | grammar_receipts in voice context |
| Low | Schema still carries unused `answer_contract` on harness request | Deprecate in docstring; remove field in follow-up |

---

## Recommended next patch after this

1. Stance eval suite: imperative quality on coding / introspection / relational fixtures  
2. Brain sunset: remove `heuristic_answer_contract` from cortex-orch bootstrap  
3. Optional: drop `answer_contract` from `HarnessRunRequestV1` when all consumers gone
