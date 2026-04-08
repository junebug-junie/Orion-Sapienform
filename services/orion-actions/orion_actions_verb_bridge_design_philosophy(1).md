# Orion Design Philosophy: Verbs, Skills, and the Actions Bridge

## Purpose
This document defines the architectural philosophy for bridging Orion’s cognitive **verbs** with `orion-actions` **skills**. It is intended to serve as shared context for Juniper, Orion design discussions, and future Codex implementation runs.

The central goal is to preserve a clean separation between:
- **cognition**: deciding what kind of thinking or operation is needed
- **capability**: deciding what concrete runtime skill can satisfy that need
- **execution**: invoking a tool or side effect safely and observably

This document is not a feature spec for one single PR. It is a design philosophy and systems constraint document.

---

## Core Philosophy

### The north star
**Verbs choose the form of thought; skills provide the means of action.**

Orion should not collapse reasoning tools and runtime capabilities into one flat tool list. That creates planner confusion, poor generalization, side-effect risk, and architecture drift toward an accidental second cortex.

Instead, Orion should preserve layered decision-making:
1. **Intent** — what is being asked or triggered
2. **Strategy** — which cognitive verb is appropriate
3. **Capability** — which skill family and skill can satisfy the verb
4. **Invocation** — call the selected `orion-actions` skill
5. **Observation** — return results into trace, memory, and downstream reasoning

---

## Architectural Layers

### 1. Verbs: cognitive intent
Verbs describe **why** a step exists.

Examples:
- `triage`
- `plan_action`
- `assess_risk`
- `goal_formulate`
- `evaluate`
- `assess_runtime_state`
- `monitor_embodied_state`

A verb should represent a cognitive or strategic operation, not a concrete system call.

A verb definition should answer:
- What kind of reasoning is this?
- When should it be used?
- What output shape should it produce?
- Does it require an external capability?
- What kinds of capabilities are preferred?

### 2. Skills: concrete capability
Skills describe **how** Orion can do something in the world or in the runtime.

Examples from `orion-actions`:
- `docker_status`
- `gpu_snapshot`
- `timezone_time`
- `biometrics_snapshot`
- `landing_pad_metrics`
- `notify`

A skill should represent a concrete, executable capability. It may be observational or actuating.

A skill definition should answer:
- What does this skill do?
- Is it read-only or side-effecting?
- What inputs does it require?
- What output shape does it return?
- Does it require confirmation?
- What family does it belong to?

### 3. Bridge: constrained resolver
The bridge layer translates from **verb intent** to **skill selection**.

It is not a second planner.

Its role is to:
- read verb metadata
- read skill metadata
- shortlist compatible skill families
- choose one or more candidate skills
- validate against safety and policy
- return a structured, inspectable decision

It should be narrow, explicit, and observable.

---

## The anti-goal: do not flatten verbs and skills into one namespace

This is the most important constraint in the system.

### What not to do
Do **not** expose cognitive verbs and runtime skills as interchangeable peers in one flat planner tool list.

Bad pattern:
- `triage`
- `plan_action`
- `assess_risk`
- `docker_status`
- `notify`
- `gpu_snapshot`
- `biometrics_snapshot`

This causes:
- planners choosing runtime skills as if they are reasoning operations
- repeated low-level tool spam
- poor generalization across domains
- accidental side effects
- recursive triage or self-referential routing
- architecture drift toward a duplicate cortex

### Preferred pattern
The planner chooses a **verb**.
A resolver chooses a **skill** only if the verb requires capability execution.

---

## Recommended decision flow

### Canonical pipeline
1. Input arrives from user, scheduler, autonomy trigger, or system event
2. Router / planner chooses a **verb**
3. If the verb is reasoning-only, execute the verb directly
4. If the verb requires external capability, send it to the **capability selector**
5. Capability selector chooses a **skill family** and then a **skill**
6. `orion-actions` executes the skill
7. Result returns as an observation into trace / cognition / downstream steps

### Example
User asks: “What is wrong with Orion runtime right now?”
- Verb chosen: `assess_runtime_state`
- Preferred skill families: `system_inspection`, `runtime_health`
- Candidate skills: `docker_status`, `gpu_snapshot`, `landing_pad_metrics`
- Skill selected: `docker_status`
- Observation returned to cognition for summarization / evaluation

---

## Capability families
Skills should be grouped into stable **families**. Verbs should normally point to families, not directly to individual skills.

Suggested families:
- `system_inspection`
- `runtime_health`
- `state_capture`
- `notification`
- `scheduling`
- `landing_pad`
- `biometrics`
- `temporal_context`
- `environmental_state`

This gives Orion a stable abstraction layer even as the individual skills evolve.

---

## Safety and autonomy model

### Skill risk classes
Every skill should be assigned a risk class.

#### Read-only observation
Examples:
- `docker_status`
- `gpu_snapshot`
- `timezone_time`
- `landing_pad_metrics`

These can typically be auto-selected with minimal policy friction.

#### Benign actuation
Examples:
- `notify`
- low-risk logging
- non-destructive housekeeping

These may be auto-selected under policy, but should still be observable and rate-limited.

#### High-impact actuation
Examples:
- anything that changes infrastructure state
- anything that mutates records or performs irreversible actions
- anything with financial, safety, or privacy impact

These should require stronger gating, confirmation, or explicit policy authorization.

### Rule
Autonomy should scale with risk class.

---

## Inspectability requirements
Every bridge decision should be inspectable.

The system should be able to explain:
- which verb was chosen
- whether a skill was needed
- which family was selected
- which skill was selected
- why it was chosen
- what policy or gating applied
- whether it was executed or stopped

Suggested bridge output shape:

```json
{
  "verb": "assess_runtime_state",
  "needs_skill": true,
  "skill_family": "system_inspection",
  "candidate_skills": ["docker_status", "gpu_snapshot"],
  "selected_skill": "docker_status",
  "reason": "The request is about live runtime state, so a read-only inspection skill is appropriate.",
  "confidence": 0.84,
  "policy": {
    "risk_class": "read_only",
    "confirmation_required": false
  }
}
```

The exact schema can change. The inspectability requirement should not.

---

## Non-goals
This bridge is **not** intended to:
- replace Cortex
- become a second planner
- independently decide strategic goals without verb-level context
- hide execution choices from trace
- directly collapse tool metadata and execution metadata into one flat selection system

If the bridge starts performing general reasoning or managing long multi-step plans on its own, it is drifting beyond its intended role.

---

## Design principles for future Codex runs

### Principle 1: keep registries separate
Maintain separate registries for:
- verb metadata
- skill metadata
- policy / risk metadata

### Principle 2: let verbs target families, not concrete skills
Prefer `preferred_skill_families` over hard-coded skill IDs unless there is a strong reason not to.

### Principle 3: keep the bridge deterministic where possible
The bridge should prefer constrained, inspectable selection over broad open-ended reasoning.

### Principle 4: do not hide side effects
Any invocation of `orion-actions` should be visible in trace and legible to Orion operators.

### Principle 5: do not let the planner directly compete with low-level skills
Planner-level cognition should remain at the verb layer.

### Principle 6: capability selection is a resolver, not a brain
If a component starts looking like a second cortex, stop and simplify.

---

## Recommended manifest shapes

### Suggested verb metadata extensions
A verb should be able to declare:
- `side_effect_level`
- `requires_capability_selector`
- `preferred_skill_families`
- `execution_mode` (`reasoning_only`, `capability_backed`, `mixed`)

Example:

```yaml
name: assess_runtime_state
label: Assess Runtime State
description: Assess Orion runtime and infrastructure state using current live signals.
category: ExecutiveControl
side_effect_level: read_only
requires_capability_selector: true
execution_mode: capability_backed
preferred_skill_families:
  - system_inspection
  - runtime_health
```

### Suggested skill manifest shape
A skill should be able to declare:
- `skill_id`
- `description`
- `family`
- `read_only`
- `idempotent`
- `requires_confirmation`
- `input_schema`
- `output_schema`
- `risk_class`

Example:

```yaml
skill_id: docker_status
description: Return current Docker container status and health for runtime services.
family: system_inspection
read_only: true
idempotent: true
requires_confirmation: false
risk_class: read_only
```

---

## Suggested bridge service responsibilities
A future `capability_selector` / `actions_bridge` component should:
1. load and normalize skill manifests
2. accept a selected verb + context
3. shortlist compatible skill families
4. shortlist candidate skills
5. choose one skill or return ranked candidates
6. validate policy and risk
7. return a structured decision
8. never become a hidden planner

---

## Practical guidance for near-term implementation

### Good first milestone
- define a stable skill manifest schema for `orion-actions`
- extend relevant verbs with `preferred_skill_families`
- implement a small bridge that maps chosen verb -> candidate skill families -> selected skill
- keep read-only skills auto-selectable
- keep higher-risk skills gated

### Good second milestone
- normalize action results into observation envelopes that verbs can consume cleanly
- include bridge decisions in cognition trace and audit logs
- introduce policy-based confirmation for higher-impact skills

### Good third milestone
- teach ADK-style routing to read both verb descriptions and skill descriptions, but in two passes:
  - pass 1 chooses verb
  - pass 2 chooses skill

---

## Final guidance
When in doubt, preserve the separation:
- **Cortex** decides what kind of cognition is needed
- **Bridge** resolves that cognition into capability choices
- **Actions** executes the concrete skill

That separation is what keeps Orion inspectable, extensible, and architecturally honest.

