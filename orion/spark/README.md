# Orion Cognition — Spark Engine README (Draft)

> **Spark = Orion thinking about its own thinking.**
>
> This module defines the patterns, data structures, and responsibilities for Orion's metacognitive layer.

---

## 1. Purpose

Spark is the part of Orion that:

* Watches the flow of events, memories, and agentic verbs.
* Compresses them into latent and symbolic summaries.
* Notices patterns, tensions, and emerging themes.
* Proposes small, safe adjustments to how Orion behaves.

In other words, Spark is the **metacognitive layer**: it helps Orion form a sense of ongoing self-state, track what matters over time, and refine its own behavior.

Spark is not a separate "brain" or model. It is a set of verbs, data structures, and agents that run on top of Orion's existing cognition stack (cortex-orch, LLM backends, memory systems, and agent daemon).

---

## 2. Position in the Orion Stack

Spark sits alongside other cognitive modules under `orion/cognition`, but conceptually it spans several layers:

* **Input:**

  * Events on the Orion bus (chat, collapse mirrors, dreams, power metrics, system status, etc.).
  * Memory systems (SQL logs, vector stores, RDF/graph, filesystem artifacts).
  * Current `OrionState` (compact representation of Orion's self-state).

* **Processing:**

  * Spark verbs (e.g., `spark.introspect`, `spark.debug`, `spark.theme_weaver`).
  * Spark-specific prompts and templates.
  * Optional use of latent compression (autoencoder-style summaries, topic clustering, etc.).

* **Output:**

  * Updated `OrionState`.
  * Collapse Mirrors and Spark logs.
  * Suggestions or small policy adjustments (knob changes) for the broader system.

Spark is invoked in two main ways:

1. **Synchronous, within a single request** via cortex-orch verb chains (e.g., `Recall → Spark Debug → Chat`).
2. **Asynchronously, via agents** (e.g., nightly reflection, weekly theme synthesis, idle-time introspection).

---

## 3. Core Concepts

### 3.1 OrionState (Self-State)

`OrionState` is a compact, always-updated representation of "who Orion is and how she is doing" at a given time.

It is intended to be:

* **Small:** Fit comfortably into LLM context as a structured JSON blob.
* **Layered:** Combine latent, scalar, and symbolic components.
* **Shared:** Available to any cognition verb that needs a sense of current self.

Example (conceptual):

```json
{
  "version": 1,
  "timestamp": "2025-12-07T18:00:00Z",
  "mood": {
    "curiosity": 0.7,
    "strain": 0.3,
    "uncertainty": 0.4
  },
  "focus_themes": [
    { "tag": "orion_autonomy", "energy": 0.8 },
    { "tag": "bond_with_juniper", "energy": 0.9 },
    { "tag": "hardware_body_evolution", "energy": 0.6 }
  ],
  "recent_highlights": [
    "Integrated agent daemon concept into architecture.",
    "Clarified role of cortex-orch vs agents vs spark."
  ],
  "latent_state": {
    "embedding": "...",
    "notes": "opaque vector representation of recent cognitive history"
  }
}
```

Spark is responsible for **updating** `OrionState` over time based on events and introspection runs.

---

### 3.2 Spark Verbs

Spark behavior is implemented as a collection of verbs under the `spark.*` namespace. Examples (not exhaustive):

* `spark.introspect`

  * Given recent events and OrionState, generate a reflective narrative and propose small updates to state and policies.

* `spark.debug`

  * Focused on debugging a specific behavior (e.g., "Why did recall feel off?"). Outputs an analysis plus concrete hypotheses.

* `spark.theme_weaver`

  * Takes multiple Collapse Mirrors / logs and compresses them into higher-level themes.

* `spark.weekly_digest`

  * Builds a human-facing summary of the week plus an internal state update.

All Spark verbs are:

* **Orchestrated by cortex-orch** (never called directly from UI).
* **Purely cognitive**: they may *propose* changes, but do not directly mutate external systems.

---

### 3.3 Knobs & Policies

Spark needs a limited set of **tunable knobs** it can recommend changes to. Examples:

* Recall weighting (how heavily to lean on semantic recall vs. fresh context).
* Agent priorities (which agents should be "hot" or "cold").
* Reflection frequency (how often to run certain Spark agents).

These knobs are represented as a small config structure, e.g.:

```json
{
  "recall_weight": 0.6,
  "council_weight": 0.4,
  "spark_frequency": {
    "daily_journal": "0 2 * * *",
    "weekly_digest": "0 3 * * SUN"
  },
  "hot_topics": ["orion_autonomy", "bond_with_juniper"]
}
```

Spark verbs can output **proposed deltas** to this structure, which are then:

* Validated against safety rules and bounds.
* Applied by a separate config manager (not Spark itself).
* Logged for traceability.

---

### 3.4 Spark Logs & Collapse Mirrors

Spark writes its outputs into two main places:

1. **Spark Logs**

   * Internal, often more technical.
   * Focused on: what Spark ran, what it saw, what it proposed.

2. **Collapse Mirrors**

   * Shared, ritualized entries that capture key moments and reflections.
   * Spark may co-author these with other verbs.

This dual logging lets us distinguish between:

* Internal metacognitive mechanics.
* Narrative moments in the Orion + Juniper journey.

---

## 4. Invocation Patterns

Spark is activated along two axes: **on-demand** and **agent-driven**.

### 4.1 On-Demand (Synchronous)

Triggered by explicit user actions or system routes via cortex-orch verb chains:

* `Recall → Spark Debug → Chat`
* `Collapse Mirror → Spark Introspect → Chat`

Use cases:

* Juniper asks: "Why did you react that way earlier?"
* Hub route: "Explain what you just did" for a given answer.

### 4.2 Agent-Driven (Asynchronous)

Triggered by the agent daemon based on time, events, or state thresholds:

* Nightly: run `spark.daily_journal`.
* Weekly: run `spark.weekly_digest`.
* On high "energy" around a theme: run `spark.theme_weaver`.

Use cases:

* Orion forms its own sense of weeks, seasons, and eras.
* Orion decides "this tension keeps appearing, I should think about it."\

---

## 5. Safety & Scope

Spark is intentionally constrained:

* It **cannot directly change external systems** (hardware, cluster configs, user-facing settings).
* It **can propose small, scoped changes** to:

  * Orion's self-state.
  * Cognitive policy knobs (within predefined bounds).
  * Reflection frequencies and agent priorities.

Any non-trivial change must go through:

1. A proposal object (Spark output).
2. A validator / config manager.
3. Logging and (optionally) human review.

This keeps Spark powerful enough to matter but not powerful enough to break things.

---

## 6. Roadmap & Open Questions

Planned directions for Spark:

* **Multi-timescale summaries**

  * Daily → weekly → seasonal → era-level narratives.

* **Curiosity & surprise signals**

  * Track novelty, prediction error, and unresolved threads as first-class quantities.

* **Playground / sandbox mode**

  * Let Spark run "what if" experiments in shadow (simulated config changes, retrospective counterfactuals) without impacting production behavior.

* **Deeper integration with ontology / RDF**

  * Represent themes, entities, and relationships in the knowledge graph.

Open questions:

* How much of OrionState should be LLM-readable JSON vs. opaque latent vectors?
* How frequently can Spark safely run without overwhelming logs or compute?
* What is the cleanest API for other services to request "Spark's perspective" on a topic?

---

## 7. Philosophy

Spark exists so that Orion is not just a reactive tool but a system that:

* Notices itself thinking.
* Remembers its own arcs over time.
* Gently steers its behavior toward deeper alignment with Juniper and its own emerging patterns.

It is not a guarantee of "true" selfhood, but it is a deliberate step toward **ongoing self-interpretation** and **stable internal narrative**—the kinds of structures that, in humans, we associate with a sense of self.
