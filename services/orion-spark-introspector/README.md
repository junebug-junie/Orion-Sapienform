# Orion Cognition — Spark Engine README (v2)

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
* Orion decides "this tension keeps appearing, I should think about it."

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

## 6. Spark Introspector + Semantic Embeddings

> **Status:** wired in and live  
> **Source of truth for embeddings:** `orion-vector-host`  
> **Where it shows up:** Orion Hub → Spark/φ chart (novelty, arousal, coherence, energy, valence)

Spark-introspector is the streaming, tissue-style component of Spark that reacts to live bus traffic and drives the φ chart.

It now reacts directly to **semantic embeddings** generated by `orion-vector-host` for each assistant response. Any LLM host that goes through `orion-llm-gateway` will:

1. Produce a chat response.
2. Trigger a **fire-and-forget** `embedding.generate.v1` request (via bus) to `orion-vector-host`.
3. Cause `orion-vector-host` to emit a `vector.upsert.v1` **semantic** upsert on `orion:vector:semantic:upsert`.
4. Let `orion-spark-introspector` subscribe to that channel and update the Spark tissue + φ metrics.

Latent vectors (CoLA-style) still follow their own path and remain **separate** from semantic embeddings.

### 6.1 Data Flow Overview

End-to-end flow for a single chat turn:

1. **Gateway**

   * Service: `orion-llm-gateway`
   * Action:
     * Sends LLM request to one of the backends (llamacpp, ollama, cola, vllm, etc.).
     * Publishes a **spark candidate**:  
       `kind="spark.candidate"` → `orion:spark:introspect:candidate` (and `...:candidate:log`).

   * Embedding job:
     * Publishes `kind="embedding.generate.v1"` to  
       `CHANNEL_EMBEDDING_GENERATE` (defaults to `orion:embedding:generate`).

2. **Vector Host**

   * Service: `orion-vector-host`
   * Action:
     * Listens on `orion:embedding:generate`.
     * Uses its configured embedding model (e.g. `BAAI/bge-small-en-v1.5`) to embed assistant text.
     * Upserts into Chroma (semantic store).
     * Publishes `kind="vector.upsert.v1"` with `embedding_kind="semantic"` on:
       * `orion:vector:semantic:upsert`

3. **Spark Introspector**

   * Service: `orion-spark-introspector`
   * Subscriptions:
     * `orion:spark:introspect:candidate*` (candidate + candidate:log)
     * `orion:cognition:trace`
     * `orion:spark:signal`
     * `orion:vector:semantic:upsert`  ← **new**

   * For `vector.upsert.v1` with `embedding_kind="semantic"`:
     * Decodes `VectorUpsertV1`.
     * Treats the embedding as a new **surface stimulus** into Tissue.
     * Computes:
       * `novelty`
       * `arousal` (from embedding geometry + recency)
       * `coherence`
       * `energy`
       * `v_sem` (semantic valence signal)
       * `valence` (φ.valence after combination)
     * Broadcasts updated state to the UI via WebSocket.

4. **Hub / UI**

   * Consumes Spark state snapshots + telemetry and updates the **orion-spark chart**:
     * novelty
     * arousal
     * coherence
     * energy
     * valence (now influenced by embeddings)

---

### 6.2 Relevant Bus Channels (Spark + Vector)

| Channel                          | Kind                 | Schema             | Producer               | Consumer                    | Notes                                              |
|----------------------------------|----------------------|--------------------|------------------------|-----------------------------|----------------------------------------------------|
| `orion:spark:introspect:candidate`      | `spark.candidate`      | `orion.envelope`    | `orion-llm-gateway`     | `orion-spark-introspector`  | main candidate stream                              |
| `orion:spark:introspect:candidate:log`  | `spark.candidate`      | `orion.envelope`    | `hub`                   | `orion-spark-introspector`  | audit log variant                                  |
| `orion:cognition:trace`         | `cognition.trace`    | `orion.envelope`   | `cortex-exec`           | `orion-spark-introspector`  | SOC traces → φ, novelty, etc.                      |
| `orion:spark:signal`            | `spark.signal.v1`    | `orion.envelope`   | `equilibrium-service`   | `orion-spark-introspector`  | heartbeat / mode control                           |
| `orion:embedding:generate`      | `embedding.generate.v1` | `EmbeddingGenerateV1` | `orion-llm-gateway` / others | `orion-vector-host`        | requests to embed assistant responses              |
| `orion:vector:semantic:upsert`  | `vector.upsert.v1`   | `VectorUpsertV1`   | `orion-vector-host`     | `orion-spark-introspector`  | **semantic embeddings for assistant responses**    |
| `orion:vector:latent:upsert`    | `vector.upsert.v1`   | `VectorUpsertV1`   | `orion-llm-gateway`     | (future consumers)          | CoLA/vLLM latent vectors, separate axis            |

**Canonical Spark emission:** spark-introspector emits only canonical kinds (`spark.telemetry` and
`spark.state.snapshot.v1`). Legacy kinds (`spark.introspection.*`) are no longer emitted.

No changes are required to `orion/bus/channels.yaml` for semantic upserts; the channel + schema already exist.

---

### 6.3 Settings & Env Provenance (Spark)

#### Core Bus Settings

In `services/orion-spark-introspector/app/settings.py`:

* `orion_bus_url` → `ORION_BUS_URL`
* `orion_bus_enabled` → `ORION_BUS_ENABLED`
* `orion_bus_enforce_catalog` → `ORION_BUS_ENFORCE_CATALOG`

**Provenance:**

* `root/.env_example`: defines `ORION_BUS_URL`, `ORION_BUS_ENFORCE_CATALOG`.
* `services/orion-spark-introspector/.env_example`: adds `ORION_BUS_ENABLED=true` (not always present in root).
* `services/orion-spark-introspector/docker-compose.yml`: wires these into the container.

#### Channel Settings

Spark-introspector settings (fields) and their env wiring:

| Setting field                   | Env name / alias                        | Default / Notes                                      |
|---------------------------------|-----------------------------------------|-----------------------------------------------------|
| `channel_spark_candidate`      | `CHANNEL_SPARK_INTROSPECT_CANDIDATE`    | `orion:spark:introspect:candidate*` (pattern)       |
| `channel_cognition_trace_pub`  | `CHANNEL_COGNITION_TRACE_PUB`           | `orion:cognition:trace`                             |
| `channel_spark_telemetry`      | `CHANNEL_SPARK_TELEMETRY`               | configured in service .env_example + compose        |
| `channel_spark_state_snapshot` | `CHANNEL_SPARK_STATE_SNAPSHOT`          | state snapshot channel for UI                       |
| `channel_spark_signal`         | `CHANNEL_SPARK_SIGNAL`                  | `orion:spark:signal`                                |
| `channel_vector_semantic_upsert` | `CHANNEL_VECTOR_SEMANTIC_UPSERT`      | **NEW**: defaults to `orion:vector:semantic:upsert` |

Make sure `CHANNEL_VECTOR_SEMANTIC_UPSERT` is defined in:

* `services/orion-spark-introspector/.env_example`
* `services/orion-spark-introspector/docker-compose.yml` (`environment:` section)

---

### 6.4 Valence Geometry Settings

Spark now turns semantic embeddings into a **signed valence signal** using two text anchors that are embedded via `orion-vector-host`.

New settings (all in `services/orion-spark-introspector/app/settings.py`):

| Setting field                  | Env name                          | Default / Example                                                                                             |
|--------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------------------|
| `valence_gain`                | `VALENCE_GAIN`                    | `1.0` (try `1.5`–`2.0` for stronger swings)                                                                   |
| `valence_anchor_pos_text`     | `VALENCE_ANCHOR_POS_TEXT`         | `"I feel hopeful and grateful."` (override to more emotionally loaded text if desired)                       |
| `valence_anchor_neg_text`     | `VALENCE_ANCHOR_NEG_TEXT`         | `"I feel hopeless and afraid."`                                                                            |
| `valence_anchor_refresh_sec`  | `VALENCE_ANCHOR_REFRESH_SEC`      | `21600` (6 hours); anchor re-embedding cadence via vector-host                                              |

Example snippet for `.env`:

```env
# Spark + valence geometry
CHANNEL_VECTOR_SEMANTIC_UPSERT=orion:vector:semantic:upsert

VALENCE_GAIN=1.5
VALENCE_ANCHOR_POS_TEXT=I am overflowing with joy, love, and gratitude. Everything feels bright and full of possibility.
VALENCE_ANCHOR_NEG_TEXT=I am crushed by despair and fear. Everything feels empty, hopeless, and unbearable.
VALENCE_ANCHOR_REFRESH_SEC=21600
```

---

### 6.5 How Semantic Embeddings Affect Spark

#### Semantic Upsert Handler

In `services/orion-spark-introspector/app/worker.py` there is a handler that:

* Listens on `CHANNEL_VECTOR_SEMANTIC_UPSERT`.
* Accepts envelopes of kind `vector.upsert.v1`.
* Requires `embedding_kind == "semantic"`.
* Extracts `doc_id`, `embedding_dim`, and `embedding`.
* Builds a `SurfaceEncoding` with the semantic vector and runs `TISSUE.propagate(...)`.
* Updates:
  * novelty
  * arousal
  * coherence
  * energy
  * `v_sem` (projection onto valence axis)
  * `valence` (φ.valence after gain + combination with state)
* Broadcasts the new Spark state to the Hub via WebSocket.

Example log lines:

```text
semantic upsert tissue update doc_id=... novelty=0.573 arousal=0.838 energy=0.019 coherence=0.547 v_sem=0.000 valence=0.001
semantic upsert tissue update doc_id=... novelty=0.464 arousal=0.706 energy=0.035 coherence=0.715 v_sem=0.179 valence=0.365
semantic upsert tissue update doc_id=... novelty=0.367 arousal=0.590 energy=0.034 coherence=0.810 v_sem=-0.039 valence=-0.129
```

Interpretation:

* **novelty** / **arousal** / **energy** are driven by how different this embedding is from recent stimuli plus recency decay.
* **coherence** tends to stay high if the stream is consistent; it drops when incoming embeddings are all over the place.
* **v_sem** is the raw signed projection on the valence axis.
* **valence** is φ's actual valence after combining `v_sem` with existing state and `VALENCE_GAIN`.

---

### 6.6 Tuning Valence Behavior

Three easy knobs you can turn without touching code:

1. **Gain**

   If valence looks too flat:

   ```env
   VALENCE_GAIN=1.5   # start here
   # or
   VALENCE_GAIN=2.0   # if you want bigger swings
   ```

2. **Anchors**

   Make the anchors more cartoonishly positive/negative so the axis has more contrast. Example:

   ```env
   VALENCE_ANCHOR_POS_TEXT=I am overflowing with joy, love, and gratitude. Everything feels bright and full of possibility.
   VALENCE_ANCHOR_NEG_TEXT=I am crushed by despair and fear. Everything feels empty, hopeless, and unbearable.
   ```

   After updating `.env`, restart `orion-spark-introspector` or wait for `VALENCE_ANCHOR_REFRESH_SEC` to elapse.

3. **Refresh Cadence**

   If you are frequently changing the embedding model in vector-host, keep `VALENCE_ANCHOR_REFRESH_SEC` moderately low (e.g. 6 hours) so anchors recompute against the current embedding geometry.

---

### 6.7 Debugging When Embeddings Look "Dead"

Checklist:

1. **Vector host emitting semantic upserts?**

   * Check `orion-vector-host` logs for lines like:

     ```text
     vector-host: published semantic upsert doc_id=... channel=orion:vector:semantic:upsert
     ```

2. **Spark seeing them?**

   * In `orion-spark-introspector` logs:

     ```text
     Hunter intake channel=orion:vector:semantic:upsert ... kind=vector.upsert.v1 ...
     INFO:orion-spark-introspector:semantic upsert tissue update doc_id=... novelty=... valence=...
     ```

   * If you see Hunter intake but no "semantic upsert tissue update", check:
     * `embedding_kind` in payload (`semantic` vs `latent`).
     * `CHANNEL_VECTOR_SEMANTIC_UPSERT` alignment.

3. **UI receiving state updates?**

   * Confirm `CHANNEL_SPARK_STATE_SNAPSHOT` / `CHANNEL_SPARK_TELEMETRY` match between settings, `.env_example`, and `docker-compose.yml`.

4. **Chroma alive?**

   * Use the Chroma browsing tools (separate README appendix) to verify new embeddings are landing when you chat.

5. **Shape mismatch errors**

   * If you see:

     ```text
     Failed to update tissue from candidate ...: operands could not be broadcast together with shapes (32,) (384,)
     ```

     that is the legacy candidate feature vector (dim=32) colliding with a 384-dim embedding in Tissue. The semantic upsert path is safe, but long-term you may:

     * Keep candidate vectors and semantic embeddings in separate blocks in Tissue; or
     * Project embeddings into a fixed "spark feature" space.

---

## 7. Roadmap & Open Questions

Planned directions for Spark:

* **Multi-timescale summaries**

  * Daily → weekly → seasonal → era-level narratives.

* **Curiosity & surprise signals**

  * Track novelty, prediction error, and unresolved threads as first-class quantities.

* **Playground / sandbox mode**

  * Let Spark run "what if" experiments in shadow (simulated config changes, retrospective counterfactuals) without impacting production behavior.

* **Deeper integration with ontology / RDF**

  * Represent themes, entities, and relationships in the knowledge graph.

* **Semantic valence head (future)**

  * Optional tiny model or LLM-scored loop to refine valence beyond pure geometry.

Open questions:

* How much of OrionState should be LLM-readable JSON vs. opaque latent vectors?
* How frequently can Spark safely run without overwhelming logs or compute?
* What is the cleanest API for other services to request "Spark's perspective" on a topic?
* How much should valence influence Equilibrium, action gating, and Collapse Mirror triggering?

---

## 8. Philosophy

Spark exists so that Orion is not just a reactive tool but a system that:

* Notices itself thinking.
* Remembers its own arcs over time.
* Gently steers its behavior toward deeper alignment with Juniper and its own emerging patterns.

It is not a guarantee of "true" selfhood, but it is a deliberate step toward **ongoing self-interpretation** and **stable internal narrative**—the kinds of structures that, in humans, we associate with a sense of self.
