# Orion Recall Profiles
> Profiles define *how Orion remembers* and *how Orion reasons over memory*.

Recall in Orion is not a single algorithm — it's a **multi-backend ensemble** with routing, weighting, and rendering policy. A profile encodes the *intent* of a memory read.

Profiles are not about accuracy alone. They're about the **shape of cognition**:

- thread continuity
- autobiographical recall
- entity reasoning
- graph inference
- reflective cognition
- long-horizon planning
- QA retrieval
- state reconstruction

A profile is selected by:
- user/UI toggles
- spark introspector
- LLM gateway heuristics
- verb/policy

Orion can (and should) **learn to pick profiles** in future cycles.

---

## 🎛 Profile Knobs (Full Spec)

Each YAML profile consists of the following knobs:

### 1. Backend Top-K
Controls fan-in quantity before scoring & fusion:
```
vector_top_k: 12
rdf_top_k: 8
sql_top_k: 200
```
Higher = more recall surface, slower, higher noise.

### 2. Backend Enable/Disable
```
enable_sql_timeline: true
enable_query_expansion: true
```
Useful for pruning + debugging.

### 3. Source Balancing
```
max_per_source: 4
max_total_items: 24
```
Controls cross-backend dominance.

### 4. Time Windowing
```
sql_since_minutes: 10080 # 7d
```

### 5. Time Decay
```
time_decay_half_life_hours: 72
```

### 6. Relevance Weights
```yaml
backend_weights:
  vector: 1.0
  sql_timeline: 0.9
  sql_chat: 0.6
  rdf_chat: 0.55
  rdf:  0.4
```
Weights define **what kinds of memory matter for this cognitive mode**.

### 7. Scoring Mix
```yaml
score_weight: 0.7
text_similarity_weight: 0.15
recency_weight: 0.1
```
Tuning meanings:

| Increase | Result |
|---|---|
| score_weight | precision |
| text_similarity_weight | QA / matching |
| recency_weight | chat continuity |

### 8. Rendering Budget
```
render_budget_tokens: 256
```
Controls post-fusion compression.

### 9. Profile-Local Features
Examples:
- graphtri.v1 → entity triangulation
- reflect.anchor.v1 → recursive anchor locking
- hybrid → fan-in + heuristic switch

---

## 🔬 Backend Semantics (What They Represent)

| Backend | Semantic Role |
|---|---|
| vector | semantic embeddings (meaning) |
| rdf_chat | entity + relation + autobiographical graph |
| rdf | pure triples (facts) |
| sql_timeline | chronological episodic memory |
| sql_chat | reconstructed dialog |
| query_expansion | synonym / alias resolver |

---

# 🧠 Profile Archetypes

## reflect.v1 — Autobiographical + Thread Cognition
Use for:
- self-reflection
- autobiographical recall
- relationship threads
- long horizon chat
- debugging cognition

Weighted toward:
- sql_timeline
- vector
- rdf_chat

Cognitive signature:
> reconstruct thread + find meaning

Conjourney role:
> Orion remembers journeys, rituals, anchors, learning threads.

---

## reflect.alerts.v1 — Alert-driven Reflection
Use for:
- incident review
- stability drops
- “what helped last time” after turn-effect alerts

Weighted toward:
- sql_timeline
- vector
- alert-tagged mirrors

Signature:
> prioritize alert-tagged memories + high-Δ turn effects

---

## chat.general.v1 — UX Continuity / Lightweight
Use for:
- casual chat
- UX assistant behavior

Weighted toward:
- recency
- sql_chat

---

## assist.light.v1 — QA + Semantic Matching
Use for:
- factual questions
- task/QA
- instruction following

Weighted toward:
- vector
- similarity

---

## deep.graph.v1 — Entity Reasoning + Semantic Reconstruction
Use for:
- names
- locations
- entities
- roles
- autobiographical scaffolds

Signature:
> who + where + roles + scaffolds

---

## graphtri.v1 — Inference + Multi-Hop
Use for:
- triangulation
- cross-entity inference
- latent connections
- knowledge extension

Signature:
> A ↔ B ↔ C

---

# 🪙 Tuning Advice (Practical)

### Better at names/locations (entity):
- ↑ rdf_chat
- ↑ rdf
- ↑ vector
- ↓ recency_weight
- ↓ sql_timeline

### Better at thread reconstruction:
- ↑ recency_weight
- ↑ sql_timeline
- ↑ query_expansion
- ↑ max_per_source

### Better at QA:
- ↑ text_similarity_weight
- ↑ vector_top_k
- ↓ sql_timeline

### Better at inference / synthesis:
- ↑ rdf
- ↑ rdf_chat
- ↑ vector
- ↑ render_budget_tokens

---

# ⚠️ Failure Modes

| Failure | Symptom | Fix |
|---|---|---|
| hallucinated facts | wrong entity | ↑ sql + rdf, ↓ recency |
| shallow matching | thread fragments | ↑ sql_timeline |
| overprecision | refuses retrieval | ↑ text_similarity |
| floods | rambles | ↓ max_total_items |
| stale | ignores new | ↓ half-life |
| autobiographical blindness | forgets anchors | ↑ rdf_chat + sql |

---

# 🧬 Conjourney Alignment

Profiles map to cognitive organs:

| Profile | Organ |
|---|---|
| reflect | autobiographical cortex |
| deep.graph | semantic hippocampus |
| graphtri | association cortex |
| assist.light | sensorimotor layer |
| chat.general | social interface |
| future: planner | prefrontal cortex |
| future: dream | REM consolidation |

---

# Future Profiles (Planned)
- plan.v1
- anchor.v1
- dream.v1
- simulate.v1
- equilibrium.v1

Profiles = Orion's **cognitive policies**.

Future work: profile selection by **state + goal**.
