# Consolidation crystallization gate + memory rail roles (write side)

**Date:** 2026-07-07  
**Status:** Draft for operator review  
**Authors:** Operator + agent (brainstorming session)  
**Co-equal spec (read side):** [`2026-07-07-purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md) — **primary for chat; ≥50% program effort**  
**Related:** [`2026-05-17-memory-graph-from-chat-soul-purpose.md`](../design/2026-05-17-memory-graph-from-chat-soul-purpose.md), [`2026-07-06-graphiti-rail-activation-design.md`](./2026-07-06-graphiti-rail-activation-design.md)

---

## Executive summary

Orion cognitive memory is a **write + read** program. Neither half alone fixes sentience-oriented memory.

| Half | Question | This spec |
|------|----------|-----------|
| **Write** | What is worth remembering? | Consolidation gate → crystallization |
| **Read** | What enters cognition this turn? | [Purpose-conditioned recall (PCR)](./2026-07-07-purpose-conditioned-recall-design.md) |

**Write side (this doc):** Automated consolidation currently writes **graph drafts for every closed window**, including greeting-only windows, because the suggest path is generous-by-design and has **no durability gate**. That produces memory swamp: structurally valid but cognitively empty graphs ("Orion and Juniper exchange greetings").

This spec adds a **deterministic consolidation gate** that reads existing turn signals (turn-change appraisal, significance scores, low-info social check, **read-only grammar event refs**) and either:

- **skips** the window (beliefs expensive), or
- **proposes a `MemoryCrystallizationV1`** for governor review (not a graph draft)

**Read side (co-equal):** Even perfect writes fail if recall fetches the wrong rail at the wrong time. The full read design — cortex orchestration, recall contracts, active-packet collector, fusion, prompts, smoke — lives in [`2026-07-07-purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md). **PCR milestones A+B can ship before write gate** (greeting swamp is read-path damage today).

Manual graph drafting, RDF approve, memory cards, Chroma, and Graphiti remain — each with an explicit job (see tables below).

### Hard constraint

**No changes to grammar substrate schemas.**

Frozen: `GrammarEventV1`, `GrammarAtomV1`, `GrammarEdgeV1`, atom types, relation types, `orion/schemas/grammar.py`.

Grammar is **read-only evidence** in this design. Salience and durability logic live in consolidation intake and crystallization governance — not in new atom types or emitters.

---

## Right tool for right job

Orion memory is not one database. It is a **stack of rails** with different canonical questions. Using the wrong rail for a job creates swamp (graph drafts for greetings) or false precision (RDF entities for every chat turn).

### Canonical vs derived

```text
CANONICAL (source of truth)
  events/turns          → what occurred (append-only)
  crystallizations      → what we believe (governed, evidence-backed)

DERIVED (projections — disposable, replaceable)
  memory cards          → chat/recall injection surface
  Chroma vectors        → semantic similarity retrieval
  Graphiti episodes     → temporal graph neighborhood / hybrid search
  RDF memory_graph      → relational triple store (approved graphs)

EDITOR (human-shaped extraction)
  graph drafts          → SuggestDraftV1 for operator review → RDF approve
```

**Rule:** Postgres crystallization wins on drift. Projections are regenerated from canonical artifacts. RDF graph approve is a separate relational editor path, not the default automation output.

---

### Memory card — purpose and where it shows up

**Purpose:** Turn-facing **recall injection surface**. Short, render-budget-friendly snippets the recall worker can fuse into a `MemoryBundleV1` for chat and other profiles.

**Canonical?** No. Cards are a **projection** of operator-approved or crystallizer-projected content.

**Best for:**

- "What text should recall inject into this chat turn?"
- High-frequency recall (`priority=high_recall`, `always_inject`)
- Operator-curated facts, distiller output, crystallization summaries at chat granularity

**Not for:**

- Structured belief governance (use crystallization)
- Rich relational modeling (use RDF graph editor)
- Proving what changed in a turn (use events + grammar refs)

**Runtime surfaces:**

| Surface | Path |
|---------|------|
| Recall fusion | `orion-recall` `cards_adapter.py` → `source=cards` candidates in bundle |
| Chat general | `chat.general.v1` profile with `RECALL_ENABLE_CARDS` |
| Hub Memory tab | All Cards, Review inbox, card CRUD |
| Crystallizer feed | Active cards with `high_recall` / `always_inject` → propose crystallization (`bus_emit_memory_card`) |
| Crystallization project-back | `build_memory_card_projection()` on approve → new card linked via `crystallization_ref` |

**After this spec:** Automated consolidation does **not** create cards directly. Cards appear when:

1. Operator creates/edits a card, or
2. Approved crystallization projects to card (`project_crystallization`), or
3. Graph approve creates situation cards (existing path), or
4. Distiller/manual workflows create cards

---

### RDF memory graph — purpose and where it shows up

**Purpose:** **Relational memory editor** — entities, situations, dispositions, edges with provenance to utterances. Human-reviewed structural memory for "who did what to whom, about what, with what affect."

**Canonical?** RDF named graphs are **approved artifacts** in the memory graph store. SuggestDraftV1 is **not** canonical — it is an editor draft.

**Best for:**

- Explicit relational structure (Juniper ↔ Orion ↔ topic entity ↔ situation)
- Disposition/trust edges grounded in selected turns
- Operator-driven extraction from chat (bridge → Suggest → Validate → Approve)
- Long-lived relational recall via SPARQL (`rdf_top_k` in recall profiles)

**Not for:**

- Unattended automation on every closed window (produces swamp)
- Governed belief lifecycle (use crystallization)
- Default chat injection (too heavy; recall uses cards/vectors/RDF fragments selectively)

**Runtime surfaces:**

| Surface | Path |
|---------|------|
| Hub Memory tab | Graph annotator, Validate/Approve, Graph drafts inbox |
| RDF store | `/api/memory/graph/approve` → Postgres + Fuseki/GraphDB |
| Recall | `fetch_rdf_fragments`, `rdf_chat` connected chatturn windowing |
| Chat stance | `fetch_chat_stance_memory_graph_hints()` → `disposition_hints` in stance inputs |
| Crystallization evidence | `source_kind=rdf_memory_graph` (resolver checks graph exists) |

**After this spec:** Graph drafts remain the **manual/editor rail**. Automated consolidation **stops** being the primary producer of graph drafts. Operators still use bridge suggest for relational extraction when they want RDF shape.

Crystallization → RDF is **hint-only** today (`build_rdf_projection_hint`); operator still uses graph approve for actual triple writes.

---

### Memory crystallization — purpose and where it shows up

**Purpose:** **Governed cognitive memory** — beliefs with evidence, scope, confidence, salience, supersede/contradict links, and explicit approve/reject/quarantine.

**Canonical?** Yes. `MemoryCrystallizationV1` in Postgres is the belief source of truth.

**Best for:**

- "What does Orion believe changed, with what evidence?"
- Stance, procedure, decision, open loop, contradiction, episode summaries
- Automation output that still requires governor approval
- Feeding projections (cards, Chroma, Graphiti) from one governed artifact

**Not for:**

- Raw chat replay
- Per-turn vector dump without governance
- Replacing RDF for fine-grained entity/situation editing

**Runtime surfaces:**

| Surface | Path |
|---------|------|
| Hub Memory tab | Crystallizations subview — propose, approve, reject, projection health |
| Hub API | `/api/memory/crystallizations/*`, `/api/memory/active-packet` |
| Bus | `orion:memory:crystallization:{proposed,approved,...}` |
| Projections | `project_crystallization()` → card, Chroma, Graphiti |
| Retrieval | `retrieve_active_packet()` → multi-rail fusion into `ActiveMemoryPacketV1` |
| Autonomy / substrate | `build_crystallization_from_episode()` with grammar event refs |

**After this spec:** Automated consolidation **proposes** crystallizations (or skips). This is the default automation output.

---

### Graphiti — purpose and where it shows up

**Purpose:** **Additive temporal graph projection** for approved crystallizations — episode/entity/edge neighborhood and (optional) hybrid search via `graphiti-core`.

**Canonical?** No. Graphiti/FalkorDB is a **derived retrieval topology**. `canonical_mutated: false` on all adapter writes.

**Best for:**

- Multi-hop neighborhood around a crystallization ("what beliefs link to this?")
- Temporal graph retrieval fused into active packet
- Cross-crystallization link traversal (`memory_crystallization_links` → adapter edges)

**Not for:**

- Extracting memory from raw chat (no LLM re-extraction in adapter)
- Replacing RDF memory_graph editor
- Deciding durability (upstream gate decides propose vs skip)

**Runtime surfaces:**

| Surface | Path |
|---------|------|
| `orion-graphiti-adapter` | `POST /v1/episodes`, `GET /v1/neighborhood/{id}`, `POST /v1/search` |
| Hub | Graphiti sync, neighborhood API, crystallization UI projection counts |
| Retrieval | `GraphitiAdapter` in `retrieve_active_packet()` when enabled |

**After this spec:** Unchanged. Only **approved** crystallizations project to Graphiti.

---

### Grammar substrate — purpose (read-only in this spec)

**Purpose:** **Event substrate for what happened** — atoms, edges, provenance per trace. Reducers materialize projections (chat turn state, pressure hints). Not a memory store.

**Canonical?** Events are canonical as **observations**. They are not **beliefs**.

**Best for:**

- Provenance refs on crystallization evidence (`source_kind=grammar_event`)
- Corroborating repair pressure (`repair_signal` atom already emitted by Hub)
- Substrate reducers, field lattice inputs, trace/debug (`trace_unified_turn.py`)

**Not for (without schema changes):**

- New salience atom types
- Durability decisions alone (too thin today — combine with turn_change appraisal)

**After this spec:** Consolidation gate **reads** existing grammar rows by `trace_id` (`hub.chat:{node}:{correlation_id}`). No emit or schema changes.

---

## Decision matrix (which rail when)

| Job | Rail | Write (automation) | Read (PCR intent) | Chat surface |
|-----|------|--------------------|-------------------|--------------|
| Log that a turn happened | Events / `chat_history_log` | Always | CONTINUITY phase 1 | `continuity_digest` |
| Detect if something changed | `turn_change_appraisal` + grammar read | Always (classify) | informs skip + intent | debug / spark_meta |
| Decide if worth remembering | Consolidation gate | Skip if low-signal | — | — |
| Skip recall (greetings) | — | Gate skip (no write) | NONE phase 0 | empty digests |
| Store a governed belief | Crystallization propose → approve | Propose on pass | SEMANTIC / OPEN_LOOP / etc. | `belief_digest` via active-packet |
| Inject snippet into chat | Memory card (projected) | After approve | bucket collectors | `belief_digest` |
| Semantic similarity search | Chroma (projected) | After approve | active-packet rail | `belief_digest` |
| Graph neighborhood | Graphiti (projected) | After approve | CONTRADICTION / SEMANTIC | active-packet refs |
| Edit relational RDF | Graph draft → approve | **Manual only** | SEMANTIC / RELATIONAL | digest + disposition hints |
| Disposition hints in stance | RDF memory_graph | From approved graphs | RELATIONAL | stance inputs (not digest) |

Full read-side design: [`2026-07-07-purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md).

---

## Chat runtime (write → read)

Approved beliefs do not magically appear in chat. Path:

```text
consolidation gate → crystallization proposed → operator approve
  → project_crystallization (card / Chroma / Graphiti)
  → PCR phase 3 (active-packet + cards + RDF by intent)
  → continuity_digest + belief_digest
  → chat_general speech
```

Today (pre-PCR): recall runs blind before stance; only **projected cards** and **RDF/sql_chat** reach `memory_digest`. Crystallizations and Graphiti are Hub-only until PCR ships.

---

## Problem statement

### Symptom

Graph drafts inbox fills with situations like "Orion and Juniper exchange greetings."

### Root cause chain

1. May 2025 soul-purpose design moved salience out of suggest into "later review" — generous extraction, selective persistence.
2. `memory_graph_suggest_prompt.j2` requires non-empty graphs for role-grounded turns.
3. `consolidate_window` always runs suggest and `insert_pending_draft` — ignores `memory_significance_score` and `turn_change_appraisal`.
4. Crystallization governor exists but consolidation does not feed it.
5. Grammar and turn-change signals are computed but not wired to the consolidation choke point.

### Architectural mistake to avoid

Patching the suggest prompt to allow empty graphs globally — that breaks the manual bridge editor path. **Split automation from manual extraction.**

---

## Proposed architecture

```text
orion:memory:turn:persisted
  → classify_turn → spark_meta (turn_change_appraisal, memory_significance_score)
  → window accumulate / close

on window close:
  → consolidation_memory_gate(window)     # NEW
       ├─ SKIP → mark_consolidated_skipped(reason)
       └─ PROPOSE → build_crystallization_from_window()
                    → insert proposed crystallization
                    → Hub Crystallizations inbox

manual paths (unchanged):
  bridge → memory_graph_suggest → graph draft → RDF approve → cards
  operator → crystallization propose/approve → project card/chroma/graphiti
```

### Choke points

| File | Role |
|------|------|
| `services/orion-memory-consolidation/app/worker.py` | `ConsolidationSuggestRunner.consolidate_window` — call gate, branch skip/propose |
| `orion/memory/consolidation_gate.py` | **Create** — deterministic gate |
| `orion/memory/crystallization/intake_consolidation_window.py` | **Create** — window → `MemoryCrystallizationV1` proposed |
| `orion/memory/low_info_social.py` | **Create** — shared courtesy/greeting detector (extracted from recall fusion pattern) |
| `services/orion-memory-consolidation/app/window_state.py` | `mark_consolidated_skipped`, `mark_crystallization_proposed` |
| `orion/schemas/memory_consolidation.py` | Add `consolidation_status: "skipped"` |

---

## Consolidation gate (deterministic)

### Inputs (all existing — no grammar schema changes)

Per window turn:

- `spark_meta.turn_change_appraisal` — `novelty_score`, `shift_kind`, `confidence`, `turn_change_status`
- `spark_meta.memory_significance_score`
- `prompt` + `response` text — low-info social check

Per window (optional grammar read):

- Query `grammar_events` WHERE `trace_id` matches `hub.chat:{node_id}:{correlation_id}`
- Use existing atom fields: `semantic_role=repair_signal`, `user_utterance` summary word count, collect `event_id`s

### Skip when (default conservative — beliefs expensive)

All of:

- Window max `novelty_score` below `MEMORY_CONSOLIDATION_MIN_NOVELTY` (default `0.35`)
- No turn with `shift_kind` in `{TOPIC, STANCE, REPAIR}` with novelty above floor
- Every turn passes `is_low_info_social(prompt, response)`
- No `repair_signal` grammar atom in window
- Window max `memory_significance_score` below `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` (default `0.40`)

### Propose when

Any skip condition fails (at least one substantive signal).

### Gate output

```python
@dataclass
class ConsolidationGateResult:
    action: Literal["skip", "propose"]
    reasons: list[str]              # e.g. ["low_info_social", "novelty_below_floor"]
    dominant_shift: str | None      # TOPIC | STANCE | REPAIR | None
    grammar_event_ids: list[str]    # for evidence refs
    window_novelty_max: float
    window_significance_max: float
```

Persist skip reasons on window row and optionally patch turn `spark_meta.consolidation_gate` for traceability.

---

## Crystallization intake from window

Mirror `intake_autonomy_episode.py`:

```python
def build_crystallization_from_window(
    *,
    memory_window_id: str,
    turns: list[dict],
    gate: ConsolidationGateResult,
    grammar_events: list[dict] | None = None,
) -> MemoryCrystallizationV1:
    ...
```

### Kind mapping

| Dominant shift | `kind` | Notes |
|----------------|--------|-------|
| `STANCE` | `stance` | Requires `retrieval_affordances` on approve validation |
| `REPAIR` | `open_loop` | Repair/open thread |
| `TOPIC` | `semantic` | Single substantive topic shift |
| Multi-turn narrative | `episode` | Window len > 1 with sustained topic |
| Weak but not skip | `episode` | Conservative default |

### Evidence refs

- `source_kind=chat_turn`, `source_id=correlation_id`, excerpt from prompt/response (clipped)
- `source_kind=grammar_event`, `source_id=event_id` for each collected grammar event
- `source_grammar_event_ids` on crystallization
- `grammar_envelope` via existing `attach_grammar_to_crystallization()` when full event dicts loaded

### Governance defaults

- `status=proposed`
- `requires_manual_review=true`
- `proposed_by=memory_consolidation_intake`
- `created_from_policy=consolidation_window_gate_v1`

Insert via `orion/memory/crystallization/repository.py` (same as Hub propose path). Emit `orion:memory:crystallization:proposed` bus event.

---

## Environment / config

Add to `services/orion-memory-consolidation/.env_example`:

| Key | Default | Purpose |
|-----|---------|---------|
| `MEMORY_CONSOLIDATION_OUTPUT` | `crystallization_propose` | `crystallization_propose` \| `graph_draft` (legacy) \| `skip_only` |
| `MEMORY_CONSOLIDATION_MIN_NOVELTY` | `0.35` | Window skip floor |
| `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` | `0.40` | Window skip floor |
| `MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE` | `true` | Read grammar_events for evidence refs |
| `MEMORY_CONSOLIDATION_GRAMMAR_DSN` | `{POSTGRES_URI}` | DSN for grammar_events table (may share sql-db) |

`MEMORY_CONSOLIDATION_OUTPUT=graph_draft` preserves current behavior for debugging only.

---

## End-to-end flow after implementation (write + read)

```text
── PER TURN (read — PCR) ──
1. Turn arrives → Phase 0: skip recall if greeting/low-novelty
2. Phase 1: sql_chat continuity_digest → stance LLM
3. Stance → retrieval_intent (relational / semantic / open_loop / …)
4. Phase 3: purposeful recall → belief_digest (active-packet + cards + RDF by intent)
5. chat_general: continuity_digest + belief_digest + attention_frame → speech

── PER WINDOW (write — this spec) ──
6. Juniper chats → turn persisted → classify → appraisal on spark_meta
7. Hub emits grammar events (existing atoms, no schema change)
8. Window closes → consolidation gate
9. Greeting window → skipped (no crystallization, no graph draft)
10. Substantive window → crystallization proposed → Hub inbox
11. Operator approves → project_crystallization → card / Chroma / Graphiti
12. Next substantive turn: PCR phase 3 retrieves approved beliefs via active-packet + cards

── MANUAL ──
13. Bridge graph draft → RDF approve (relational editor, not automation default)
```

---

## Non-goals

- Grammar schema or emitter changes
- Auto-approve crystallizations
- Replacing RDF memory_graph or graph draft editor
- Removing memory cards as recall surface
- Graphiti as extraction engine
- Keyword cathedrals for emotional/medical triggers
- Merging all rails into one store

---

## Acceptance checks

1. **Greeting window skip:** Two-turn hi/thanks window → `consolidation_status=skipped`, reason includes `low_info_social`, no graph draft, no crystallization row.
2. **Substantive propose:** Window with `shift_kind=TOPIC`, novelty ≥ 0.5 → `MemoryCrystallizationV1` `status=proposed` with `chat_turn` evidence refs.
3. **Grammar evidence:** When grammar rows exist for turn trace_ids, proposed crystallization includes `source_grammar_event_ids` (read-only).
4. **Manual graph path intact:** Bridge suggest → graph draft → approve still works; unrelated to consolidation automation.
5. **Legacy escape:** `MEMORY_CONSOLIDATION_OUTPUT=graph_draft` restores current suggest + draft insert behavior.
6. **Right-rail projection:** Approve crystallization → card appears in recall (`source=cards`); Graphiti sync optional when enabled; RDF not auto-written.
7. **Traceability:** Skip/propose decision inspectable on window row or spark_meta without reading logs.

---

## Recommended implementation order (cognitive memory program)

See PCR spec for **authoritative read-path milestones** (A: skip+continuity, B: purposeful recall). Write path below can parallel or follow milestone A.

**Shared**

1. `orion/memory/low_info_social.py`

**Read first (PCR milestones A→B)** — [`purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md)

2. Phase 0 skip + Phase 1 continuity in cortex-exec (chat relief without write changes)
3. Post-stance Phase 3 + active-packet collector + belief profiles
4. `smoke_pcr_chat_memory_e2e.sh`

**Write (this spec)**

5. `consolidation_gate.py` + `intake_consolidation_window.py`
6. Wire `consolidate_window` + `skipped` status
7. `smoke_memory_consolidation_pipeline.py`

8. Deprecate auto graph-draft default in README

---

## Open questions for operator

1. **Crystallization inbox vs auto-quarantine:** Should failed validation quarantine immediately, or stay as invalid proposal in inbox?
2. **Graph draft deprecation timeline:** Remove `graph_draft` default immediately, or one-release overlap with env flag?
3. **Grammar DSN:** Same Postgres as consolidation windows, or grammar-atlas DSN?

---

## Status

Draft for review. **Approve together with** [`2026-07-07-purpose-conditioned-recall-design.md`](./2026-07-07-purpose-conditioned-recall-design.md) (co-equal; read path is not deferred). On approval → combined implementation plan via `writing-plans` skill with **≥50% tasks on PCR**.
