# Memory graph chat entry bridge + spec gap closure — design

**Date:** 2026-05-02  
**Status:** Draft — operator review  
**Related:** [Memory graph annotator (Hub) + dual-write GraphDB](./2026-05-02-memory-graph-annotator-hub-design.md), [Draft viz + bridge turn depth / ids (2026-05-03)](./2026-05-03-memory-graph-draft-viz-and-bridge-turns-design.md)  
**Decision:** Smallest change that removes paste-as-primary workflow (**option C**): optimize for minimal surface area; Approve location unconstrained. **Normative depth, cap, persistence, “Select last K”, and user-bubble `hub-utterance:*` ids** are defined in the 2026-05-03 spec; the “e.g. 5 turns” below is a **default token target**, not the hard maximum.

---

## 1. Problem statement

The shipped Hub **Memory** tab exposes a **freeform textarea** for graph annotation. The parent spec promised a **turn picker** and provenance via **session + utterance ids** ([§6 data flow](./2026-05-02-memory-graph-annotator-hub-design.md)). Operators without pre-built Appendix C JSON cannot bridge **live chat history** into `Suggest` / `Validate` / `Approve` without manual copy-paste and guesswork.

The backend shape **`SuggestDraftV1`** already includes **`utterance_ids`** and **`utterance_text_by_id`** (`orion/memory_graph/dto.py`), matching **Appendix C** intent — the missing piece is **UI + prompt wiring** from Hub conversation turns.

---

## 2. Goals and non-goals

### Goals

1. **Primary entry from Hub chat:** Select one assistant reply or a **short chain** (user + assistant turns) → open a focused flow that **feeds structured evidence** into `memory_graph_suggest` without requiring operators to paste Appendix C.
2. **Reuse existing rails:** Same `/api/chat` brain payload with `verbs: ["memory_graph_suggest"]` and same `/api/memory/graph/validate` + `/api/memory/graph/approve` — no parallel ontology path.
3. **Handoff to Memory tab without duplication of truth:** Prefill or sync the Memory textarea + output panel so operators who prefer **Validate / Approve** next to cards can continue there (optional one-click “Edit on Memory tab”).
4. **Track parent-spec gaps** (§5, §12, appendices, downstream §9.x) in an explicit **closure matrix** with phased work — this doc names owners-by-phase; the [implementation plan](../plans/2026-05-02-memory-graph-chat-entry-bridge-implementation.md) sequences tasks.

### Non-goals (this bridge track)

- Replacing Memory Cards **policies** or **recall routing** — sibling tracks remain per parent spec §8.5 / §9.7.
- OWL reasoning beyond validation already in **memory-graph** — unchanged.
- Perfect **during-chat** ergonomics for every edge case — v1 targets the common path: **annotate after reading a reply**, optionally including preceding user message(s).

---

## 3. Recommended UX (smallest lift)

### 3.1 Hub conversation surface

- On **assistant** bubbles (same row as existing feedback / inspect affordances where layout fits), add **“Memory graph…”** (or shorter **“Graph…”**).
- Click opens a **modal** that shows:
  - **Chain selector:** Default = clicked assistant turn + optional **previous N user/assistant turns** (checkboxes or “include prior user message”). **Default N ≈ 5** for token control on first visit; operators may raise **N** up to a **hard cap** with persistence — see [2026-05-03 spec §5](./2026-05-03-memory-graph-draft-viz-and-bridge-turns-design.md#5-bridge-configurable-depth-and-selection).
  - **Evidence preview:** Read-only concatenation of turn texts with **stable ids** displayed monospaced (`message_id` / `turnId` / `correlationId` — whatever `resolveFeedbackLinkage` and message meta already expose for feedback).

### 3.2 Suggest from modal

- **Primary button:** **Suggest draft**.
- Implementation composes a **single user message** to `/api/chat` that includes:
  - Explicit **`ontology_version`** target (from Hub settings constant or env surfaced once on load).
  - **`utterance_ids`** array and **`utterance_text_by_id`** map mirroring **`SuggestDraftV1`** fields so the LLM step is grounded (verb remains `memory_graph_suggest`; optional follow-up: enrich **Jinja** prompt in `memory_graph_suggest_prompt.j2` to prioritize these fields — plan covers both “prompt-only” and “template tweak” as sequential tasks).
- **Result handling:** Same behavior as today’s Memory tab — extract JSON into an editable textarea **inside the modal**; show the same “trim prose wrapper” hint.

### 3.3 Validate / Approve placement (option C)

To minimize duplicate logic and respect **GraphDB on Hub** for approve:

| Step | Location |
|------|-----------|
| Suggest | Hub modal (primary) **or** Memory textarea (unchanged) |
| Validate | Prefer **shared helper**: POST `/api/memory/graph/validate` from either surface |
| Approve | **Memory tab** remains the default location for v1 (banner already explains GraphDB). Modal offers **“Continue on Memory tab”** which copies draft + optional `named_graph_iri` hint into **sessionStorage** + fires **`CustomEvent`** so `memory.js` prefills `#memoryGraphDraftJson` and optionally switches panel to **Memory**. |

If a later iteration wants **Approve in-modal**, it is a thin wrapper calling the same POST — not required for smallest lift.

---

## 4. Data flow (bridge)

```text
Hub DOM (turn meta) → modal selection
  → compose chat payload (utterance_ids + utterance_text_by_id + ontology hint)
  → POST /api/chat (memory_graph_suggest)
  → JSON draft in modal
  → optional sessionStorage + event → Memory tab textarea
  → Validate / Approve unchanged (memory_graph_routes.py)
```

**Provenance:** Turn ids must match what **approve** / **json_to_rdf** can attach as `prov:wasDerivedFrom` — if meta only has correlation ids, document the **canonical id choice** in the implementation plan (single rule: prefer `message_id` when present, else `turnId`, else synthetic session-scoped index with warning banner). **User bubbles without server meta:** use `hub-utterance:<uuid>` and session-local semantics per [2026-05-03 spec §3](./2026-05-03-memory-graph-draft-viz-and-bridge-turns-design.md#3-id-strategy-for-user-and-assistant-turns-decision).

---

## 5. Parent-spec gap closure matrix

Cross-reference: [Memory graph annotator — §2 Non-goals, §5, §12, Appendices A–D](./2026-05-02-memory-graph-annotator-hub-design.md).

| Area | Parent reference | Current state | Closure approach |
|------|------------------|---------------|-------------------|
| Turn picker + ids | §6, §10 Hub UI | Textarea only | **This bridge** + meta wiring |
| Appendix C shape | Appendix C | DTO matches; UI did not supply ids | Modal + prompt payload |
| Appendix D projection | §4.11, Appendix D | `project.py` builds subschema | Covered by approve path when draft valid — verify golden row in §5 track |
| Appendix A exemplar | Appendix A | Informative | **Golden fixture** in tests (§5 deliverables) |
| Appendix B SHACL | Appendix B | Partial / evolving | **Ontology + SHACL package** (§5) |
| §5 Schema alignment package | §5 normative | Incomplete | **Blocking** for production-grade approve — phased deliverables |
| §12 Open points | §12 | Open | Each item → task in implementation plan (verbs, compensation, deploy shape, enum, distiller order, recall routing) |
| §8.5 Recall routing | §8.5 | Sibling `orion-recall` | Explicit **follow-on plan**; not in bridge PR |
| §9.4–9.6 Journals / metacog / spark | §9.4–9.6 | Not wired | Prioritized epics after §9.1–9.3 per parent spec |
| §8.3 Distiller (D) vs projector (P) | §8.2–8.3 | Default P | Gate **D** behind operator flag when §5 stable |
| Non-goal “during-chat primary” | §2 | — | **Interpretation:** entry can be Hub-adjacent; **commit** remains operator-controlled post-hoc — aligns with parent intent |

---

## 6. Testing and acceptance (bridge)

| Check | Acceptance |
|-------|------------|
| No manual JSON to **start** | Operator opens modal from a reply and runs Suggest with pre-filled evidence |
| Provenance | Suggested draft JSON includes same **utterance_ids** keys as selected turns (modulo canonical id rule) |
| Handoff | “Continue on Memory tab” prefills Memory textarea and validates without re-paste |
| Regression | Memory-only workflow (type/paste in textarea) still works |

---

## 7. Out of scope for this design file

Detailed **ontology TTL**, **projector_mapping** line-by-line, and **recall fusion** algorithms remain in the parent spec and **§5 / §12** implementation tasks — this document only **routes** them into the plan backlog.

---

## 8. Approval

After operator sign-off, implementation follows [memory-graph chat entry bridge implementation plan](../plans/2026-05-02-memory-graph-chat-entry-bridge-implementation.md).
