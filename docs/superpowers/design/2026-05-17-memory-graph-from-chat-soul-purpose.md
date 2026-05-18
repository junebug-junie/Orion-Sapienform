# Memory graph from chat — soul-purpose review (2026-05-17)

## 1. What should this feature accomplish?

**Keep.** It converts operator-selected conversation turns into **relational memory candidates** (SuggestDraftV1) for human review and approval—not arbitrary JSON visualization, not chat replay, not evidence dumps in the draft editor.

## 2. Extraction vs salience vs approval

**Change.** Do not treat ordinary selected turns as “no memory” when the model returns empty graph arrays.

| Layer | Question | Owner |
|-------|----------|-------|
| **Extraction** | What relational structure is present or reasonably implied from selected turns? | Suggest prompt + validator (generous) |
| **Salience / durability** | Is this worth long-term memory? | Future metadata / review workflow—not suggest |
| **Approval** | Should the operator persist it? | Approve action |

**Core principle:** Extraction should be **generous**. Persistence should be **selective**.

The Memory graph from chat feature is an **extraction/editor surface**, not the salience judge.

## 3. Role-grounded extraction

Selected chat **roles are semantic evidence**:

- User-turn speaker → user / Juniper
- Assistant-turn speaker → Orion / assistant
- First-person language in a **user** turn generally maps to the user
- First-person language in an **assistant** turn generally maps to Orion
- “You” in **assistant** text generally refers to the user
- “You” in **user** text generally refers to Orion
- Ordinary/routine turns still deserve a **minimal faithful semantic projection** (entities, situations, edges grounded in the selected text)
- Salience/durability is **later review metadata**, not a precondition for extraction

**Example (shower):** User: “k, off to shower. Be back soon!” / Assistant: “Shower well. I’ll be here when you’re back.”

Expected minimal graph (not hallucination—role-grounded discourse inference):

- user / Juniper as user-turn speaker and implied subject of departure
- Orion / assistant as assistant-turn speaker and implied subject of availability
- situations for temporary departure/shower/return expectation and assistant remaining available
- edges linking entities to situations using allowed predicates only

## 4. Object boundaries (UI contract)

| Object | Role | Where it lives |
|--------|------|----------------|
| Evidence envelope | Request input only (ids + text) | Bridge selection + suggest POST body |
| Draft | Valid SuggestDraftV1 | Draft JSON textarea only |
| Diagnostics | Parse/validation/route metadata | Status / output panel only |
| Preview graph | Derived from draft only | Cytoscape panel |
| Approved memory | Persisted RDF/Postgres | After explicit approve |

**Keep:** Enforce separation in code; evidence must never populate Draft JSON.

## 5. Operator control

**Change.** Status lines must answer: what was selected, that suggest ran, whether output is a valid role-grounded draft, why graph is empty (no extractable role evidence vs extractor failure), and where diagnostics live.

Do **not** show “no durable memory candidate” from the extraction path—that conflates salience with extraction.

## 6. Orion’s larger purpose

**Keep** discernment at **persistence** time: relational, meaningful, provenance-preserving, operator-reviewable memory. **Extract faithfully first**; let the operator decide what to keep.

---

## Conclusion

| | |
|---|---|
| **Keep** | Chat→suggest→review→approve path; `/api/memory/graph/suggest`; operator review |
| **Change** | Mandatory role-grounded extraction for selected turns; strict SuggestDraftV1-only Draft JSON; failure vs empty-without-evidence status separation |
| **Remove** | Treating banal turns as successful “empty candidate”; “no durable memory” language in suggest UI |

### Feature promise (v1)

Selected turns always produce a **valid SuggestDraftV1**. If the selected turns contain role-grounded semantic content, the draft contains a **minimal faithful graph** with entities/situations/edges. If extraction fails, the UI loads an **empty valid fallback draft** and surfaces diagnostics **outside** the editor. Durability/salience is **separate** from extraction.

### Must never do again

- Put assistant prose in Draft JSON
- Put evidence envelope (`utterance_ids` + `utterance_text_by_id` only) in Draft JSON
- Treat diagnostics or chat display text as draft
- Label extractor failure as “no durable memory candidate”
- Accept empty graph arrays as successful extraction when role-grounded selected turns imply structure

### Minimal product shape (v1)

1. Bridge: select turns → Suggest → validated role-grounded draft or empty fallback + diagnostics on failure
2. Memory tab: same coalescer; import gate on sessionStorage
3. Preview: from draft only; empty preview explains extraction outcome via status line
4. Approve: unchanged downstream of valid draft

### Status copy (extraction path)

- Success (nonempty graph): “Loaded validated role-grounded SuggestDraftV1 JSON.”
- Success (empty graph, no role-grounded evidence to extract): “Loaded valid empty SuggestDraftV1 JSON.”
- Extractor failure: “Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics.”
- Evidence blocked: “Blocked selected-turn evidence envelope from Draft JSON; evidence is request input, not graph output.”
