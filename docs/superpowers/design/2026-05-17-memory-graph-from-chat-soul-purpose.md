# Memory graph from chat — soul-purpose review (2026-05-17)

## 1. What should this feature accomplish?

**Keep.** It converts operator-selected conversation turns into **relational memory candidates** (SuggestDraftV1) for human review and approval—not arbitrary JSON visualization, not chat replay, not evidence dumps in the draft editor.

## 2. Banal / non-memory-worthy turns

**Change.** When turns lack durable relational content (e.g. “off to shower” / “see you soon”), the pipeline must return a **valid empty SuggestDraftV1** with provenance (`utterance_ids`, `utterance_text_by_id`) and an explicit status: **“No durable memory candidate found.”** Never a broken graph, never an evidence-only object in Draft JSON, never ambiguous empty preview without explanation.

## 3. Object boundaries (UI contract)

| Object | Role | Where it lives |
|--------|------|----------------|
| Evidence envelope | Request input only (ids + text) | Bridge selection + suggest POST body |
| Draft | Valid SuggestDraftV1 | Draft JSON textarea only |
| Diagnostics | Parse/validation/route metadata | Status / output panel only |
| Preview graph | Derived from draft only | Cytoscape panel |
| Approved memory | Persisted RDF/Postgres | After explicit approve |

**Change:** Enforce separation in code; evidence must never populate Draft JSON.

## 4. Operator control

**Change.** Status lines must answer: what was selected, that suggest ran, whether output is valid draft, why graph is empty (no candidate vs failure), and where diagnostics live. Remove misleading “model reply” / prose-in-draft flows.

## 5. Orion’s larger purpose

**Keep** discernment: relational, meaningful, provenance-preserving, operator-reviewable memory—not graphing everything.

---

## Conclusion

| | |
|---|---|
| **Keep** | Chat→suggest→review→approve path; `/api/memory/graph/suggest`; operator review |
| **Change** | Strict SuggestDraftV1-only Draft JSON; explicit empty-candidate vs failure states; object separation |
| **Remove** | Writing evidence/prose/chat text into Draft JSON; permissive draft-shape heuristics |

### Feature promise (v1)

Selected turns → suggest extractor → **always** a valid SuggestDraftV1 in Draft JSON (possibly empty graph arrays) with clear status.

### Must never do again

- Put assistant prose in Draft JSON
- Put evidence envelope (`utterance_ids` + `utterance_text_by_id` only) in Draft JSON
- Treat diagnostics or chat display text as draft
- Leave operator guessing whether empty graph means “nothing to remember” vs “pipeline broke”

### Minimal product shape (v1)

1. Bridge: select turns → Suggest → validated draft or empty draft + status
2. Memory tab: same coalescer; import gate on sessionStorage
3. Preview: from draft only; empty graph + status explains “no candidate”
4. Approve: unchanged downstream of valid draft
