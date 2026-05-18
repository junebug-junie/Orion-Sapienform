# PR: Strict SuggestDraftV1 enforcement for Memory graph from chat

**Branch:** `fix/memory-graph-strict-suggest-draft-v1`  
**Worktree:** `.worktrees/fix-memory-graph-strict-suggest-draft-v1`  
**Base:** `main` @ `f4a504bb`

## Summary

Stops whack-a-mole patching of Memory graph from chat by separating **evidence** (request input) from **draft** (SuggestDraftV1 only). The Draft JSON textarea now always receives a structurally valid `SuggestDraftV1` object—never assistant prose, never a bare `{ utterance_ids, utterance_text_by_id }` evidence envelope. Operator status lines distinguish success, no durable candidate, extractor failure, and blocked evidence.

## Soul-purpose (PASS 1)

Design note: `docs/superpowers/design/2026-05-17-memory-graph-from-chat-soul-purpose.md`

| Decision | |
|----------|---|
| **Keep** | Chat → suggest → review → approve; relational discernment |
| **Change** | Strict draft shape; explicit empty-candidate vs failure states |
| **Remove** | Prose/evidence/diagnostics in Draft JSON |

**v1 promise:** Selected turns → suggest → always valid SuggestDraftV1 in editor (possibly empty graph arrays) + clear status.

## Changes

| File | Change |
|------|--------|
| `memory-graph-draft-ui.js` | Strict `looksLikeMemoryGraphDraftObject`; `emptySuggestDraft`; `looksLikeEvidenceEnvelopeOnly`; hardened `coalesceMemoryGraphSuggestEnvelope`; `formatSuggestCoalesceUserStatus` |
| `app.js` | Bridge suggest uses coalescer with `{ utteranceIds, utteranceTextById }`; product-true status lines |
| `memory.js` | Same coalescer/status; import gate `invalid_import_not_suggest_draft_v1`; provenance on suggest fallback |
| Tests | Prose/evidence/empty/non-empty draft regression; bridge wiring asserts |

## Hard requirements checklist

- [x] Draft JSON only valid SuggestDraftV1 shape
- [x] Evidence envelope rejected (`evidence_envelope_not_draft`)
- [x] Strict shape check (ontology + all arrays)
- [x] `emptySuggestDraft` with `utterance_text_by_id`
- [x] Coalescer rejects prose, evidence, arbitrary JSON
- [x] Bridge + Memory tab use `POST /api/memory/graph/suggest` (not `/api/chat`)
- [x] Import listener parse-gates sessionStorage
- [x] Product-true status copy (no “model reply” message)
- [x] Regression tests A–F

## Verification

```bash
cd .worktrees/fix-memory-graph-strict-suggest-draft-v1
PYTHONPATH=. ../../venv/bin/python -m pytest \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py -q --tb=short
```

**Result:** 13 passed, exit 0

## Manual acceptance (shower case)

1. Open chat → Memory graph from chat on banal turns:
   - user: “k, off to shower. Be back soon!”
   - assistant: “Shower well. I'll be here when you're back.”
2. Click **Suggest draft**.
3. **Expected:**
   - Draft JSON is valid SuggestDraftV1 with empty `entities`/`situations`/`edges`/`dispositions`
   - Status: “No durable memory candidate found. Empty valid draft loaded.” (or “Loaded validated…” if model returns graph rows)
   - Bare evidence envelope **not** in Draft JSON
   - No “Replaced the box with the model reply” text

**Live UI:** UNVERIFIED in this session (no hub stack run). Static + Node coalescer tests cover the failure modes.

## Commits

1. `docs: soul-purpose review for memory graph from chat`
2. `fix(hub): strict SuggestDraftV1 coalescer and empty draft helper`
3. `fix(hub): product-true memory graph suggest statuses in bridge and tab`
4. `test(hub): regression for strict memory graph suggest draft coalescing`
5. `fix(hub): preserve utterance provenance on Memory tab suggest failure`

## Risks / follow-ups

- Client shape check is heuristic, not full Pydantic parity; **Validate** remains authoritative.
- Bridge `!res.ok` path writes empty draft without coalescing response body (minor; unlikely to contain valid draft).
- Optional: test for `missing_required_suggest_draft_fields` rejection path.

## Test plan

- [x] Automated coalescer + bridge UI tests
- [ ] Manual shower-turn suggest in running hub
- [ ] Bridge → Memory tab import with valid draft
- [ ] Bridge → Memory tab import with evidence-only JSON (should block + empty draft)
