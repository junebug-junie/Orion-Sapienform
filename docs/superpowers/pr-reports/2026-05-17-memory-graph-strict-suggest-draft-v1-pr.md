# PR: Memory graph from chat — strict draft gating + role-grounded extraction

**Branch:** `fix/memory-graph-strict-suggest-draft-v1`  
**Worktree:** `.worktrees/fix-memory-graph-strict-suggest-draft-v1`  
**Base:** `main`

## Summary

Two layers on the same branch:

1. **Strict SuggestDraftV1 gating** — Draft JSON never accepts prose, diagnostics, or selected-turn evidence envelopes; coalescer + import gate enforce shape.
2. **Mandatory role-grounded extraction** — Ordinary selected user/assistant turns must yield a minimal faithful graph (not “empty = no memory”). Extraction is generous; salience/durability is deferred to review/approve.

**Core principle:** Extraction should be generous. Persistence should be selective.

## Soul-purpose correction

Design note: `docs/superpowers/design/2026-05-17-memory-graph-from-chat-soul-purpose.md`

| Before (wrong) | After (correct) |
|----------------|-----------------|
| Banal turns → empty graph + “no durable candidate” | Role-grounded turns → minimal user/Orion/situation graph |
| Suggest path judges salience | Suggest extracts structure; operator judges persistence |

## Changes

| Area | Files |
|------|--------|
| Design | `docs/superpowers/design/2026-05-17-memory-graph-from-chat-soul-purpose.md` |
| Prompt | `orion/cognition/prompts/memory_graph_suggest_prompt.j2` — role rules + shower few-shot (`urn:uuid` ids) |
| Validator | `orion/memory_graph/suggest_validate.py` — `extract_selected_role_evidence`, role-grounded escalation errors |
| Fixture | `tests/fixtures/memory_graph/shower_role_grounded_draft.json` |
| UI coalescer | `memory-graph-draft-ui.js` — no `no_durable_memory_candidate`; role-grounded status strings |
| Bridge / Memory tab | `app.js`, `memory.js` — matching status copy |
| Tests | `test_memory_graph_suggest_validate.py`, `test_memory_graph_suggest_escalation.py`, coalesce + bridge UI tests |

## Validator escalation (shower case)

Empty draft + bridge prompt with `role=user` / `role=assistant` and shower text → escalates with:

- `no_entities_when_role_grounded_subjects_expected`
- `no_situations_when_role_grounded_context_expected`
- `missing_user_role_entity` / `missing_assistant_role_entity` (when both roles present)

Quick empty → Brain minimal graph → `ok: true`, `route_used: brain`.

Both routes empty → `ok: false`, `memory_graph_suggest_failed` (UI loads empty fallback + extractor failure status).

## UI status copy (extraction path)

| Outcome | Status |
|---------|--------|
| Success, nonempty graph | Loaded validated role-grounded SuggestDraftV1 JSON. |
| Success, empty graph (no role evidence) | Loaded valid empty SuggestDraftV1 JSON. |
| Extractor failure | Extractor did not return a valid role-grounded SuggestDraftV1. Empty valid fallback draft loaded; see diagnostics. |
| Evidence blocked | Blocked selected-turn evidence envelope from Draft JSON; evidence is request input, not graph output. |

**Removed:** “No durable memory candidate found” from suggest/coalesce paths.

## Verification

```bash
cd .worktrees/fix-memory-graph-strict-suggest-draft-v1
PYTHONPATH=. ../../venv/bin/python -m pytest \
  tests/test_memory_graph_suggest_validate.py \
  services/orion-hub/tests/test_memory_graph_suggest_coalesce_ui.py \
  services/orion-hub/tests/test_memory_graph_bridge_ui.py \
  services/orion-hub/tests/test_memory_graph_suggest_escalation.py \
  -q --tb=short
```

**Result:** 38 passed, exit 0

## Manual acceptance (shower case)

**UNVERIFIED** — hub stack not run this session.

1. Select turns:
   - user: “k, off to shower. Be back soon!”
   - assistant: “Shower well. I’ll be here when you’re back.”
2. Click **Suggest draft** (bridge or Memory tab).
3. **Expected:**
   - Valid SuggestDraftV1 with User + Orion entities and departure/availability situations
   - No bare evidence envelope in Draft JSON
   - No prose in Draft JSON
   - Status is role-grounded success or extractor failure — never “no durable memory candidate”
4. If both Quick and Brain return empty graphs: empty fallback draft + extractor failure diagnostics

## Env / compose

No new settings keys — no `.env` / `docker-compose` changes in this slice.

## Test plan

- [x] Role-grounded validator unit tests (shower empty, shower minimal, bikes user-only)
- [x] Quick→Brain escalation on empty role-grounded draft
- [x] Both-fail returns `memory_graph_suggest_failed`
- [x] Strict coalescer regressions (prose/evidence rejected)
- [x] Status string regression (no “durable memory candidate”)
- [ ] Live hub shower suggest with real LLM routes

## Commits (this branch)

Includes prior strict-draft commits plus:

- `docs: role-grounded extraction principle for memory graph from chat`
- `feat(memory-graph): mandatory role-grounded suggest prompt and validator`
- `fix(hub): role-grounded suggest status copy; remove no-durable-candidate`
- `test: role-grounded memory graph suggest validation and escalation`
