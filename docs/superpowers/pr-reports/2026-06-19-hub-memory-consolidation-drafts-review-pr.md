# PR: Hub Memory tab — consolidation graph drafts review

**Branch:** `feat/hub-memory-consolidation-drafts`  
**Service:** `orion-hub` (+ shared `orion/memory_graph/draft_repository.py`)

---

## Summary

Wires automated memory consolidation output into the existing Hub Memory tab review flow:

- **Backend:** list/get/status APIs over `memory_graph_suggest_drafts` (same Postgres pool as memory cards via `RECALL_PG_DSN`).
- **UI:** Memory tab → **Graph drafts** subview lists `pending_review` drafts; **Load in editor** prefills the graph annotator; **Reject** marks draft rejected.
- **Approve hook:** `POST /api/memory/graph/approve` accepts optional `consolidation_draft_id` and marks draft `approved` after successful graph approve.

No new bus channels or env keys — uses existing Hub memory Postgres pool and graph validate/approve endpoints.

---

## Operator flow

1. Open Hub → **Memory** → **Graph drafts**
2. Click **Load in editor** on a draft
3. **Validate** / edit in graph annotator → **Approve**
4. Draft leaves pending queue; active situation cards created as today

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/memory/consolidation/drafts?status=pending_review&limit=50` | Inbox list (summary only) |
| GET | `/api/memory/consolidation/drafts/{draft_id}` | Full draft JSON |
| POST | `/api/memory/consolidation/drafts/{draft_id}/status` | `{ "status": "rejected" \| "pending_review" }` only (`approved` via graph approve) |
| POST | `/api/memory/graph/approve` | Optional `consolidation_draft_id` in body |

---

## Prerequisites

- `RECALL_PG_DSN` on Hub (memory pool)
- `manual_migration_memory_consolidation_v1.sql` applied
- `orion-memory-consolidation` running (produces drafts)

---

## Verification

```bash
cd .worktrees/feat/hub-memory-consolidation-drafts
PYTHONPATH=.:services/orion-hub ./venv/bin/python -m pytest \
  services/orion-hub/tests/test_memory_consolidation_draft_routes.py \
  services/orion-hub/tests/test_memory_review_ui.py \
  tests/test_memory_graph_draft_repository.py -q
```

**Result:** 12 passed, exit 0

```bash
PYTHONPATH=.:services/orion-hub ./venv/bin/python -m pytest \
  services/orion-hub/tests/test_memory_consolidation_draft_routes.py \
  services/orion-hub/tests/test_memory_graph_consolidation_draft_approve.py \
  services/orion-hub/tests/test_memory_review_ui.py \
  tests/test_memory_graph_draft_repository.py -q
```

---

## Code review (second pass — after fixes, commit `c0664381`)

**Verdict:** Ready to merge

| Prior issue | Fix verified |
|-------------|--------------|
| Status POST allowed `approved` without graph approve | Only `rejected` / `pending_review`; test `test_set_consolidation_draft_status_rejects_approved` |
| `activeConsolidationDraftId` not cleared on manual Suggest | Cleared at start of Suggest handler |
| Approve hook silently succeeded if draft update failed | Returns `consolidation_draft_marked: false`, `consolidation_draft_status: update_failed`; UI shows error status |
| Missing `memory_schema_missing` for undefined table | `_http_if_missing_memory_schema` on list/get/status |
| No test for approve + `consolidation_draft_id` hook | `test_memory_graph_consolidation_draft_approve.py` (success + update_failed) |

**Strengths:** Minimal diff; approve path still succeeds when graph/cards succeed even if inbox mark fails (operator gets explicit signal). Consolidation draft id captured before async approve so Suggest cannot race.

**Remaining minor (non-blocking):**
- Live stack checklist still unchecked (requires Hub deploy on branch).
- `gh` auth unavailable — PR must be opened manually.

---

## Test plan (live)

- [ ] Hub restart with branch deployed
- [ ] Memory tab → Graph drafts shows existing `pending_review` rows
- [ ] Load in editor → graph preview renders
- [ ] Approve → draft status `approved` in Postgres; cards created
- [ ] Reject → draft status `rejected`; removed from inbox

---

## Non-goals

- Auto-promotion without operator approve
- Backfill/repair of historical drafts
- Separate review queue for memory_cards changes
