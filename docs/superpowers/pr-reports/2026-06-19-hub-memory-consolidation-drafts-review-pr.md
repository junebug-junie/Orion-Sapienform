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

**Result:** 13 passed, exit 0

---

## Code review (third pass — all review items addressed)

**Verdict:** Ready to merge

| Issue | Fix |
|-------|-----|
| Reject did not clear `activeConsolidationDraftId` | `onRejected` callback clears id when rejected draft was loaded |
| Unrelated `fb53b08a` on branch | Rebased onto `main`; mesh email plan commit dropped |
| No `memory_schema_missing` test | `test_list_consolidation_drafts_memory_schema_missing` |
| List `limit` not clamped at route | `_clamp_limit(limit, default=50, cap=200)` |
| Approve hook no `pending_review` guard | `update_consolidation_draft_status` requires `pending_review` for `approved` |
| Non-atomic approve undocumented | README operator note + UI warning behavior documented |

**Remaining:** Live stack smoke (operator deploy).

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
