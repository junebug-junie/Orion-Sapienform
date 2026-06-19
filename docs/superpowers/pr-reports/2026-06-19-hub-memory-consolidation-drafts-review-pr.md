# PR: Hub Memory tab — consolidation graph drafts review

**Branch:** `feat/hub-memory-consolidation-drafts`  
**Service:** `orion-hub` (+ shared `orion/memory_graph/draft_repository.py`)

---

## PR description (GitHub — copy below)

## Summary

Adds a Hub Memory tab inbox for automated memory consolidation graph drafts. `orion-memory-consolidation` already persists `memory_graph_suggest_drafts` rows when conversation windows close; this PR wires those drafts into the existing graph annotator review flow so operators can load, reject, and approve them without a separate tool.

No new env keys, bus channels, or schema registry changes. Uses the existing Hub memory Postgres pool (`RECALL_PG_DSN`) and graph validate/approve endpoints.

## Motivation

Consolidation was producing real `pending_review` drafts in Postgres, but the Memory tab only had a review queue for `memory_cards`, a manual graph editor, and crystallizations — no inbox for consolidation output. Operators had no UI path from pipeline output to approve.

## What changed

**Backend**
- `orion/memory_graph/draft_repository.py` — `list_consolidation_drafts`, `get_consolidation_draft`, `update_consolidation_draft_status` (approve requires `pending_review`)
- `memory_consolidation_draft_routes.py` — list / get / status APIs with session gate, `_clamp_limit`, `memory_schema_missing` mapping
- `memory_graph_routes.py` — optional `consolidation_draft_id` on approve; marks draft approved after successful graph/cards; returns `consolidation_draft_marked` / `update_failed` when inbox update fails (graph success is preserved)

**UI**
- Memory tab → **Graph drafts** subview (list, Load in editor, Reject)
- Load prefills graph annotator; Approve sends `consolidation_draft_id` when loaded from inbox
- Reject clears active draft link and editor content; manual Suggest clears inbox link; manual graph approve does not touch inbox fields

**Docs**
- `services/orion-hub/README.md` — operator notes (non-atomic approve, reject behavior)

## Operator flow

1. Hub → **Memory** → **Graph drafts**
2. **Load in editor** on a draft
3. **Validate** / edit → **Approve**
4. Draft status becomes `approved`; active situation cards created via existing graph approve path

Reject removes the draft from the inbox (`status=rejected`) and clears it from the editor if loaded.

## API

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/memory/consolidation/drafts?status=pending_review&limit=50` | Summary only (no full draft JSON) |
| GET | `/api/memory/consolidation/drafts/{draft_id}` | Full draft JSON |
| POST | `/api/memory/consolidation/drafts/{draft_id}/status` | `{ "status": "rejected" \| "pending_review" }` — not `approved` |
| POST | `/api/memory/graph/approve` | Optional `consolidation_draft_id` in body |

## Prerequisites

- `RECALL_PG_DSN` on Hub
- `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql` applied
- `orion-memory-consolidation` running

## Test plan

**Automated (16 passed)**

```bash
cd .worktrees/feat/hub-memory-consolidation-drafts
PYTHONPATH=.:services/orion-hub ./venv/bin/python -m pytest \
  services/orion-hub/tests/test_memory_consolidation_draft_routes.py \
  services/orion-hub/tests/test_memory_graph_consolidation_draft_approve.py \
  services/orion-hub/tests/test_memory_review_ui.py \
  tests/test_memory_graph_draft_repository.py -q
```

**Live (operator)**

- [ ] Hub restart on branch
- [ ] Memory → Graph drafts lists `pending_review` rows
- [ ] Load in editor → graph preview renders
- [ ] Approve → draft `approved` in Postgres; cards created
- [ ] Reject → draft `rejected`; removed from inbox; editor cleared if loaded

## Non-goals

- Auto-promotion without operator approve
- Backfill/repair of historical drafts
- Changes to memory_cards review queue semantics

## Review status

Multiple code-review passes completed; no open Critical or Important issues. Ready to merge pending live smoke.

---

## Commits

```
116198e4 fix(hub): clear editor on reject; expand schema-missing tests.
76f7d95a fix(hub): omit consolidation inbox fields on manual graph approve.
37d48446 fix(hub): address third-pass consolidation draft review items.
5e3f4998 fix(hub): harden consolidation draft review per code review.
d8bc8e42 feat(hub): add Memory tab inbox for consolidation graph drafts.
```

**Diff:** 12 files, +890 / −38 vs `main`
