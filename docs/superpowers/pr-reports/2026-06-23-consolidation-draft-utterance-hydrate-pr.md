# PR: Hydrate consolidation graph draft utterance text

**Branch:** `fix/consolidation-draft-utterance-hydrate`  
**Services:** `orion-hub`, `orion-memory-consolidation`, shared `orion/memory_graph`

---

## PR description (GitHub — copy below)

## Summary

Fixes `utterance_text_missing:turn-id-1,...` when validating or approving consolidation graph drafts from the Hub Memory tab.

Consolidation cortex output uses placeholder `utterance_ids` (`turn-id-1`, `turn-id-2`) with empty `utterance_text_by_id`. The graph preview renders from entities/situations/edges, but graph validate/approve requires utterance excerpts for RDF `schema:text`.

This PR hydrates `utterance_text_by_id` from `chat_history_log` via index-aligned `turn_correlation_ids` stored on each draft row.

## What changed

- **`orion/memory_graph/consolidation_draft_hydrate.py`** — map utterance ids to turn prompt/response text (`User: …\nOrion: …`)
- **Hub GET consolidation draft** — returns hydrated draft JSON on load into editor
- **Hub graph validate/approve** — when `consolidation_draft_id` is sent, merges DB-hydrated text as supplemental before `ensure_draft_utterance_text`
- **Hub UI** — validate sends `consolidation_draft_id` when a graph draft is loaded (same as approve)
- **Consolidation worker** — populates `utterance_text_by_id` at insert time for new drafts

## Operator flow (after deploy)

1. Memory → **Graph drafts** → **Load in editor** (must reload if editor had stale JSON)
2. Validate → Approve — no `utterance_text_missing` error

## Test plan

```bash
cd services/orion-hub
PYTHONPATH=../../.:. ../../venv/bin/python -m pytest \
  tests/test_memory_consolidation_draft_routes.py \
  tests/test_memory_graph_consolidation_draft_approve.py -q

cd ../..
PYTHONPATH=. ./venv/bin/python -m pytest \
  tests/test_memory_graph_consolidation_draft_hydrate.py -q
```

**Result:** 15 passed

**Live smoke**

- [ ] Hub + memory-consolidation restarted on branch
- [ ] Load consolidation draft → `draft.utterance_text_by_id` populated in network response
- [ ] Validate + Approve succeed (no `utterance_text_missing`)

## Non-goals

- Rewriting cortex suggest to emit real utterance ids (future improvement)
- Backfill persisted draft JSON in Postgres (hydration at read/approve is sufficient)

---

## Files changed

- `orion/memory_graph/consolidation_draft_hydrate.py` (new)
- `services/orion-hub/scripts/memory_consolidation_draft_routes.py`
- `services/orion-hub/scripts/memory_graph_routes.py`
- `services/orion-hub/static/js/memory.js`
- `services/orion-memory-consolidation/app/worker.py`
- `tests/test_memory_graph_consolidation_draft_hydrate.py` (new)
- `services/orion-hub/tests/test_memory_consolidation_draft_routes.py`
- `services/orion-hub/tests/test_memory_graph_consolidation_draft_approve.py`
