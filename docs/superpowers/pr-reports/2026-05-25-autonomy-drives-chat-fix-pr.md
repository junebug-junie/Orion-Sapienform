# PR: Autonomy drives chat_stance timeout fix

**Branch:** `feat/autonomy-drives-chat-fix`  
**Status:** Ready for review

---

## Summary

Fixes the Hub autonomy compact card showing **"Orion drives facet timed out"** after every chat turn when Fuseki cannot finish Orion's drives SPARQL within budget.

| Area | Change |
|------|--------|
| `orion/autonomy/repository.py` | Default defer Orion drives on `chat_stance`; prefer relationship when Orion drives fail/defer; fix `_drives_facet_ok` row_count path; reuse injected `query_client` for drives |
| `services/orion-cortex-exec` | `docker-compose.yml` + `.env_example`: defer=true, subquery workers=1 |
| `services/orion-hub/static/js/app.js` | Clearer degraded labels when using relationship contextual fallback |
| `orion/autonomy/README.md` | Chat stance drives config table |
| Tests | Selection/defer/default coverage; 33+ autonomy tests pass |

---

## Root cause

Orion's drives graph is large. Chat stance ran a full Orion `drives` subquery (12s budget) on every turn. When it timed out, Orion was still selected with partial identity/goals state, so the Hub card showed degraded drive fields instead of relationship drives.

---

## Behavior after fix

1. **`AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=true` (default)** — skips Orion drives SPARQL on `chat_stance`; uses relationship drives + Orion goals.
2. **Selection** — if Orion drives timeout/defer, `select_preferred_autonomy_lookup` picks relationship when its drives facet is ok.
3. **Hub** — when contextual fallback applies, card shows relationship drives; context note explains Orion drives were skipped/unavailable.

Set `AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=false` only if you need Orion drives on every chat turn and Fuseki p95 stays under `AUTONOMY_DRIVES_SUBQUERY_TIMEOUT_SEC`.

---

## Config (cortex-exec)

```bash
AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=true
AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS=1
AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT=20
AUTONOMY_DRIVES_SUBQUERY_TIMEOUT_SEC=12
AUTONOMY_GRAPH_TIMEOUT_SEC=20.0
```

**Ops:** Restart `orion-cortex-exec` after deploy. Optional one-time graph hygiene:

```bash
PYTHONPATH=. python scripts/autonomy/archive_stale_goal_proposals.py --apply --all-subjects
```

---

## Tests

```bash
PYTHONPATH=. python -m pytest \
  orion/autonomy/tests/test_repository_selection_and_query.py \
  orion/autonomy/tests/test_repository_defer_orion_drives.py \
  tests/test_autonomy_summary_degraded.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py -q
```

---

## Test plan

- [ ] Send a `chat_general` turn; Hub autonomy card shows dominant/top drives (relationship subject), not "Orion drives facet timed out"
- [ ] `chat_autonomy_debug` shows `selected_subject: relationship` when Orion drives deferred
- [ ] With `AUTONOMY_CHAT_STANCE_DEFER_ORION_DRIVES=false`, verify Orion drives query still runs (may degrade if graph slow)
- [ ] Nightly goal archive still runs in orion-actions (unchanged)

---

## Sign-off

- [x] Defer Orion drives on chat_stance (default on)
- [x] Relationship fallback selection hardened
- [x] Critical `_drives_facet_ok` bug fixed
- [x] Local `.env` synced (not committed)
- [ ] Phase 2 Fuseki index / latest-pointer (deferred)
