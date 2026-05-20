# PR: Autonomy drives query fix + degraded semantics + contextual fallback

**Branch:** `feat/autonomy-compact-degraded-state`  
**Worktree:** `.worktrees/feat-autonomy-compact-degraded-state`

## Summary

Fixes the **root causes** behind silent autonomy compact card blanks, not just the labels:

1. **Drives SPARQL rewritten** — latest-artifact subquery before OPTIONAL joins (avoids scanning/joining across many artifacts)
2. **`availability=degraded`** — partial facet failure is no longer masqueraded as `available`
3. **Contextual fallback** — when Orion drives fail but relationship drives succeed, chat stance selects relationship drives with explicit provenance (Orion goals preserved, not substituted)
4. **Parallel chat-stance subqueries** — `AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS=3` (default)
5. **Separate compact drives limit** — `AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT=20` (not overridden by probe `AUTONOMY_DRIVES_QUERY_LIMIT=80`)

Prior commit on this branch added structured degraded UI fields; this commit adds the substantive runtime/query fixes.

## Key changes

| Area | Change |
|------|--------|
| `orion/autonomy/repository.py` | Latest-artifact-first drives query; `availability=degraded`; `select_preferred_autonomy_lookup()`; stricter `_drives_facet_ok()` |
| `orion/autonomy/summary.py` | `contextual_fallback` → `stance_mode=fallback_contextual`; missing facet diagnostics treated as `empty` |
| `services/orion-cortex-exec/app/chat_stance.py` | Uses selection helper; merges Orion goals on fallback; chat-stance env split |
| Config | `AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT`, `AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS`, default subquery workers 3 |

## Behavior: Orion drives timeout + relationship drives ok

**Before:** Orion selected (partial `available`), drive fields blank, relationship ignored.

**After:**
- `orion.availability = degraded`
- `selected_subject = relationship`
- `stance_mode = fallback_contextual`
- `dominant_drive` from relationship state
- Orion `proposal_headlines` merged in
- `context_note`: "Orion drives unavailable; stance context from relationship drives (not substituted as Orion drives)"

## Tests

```bash
cd .worktrees/feat-autonomy-compact-degraded-state
PYTHONPATH=. orion_dev/bin/python -m pytest \
  tests/test_autonomy_summary_degraded.py \
  orion/autonomy/tests/test_repository_selection_and_query.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py \
  services/orion-cortex-exec/tests/test_autonomy_repository_graph_diagnostics.py \
  services/orion-cortex-exec/tests/test_autonomy_repository_list_latest_short_circuit.py \
  tests/test_autonomy_summary.py \
  services/orion-hub/tests/test_autonomy_runtime_ui_panel.py -q
# 60 passed
```

## Verification

| Command | Exit | Result |
|---------|------|--------|
| Targeted pytest suite | 0 | 60 passed |

Live Fuseki verification: **UNVERIFIED** — deploy cortex-exec and confirm `orion.drives` returns `status=ok` under load.

## Operator config (after merge)

In `services/orion-cortex-exec/.env`:

```env
AUTONOMY_SUBQUERY_MAX_WORKERS=3
AUTONOMY_CHAT_STANCE_SUBQUERY_MAX_WORKERS=3
AUTONOMY_DRIVES_QUERY_LIMIT=80
AUTONOMY_CHAT_STANCE_DRIVES_QUERY_LIMIT=20
```

Rebuild `orion-cortex-exec` after merge.

## Remaining risks

- If Orion drives query still times out at 20s after SPARQL rewrite, investigate Fuseki data volume/indexing for `self:orion` drive audits
- Quick lane (`chat_quick`) still defaults to identity-only subqueries — drive blanks expected there unless config changed

## Test plan

- [ ] Rebuild cortex-exec + hub
- [ ] Reproduce original scenario (Orion drives timeout under load)
- [ ] Confirm relationship fallback yields real `dominant_drive` in compact card
- [ ] Confirm Orion proposals still visible with contextual fallback
- [ ] Confirm `orion` debug shows `availability: degraded` not `available`
