## Summary

Companion to the `orion-sql-writer` cutover PR (#1743): `orion-recall` needs to read AI-Town content from `aitown_chat_history_log` now that it's been moved out of `chat_history_log`. This is the migration part of Phase 2's own consumer audit -- 7 `orion-recall` files needed a fix, and it turned out to be **4 distinct query shapes across 2 Postgres drivers**, not the "one shared join point" the audit's open question assumed.

- **Id-batch lookups** (`sql_chat.py::fetch_chat_turn_timestamps`/`fetch_chat_turns_by_id`): two separate queries + Python merge, mirror table wins on conflict. Correct regardless of whether the sql-writer cutover has merged yet.
- **Recency/search scans** (`sql_timeline.py`'s 3 functions, `sql_adapter.py`, `sql_chat.py::fetch_chat_history_pairs`): `UNION ALL`, with an honestly-documented, bounded, self-resolving duplicate-row risk while Phase 1's dual-write is still the live write path (accepted, not eliminated -- the original design doc's own recommendation for this shape).
- **Correlation-id -> platform resolution** (2 crystallization-gate scripts): extended `LEFT JOIN` + `COALESCE`.
- **Unfiltered snapshot** (1 backfill script): second query, concatenated.
- **A real, previously-undiscovered production bug found and fixed while touching this code**: `storage/sql_adapter.py::fetch_sql_fragments`'s chat branch referenced a `trace_id` column that does not exist on `chat_history_log` at all -- confirmed live, this branch had been silently contributing **zero chat fragments**, unconditionally, the whole time (swallowed by a bare `except Exception`). Fixed by selecting the real `id` column instead.
- **Code review (2 rounds, deep -- 8 finder angles) caught a real, confirmed staleness gap**: comments asserting "an id lives in exactly one table" as settled fact, when that's only true once the sql-writer cutover PR (#1743) actually merges. Fixed by making the id-batch-lookup code correct by construction (two-query merge) rather than relying on the invariant, and by making every comment honest about current-vs-pending state.

## Outcome moved

`orion-recall` can see AI-Town content again, live-verified across all 4 shapes. The `trace_id` bug fix alone took `fetch_sql_fragments`'s chat-fragment output from a silent, permanent zero to 300 real fragments (183 of them correctly platform-tagged `aitown`).

## Current architecture

`orion-recall` reads `chat_history_log` via two different drivers: `asyncpg` (`sql_chat.py`) and a synchronous `psycopg2`-style cursor wrapped in `asyncio.to_thread` (`sql_timeline.py`, `storage/sql_adapter.py`), the latter with dynamic per-call column introspection (`_pick_id_col`/etc., since `RECALL_SQL_TIMELINE_TABLE` can point at an unrelated table like `collapse_mirror`).

## Files changed

- `services/orion-recall/app/settings.py`: new `RECALL_SQL_AITOWN_CHAT_TABLE` (default `aitown_chat_history_log`). Review-fixed comment to state the actual current-vs-pending merge state instead of an unconditional invariant.
- `services/orion-recall/app/sql_chat.py`: `fetch_chat_turn_timestamps`/`fetch_chat_turns_by_id` rewritten to two-query-merge (mirror wins); `fetch_chat_history_pairs` gained real `id`/`correlation_id`/per-branch `source_ref` (previously always mislabeled as the primary table, a pre-existing gap this file's own row-id fallback never actually reached).
- `services/orion-recall/app/sql_timeline.py`: `fetch_recent_fragments`/`fetch_related_by_entities`/`fetch_exact_fragments` union the aitown table in, with per-branch `source_ref` literals. Review-fixed: two duplicated param-building loops replaced with list reuse; comments corrected.
- `services/orion-recall/app/storage/sql_adapter.py`: same union, plus the `trace_id` bug fix. Comment corrected.
- `services/orion-recall/.env_example`: `RECALL_SQL_AITOWN_CHAT_TABLE=aitown_chat_history_log`.
- `scripts/smoke_aitown_crystallization_gate.py` (and `scripts/bulk_reject_aitown_proposals.py`, which imports its `QUERY`): extended `LEFT JOIN` + `COALESCE`. Comment corrected to describe the real live-data state (moved via a separate backfill script) vs. the pending code-merge state.
- `scripts/backfill_recall_falkor_chat_tags_snapshot.py`: unions the aitown table into its full snapshot. Comment corrected.
- `docs/superpowers/specs/2026-08-19-aitown-table-split-phase2-recall-migration-design.md`: correction note added -- the original draft's "dual-write transition window" framing was superseded same day by the cutover decision; the Shape 1/2/3 recommendations still stand, the reasoning for them changed.
- Tests: `test_sql_chat_fetch_by_id.py` (new mirror-wins-on-conflict test, two-query-not-UNION assertions), `test_sql_chat_self_hit_suppression.py` (new union-assertion test for `fetch_chat_history_pairs`).

## Schema / bus / API changes

None -- `aitown_chat_history_log` already exists (PR #1734/#1743).

## Env/config changes

- Added keys: `RECALL_SQL_AITOWN_CHAT_TABLE=aitown_chat_history_log`.
- `.env_example` updated: yes.
- local `.env` synced: yes (was already present from the design-doc PR's earlier prep).

## Tests run

```
.venv/bin/python3 -m pytest services/orion-recall/tests/ -q
  → 269 passed, 3 failed (pre-existing, confirmed identical on unmodified main), 13 warnings
```

## Evals run

No eval harness exists for this pipeline.

## Docker/build/smoke checks

Rebuilt and redeployed `orion-athena-recall` live 3 times across two review rounds, against real Postgres with real migrated data:

```
fetch_chat_turns_by_id([aitown_id])       -> found, correct client_meta
fetch_chat_turn_timestamps([aitown_id])   -> correct epoch timestamp
fetch_related_by_entities(['shadows'])    -> 10/10 hits from aitown_chat_history_log,
                                              correctly [ai-town]-labeled text
fetch_sql_fragments(include_chat=True)    -> 300 chat fragments (was silently 0 before
                                              the trace_id fix), 183 aitown-tagged
fetch_chat_history_pairs(limit=2000)      -> 1,577 aitown-sourced pairs, real ids,
                                              correct source_ref
```

## Review findings fixed

- Finding (CONFIRMED by 3+ independent review angles): comments asserted "an id lives in exactly one table" as settled fact, describing a state that depends on the separate sql-writer cutover PR merging first -- not yet true of what's on `main`.
  - Fix: id-batch lookups made correct by construction (two-query merge, mirror wins) instead of relying on the invariant; every affected comment corrected to state the real current-vs-pending merge state, with an explicit note in each that the sql-writer cutover PR (#1743) is a separate, not-yet-merged dependency.
  - Evidence: `test_mirror_table_wins_on_id_conflict` (new); live re-verification against real Postgres after the rewrite.
- Finding: `fetch_chat_history_pairs` never selected `id`/`correlation_id` at all (a pre-existing gap), and its `source_ref` was hardcoded to the primary table regardless of which table a row actually came from.
  - Fix: both columns now genuinely selected, `source_ref` is a real per-branch literal.
  - Evidence: live query showing 1,577 correctly-labeled `aitown_chat_history_log`-sourced pairs with real ids.
- Finding: `fetch_exact_fragments`/`fetch_related_by_entities` each hand-duplicated a param-building loop/list instead of reusing the already-built one.
  - Fix: both now derive the second branch's params from the first (`params.extend(params)`, `branch_params + branch_params`).
- Not fixed, noted as an accepted, out-of-scope risk: no existence check for `aitown_chat_history_log` -- if this deployment's table were ever missing (a fresh/different Postgres, an environment that never ran the migration), the combined `UNION ALL` queries would raise and the broad `except Exception` would degrade the WHOLE query (primary table included) to empty, not just the aitown half. Consistent with every other `manual_migration_*.sql`-gated feature in this repo (none have this kind of graceful degradation either); this deployment does have the table (verified).
- Not fixed, noted: 6+ near-identical `UNION ALL` query bodies across `sql_chat.py`/`sql_timeline.py`/`sql_adapter.py` share no query-builder helper. A real refactor opportunity, deferred given the genuinely different shapes (async vs. sync driver, dynamic vs. static columns, id-batch vs. content-search) would make a fully general helper non-trivial.

## Restart required

Already done live on this session's Athena host as part of verification -- `orion-athena-recall` is running the new image now.

## Risks / concerns

- Severity: low
- Concern: this PR should merge together with or after #1743 (the sql-writer cutover). Before that, the code here is still correct (it doesn't assume the invariant, it queries both tables regardless), but the comments' "current state" framing would be describing a still-pending dependency.
- Mitigation: explicit note added to both PR descriptions.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1744
