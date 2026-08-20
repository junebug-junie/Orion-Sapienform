## Summary

`orion-recall`'s own piece of the AI-Town-mirror-table-blind-spot arc: a real, previously-confirmed-but-deferred bug from the earlier `orion-sql-writer` routing-follow-up review (PR #1750, round 1's finding on a different service). `services/orion-recall/app/sql_chat.py`'s `fetch_chat_turn_timestamps` and `fetch_chat_turns_by_id` queried `chat_history_log` and its AI-Town mirror `aitown_chat_history_log` sequentially inside one shared `try/except`, so a mirror-table failure (missing table, permission error, transient fault) discarded an already-successful primary-table result too — silently emptying the whole response instead of degrading to primary-only.

- Each table query is now isolated (a mirror failure can't discard a primary success or vice versa), with a warning-level log on failure so a real query error is no longer silently indistinguishable from "id genuinely not in this table" (this file's own documented 0-row-miss contract).
- **Code review caught a real regression in this branch's own first draft and a real duplication smell — both fixed** (see below).

## Outcome moved

A transient or persistent failure on the AI-Town mirror table (`aitown_chat_history_log`) no longer blanks out real-content chat-turn recall for every id, including ones that live only in the primary table. Previously this failure mode would have silently degraded `fetch_falkor_chatturn_fragments` and RDF chat-turn windowing to "nothing found" with zero error signal beyond a generic empty result.

## Current architecture

`orion-recall` resolves chat-turn text/timestamps by id via two independent Postgres tables (the AI-Town table split, `docs/superpowers/specs/2026-08-19-aitown-table-split-phase2-recall-migration-design.md`), merged in Python with mirror-wins-on-conflict semantics — deliberately not a single `UNION ALL`, since "an id lives in exactly one table" isn't yet a guaranteed invariant of what's live on `main`. Three call sites depend on these two functions: `storage/falkor_chat_adapter.py`, `storage/falkor_neighborhood_adapter.py`, and `worker.py`'s RDF/SQL chat-turn windowing.

## Architecture touched

`services/orion-recall` only. No schema, contract, or config changes.

## Files changed

- `services/orion-recall/app/sql_chat.py`:
  - New `_fetch_primary_and_mirror_rows()`: shared helper (both functions previously duplicated the same scaffold near-verbatim), fetches primary + mirror concurrently via `asyncio.gather(..., return_exceptions=True)`, isolating each table's failure and logging a warning on failure rather than silently returning nothing.
  - `fetch_chat_turn_timestamps` / `fetch_chat_turns_by_id`: both now call the shared helper; both wrap `conn.close()` in its own best-effort `try/except` (a `close()` failure must not discard already-fetched results either).
- `services/orion-recall/tests/test_sql_chat_fetch_by_id.py`: 5 new regression tests — mirror failure preserves primary results (both functions, one covers `fetch_chat_turns_by_id`, one covers `fetch_chat_turn_timestamps`), primary failure preserves mirror results, and connection-close failure still returns already-fetched results.

## Schema / bus / API changes

None. Pure internal reliability fix — both functions' external contract (`Dict[str, ...]`, ids not present anywhere absent from the map) is unchanged.

## Env/config changes

None.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-recall/tests/test_sql_chat_fetch_by_id.py -q
  -> 11 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-recall/tests -q
  -> 272 passed, 3 failed
  The 3 failures are pre-existing and unrelated: confirmed identical
  (same 3 test names) via git-stash comparison against this branch's own
  base, in test_process_recall_active_turn_exclusion.py,
  test_recall_policy_harness.py, and test_recall_vector_amputation.py --
  none touch sql_chat.py or this branch's diff.
```

## Evals run

No eval harness exists for `orion-recall` beyond `tests/`. This is an internal reliability fix to an existing, already-tested code path, not a new capability — judged not to need a new eval harness.

## Docker/build/smoke checks

No runtime/config/Docker-boot-path changes. Not run.

## Review findings fixed

Code review ran once (`Skill("code-review")` on branch `fix/orion-recall-mirror-fallback-swallow`), found 4 findings; 2 fixed directly, 1 picked up for free by the fix for the other, 1 documented as a pre-existing, out-of-scope gap:

- Finding (CONFIRMED, most severe): `finally: await conn.close()` in both functions had no enclosing `except`. `asyncpg`'s own `Connection.close()` can raise on a failed graceful close (transport reset, etc.) — left unguarded, that propagated uncaught out of the `finally` **after** both queries had already succeeded, discarding the already-built result. The exact same failure class this whole patch exists to fix, just moved to the close() point.
  - Fix: `conn.close()` is now wrapped in its own best-effort `try/except` (warning log on failure), never blocking the return of already-fetched results.
  - Evidence: `test_connection_close_failure_still_returns_already_fetched_results` (new).
- Finding (CONFIRMED): the connect/query-isolate/close scaffold (~20 lines) was duplicated near-verbatim across both functions, differing only in table names and log-message prefixes — flagged as a two-site hand-edit hazard, which is exactly the failure mode that produced the original bug (it had to be fixed in two places).
  - Fix: factored into a shared `_fetch_primary_and_mirror_rows()` helper.
  - Evidence: `services/orion-recall/app/sql_chat.py` — both functions now call the same helper.
- Finding (CONFIRMED, pre-existing, not a regression): the two independent table queries were awaited sequentially instead of concurrently, unlike the analogous `worker.py::_compute_entity_relatedness_boost_map`'s existing `asyncio.gather(..., return_exceptions=True)` pattern for the same "N independent lookups, isolate failures" shape.
  - Fix: picked up for free by the shared-helper refactor above — `_fetch_primary_and_mirror_rows()` uses `asyncio.gather`.
  - Evidence: same helper; existing `test_*_queries_both_tables_separately` tests still pass under the concurrent implementation.
- Finding (CONFIRMED, pre-existing, not a regression — documented, not fixed): `worker.py`'s windowing consumers (`_window_rdf_chatturn_candidates`, `_window_sql_chat_candidates`) still can't distinguish "id outside window"/"id not in this table" from "this table's query failed" via the returned map alone. This patch narrows the blast radius (a table failure now only silently drops ids that live solely in that failed table, not the entire map, as it did before this patch), but doesn't close the ambiguity — that would mean changing `fetch_chat_turn_timestamps`'s return contract for every caller, out of proportion for a thin bugfix patch.
  - Not fixed: documented here as a real, known, accepted limitation rather than left unexplained.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `worker.py`'s windowing consumers still can't distinguish a table-query failure from a genuine "not in window"/"not present" result (documented finding above, not fixed here).
- Mitigation: this patch already narrows the blast radius from "whole map silently emptied" to "only the ids that live solely in the failed table silently dropped," plus adds warning-level logs on the query-failure path itself for anyone actually watching logs. A full fix (an explicit partial-failure signal in the return contract) is a real but separately-scoped follow-up.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1757
