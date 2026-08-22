## Summary

- Root-caused and fixed the crystallization-observatory bug where evidence entries duplicate per turn, and where deleting one "duplicate" turn card wipes every duplicate sharing that turn.
- `WindowStore.append_turn` (orion-memory-consolidation) blindly appended a new `turn_correlation_ids` entry every time `orion:memory:turn:persisted` fired again for an already-tracked correlation_id — now it replaces in place and collapses any pre-existing duplicates for that id.
- `build_crystallization_from_window` now dedupes the full evidence list by `(source_kind, source_id)` at window-close time, as a second guard for windows already mid-flight.
- `memory_crystallizations.sql` gets an idempotent one-time cleanup DELETE (gated so it doesn't full-table-scan on every future boot) plus a real unique index on `(crystallization_id, source_kind, source_id)`.
- `insert_crystallization()`'s evidence insert is now `ON CONFLICT ... DO NOTHING` against that index, so a leftover duplicate evidence list can never 500 the propose endpoint.

## Outcome moved

Crystallization proposals will stop showing the same chat turn twice (or the same grammar event twice) in their evidence list. The drop-turn button in the memory-crystallization-observatory UI will delete exactly the row clicked, not every row sharing that turn's `source_id` — because after this fix, at most one row can ever exist per `(crystallization_id, source_kind, source_id)`.

## Current architecture

`orion-sql-writer` writes a chat turn to Postgres and publishes `orion:memory:turn:persisted`. `orion-memory-consolidation`'s `WindowStore.append_turn()` appends each received turn into an open `memory_consolidation_windows.turn_correlation_ids` JSON array. When the window closes, `build_crystallization_from_window()` (shared code in `orion/memory/crystallization/`) turns each tracked turn into one `chat_turn` evidence entry (via `intake_consolidation_window.py`) and looks up each turn's grammar events (`fetch_grammar_evidence_for_window`, one query per turn in the window), then `insert_crystallization()` writes all evidence rows to `memory_crystallization_sources`.

## Architecture touched

- `services/orion-memory-consolidation/app/window_state.py`
- `orion/memory/crystallization/intake_consolidation_window.py`
- `orion/memory/crystallization/repository.py`
- `orion/core/storage/sql/memory_crystallizations.sql`

## Root cause

`orion-sql-writer/app/worker.py` has two structurally independent branches that each publish `orion:memory:turn:persisted` for the same correlation_id: one when writing a `chat.history` envelope, and a separate one (`_maybe_emit_memory_turn_from_row`) when writing the matching `chat.history.message.v1` assistant-role envelope. Neither branch reclassifies or rescopes anything — it's a producer-side double publish, not a legitimate two-phase classify design (an earlier version of this fix's own code comment mischaracterized it that way; caught in review and corrected).

`WindowStore.append_turn()` had no dedup against `turn.correlation_id`, so the second publish appended a second window entry instead of updating the first. That duplicated entry then:
1. Made `fetch_grammar_evidence_for_window()` query the same `grammar_events` trace_id twice (it loops once per turn in the window), doubling every grammar_event evidence row for that window.
2. Made `build_crystallization_from_window()`'s per-turn evidence loop mint two `chat_turn` evidence rows for the same turn.

Confirmed live 2026-08-20 against production Postgres (`conjourney`): 571 duplicate `memory_crystallization_sources` groups — 55 `chat_turn`, 516 `grammar_event` — every duplicate pair sharing one exact insert timestamp, i.e. minted together from one already-duplicated evidence list in a single `insert_crystallization()` call, not from two separate writes. The specific card Juniper reported (`53588b5f-298c-4e4d-be83-7508fd6dac9f`, the "Heck yeah! What else?" stance memory) is one of these 571 groups.

The delete-wipes-all symptom: `crystallization_delete_evidence()` (`services/orion-hub/scripts/crystallization_routes.py`) does `DELETE FROM memory_crystallization_sources WHERE crystallization_id = $1 AND source_id = $2` with no row limit. With duplicate rows sharing that `(crystallization_id, source_id)`, one click removed all of them.

## What's fixed vs. deferred

**Fixed in this PR:** the consumer-side accumulation that turns one double-publish into duplicate persisted evidence, at three independent layers (window append, evidence-build dedup, DB uniqueness + conflict handling).

**Not fixed here, flagged as a follow-up:** the `orion-sql-writer` double-publish itself. Fixing it means picking which of the two emission branches is authoritative and removing the other, which touches the core turn-persistence path other consumers may also depend on — a separate, larger, higher-blast-radius change that deserves its own investigation rather than riding on this bug fix. Until it's fixed, any *other* future consumer of `orion:memory:turn:persisted` needs its own dedup-by-correlation_id, same as this one now has.

## Files changed

- `services/orion-memory-consolidation/app/window_state.py`: `append_turn()` replaces an existing entry for a reclassified correlation_id in place (collapsing ALL pre-existing duplicates for that id, not just the first match) instead of blindly appending
- `orion/memory/crystallization/intake_consolidation_window.py`: `build_crystallization_from_window()` dedupes the full evidence list by `(source_kind, source_id)` after building it, keeping the last (freshest) occurrence; also dedupes `gate.grammar_event_ids` itself
- `orion/core/storage/sql/memory_crystallizations.sql`: one-time cleanup DELETE (guarded on the unique index not existing yet) + `CREATE UNIQUE INDEX idx_mcr_sources_unique ON memory_crystallization_sources (crystallization_id, source_kind, source_id)`
- `orion/memory/crystallization/repository.py`: `insert_crystallization()`'s evidence `INSERT` now `ON CONFLICT (crystallization_id, source_kind, source_id) DO NOTHING`
- `services/orion-memory-consolidation/tests/test_window_state_turn_dedup.py` (new): dedup/collapse regression tests for `append_turn()`
- `services/orion-memory-consolidation/tests/test_intake_consolidation_window.py`: duplicate-turn and duplicate-grammar-event-id regression tests for `build_crystallization_from_window()`
- `tests/test_memory_crystallization_repository_evidence_conflict.py` (new): asserts the evidence insert SQL targets the new unique index with `DO NOTHING`

## Schema / bus / API changes

- Added: `idx_mcr_sources_unique` unique index on `memory_crystallization_sources (crystallization_id, source_kind, source_id)`
- Removed: none
- Renamed: none
- Behavior changed: `insert_crystallization()` silently drops a duplicate evidence row instead of writing it (previously it would write a duplicate; there was no constraint to conflict against)
- Compatibility notes: the cleanup DELETE runs once per database (gated on the index not existing yet), self-applies via the existing `apply_memory_crystallizations_schema()` startup path used by both `orion-hub` and `orion-memory-crystallizer` — no manual migration step needed beyond a normal restart

## Env/config changes

None.

## Tests run

```
$ python -m pytest services/orion-memory-consolidation/tests -q
112 passed

$ python -m pytest tests/test_memory_crystallization_repository_evidence_conflict.py tests/test_memory_crystallization_concept_relation.py tests/test_memory_crystallization_dynamics.py services/orion-hub/tests/test_crystallization_routes_contract.py -q
57 passed

(pre-existing, unrelated failure confirmed present on main before this branch:
tests/test_memory_crystallization.py::TestMemoryCardBackwardCompat::test_memory_card_v1_unchanged_in_registry_gap)
```

## Evals run

No dedicated eval harness for this seam; covered by the unit/regression tests above plus a live migration dry-run (below).

## Docker/build/smoke checks

No new deps, ports, or compose wiring — not applicable. Instead, verified the actual migration against live production Postgres inside a transaction that was rolled back (nothing committed):

```
$ docker exec orion-athena-sql-db psql -U postgres -d conjourney <<'SQL'
BEGIN;
\i memory_crystallizations.sql
SELECT count(*) FROM (SELECT crystallization_id, source_kind, source_id, count(*) c
  FROM memory_crystallization_sources GROUP BY 1,2,3 HAVING count(*) > 1) t;  -- 0 (was 571)
SELECT to_regclass('idx_mcr_sources_unique') IS NOT NULL;                     -- t
ROLLBACK;
SQL
```

Also snapshotted the live table before any dry run: `/tmp/crystallization-evidence-dedup/mcs_before.csv` (4,909 rows).

## Review findings fixed

- Finding: `append_turn()`'s replace loop only fixed the first duplicate for a correlation_id (`break` on first match); a window already carrying legacy duplicates at deploy time would end up with a stale entry positioned after the fresh one, which `build_crystallization_from_window()`'s last-wins dedup would then keep instead of the fresh data.
  - Fix: rewrote the loop to collapse every existing entry for that correlation_id into the fresh one, preserving the first occurrence's position.
  - Evidence: new test `test_reclassified_turn_collapses_preexisting_legacy_duplicates`.
- Finding: the fix's own code comment claimed duplicate publishes come from a legitimate "fast pass then deep reclassification" design; the real producer (`orion-sql-writer/app/worker.py`) has two independent, non-reclassifying emission branches instead — a producer-side double-publish bug, and the consumer-side fix here doesn't cover any other future consumer of the channel.
  - Fix: corrected the comment to name the real producer code paths and explicitly scope this as a consumer-side guard, not a fix for the double-publish itself; documented the gap as a follow-up in this PR description.
  - Evidence: `services/orion-memory-consolidation/app/window_state.py` comment + "What's fixed vs. deferred" section above.
- Finding: the one-time SQL cleanup DELETE ran unconditionally on every service boot forever (no completion marker), taking a full self-join + `RowExclusiveLock` over `memory_crystallization_sources` for a result guaranteed empty after the first run.
  - Fix: wrapped it in `IF to_regclass('idx_mcr_sources_unique') IS NULL THEN ... END IF` — after the first successful run the guard is a cheap catalog lookup, not a table scan.
  - Evidence: dry-run above shows the guard firing correctly (0 dup groups, index present) inside a single rolled-back transaction.
- Finding (PLAUSIBLE, documented not re-engineered): the SQL cleanup's earliest-row tie-break and the application-level last-occurrence tie-break resolve historical duplicates in opposite directions; since duplicate rows share one insert transaction, the SQL side's real tie-break degrades to a random `source_ref_id` UUID.
  - Fix: documented honestly in the migration's own comment and in this PR rather than presenting a fabricated meaningful ordering — no true insertion-order column exists to fix this properly, and the only field it can affect is a display-only `note` string on already-corrupted historical rows (516 of 571 groups are byte-identical `grammar_event` dupes where this doesn't apply at all).
  - Evidence: `orion/core/storage/sql/memory_crystallizations.sql` comment above the `DO $$` block.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-memory-crystallizer/.env \
  -f services/orion-memory-crystallizer/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-memory-consolidation/.env \
  -f services/orion-memory-consolidation/docker-compose.yml \
  up -d --build
```

`orion-hub` or `orion-memory-crystallizer` (whichever restarts first) applies the schema migration — cleaning up the 571 live duplicate groups and creating the unique index — via the existing `apply_memory_crystallizations_schema()` startup call. `orion-memory-consolidation` needs its own restart separately to pick up the `window_state.py` fix.

## Risks / concerns

- Severity: low
- Concern: the migration's DELETE touches production data (571 rows removed).
- Mitigation: gated to run at most once (verified via dry-run), keeps one row per duplicate group rather than deleting a group entirely, full table snapshotted beforehand (`/tmp/crystallization-evidence-dedup/mcs_before.csv`), and the exact statement was dry-run against live Postgres inside a rolled-back transaction before shipping.

- Severity: medium (pre-existing, not introduced by this PR)
- Concern: `orion-sql-writer` will keep double-publishing `orion:memory:turn:persisted` for every chat turn; this PR only guards the one known consumer.
- Mitigation: documented above as a follow-up; any new consumer of that channel should dedup by `correlation_id` until the producer itself is fixed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/crystallization-evidence-dupes

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1772
