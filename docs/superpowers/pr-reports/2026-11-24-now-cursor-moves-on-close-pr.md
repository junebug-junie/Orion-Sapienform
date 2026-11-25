## Summary

- Reading Orion's now no longer moves the reader's place in it. Closing the read does. A read the model
  never used (a GPU refusal, a timeout, a preempted turn) is now offered again on the next read, instead of
  its experiences being skipped forever. Fixes #3498.
- Closing = `POST /temporal-self/now/use` with zero or more cites. It stamps `experience_shown.closed_at`
  (first close wins). The organ's cursor is the newest closed read's cut-off (`max`), so a late close of an
  older read never moves it backwards. A `since=` read still never moves it.
- Every cursor reader closes once its model call ran, even with no cites, and never before: reverie,
  curiosity, dream, play (retro lines). Prime keeps its own `since=` (see below) and now also closes, so its
  reads count honestly.
- The coverage report and the daily `dig organs` roundup now split "read and cited nothing" (closed, no
  cite) from "never closed" (no model call used it).
- One additive migration: `manual_migration_experience_shown_closed_v1.sql` (`ADD COLUMN closed_at`).

## Outcome moved

Failure mode fixed: an unused read skipped its window for that organ forever, and the `experience_shown` row
claimed the experiences were shown when no model saw them. Now an unused read leaves the cursor where it
was, and the coverage metric's "0 cites" splits into its two real causes.

## Current architecture

`GET /temporal-self/now?organ=<id>` without `since=` started at `max(shown_at)` over the organ's
`advances_cursor` rows, so every default read moved the cursor. `POST /temporal-self/now/use` only stored
valid cites. Reverie, curiosity, dream and play posted only when they had cites. Prime already avoided the
cursor: it always sends `since=` taken from the cut-off (`as_of`) of the last now a model call saw, and reuses
an unseen now on a retry.

## Architecture touched

- orion-durable-runs: cursor SQL (`LAST_SHOWN_SQL` now requires `closed_at IS NOT NULL`), close SQL in
  `ExperienceStore.record_cites` (same transaction, row already locked), route docs; dream's persist node.
- orion-thought (reverie), orion-hub (curiosity, retro play lines), orion-mind (Prime): close on a real
  model answer.
- `orion/integration/coverage.py` `NOW_READS_SQL` (new `reads_unclosed` column; `reads_no_cite` now means
  closed with no cite), `scripts/report_integration_coverage.py`, `orion/dig/organs.py`.

### Prime: kept on `since=` (decision)

Prime stays on its own `since=` path. It is already correct: the next read starts at the `as_of` of the last
now the model saw, and a try that never reached the model reuses its unseen now
(`test_the_next_read_starts_where_the_seen_now_ended_and_a_failed_read_skips_nothing`,
`test_a_try_that_never_reached_the_model_reuses_its_now_instead_of_reading_again`). Moving Prime onto the
server cursor would add a dependency on every close succeeding, for no gain. The one change: Prime now also
posts the close with zero cites after the model answered, so its reads are counted as "read, cited nothing"
instead of "never closed". A since= read's close never moves any cursor
(`test_closing_a_since_read_never_moves_the_cursor`). An unparseable or preempted Prime thought never
reaches the post, so it reads as "never closed", which is the truth.

### The bound on re-offer

- An organ that never closed a read sees a rolling last 24 h (`NOW_DEFAULT_LOOKBACK_HOURS`), as before.
- An organ that closed once and then stopped closing sees back to its last close, at most 30 days
  (`NOW_MAX_LOOKBACK_DAYS`, the event retention).
- Either way a read still shows at most 12 items, 2 per organ, in the #3841 fair order (ranking untouched).
- `test_an_organ_that_never_closes_sees_a_rolling_day_not_an_ever_growing_window` pins both.

### Per reader: when the read is closed

| Reader | Closes (model ran) | Leaves open |
|---|---|---|
| reverie | answer persisted or not, any refusal reason, cites or none | `ReverieLlmUnavailable` (refusal, timeout, exec failure) |
| curiosity | turn produced text (cites or none) | turn with no text (existing `_schedule_now_cites` gate) |
| dream | persist node ran (also the run that lost the day) | no answer / not read: the retry takes a fresh read |
| play | delivered (with cites), pass, frames not staged, not delivered | empty generation, yielded mid-turn |
| Prime | answered and applied (cites or none) | no seat, unparseable, preempted |

## Files changed

- `services/orion-sql-db/manual_migration_experience_shown_closed_v1.sql`: new column `closed_at`.
- `services/orion-durable-runs/app/temporal_self_store.py`: cursor reads closed rows only; `record_cites` closes.
- `services/orion-durable-runs/app/main.py`: route docs, bound comment, `closed_at` passed, 503 names the new file.
- `services/orion-durable-runs/app/dream_day_graph.py`, `dream_day_store.py`: dream closes after persist.
- `services/orion-durable-runs/evals/temporal_self_fixture_db.py`: applies the new migration.
- `services/orion-durable-runs/README.md`, `docs/superpowers/specs/2026-11-15-orion-integration-and-a-life-design.md`: contract.
- `services/orion-thought/app/reverie.py`, `temporal_self_cue.py`: reverie closes with no cites too.
- `services/orion-hub/scripts/curiosity_now.py`: posts the close with no cites when the turn has text.
- `services/orion-hub/scripts/retro_commentary.py`: play closes once the model answered.
- `services/orion-mind/app/prime_loop.py`: Prime posts the empty close.
- `orion/integration/coverage.py`, `scripts/report_integration_coverage.py`, `orion/dig/organs.py`: split.
- `orion/schemas/temporal_self.py`: `ExperienceCitesV1` docstring (no field change).
- Tests: `test_experience_now.py`, `test_experience_now_postgres.py`, `test_dream_day.py`,
  `test_reverie_questions.py`, `test_curiosity_now.py`, `test_retro_commentary_service.py`,
  `test_prime_now.py`, `test_organs.py`, `tests/test_report_integration_coverage.py`.

## Schema / bus / API changes

- Added: `experience_shown.closed_at timestamptz` (DB). `NOW_READS_SQL` returns `reads_unclosed`.
- Removed: none.
- Renamed: none.
- Behavior changed: `GET /temporal-self/now` (no `since=`) no longer moves the cursor; `POST
  /temporal-self/now/use` closes the read and moves it. `reads_no_cite` now counts closed reads with no cite.
- Compatibility notes: no pydantic schema change (empty `cites` was already valid), so no registry change.
  Rows written before the migration read as unclosed: each cursor reader's first read after deploy starts
  24 h back once. Durable-runs must not run without the column (the deploy wrapper refuses `up` while the
  migration is unapplied; the post-pull applier applies it first, it is on the additive allowlist).

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no. Local `.env` sync: not needed. Skipped keys: none.

## Tests run

```text
services/orion-durable-runs: pytest tests (ORION_ADMISSION_TEST_DSN on a disposable local postgres:16)  1264 passed, 1 skipped
services/orion-durable-runs: pytest tests (no DSN)        1108 passed, 157 skipped
services/orion-mind: pytest tests                          455 passed
services/orion-thought: pytest tests                       578 passed, 31 skipped
services/orion-hub: pytest tests -k "curiosity_now or retro_commentary or curiosity_investigation"  211 passed
orion/temporal_self orion/dig/tests orion/integration tests/test_check_integration_seam.py tests/test_report_integration_coverage.py  764 passed
scripts/check_integration_seam.py   OK (16 organs)
scripts/check_migration_required_by.py  OK (1 new migration file)
orion.sql_migration_drift.auto_apply_refusals(new file)  []  (auto-appliable)
```

## Evals run

```text
No new eval. The real-Postgres lifecycle test (test_experience_now_postgres.py) is the end-to-end proof:
unclosed read re-offered, empty close moves the cursor, a late close of an older read does not move it back,
and NOW_READS_SQL returns (reads 5, closed-no-cite 2, unclosed 3).
```

## Docker/build/smoke checks

```text
No image build: no dependency, Dockerfile or compose change. Live path UNVERIFIED (not deployed, by instruction).
```

## Review findings fixed

(filled after review)

## Restart required

Apply the migration first (the post-pull rebuild does this on its own; by hand):

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_experience_shown_closed_v1.sql
```

Then rebuild, from the clean primary main checkout after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: after deploy each cursor reader re-reads the last 24 h once (old rows unclosed).
  Mitigation: at most 12 items per read; one-time.
- Severity: low. Concern: a reader that stops closing (a bug) sees a window back to its last close, up to
  30 days, and the same top items each time. Mitigation: visible in the coverage report and dig as
  "never closed".
- Severity: medium. Concern: live proof is pending. Check after deploy: `SELECT reader_organ, count(*)
  FILTER (WHERE closed_at IS NOT NULL), count(*) FILTER (WHERE closed_at IS NULL) FROM experience_shown
  WHERE shown_at > now() - interval '6 hours' GROUP BY 1;` should show closed rows for reverie, curiosity,
  prime; and a reverie refusal in the log followed by a read whose `since` equals the previous one.
- Issues opened: (filled after filing). Issues closed: #3498.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
