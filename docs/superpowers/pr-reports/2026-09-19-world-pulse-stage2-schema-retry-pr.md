# World-pulse reading: Stage 2 stopped rejecting its own good work, and failures can retry

## Summary

- Stage 2's prompt asked the model to "form/test priors, note hops" but the schema only accepted a
  few fields (`extra="forbid"`) — 3 of the last 8 live Stage 2 completions were thrown away for
  doing exactly what they were told. The schema now has the fields the prompt actually asks for
  (`priors_tested`, `candidate_priors`, `concept_candidates`, `open_threads`, `hops`), and the
  prompt was rewritten to list exactly those fields as a JSON skeleton.
- Anything the model still invents outside that list is dropped with a logged warning naming the
  keys (not a silent `extra="ignore"`), so future prompt/schema drift stays visible instead of
  throwing the whole turn away or hiding silently.
- A failed reading turn used to be permanent (`status='failed'`, no way back). Both stages now
  retry a bounded number of times (`HUB_WORLD_PULSE_READ_MAX_ATTEMPTS`, default 3) when the failure
  looks like an infrastructure/turn problem (GPU capacity refusal, timeout, blank response, etc.)
  rather than a bad seed (schema rejection, bad URL, unparseable JSON).
- The `/world-pulse-read/api/status` dashboard now shows how many seeds are waiting on a retry and
  how many burned through their whole retry budget, separately for each stage.
- A new migration adds two small counter columns (`attempts`, `stage2_attempts`); nothing existing
  is renamed or removed.

## Outcome moved

Live on 2026-09-19, Stage 2 had had nothing to read since 2026-09-15 (3 done, 12 permanently
failed, 0 waiting) even though Stage 1 kept running normally — the second-pass loop was starved.
51 of the 54 dead Stage 1 rows and 8 of the 12 dead Stage 2 rows match the transient-failure
pattern this patch retries; they were killed by one recurring GPU-capacity refusal
(`turn_deferred:stance_react_failed: ...`), not by anything wrong with the seeds themselves.

## Current architecture

Stage 1 (`world_pulse_read_pipeline.py`) reads a page into `world_pulse_read_seed.handoff_json`.
Stage 2 (`world_pulse_read_stage2.py`) runs a second turn over that handoff and must return a
`WorldPulseReadStage2ResultV1` (schema in `orion/schemas/world_pulse_read.py`, `extra="forbid"`).
`orion/world_pulse_read/queue.py` holds the claim/mark-done/mark-failed SQL for both stages. Before
this patch, a failed turn set `status`/`stage2_status` to `'failed'` with no way back to `pending`,
and the Stage 2 prompt promised fields the schema didn't accept.

## Architecture touched

- `orion/schemas/world_pulse_read.py` — extended `WorldPulseReadStage2ResultV1`; new
  `WorldPulseReadPriorTestV1` model.
- `orion/schemas/registry.py` — registered `WorldPulseReadPriorTestV1`.
- `orion/world_pulse_read/retry.py` (new) — `is_transient_failure` predicate, `FailureOutcome`.
- `orion/world_pulse_read/queue.py` — retry-aware `mark_seed_failed`/`mark_stage2_failed`
  (now return `FailureOutcome` via `RETURNING` instead of `None`), `count_retry_state`, claim-order
  tiebreak on attempts, inline `WORLD_PULSE_READ_RETRY_SQL` mirror of the new migration.
- `orion/world_pulse_read/journal.py` — journal body now includes `priors_tested`/`open_threads`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py` — new prompt skeleton,
  `_as_stage2_result` unknown-key drop + logging, `max_attempts` wiring, retry-scheduled logging.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py` — same `max_attempts` wiring for
  Stage 1, retry-scheduled logging.
- `services/orion-hub/scripts/world_pulse_read_routes.py` — new `retries` block on
  `/world-pulse-read/api/status`.
- `services/orion-hub/scripts/main.py` — wires `HUB_WORLD_PULSE_READ_MAX_ATTEMPTS` into both
  pipeline constructors.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example` — new
  `HUB_WORLD_PULSE_READ_MAX_ATTEMPTS` (default 3, bounds 1–10).
- `services/orion-sql-db/manual_migration_world_pulse_read_retry_v1.sql` (new) — `attempts`,
  `stage2_attempts` columns.
- Tests: `orion/world_pulse_read/tests/test_retry.py` (new), extended
  `test_world_pulse_read_stage2.py`, `test_world_pulse_read_pipeline.py`,
  `test_world_pulse_read_routes.py`, `test_reading_postgres.py` (new real-Postgres retry lifecycle
  test, and the pre-existing migration test now applies the new migration too),
  `reading_queue_fakes.py` (shared fake now understands the retry-aware failure SQL).

## Files changed

- `orion/schemas/world_pulse_read.py`: new `priors_tested`/`candidate_priors`/`concept_candidates`/`open_threads`/`hops` fields on `WorldPulseReadStage2ResultV1`; new `WorldPulseReadPriorTestV1` model and its coercers.
- `orion/schemas/registry.py`: register `WorldPulseReadPriorTestV1`.
- `orion/world_pulse_read/retry.py`: new — the transient-vs-terminal failure predicate.
- `orion/world_pulse_read/tests/test_retry.py`: new — predicate tested against literal live error strings.
- `orion/world_pulse_read/queue.py`: retry-aware `mark_seed_failed`/`mark_stage2_failed`, `count_retry_state`, claim-order tiebreak, inline retry-migration mirror.
- `orion/world_pulse_read/journal.py`: journal body carries `priors_tested`/`open_threads`.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: prompt/schema agreement, unknown-key drop + log, `max_attempts`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: `max_attempts` wiring, retry logging.
- `services/orion-hub/scripts/world_pulse_read_routes.py`: `retries` status block.
- `services/orion-hub/scripts/main.py`: wire `HUB_WORLD_PULSE_READ_MAX_ATTEMPTS`.
- `services/orion-hub/app/settings.py`: new setting.
- `services/orion-hub/.env_example`: new key + comment.
- `services/orion-sql-db/manual_migration_world_pulse_read_retry_v1.sql`: new migration.
- `services/orion-hub/tests/reading_queue_fakes.py`: shared fake understands the new SQL; removed an earlier `_claim_pending`/`_claim_stage2` override that shadowed per-test-file tracking via MRO (caught by a failing test, not left in).
- `services/orion-hub/tests/test_world_pulse_read_stage2.py`, `test_world_pulse_read_pipeline.py`: schema-agreement and retry-lifecycle tests.
- `services/orion-hub/tests/test_world_pulse_read_routes.py`: `retries` block coverage, new env default.
- `services/orion-hub/tests/test_reading_postgres.py`: real-Postgres retry lifecycle test; migration test applies the new migration.

## Schema / bus / API changes

- Added: `WorldPulseReadStage2ResultV1.candidate_priors`, `.priors_tested`, `.concept_candidates`, `.open_threads`, `.hops` (all defaulted — an old `stage2_result_json` row with only `summary` still validates). New model `WorldPulseReadPriorTestV1`.
- Added: `world_pulse_read_seed.attempts`, `.stage2_attempts` columns (default 0).
- Added: `/world-pulse-read/api/status` response gets a new `retries` object (`max_attempts`, `stage1_pending_retry`, `stage2_pending_retry`, `stage1_exhausted`, `stage2_exhausted`).
- Removed: nothing.
- Renamed: nothing.
- Behavior changed: `mark_seed_failed`/`mark_stage2_failed` now return a `FailureOutcome` (previously `None`) and accept a `max_attempts` keyword (default 1 = old terminal-on-first-failure behavior, so any external caller not yet updated keeps working). A transient failure under the attempt cap now sets status back to `pending` (clearing the claim) instead of `failed`. `_as_stage2_result` now drops unrecognized top-level keys with a logged warning instead of raising (this only fires when the schema itself doesn't recognize the key — validation errors on recognized fields, or non-dict/non-object payloads, still raise exactly as before).
- Compatibility notes: `extra="forbid"` is unchanged on the schema itself — the drop happens in application code before validation, so an unregistered payload shape is still visible in logs, just not fatal to the turn.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_MAX_ATTEMPTS` (default `3`, valid range 1–10; `1` reproduces the old no-retry behavior).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes — confirmed
  `HUB_WORLD_PULSE_READ_MAX_ATTEMPTS=3` landed in the primary checkout's root `.env` and
  `services/orion-hub/.env` (the sync script writes to the primary checkout, not the worktree —
  a known repo behavior, not a new issue).
- skipped keys requiring operator action: none.

## Tests run

```text
# Unit/offline (against the repo venv)
python -m pytest -q orion/world_pulse_read/tests/test_retry.py \
  services/orion-hub/tests/test_world_pulse_read_stage2.py \
  services/orion-hub/tests/test_world_pulse_read_pipeline.py \
  services/orion-hub/tests/test_world_pulse_read_queue.py \
  services/orion-hub/tests/test_world_pulse_read_routes.py \
  services/orion-hub/tests/test_world_pulse_read_backfill.py \
  services/orion-hub/tests/test_reading_ingress.py \
  tests/test_world_pulse_read_schemas.py \
  services/orion-hub/evals/test_reading_handoff_eval.py
-> 191 passed

# Full CI reading-tests workflow (orion-reading-tests.yml), reproduced locally
RUN_READING_POSTGRES=1 python -m pytest -q \
  services/orion-hub/tests/test_reading_ingress.py \
  services/orion-hub/tests/test_reading_postgres.py \
  services/orion-hub/tests/test_world_pulse_read_*.py \
  services/orion-hub/tests/test_curiosity_investigation.py \
  services/orion-hub/tests/test_curiosity_self_inquiry.py \
  services/orion-hub/tests/test_turn_orchestrator_ws_frames.py \
  orion/fcc/tests/test_reading_mcp.py orion/fcc/tests/test_mcp_config.py \
  orion/harness/tests/test_fcc_motor_mcp.py \
  tests/test_world_pulse_read_*.py tests/test_unified_turn_schemas.py \
  tests/test_unified_turn_bus_catalog.py \
  orion/world_pulse_read/tests/test_retry.py
-> 482 passed, 1 skipped

python scripts/check_env_template_parity.py
-> PASS (88 services compared; unrelated pre-existing WARNs on other services, not blocking)
```

`scripts/check_schema_registry.py` and `scripts/check_bus_channels.py` (named in CLAUDE.md section
1a) do not exist in this repo as of this branch — ran the env-parity check that does exist instead.

## Evals run

```text
python -m pytest -q services/orion-hub/evals/test_reading_handoff_eval.py
-> 8 passed (offline behavioral fixtures — this does not measure live model quality; unchanged
   by this patch except for continuing to pass)
```

## Docker/build/smoke checks

```text
bash scripts/safe_docker_build.sh orion-hub config
-> rendered cleanly with the new env key present (HUB_WORLD_PULSE_READ_MAX_ATTEMPTS=3 confirmed
   in the composed config's environment block)
```

No live Docker rebuild/up was run (no runtime dependency, port, or health-check change).

## Review findings fixed

Self-caught during development, plus a code-review subagent pass over the finished diff:

- Finding (self-caught): a shared test fake's `_claim_pending`/`_claim_stage2` override (added
  mid-patch for retry-ordering realism) shadowed each test file's own claim tracking via MRO,
  breaking `test_tick_reclaims_stale_claimed_seed`'s `claimed_ids` assertion.
  - Fix: removed the override from the shared mixin; retry claim-ordering is instead exercised at
    the real SQL level (see the fresh-vs-retried finding below).
  - Evidence: `test_tick_reclaims_stale_claimed_seed` passes; full suite green throughout.
- Finding (self-caught, pre-review cleanup): an early draft of `retry.py` had an unused
  `retry_allowed` helper duplicating the SQL's own transient/attempts-cap logic with no caller — a
  keyword-cathedral risk (CLAUDE.md section 0A). Removed before the review pass ran (the reviewer
  independently confirmed no such function exists on disk).
  - Fix: removed it; `is_transient_failure` (the part that IS used, by `queue.py`'s `fetchrow`
    calls) has its own dedicated test file against literal live error strings.
  - Evidence: `orion/world_pulse_read/tests/test_retry.py`, 4 tests, all passing.
- Finding (MUST-FIX, review): the comment above `WORLD_PULSE_READ_RETRY_SQL` in `queue.py` claimed
  a test asserts it stays in sync with the standalone migration file — no such test existed, so
  the two hand-maintained copies could silently drift.
  - Fix: added `test_retry_migration_matches_inline_sql` (byte-for-byte comparison, ignoring the
    migration file's header comment) and corrected the comment to name it.
  - Evidence: `test_world_pulse_read_queue.py::test_retry_migration_matches_inline_sql` passes.
- Finding (SHOULD-FIX, review): `CLAIM_SQL`/`CLAIM_STAGE2_SQL`'s new `attempts ASC` tiebreak
  (fresh seeds claimed before retried ones at the same priority) is a real, deliberate trade-off
  but was named nowhere in code, and nothing tested the ordering itself.
  - Fix: added a comment on both queries naming the trade-off and pointing at
    `retries.stage1_pending_retry`/`.stage2_pending_retry` as the live signal to watch for real
    starvation; added a real-Postgres test with two same-priority rows (one fresh, one retried)
    asserting claim order for both stages.
  - Evidence: `test_reading_postgres.py::test_claim_prefers_fresh_seed_over_retried_seed_at_same_priority` passes.
- Finding (SHOULD-FIX, review): the "`exhausted` ≠ every terminal failure" distinction was only
  explained in a test comment, not in `queue.py` where a future reader of `count_retry_state` or
  the `/world-pulse-read/api/status` JSON would actually look.
  - Fix: moved/duplicated the explanation onto `COUNT_RETRIES_SQL` and `count_retry_state`'s
    docstring in `queue.py`.
  - Evidence: code inspection; no behavior change, existing tests still pass.
- Finding (NIT, review): `test_main_wires_pipeline_from_settings_opt_in` never asserted
  `max_attempts=settings.HUB_WORLD_PULSE_READ_MAX_ATTEMPTS` was wired for either pipeline, even
  though `main.py` correctly passes it — a future regression dropping it would go uncaught.
  - Fix: added the assertion for both the Stage 1 and Stage 2 constructor call blocks.
  - Evidence: `test_world_pulse_read_routes.py::test_main_wires_pipeline_from_settings_opt_in` passes.
- Finding (NIT, review): no test covered `_failure_outcome`'s fallback branch (RETURNING finds no
  row — e.g. the seed vanished between claim and fail) to confirm it reports terminal rather than
  raising or claiming a phantom retry.
  - Fix: added `test_mark_seed_failed_on_vanished_row_reports_terminal_not_a_crash` and the Stage 2
    sibling.
  - Evidence: both pass in `test_world_pulse_read_queue.py`.

No correctness bugs were found in the SQL, the transient-failure predicate, the schema/prompt
agreement, or backward compatibility — the review traced `is_transient_failure` against every real
error string quoted in the task and confirmed `MARK_FAILED_SQL`'s CASE-expression evaluation order
is off-by-one-free, live against a real disposable Postgres cluster.

## Restart required

```bash
# 1. Apply the new migration (additive, idempotent)
PGPASSWORD=postgres psql -h 127.0.0.1 -p 55432 -U postgres -d conjourney \
  -f services/orion-sql-db/manual_migration_world_pulse_read_retry_v1.sql

# 2. Sync .env_example -> .env on the deploy host if not already done
python scripts/sync_local_env_from_example.py

# 3. Rebuild and restart Hub
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build

# 4. Confirm
curl -fsS http://localhost:8000/world-pulse-read/api/status | python3 -m json.tool
```

## One-off requeue SQL (Juniper: run only after the migration + restart above)

This does NOT touch any row automatically — the patch deliberately leaves existing `failed` rows
alone. Below is the exact SQL to re-queue the ones that died for a transient (turn/infrastructure)
reason, matching `is_transient_failure`'s prefix list. Run the count query first.

```sql
-- Count first (read-only)
SELECT count(*) FROM world_pulse_read_seed
WHERE status = 'failed'
  AND (last_error LIKE 'turn_deferred%' OR last_error LIKE 'turn_error%'
       OR last_error LIKE 'turn_exception%' OR last_error LIKE 'non_final_frame:%'
       OR last_error IN ('stage1_turn_timeout','stage2_turn_timeout','empty_generation',
                          'blank_final_response','looks_like_error_text','no_final_frame',
                          'bus_unavailable','journal_bus_unavailable',
                          'concept_atlas_store_unavailable'));
-- Live 2026-09-19: 51 of 54 failed Stage 1 rows match (3 are JSON-parse failures, left alone).

SELECT count(*) FROM world_pulse_read_seed
WHERE stage2_status = 'failed'
  AND (stage2_error LIKE 'turn_deferred%' OR stage2_error LIKE 'turn_error%'
       OR stage2_error LIKE 'turn_exception%' OR stage2_error LIKE 'non_final_frame:%'
       OR stage2_error IN ('stage1_turn_timeout','stage2_turn_timeout','empty_generation',
                            'blank_final_response','looks_like_error_text','no_final_frame',
                            'bus_unavailable','journal_bus_unavailable',
                            'concept_atlas_store_unavailable'));
-- Live 2026-09-19: 8 of 12 failed Stage 2 rows match (4 are the Bug-1 schema-validation rows,
-- left alone -- those are seed-shaped/schema-shaped failures, not turn failures, and would just
-- fail the same way again since they predate this patch's schema fix... actually they WOULD now
-- validate under the fixed schema if re-run, but this SQL intentionally only requeues
-- turn/infrastructure failures, not schema failures, to keep the two bugs' remediation separate
-- and auditable; a human can separately decide to requeue the 4 schema-shaped rows if desired).

-- Requeue Stage 1 (resets attempts to 0 -- a fresh retry budget going forward)
UPDATE world_pulse_read_seed
SET status = 'pending', attempts = 0, claimed_at = NULL, completed_at = NULL
WHERE status = 'failed'
  AND (last_error LIKE 'turn_deferred%' OR last_error LIKE 'turn_error%'
       OR last_error LIKE 'turn_exception%' OR last_error LIKE 'non_final_frame:%'
       OR last_error IN ('stage1_turn_timeout','stage2_turn_timeout','empty_generation',
                          'blank_final_response','looks_like_error_text','no_final_frame',
                          'bus_unavailable','journal_bus_unavailable',
                          'concept_atlas_store_unavailable'));

-- Requeue Stage 2
UPDATE world_pulse_read_seed
SET stage2_status = 'pending', stage2_attempts = 0, stage2_claimed_at = NULL, stage2_completed_at = NULL
WHERE stage2_status = 'failed'
  AND (stage2_error LIKE 'turn_deferred%' OR stage2_error LIKE 'turn_error%'
       OR stage2_error LIKE 'turn_exception%' OR stage2_error LIKE 'non_final_frame:%'
       OR stage2_error IN ('stage1_turn_timeout','stage2_turn_timeout','empty_generation',
                            'blank_final_response','looks_like_error_text','no_final_frame',
                            'bus_unavailable','journal_bus_unavailable',
                            'concept_atlas_store_unavailable'));
```

## Risks / concerns

- Severity: low. Concern: `CLAIM_SQL`/`CLAIM_STAGE2_SQL` now order by `attempts ASC` at the same
  priority, so a seed that already failed once could in principle wait behind an unbounded stream
  of always-fresh seeds at the same priority. Mitigation: the live queue depths (57 pending Stage 1,
  13 skipped) are small and refill slowly (digest-driven), and `max_attempts` bounds how many times
  this can happen before the row goes terminal anyway; worth revisiting only if a live starvation
  pattern actually shows up in `retries.stage1_pending_retry`/`stage2_pending_retry` staying high.
- Severity: low. Concern: a retry is a real turn — it re-debits the same wallet
  (`debit_wallet_a`/`debit_wallet_b`) and is cooldown-spaced exactly like a fresh claim, since it
  goes through the normal `tick()` claim path. This means a GPU-outage day can burn 3x the normal
  wallet budget on the same handful of seeds before giving up. Named here per the task instructions;
  not mitigated in this patch (the alternative — a separate unmetered retry lane — is a larger
  design change than this bug-fix patch should carry).

## PR link

<opened after push — see final response>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
