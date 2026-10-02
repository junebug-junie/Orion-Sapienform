# Urgent curiosity briefing: read Postgres, not blocked curl URLs

## Summary

- The urgent-run briefing (`orion/curiosity/urgent_prompt.py`) no longer tells Orion to `curl` Hub or GPU-pool URLs. The harness blocks curl/wget in the sandbox, so those five lines were guaranteed dead ends.
- Its sources are now three `psql` queries the read-only role can run, each run verbatim against live Postgres: per-minute temperatures/power/load/fans from `orion_biometrics_summary`, per-minute cabinet AC plug from `home_cooling_sample`, and "who held which GPU card in the last 3 hours, longest first" from `gpu_pool_events`, with the job name for durable-run holders.
- New migration `scripts/sql/2026-10-02_grant_orion_readonly_gpu_pool.sql`: `GRANT SELECT` on `gpu_pool_events`, plus a narrow view `durable_run_workflow` (run_id, workflow, created_at, terminal). `durable_admission_runs` itself stays closed because its `request` holds the free-text brief.
- With no read-only role (`pg_available=False`) the briefing now says plainly that the evidence bundle is all the run has, instead of pointing at HTTP readings.
- Tests: no curl/wget/URL anywhere in the prompt; every table the prompt queries must be granted by a `scripts/sql/*grant_orion_readonly*.sql` file; the view must not expose the brief. A new CI workflow runs these (nothing ran `tests/test_curiosity_urgent_prompt.py` in CI before).

## Outcome moved

Failure mode: urgent run `a153451fe423` (2026-10-01) confirmed "real heat" in 2 minutes, then spent the rest of its budget on five blocked curls and an external scraper. The cause was in Postgres all along. Replaying the new GPU query for that night (window ending 2026-10-01 22:00 UTC) puts `durable-runs:20261001T211035Z-c3e0fb | self_sense_eval | ["gpu1"] | 66.2 min` second from the top. The biometrics query for the same window shows circe gpu1 going 46 C -> 82 C at 21:11, one minute after that grant at 21:10:35.

## Current architecture

- Hub `CuriosityInvestigation.start_urgent` builds the prompt with `build_urgent_prompt(seed, run_id, own_graph, hub_url, graph_enabled, pg_available)`; `pg_available` is Hub's check that the `orion_readonly` role exists.
- The tool section listed four Hub `curl` URLs, the pool's `/v1/pool`, and two `psql` examples.
- `orion_readonly` could read `orion_biometrics_summary` and `home_cooling_sample` (2026-09-28 grant), not `gpu_pool_events` or `durable_admission_runs` (checked live: `permission denied` on both).

## Architecture touched

- Prompt builder only (`orion/curiosity/urgent_prompt.py`); `hub_url`/`pool_url` parameters removed from `build_urgent_prompt`.
- Hub urgent start call site stops passing `hub_url`. `HUB_CURIOSITY_SANDBOX_HUB_URL` stays: the self-directed kickoff and self-inquiry prompts still use it.
- Postgres: one new view and two `SELECT` grants (operator applies).

## Files changed

- `orion/curiosity/urgent_prompt.py`: psql-only sources; GPU holder query; plain no-Postgres wording.
- `scripts/sql/2026-10-02_grant_orion_readonly_gpu_pool.sql`: view + grants, with Undo and Verify.
- `services/orion-hub/scripts/curiosity_investigation.py`: drop `hub_url` from the urgent call; comment.
- `tests/test_curiosity_urgent_prompt.py`: no-HTTP test, grant-parity test, view-exposes-no-brief test, real measurement keys.
- `services/orion-hub/tests/test_curiosity_urgent_start.py`: expected prompt call without `hub_url`.
- `.github/workflows/curiosity-urgent-prompt-tests.yml`: new CI job.
- `orion/curiosity/README.md`, `services/orion-hub/README.md`: document the sources and grant.

## Schema / bus / API changes

- Added: Postgres view `public.durable_run_workflow`; `SELECT` for `orion_readonly` on `gpu_pool_events` and that view.
- Removed: `hub_url`, `pool_url` keyword parameters of `build_urgent_prompt` (one caller, updated).
- Renamed: none.
- Behavior changed: urgent prompt text.
- Compatibility notes: until the migration is applied, the GPU query returns "permission denied"; the prompt tells Orion to say which grant is missing and keep going with the other two.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_curiosity_urgent_prompt.py tests/test_curiosity_incident_report.py -q   -> 44 passed
pytest services/orion-hub/tests/test_curiosity_urgent_start.py -q                         -> 28 passed
(both also run in a fresh venv built from services/orion-hub/tests/requirements-reading.txt, the CI install)
```

Collecting the root and Hub test files in one pytest call errors at collection; same on main, so the workflow runs them as two steps.

## Evals run

```text
Live query check, 2026-10-02 (read-only):
- Every psql query extracted verbatim from the rendered prompt ran as postgres
  (with a temp view standing in for durable_run_workflow): 60 / 60 / 15 rows, 4-10 ms.
- Biometrics and cooling queries ran as orion_readonly (BEGIN READ ONLY; SET LOCAL ROLE): 60 / 60 rows.
- gpu_pool_events and durable_admission_runs as orion_readonly: permission denied (the gap this migration closes).
- EXPLAIN: biometrics uses orion_biometrics_summary_node_ts_idx; cooling uses ix_home_cooling_sample_ts;
  GPU query uses idx_gpu_pool_events_generated + idx_gpu_pool_events_lease (11 ms warm).
- Incident replay (2026-10-01 window): self_sense_eval gpu1 hold 66.2 min found; gpu1 46 C -> 82 C at 21:11.
```

No eval harness exists for urgent-run verdict quality; the live replay above is the evidence.

## Docker/build/smoke checks

```text
Not run: no container, dependency or env change. Hub picks up the new prompt on its next restart.
```

## Review findings fixed

- Finding (should): the GPU query did not count `cancelled` as ending a lease, so 5 of 12 apparent "never ended" leases in 3 days were really cancelled and would rank as long holds.
  - Fix: end events are now `released`, `aborted`, `expired`, `cancelled`; docstring corrected to 7 true ghosts.
  - Evidence: live count of unended grants dropped 12 -> 7; prompt query re-run returns 15 rows, 0 open.
- Finding (should): the view's brief-hiding depends on owner rights; `CREATE OR REPLACE VIEW` cannot narrow columns on a re-run.
  - Fix: `WITH (security_invoker = false)` explicit; header says apply as postgres, and DROP first to narrow. Test asserts the option is present.
  - Evidence: view DDL parsed live as a temp view in a rolled-back transaction.
- Finding (nit): `gpu_pool_events.detail` exposure not documented.
  - Fix: header lists what `detail` holds (worker URLs, model names, swap state; nothing secret).
- Finding (nit): grant-file parser regex too loose (no word boundary, no digits, trailing comments); view test missed a bare `request` column.
  - Fix: `orion_readonly\b`, `[a-z0-9_]+`, trailing `--` stripped; view test asserts the exact column list.
  - Evidence: 44 passed.
- Finding (nit): CI `push` paths narrower than `pull_request` paths.
  - Fix: mirrored.
- Finding (follow-up, out of scope): kickoff and self-inquiry prompts still offer `curl`. Listed under risks.

## Restart required

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < scripts/sql/2026-10-02_grant_orion_readonly_gpu_pool.sql
```

Then, after merge, redeploy Hub from a worktree on main so it builds the new prompt:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: the pool sometimes records no end for a lease (7 in 3 days), which reads as a long open hold. Mitigation: 6-hour grant lookback, and the prompt says an empty `ended_by` may be a lost record and must be checked against temperatures.
- Severity: medium (out of scope). Concern: the self-directed kickoff and self-inquiry prompts (`kickoff_prompt.py`, `self_inquiry_prompt.py`) still offer `curl` to Hub's concept endpoints, which the same hook blocks. Mitigation: follow-up.
- Severity: low. Concern: grant files under `scripts/sql/` are not watched by the unapplied-migration check (it covers `services/orion-sql-db/manual_migration_*.sql`). Mitigation: the prompt's "permission denied" instruction degrades gracefully.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2466
