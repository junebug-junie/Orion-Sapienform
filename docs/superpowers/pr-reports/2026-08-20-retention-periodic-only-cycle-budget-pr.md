# Retention: one path, and a bound on the whole cycle

## Summary

- **Removes the four synchronous startup retention blocks from `main.py`.** They ran on the
  event loop *ahead of the bus subscription*, could not converge against continuous arrival
  anyway, and covered four of the six managed tables — so two were startup-exempt purely by
  omission. The periodic loop is now the only retention path.
- **Adds a cycle-wide time budget.** `_MAX_ELAPSED_SEC` bounded one table; nothing bounded the
  cycle. Six tables × a 20s per-table cap is a 120s cycle on a 60s timer.
- **Splits that budget as a fair share**, not first-come, so whichever table is earliest in the
  fixed order and has debt cannot starve the ones behind it.
- **Surfaces the cap that actually governed** (`effective_max_elapsed_sec`) and stops
  advertising two startup caps that no longer bind anything.
- **Deliberately does not raise the batch cap**, with the I/O arithmetic written down.
- Boot went from minutes to **4.6s**.

## Outcome moved

Every restart used to stop consuming bus events while retention ran. It no longer does. And
the loop that replaced it now has a bound on the thing that actually matters — the cycle —
rather than only on its parts.

## Current architecture

Retention was startup-only until 2026-08-20, deleting ~365,000 rows per process start against
~1.1M rows/day of arrival. The periodic loop (`app/grammar_retention_loop.py`) was added to fix
that, but the startup pass was left in place beside it. This removes the vestige.

## Architecture touched

`services/orion-sql-writer` only. No new service, no contract change, no bus/schema change.

## Files changed

- `app/main.py`: four startup retention blocks deleted (~100 lines), replaced by a comment
  recording why.
- `app/grammar_truth.py`: `max_cycle_elapsed_sec` on `run_one_retention_cycle`, the
  `eligible` prefilter, fair-share split, `effective_max_elapsed_sec` on
  `GrammarRetentionState`, and the `/grammar/truth` block renames.
- `app/grammar_retention_loop.py`: reads and passes the budget; corrected `0`/negative
  handling; corrected the disabled-loop log line.
- `app/settings.py`, `.env_example`: `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC` (45).
- `tests/test_grammar_retention_periodic.py`: `TestTheCycleBudget`, and the
  startup-split test replaced by one asserting *no* table has a startup block.
- `README.md`.

## Schema / bus / API changes

- Added: `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC`; `effective_max_elapsed_sec` in each
  `/grammar/truth` retention block.
- Renamed (response only): `max_batches_per_startup` →
  `configured_max_batches_unused_startup_default`, `max_elapsed_sec` →
  `configured_max_elapsed_sec_unused_startup_default`. Verified by grep that nothing in the
  repo reads either key; `orion-substrate-runtime` has its own independent `/grammar/truth`.
- Behavior changed: retention no longer runs at startup. For the first cycle interval after a
  restart (~68s) all six tables report `*_retention_not_run` and `/health` reads `degraded`.
  There is no `healthcheck` block in this service's compose file, so nothing restarts on it.

## Env/config changes

- Added keys: `GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC` (default 45).
- `.env_example` updated: yes. Local `.env` synced **by hand** in both the primary checkout and
  this worktree — `scripts/sync_local_env_from_example.py` reads `.env_example` from the
  *primary* checkout, so a key added in a worktree is invisible to it. Confirmed present inside
  the running container via `printenv`, and in the boot log (`max_cycle_sec=45`).

## Tests run

```text
$ pytest services/orion-sql-writer/tests -q
11 failed, 419 passed, 3 skipped, 34 warnings in 11.76s
```

The 11 pre-exist on `main` (`notify_attention_*`, `chat_history_response_identity`,
`journal_entry_payload_boundary`, `biometrics_summary_sql_shape`). No new failures.

Eleven mutations, each failing exactly the intended test:

```text
N1 first-come budget instead of fair share            -> 2 failed
N2 silently skip instead of warning                   -> 1 failed
N3 budget relaxes the per-table cap                   -> 1 failed
N4 re-add a startup retention block to main.py        -> 1 failed
N5 loop stops passing the budget                      -> 1 failed
R1 divisor is all tables, not eligible                -> 1 failed
R2 non-positive budget skips everything (the old bug) -> 2 failed
R3 budget-exhausted warning drops the table name      -> 1 failed
R4 loop restores `or 45.0` (0 silently becomes 45)    -> 2 failed
R5 effective cap not recorded                         -> 1 failed (after adding its test)
```

R5 survived the first pass — the new field had no test — and is the reason a test for it
exists. That is the second time in this session mutation testing caught untested new code
rather than a bad implementation.

## Evals run

No new eval. `services/orion-sql-writer/evals/` gained one in the preceding PR; this patch's
behaviour is timing/budget arithmetic, which the unit tests cover deterministically with a
faked clock. The live half — that the loop is the only retention path and boot no longer
blocks — is evidenced under Docker checks below.

## Docker/build/smoke checks

```text
$ ./scripts/safe_docker_build.sh orion-sql-writer up -d --build
Container orion-athena-sql-writer  Started
```

Boot sequence, `docker logs -t`:

```text
container StartedAt              17:34:15.209
subscribing to channels          17:34:19.840
grammar_retention_loop starting  17:34:19.844   max_cycle_sec=45
Application startup complete     17:34:19.844
```

**4.6s**, and zero `🧹 * retention` startup lines (`grep -c` → 0). The bus subscription is no
longer behind a retention pass.

`GET /grammar/truth`, `grammar_retention`, after the rename and the new field:

```text
configured_max_batches_unused_startup_default    = 100
configured_max_elapsed_sec_unused_startup_default = 120.0
effective_max_elapsed_sec                        = 7.4999942      <-- the cap that governed
elapsed_sec                                      = 1.0281802
batches_attempted                                = 3
capped_by_startup_limit                          = True           <-- batch cap, not elapsed
capped_by_elapsed_limit                          = False
```

That is the MEDIUM finding made visible: the run's real bound was 7.5s while the endpoint used
to report 120.0 beside it.

## Review findings fixed

An adversarial review ran in a subagent against the live service, container logs and Postgres,
and returned 14 findings (4 HIGH). All were real.

- **HIGH — "`0` disables the cycle bound" was false in both directions.** `float(x or 45.0)`
  turns a configured `0` into `45` (the documented opt-out silently did nothing), and a
  negative is truthy so it passed straight through — and a non-positive value reaching
  `run_one_retention_cycle` meant *skip every table forever*, not *no bound*. Symptom would
  have been six WARNING lines a minute reading like a transient squeeze while `grammar_events`
  gained ~795k rows/day.
  - Fix: explicit `<= 0` handling in the loop with its own WARNING, `> 0` guard in
    `run_one_retention_cycle` so non-positive means *no budget*, and the `.env_example` text
    corrected.
  - Evidence: `test_a_non_positive_budget_means_no_bound_not_skip_everything` (parametrized
    0.0 / -1.0) and the parametrized loop test asserting the value passed through. Mutations
    R2 and R4.

- **HIGH — "the only table carrying real debt … versus 0 for every other table" was false.**
  At review time `substrate_proposal_frames` had **109,585 rows** of debt and took **8.21s**,
  against `grammar_events`' 1.08–3.33s. It is **last** in the tuple — precisely the table
  fair-share exists to protect. My claim was true only at the instant I wrote it, and both
  tables were at 0 by evening.
  - Fix: the argument now rests on the fixed *ordering*, not on a transient debt snapshot, in
    all four places it appeared, with an explicit "do not restate this as 'only table X has
    debt'" note and why.

- **HIGH — three places still told operators that disabling the timer falls back to a startup
  pass that no longer exists**, including the WARNING that fires exactly in that case. After
  this patch `GRAMMAR_RETENTION_INTERVAL_SEC=0` means retention *never runs*.
  - Fix: that log line is now `logger.error` and says so; `.env_example` and the README
    corrected.

- **HIGH — an inert assertion.** `assert f"table={table}" in caplog.text or True` is
  unconditionally true, leaving the one thing that test existed to check — that a dropped table
  is *identifiable*, not merely counted — unasserted.
  - Fix: real per-table assertion over an explicit expected list. Mutation R3.

- **MEDIUM — the fair share silently clips the very change the docs recommend.** At ~1.11s per
  batch, raising the batch cap 3→10 needs ~11.1s, above the 7.5s share a 45s budget gives the
  first of six tables — so the documented "~7 hours" was not reachable under the budget shipped
  in the same patch.
  - Fix: said so in `settings.py`, `.env_example` and the README, with the instruction to raise
    `_MAX_CYCLE_SEC` alongside.

- **MEDIUM — `/grammar/truth` advertised caps that govern nothing.** With the startup pass gone,
  `*_MAX_BATCHES_PER_STARTUP` / `*_MAX_ELAPSED_SEC` survive only as default args no production
  caller uses, yet the endpoint presented them as the operative bound.
  - Fix: renamed to `configured_*_unused_startup_default` and added `effective_max_elapsed_sec`.
  - Evidence: live output above; mutation R5.

- **MEDIUM — the `eligible` prefilter was untested.** A regression to
  `len(GRAMMAR_RETENTION_TABLES)` would quietly cut every table's share on a
  partially-configured deployment.
  - Fix: `test_the_divisor_counts_only_tables_with_a_window`. Its first draft used a budget
    where the per-table cap won, making it unable to distinguish a divisor of 2 from 6 —
    corrected to a budget where the share binds, and the trap noted in the test.

- **MEDIUM — a source-text gate test**, the exact idiom this file already documents as defeated
  once by a docstring.
  - Fix: replaced with a behavioural spy asserting the value handed to
    `run_one_retention_cycle`, parametrized over 45 / 12.5 / 0 / −1. It also could not have
    caught the real bug, which was the *value*.

- **MEDIUM — "~26 hours" ignored arrival, and per-table capacity was compared to aggregate
  arrival.** The loop sleeps 60s *after* each cycle, so the period is ~68s (~1,329 cycles/day,
  ~3.99M rows/day, not 4.32M), and ~795k rows/day arrive into `grammar_events`.
  - Fix: **~31 hours**, with the derivation, in all three places. Aggregate arrival corrected to
    ~1.49M/day with the per-table breakdown.

- **MEDIUM — "0.97s of its 20s budget" was one sample, below every subsequent observation.**
  Three consecutive live cycles: 1.08 / 1.41 / 3.33s. At 3.33s against a 7.5s share the headroom
  is 44% used, not ~5% — which is what makes the clipping finding material rather than
  theoretical.
  - Fix: the range, everywhere the single figure appeared.

- **LOW — the `~260s` startup-blocking figure is carried forward** from the previous PR report,
  not re-measured (the old container is gone). Calling it "measured live" overstated the
  provenance.
  - Fix: attributed to its source in both `main.py` and the README, with the 4.6s boot — which
    *is* measured on this branch — stated beside it.

- **LOW — the boot-time degraded window** widened from two tables to six (~68s).
  - Fix: documented in the README. Verified there is no `healthcheck` block in this service's
    compose file, so no restart risk.

- **LOW — `monkeypatch.setattr(grammar_truth.time, "monotonic", …)`** mutates the shared `time`
  module process-wide. No leakage was observed, but it is the wrong seam.
  - Fix: rebinds `grammar_truth.time` to a `SimpleNamespace` instead.

Two findings verified and **not** acted on: cycles cannot overlap (the loop is sleep-then-run
and fully serialized, so the missing interval-vs-budget guard is harmless), and the fair-share
arithmetic is correct on every path — `remaining_tables` is decremented exactly once, including
when `fn` raises, and the divisor is `≥ 1` by construction.

The shutdown worst case (~50s: a 20s cap plus an in-flight DELETE under a 30s statement
timeout, against Docker's 10s default grace) is **pre-existing and unchanged** by this patch —
the budget strictly reduces it for tables 1–5. Not fixed here; recorded below.

## Restart required

Already applied on athena. Elsewhere, from a worktree:

```bash
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

## Risks / concerns

- Severity: medium. Concern: `stop` is only checked *between* tables, never inside
  `_apply_bounded_table_retention`'s batch loop, so shutdown can block ~50s against Docker's
  10s grace and get SIGKILLed. Pre-existing, and this patch improves it for five of six tables.
  Mitigation: none added here; a real fix is a `stop` check inside the batch loop, or a
  `stop_grace_period` in the compose file. Worth its own patch.
- Severity: low. Concern: `/health` reads `degraded` for ~68s after every restart, now across
  six tables instead of two. Mitigation: no healthcheck consumes it; documented.
- Severity: low. Concern: two `/grammar/truth` response keys were renamed. Mitigation: grepped
  the repo for both — no consumers.
- Severity: informational. Concern: `grammar_events` still carries ~3.5M rows of debt and needs
  ~31 hours to converge. That is by design (see the batch-cap note) but is worth re-checking
  once, since the arithmetic assumes arrival stays near 795k/day.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1780
