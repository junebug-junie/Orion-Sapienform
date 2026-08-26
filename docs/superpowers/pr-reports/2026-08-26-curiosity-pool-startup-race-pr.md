# A pool that has not started yet is not an unreadable store

Found in the logs of the first real deploy of PR #1894, 139 milliseconds wide.

## Summary

- The curiosity loop's first tick fires the instant the loop starts; Hub builds
  `app.state.memory_pg_pool` a beat later. The tick loses that race.
- Behaviour was already correct — it self-heals at the next tick and consumes no
  run slot. What was wrong is that it logged the **identical WARNING an
  actually-broken store logs**, on every Hub restart.
- `no_pool` becomes its own state, `stores_not_ready`: INFO on the first
  occurrence, WARNING from the second on with the elapsed time spelled out.
- Same distinction this module already enforces twice over — unreadable is not
  empty, and now not-started-yet is not unreadable.

## Outcome moved

A warning that fires on every restart stops being a warning. This restores
`stores_unavailable` to meaning *"the memory tables could not be read"* — the
thing an operator should act on — while a pool that genuinely never arrives
still escalates to WARNING within one tick interval and can never sit at INFO
forever.

## Current architecture

`CuriosityInvestigation.tick()` read `StudyMaterial.is_unavailable`, which is
`True` both when the asyncpg pool is absent (`unavailable_reason="no_pool"`) and
when a query against the memory tables actually failed
(`"query_failed:<Exception>"`). Both mapped to the single `stores_unavailable`
block reason and the single WARNING that names it.

## The evidence

From `docker logs orion-athena-hub`, first real deploy, 2026-08-26:

```text
06:28:12,036  curiosity_investigation started tick=300.0s cooldown=14400.0s ...
06:28:12,044  curiosity_investigation_blocked reason=stores_unavailable
              detail=no_pool -- check app.state.memory_pg_pool and the two
              memory tables
06:28:12,183  memory_pg_pool_ready dsn_configured=true
```

139 ms. Not a fault, and the operator-facing text (*"check
app.state.memory_pg_pool and the two memory tables"*) sends someone to
investigate a healthy system.

**Why this is worth a patch rather than a shrug.** This repo has already paid
for the failure mode where the signal exists and has stopped meaning anything —
the 21h vision blackout was not silent, it was *unremarkable*. A WARNING
emitted on every single restart trains exactly that.

## Architecture touched

One block reason added; one branch in `tick()`; one counter on the loop. No
contract, schema, bus channel, env key, or capability surface changes.

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: `SignalGateInputs`
  gains `stores_not_ready`; `signal_block_reason` returns it first;
  `tick()` classifies `no_pool` into it, logs INFO on the first and WARNING
  from the second with elapsed time, and resets the counter the moment the
  pool answers.
- `services/orion-hub/tests/test_curiosity_investigation.py`: four new tests;
  two existing ones updated for the new reason (both still pin the real
  property — that a missing pool writes nothing).
- `orion/curiosity/README.md`, `services/orion-hub/README.md`: the gate tables.

## Schema / bus / API changes

None.

## Env/config changes

None. No keys added, removed, or renamed; `.env_example` untouched, so no sync
was required.

## Tests run

```text
pytest services/orion-hub/tests/test_curiosity_investigation.py -q
-> 64 passed
```

New tests, and what each pins:

- `test_a_pool_that_is_not_up_yet_is_not_reported_as_an_unreadable_store` — the
  two states return different reasons.
- `test_no_pool_blocks_as_not_ready_and_a_broken_query_still_blocks_as_unavailable`
  — the fault path is genuinely unchanged, not merely renamed. Asserts both.
- `test_a_pool_absent_for_more_than_one_tick_escalates_to_warning` — asserts the
  actual `levelno` on the emitted records, INFO then WARNING. **This is the one
  that matters**: without it the fix would be indistinguishable from silencing
  the warning, which is the failure it exists to prevent.
- `test_the_counter_resets_once_the_pool_answers` — otherwise one slow start
  leaves every later blip pre-escalated.

Two existing tests updated rather than deleted:
`test_there_is_no_already_studied_gate` (asserts the exact field set of
`SignalGateInputs` on purpose, so a new field must be a deliberate edit) and
`test_a_missing_pool_writes_nothing` (now expects `stores_not_ready`, still
asserts nothing was published — the property it was actually protecting).

## Evals run

```text
No eval harness exists for orion-hub. Not created here -- this is a
log-classification fix with no behavioural surface an eval could measure.
```

## Docker/build/smoke checks

No image change (Python only, no dependency or Dockerfile edit). The live
evidence above came from the already-running deploy; the fix itself needs a Hub
restart to take effect, which is not urgent — the current behaviour is a noisy
log line, not a malfunction.

## Review findings fixed

Self-review only; this is a small follow-up to PR #1894, which had a full
code-review pass. One finding from that self-review is worth recording:

- **Finding:** the first draft classified `no_pool` as not-ready and logged INFO
  unconditionally, which would have made a permanently-broken pool invisible —
  the exact silent-failure shape being fixed, reintroduced by the fix.
  - **Fix:** the consecutive counter and the escalation to WARNING.
  - **Evidence:** `test_a_pool_absent_for_more_than_one_tick_escalates_to_warning`
    asserts the log level changes, not just that a record was emitted.

## Restart required

```bash
cd <a worktree synced to main>
scripts/safe_docker_build.sh orion-hub up -d --force-recreate --no-build hub-app
```

Not urgent: without it the loop behaves identically and simply logs the old
warning on the next restart.

## Risks / concerns

- **Severity: low — the escalation threshold is one tick (300s), not tuned.**
  A pool that takes longer than a tick to build on a slow start would log one
  WARNING before recovering. That is the correct direction to err: a spurious
  warning that resolves is recoverable, a suppressed real one is not.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1896
