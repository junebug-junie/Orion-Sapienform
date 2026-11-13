# Rung 6: "one try per day" survives a Hub restart

## Summary

- Orion may propose at most one change to how it scores surprise per domain per day. That limit lived only in the Hub's memory, so every Hub rebuild reset it.
- The limit is now read from the saved proposals, which the Hub loads from Postgres at boot. A restart no longer re-arms it.
- Added a regression test that rebuilds the tuner (as a restart does) and checks the limit still holds. The test fails on the old code and passes on the new code.
- The second requested change, "widen the range of noise levels Orion may try", was measured on live data and **dropped**. It would not change any outcome. Evidence is below.

## Outcome moved

- **Before:** 9 execution proposals between 2026-10-16 21:00Z and 2026-10-18 07:45Z, about 35 hours. The design allows about 2.
- **Cause:** each proposal landed about 4 minutes after a Hub rebuild in `~/.orion/post_pull_rebuild.log`. Example: a rebuild at 01:28 was followed by a proposal at 01:32, and a rebuild at 18:48 by one at 18:52.
- **Why it matters:** retesting nearly the same data many times raises the chance that one trial passes by luck. That would let an unearned change go live.
- **No harm this time:** all 9 proposals were rejected (2 inconclusive, 7 failed), and 0 changes were adopted.
- **After:** at most one proposal per domain per 24h, across restarts. UNVERIFIED live until the next proposal after deploy.

## Why the wider noise-level range was dropped

I scanned live execution readings: 2,886 readings over 72h, with noise floors from 1e-16 to 1e-4.

- The best floor is about 1e-8. It scores the same as 1e-10: +2.37 vs +2.37 nats per reading on the fit span, and +2.12 vs +2.12 on the holdout. Floors above 1e-7 score worse.
- Almost all of the slower setting's advantage comes from **3 single readings** in 72h, worth 2,190, 1,725 and 1,288 nats. The top 1% of readings carry 106% of the total gain. Across every other reading, the two settings tie or the new one slightly loses.
- So the trial's lower bound stays negative (about -0.5) whatever the floor. The trial gate is correct to refuse: the evidence is three moments, not a pattern.
- A wider range would only add candidates that cannot pass. A real change would need more rare events in the window (a longer window), or a scoring rule that fits zero-inflated readings. Scoring is a metric-definition change, so it needs Juniper's approval first.

## Current architecture

`PredictionExpectationTuner.propose()` runs every Hub mutation-scheduler tick (30s). Per domain it checks, in order:

1. An hourly assessment gate (in memory, fine to reset).
2. A 24h proposal spacing gate. This was in memory, which is the bug.
3. The surface's active, queued, and cooldown state (all from the store).

## Architecture touched

Shared package only. No service config, schema, bus, or env change.

## Files changed

- `orion/substrate/mutation_queue.py`: new `SubstrateMutationStore.latest_proposal_at(target_surface)`. It returns the newest proposal `created_at` for a surface, read under the store lock.
- `orion/substrate/prediction_expectation_tuner.py`: the spacing gate reads `store.latest_proposal_at(surface)`. The `_last_proposed` in-process dict is removed.
- `orion/substrate/tests/test_prediction_expectation.py`: new `test_proposal_spacing_survives_a_hub_restart`.

## Schema / bus / API changes

- Added: `SubstrateMutationStore.latest_proposal_at` (Python method).
- Removed: `PredictionExpectationTuner._last_proposed` (private field).
- Renamed: none.
- Behavior changed: the proposal spacing now persists across restarts.
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
# regression test against the OLD tuner (HEAD~ version swapped in):
pytest orion/substrate/tests/test_prediction_expectation.py -k restart   -> 1 failed (expected)
# new code:
pytest orion/substrate/tests/                                             -> 1141 passed, 4 failed, 49 skipped
  (the same 4 felt_state_* failures occur on main; unrelated)
pytest services/orion-hub/tests/test_substrate_mutation_{scheduler_runtime,manual_route_routing,signal_intake}.py -> 38 passed
pytest services/orion-substrate-runtime/tests/test_worker_prediction_expectation.py \
       services/orion-substrate-runtime/evals/test_prediction_expectation_live_check.py -> 6 passed
```

## Evals run

```text
POSTGRES_URI=... python services/orion-substrate-runtime/evals/run_prediction_expectation_live_check.py
  execution: 2901 readings, live = default (alpha 0.2, min_variance 1e-10), assessment no_confident_improvement
  chat: 208 readings, insufficient_readings
  trail: proposals 9, trials failed 7 / inconclusive 2, decisions reject 9, adoptions 0, rollbacks 0
Live floor scan (scratch script, read-only): results in "Why the wider noise-level range was dropped".
```

## Docker/build/smoke checks

```text
Not run: no Dockerfile, compose, dependency or env change. The Hub picks up the shared package on rebuild.
```

## Review findings fixed

Code review ran in a subagent. It found no material findings. It confirmed four things:
- Every proposal row loads at boot. There is no LIMIT, and no compaction or pruning removes proposals.
- `created_at` is always timezone-aware on every producer path.
- No other producer creates proposals on `prediction_expectation:*` surfaces.
- No `_last_proposed` references remain.

- Finding (minor): while spacing suppresses a domain, the hourly gate never fires. So `latest_proposal_at` scanned every 30s tick for 24h.
  - Fix: a suppressed domain also stamps `_last_assessed`, so the check runs at most hourly.
  - Evidence: the spacing and restart tests still pass (see Tests run).
- Noted, not fixed (pre-existing): if the Postgres load fails at boot, the store is degraded and empty. The tuner could then propose once after that restart. The same is true of the store's queue and active-surface guards.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: `latest_proposal_at` scans every stored proposal (about 5,200 rows), at most twice per 30s tick.
- Mitigation: an in-memory scan under the existing lock, which is sub-millisecond scale.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2798

🤖 Generated with [Claude Code](https://claude.com/claude-code)
