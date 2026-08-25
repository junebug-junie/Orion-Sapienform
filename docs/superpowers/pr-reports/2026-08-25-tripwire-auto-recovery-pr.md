# Let the theater tripwire recover

## Summary

- The theater tripwire could stop Orion from acting but nothing could start it again. Live incident 2026-08-23: **45 hours of zero autonomous dispatches**, tripped by ordinary post-redeploy startup wobble, recovered only when a human noticed on the 25th.
- Adds a probe-based re-arm that preserves the original design's stated objection: the latch clears on a run of consecutive successful **probes**, never on a sample.
- Fixes the silence separately and unconditionally: a warning on every blocked frame (persisted), a throttled log line, and hourly re-notification instead of fire-once.
- Funnels all four dispatch-status recording sites through one `_record_dispatch_status` choke point.
- Review returned BLOCKED on two defects; both fixed and both reproduced against the real code.

## Outcome moved

Orion's motor could be silently disabled indefinitely by a transient. Recovery required a human to notice an outage that produced no ongoing signal of any kind. Both properties are gone.

## Current architecture (before this patch)

`services/orion-execution-dispatch-runtime/app/worker.py` pauses dispatch when >5 of the trailing 10 dispatch results are non-`success`. `theater_tripwire_active` had exactly one assignment to `True` (`:973`) and no assignment back to `False` outside `__init__`.

The 2026-08-23 sequence, from live Postgres and container logs:

| time (UTC) | event |
|---|---|
| 06:58 | whole stack redeployed; cortex-exec restarts cold |
| 06:59 | 4 dispatches return `plan_status=partial`, latency 61–91s |
| 07:33 | 6 dispatches return `plan_status=fail`, latency ~11s |
| 07:33 | tripwire latches. One notification. |
| 07:33 → 08-25 05:20 | 0 dispatches. ~1,730 ticks/hour, each logging `motor_budget mode=advisory pace=0.85x`. |

cortex-exec was serving other callers normally throughout.

## Architecture touched

`orion-execution-dispatch-runtime` only. No bus, schema, or cross-service contract change — the frame warning rides on `ExecutionDispatchFrameV1.warnings`, which already exists and is already persisted.

## Files changed

- `app/worker.py`: probe claim/evaluate/backoff/clear state machine; `_record_dispatch_status` choke point; `_abandon_tick_without_sending` shared exit; per-tick blocked-tick recording.
- `app/settings.py`: four operator knobs plus cross-field cooldown validation.
- `.env_example`, `.env` (local, synced by hand — see below), `docker-compose.yml`: the four keys.
- `README.md`: the recovery contract and the incident it exists for.
- `tests/test_tripwire_recovery.py` (new), `tests/test_theater_tripwire.py`, `tests/test_execution_dispatch_runtime_worker.py`.

## Schema / bus / API changes

None. `theater_tripwire_active` remains exposed on `GET /latest`.

## Env/config changes

- Added keys: `ORION_DISPATCH_TRIPWIRE_PROBE_ENABLED`, `ORION_DISPATCH_TRIPWIRE_PROBE_COOLDOWN_SEC`, `ORION_DISPATCH_TRIPWIRE_PROBE_MAX_COOLDOWN_SEC`, `ORION_DISPATCH_TRIPWIRE_REARM_SUCCESSES`
- Removed / renamed: none
- `.env_example` updated: yes
- local `.env` synced: **by hand.** `scripts/sync_local_env_from_example.py` reads `.env_example` from the primary checkout, so keys added in a worktree are invisible to it and it reports success having done nothing. Verified with `grep TRIPWIRE services/orion-execution-dispatch-runtime/.env`.
- **This service uses an explicit compose `environment:` allowlist, not `env_file`.** A key can be present in `.env_example`, `settings.py` and the live `.env` — passing every parity gate — and still not exist in the container. Verified by rendering `docker compose config` and by `docker exec ... env`.

## Design note: why this is not simply reversing the old decision

The manual re-arm was deliberate, and `__init__`'s comment gave the reason: *"a self-clearing tripwire could silently resume sending on a coincidentally-good sample."* That objection is preserved.

Nothing here clears on a sample. The latch opens only after `ORION_DISPATCH_TRIPWIRE_REARM_SUCCESSES` (default 3) **consecutive** probes succeed, each probe being one candidate released on an exponential backoff, with any single failure resetting the run to zero. Probes are judged solely on statuses recorded during their own tick — never on the trailing window, which at probe time still holds the failures that caused the trip.

Bounded cost, asserted rather than described: against a permanently dead motor the backoff settles at one action per `PROBE_MAX_COOLDOWN_SEC`. Weighed against 45 hours of zero actions.

## Tests run

```
pytest services/orion-execution-dispatch-runtime/tests/ \
       tests/test_execution_dispatch_runtime_worker.py \
       tests/test_dispatch_starvation.py -q
136 passed in 5.25s
```

Mutation-tested against the real file (not a synthetic fixture), 9 mutations, **9 caught**:

| mutation | caught by |
|---|---|
| never clear the latch | 8 tests |
| probe success-run reset removed | 2 |
| empty probe counts as success | 1 |
| `all` → `any` in the probe verdict | 1 |
| no backoff growth | 4 |
| drop the frame warning | 2 |
| drop the per-tick status reset | 1 |
| revert `send_budget` in the take-loop | 1 |
| holdback no longer exempts probe ticks | 1 |
| drop refund on the empty-`to_send` path | 1 |
| drop trip-branch counter reset | 1 |
| remove the claim guard | 1 |
| inconclusive probe refunds instead of backing off | 2 |

Two of those survived the first round of fixes and needed better tests, not better code: the trip-branch counter reset is invisible when asserted after a `_clear_tripwire` (which already zeroes the counters), and the claim guard is invisible when asserting only on the frame warning (the abandon path writes it either way). Fixed by seeding the counters directly and by spying on the claim.

## Evals run

No eval harness exists for this service. The recovery contract is deterministic state-machine behaviour, fully covered by the gate tests above; a periodic eval would not measure anything the tests do not. Not claiming eval coverage.

## Docker/build/smoke checks

```
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build   -> built, recreated, started
docker exec orion-athena-execution-dispatch-runtime env | grep TRIPWIRE       -> all 4 keys present
GET /latest -> theater_tripwire_active: False
substrate_dispatch_results, 5 min post-deploy -> success | 75
docker logs --since 5m | grep -ci 'error|traceback' -> 0
```

## Review findings fixed

- **Finding (BLOCKER): all four env keys unreachable in the container; the documented kill switch was dead.**
  - Fix: added to the compose `environment:` allowlist.
  - Evidence: `docker compose config` before showed zero `ORION_DISPATCH_TRIPWIRE_*` keys; `docker exec ... env` after shows all four.
- **Finding (BLOCKER): a successful re-arm was immediately re-tripped in the same tick, firing a false "dispatch resumed" notification.**
  - Fix: `_clear_tripwire` now clears the trailing window. My comment claiming the evaluate-then-check ordering already handled this was wrong.
  - Evidence: reproduced — the default 3-probe run cleared and re-tripped twice before sticking (1800s, 2 false clears). `test_a_clear_survives_the_check_that_runs_immediately_after_it` now asserts exactly one clear and zero re-trips.
- **Finding (should-fix): the refund made the next tick instantly claimable, so the blocked-tick warning was never reached again.**
  - Fix: claim only when something is preparable.
  - Evidence: 6 simulated hours produced 10,650 claim/refund cycles, 150 warnings (first 5 min only), 0 re-notifications.
- **Finding (should-fix): an inconclusive probe was refunded, uncapping the action rate.**
  - Fix: counts as a failed probe and applies the backoff.
  - Evidence: with Postgres down the old path allowed ~1,800 real dispatches/hour vs a documented 1/hour.
- **Finding (should-fix): motor-budget and risk-cap early returns consumed a probe with no refund and no warning.**
  - Fix: all three exits routed through `_abandon_tick_without_sending`.
- **Finding (should-fix): `_send_prepared_candidates` had zero test coverage; 7 mutations to it survived.**
  - Fix: real send-path tests using the existing `_make_worker` fixture.
  - Evidence: 40 → 136 tests; all 9 mutations caught.
- **Finding (note): the idempotency replay branch could vote for its own re-arm on a stored status with no motor contact.**
  - Fix: `_record_dispatch_status(..., live=False)` — counts toward the trailing window, never toward probe evidence.
- **Finding (note): no cross-field cooldown validation; a max below the base ran the backoff backwards.** Fixed with a `model_validator`.
- **Finding (note): `-1` sentinel in a persisted string.** Now renders `unscheduled`.
- **Finding (note): `_tripwire_blocked_warnings` mutated state and sent notifications despite an accessor name.** Renamed `_record_tripwire_blocked_tick`.

## Restart required

Already deployed and verified:

```bash
cd /mnt/scripts/Orion-Sapienform-tripwire-auto-recovery
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
```

## Risks / concerns

- **Severity: low.** Clearing the trailing window on re-arm means the predicate needs `THEATER_TRIPWIRE_WINDOW` fresh samples before it can trip again — roughly two ticks at `max_dispatches_per_tick=5`. Asserted explicitly in `test_re_tripping_after_recovery_works_normally`. Mitigation: the alternative is a re-arm that does not re-arm.
- **Severity: low.** A flapping motor now cycles trip → probes → clear → trip, emitting notifications each cycle, rather than latching once. That is louder, and correct: a flapping motor is worth alerting on. Each cycle needs 3 successful probes at ≥5 min spacing, so it is rate-limited by construction.
- **Severity: note.** `_record_tripwire_blocked_tick` is only reached when a policy frame exists. An upstream stall that produces no frame still leaves no per-tick record. Out of scope here; the in-process latch is no longer the only evidence, which was the goal.

## PR link

<filled on push>
