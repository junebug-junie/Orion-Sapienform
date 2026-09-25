## Summary

- Orion's "am I backed up" number (`queue_contention_score`) only compared how many items were waiting with that queue's own recent average. A queue that stops moving keeps the same length, the average catches up, and the number reads calm. It now also asks how long the item at the front of each queue has waited, compared with a fixed normal wait.
- Three queues, three normal waits taken from live data: reading seeds 48 h, durable GPU demands 12 h, GPU pool queued leases 60 s. The oldest-wait part scores 0 up to 1x the normal wait and 10 at 5x.
- When the oldest-wait part wins, the driver reads `<source>:oldest_wait`.
- A stuck reading-seed queue is reported to Orion as stalled. It is explicitly *not* a reason to hire Cursor, because a stalled reader pipeline isn't busy capacity. Stuck durable and GPU-pool queues keep the existing hire nudge.
- No new `FieldStateV1` field, so no consumer-first migration and no reader breakage. The raw ages show up in a `queue_contention_driver_changed` log line.

## Outcome moved

Before this change, a frozen queue read as calm. On 2026-09-25 there were 143 pending seeds, and the next one in line had waited 138 h, yet the score was 0.11 and sinking. With the real readers run against live Postgres, the new code gives **4.709**, driver `world_pulse_seed_pending:oldest_wait`, and the number keeps climbing while the queue stays stuck. Live after deploy: UNVERIFIED (not deployed).

## Current architecture

`services/orion-field-digester` runs `run_digestion_tick` on every tick (~2 s) and calls `update_queue_contention_pressure`. That function reads three SQL counts from `app/store.py`, scores each against its own EWMA baseline (`orion/field/queue_contention.py`), and writes score, driver and EWMA into `FieldStateV1`. Hub reads score and driver from the latest `substrate_field_state` row (`orion/hub/queue_contention_field_read.py`), and `orion/curiosity/queue_contention_disclosure.py` turns them into one line of hire-teach text.

## Architecture touched

- The digester producer and the pure scoring module.
- The Hub/curiosity disclosure text.
- Digester settings, env and compose.
- No bus, schema or registry contract changes.

## Files changed

- `orion/field/queue_contention.py`: the oldest-wait sub-score, `DEFAULT_*_EXPECTED_WAIT_SEC`, `OLDEST_WAIT_SUFFIX`, `driver_source()`, and `subs`/`oldest_wait_sec` on the reading.
- `services/orion-field-digester/app/store.py`: three oldest-age SQL readers.
  - Seeds use the head of the line in `CLAIM_SQL`'s own order.
  - Durable demands use `min(created_at)` over pending demands.
  - The GPU pool uses queued leases only.
- `services/orion-field-digester/app/digestion/queue_contention.py`: fail-open age readers, threading the ages through, and the driver-change log.
- `services/orion-field-digester/app/tensor/update_rules.py`, `app/worker.py`: wiring.
- `services/orion-field-digester/app/settings.py`, `.env_example`, `docker-compose.yml`: three expected-wait keys.
- `orion/curiosity/queue_contention_disclosure.py`: new text for the oldest-wait drivers, plus the no-hire line for a stuck seed queue.
- `orion/schemas/field_state.py`: comment only. `orion/inner_state_registry.py`: notes only.
- READMEs (digester, curiosity).
- `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`: the full metric gate, re-run.
- Tests: `tests/test_queue_contention.py`, `services/orion-field-digester/tests/test_queue_contention_digestion.py`, `orion/curiosity/tests/test_queue_contention_disclosure.py`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: `queue_contention_score` now also counts how long the oldest item has waited. `queue_contention_driver` can now be `<source>:oldest_wait`.
- Compatibility notes: the `FieldStateV1` shape is unchanged. A Hub that hasn't been redeployed maps the new driver value to its generic text instead of failing.

## Metric gate (full record in the gate doc)

1. **Provenance:** three SQL reads in `store.py`. The age is computed inside Postgres with its own `now()`. orion-gpu-pool writes `created_at` from the same host.
2. **Independence:** this is not a transform of queue length. Live today, length is at 1.05x its average while the head-of-line wait is 2.9x normal. It is also not the same as the pool's own `waited_ms`, which is only recorded when a lease is granted and so can't see a waiter that never gets one.
3. **Theory:** Little's law (items waiting = arrival rate × average wait). The number of waiting items can hold steady while the wait grows without bound once service stops. The oldest waiter's age is a lower bound on the current wait.
4. **Live data, including rest:**
   - Rest point checked by hand: an empty or fresh queue gives exactly 0 (the clip). It is re-read fresh every tick and never decays.
   - Durable replay over 11 days: 90% of samples at 0, never saturated, worst sub 2.9.
   - GPU pool replay over ~10 h: 99.5% at 0, never saturated, worst sub 7.1.
   - Seeds: the queue has never drained since it was created (296 created, 15 ever finished). The live head-of-line wait is 138 h, which scores 4.7.
5. **Existing mechanism:** no age signal exists for these queues.
6. **Reversibility:** cheap. No schema or state added. Setting a source's `*_EXPECTED_WAIT_SEC` very high mutes that source.

The three normal waits are knobs anchored in live data, not findings:
- Seeds: 48 h, the p90 claim wait of the 15 seeds that ever finished.
- Durable demands: 12 h, the p90 time-to-grant across 138 demands.
- GPU pool: 60 s, which reaches 10 at 300 s, the usual lease deadline.

## Env/config changes

- Added keys: `FIELD_QUEUE_CONTENTION_SEED_EXPECTED_WAIT_SEC=172800`, `FIELD_QUEUE_CONTENTION_DURABLE_EXPECTED_WAIT_SEC=43200`, `FIELD_QUEUE_CONTENTION_GPU_POOL_EXPECTED_WAIT_SEC=60`
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-field-digester/.env_example`).
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-field-digester`: yes. All three keys were added to the primary checkout's `services/orion-field-digester/.env`.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest tests/test_queue_contention.py orion/curiosity/tests/test_queue_contention_disclosure.py -q    -> 29 passed
pytest services/orion-field-digester/tests/test_queue_contention_digestion.py -q                      -> 17 passed
pytest services/orion-field-digester/tests -q                                                         -> 250 passed
pytest services/orion-hub/tests/test_turn_orchestrator_role_teach_disclosure.py -q                    -> 14 passed
pytest tests/test_metric_lineage.py tests/scripts/test_substrate_ladder_liveness.py tests/test_agent_trace_schema_registry.py -q -> 79 passed
pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py (with curiosity tests) -> passed
scripts/check_metric_lineage.py --gate      -> PASS
scripts/check_definition_drift.py --gate    -> PASS (no locked definition changed: the lock tracks schema/producers/consumers, not the formula; the meaning change is recorded in the gate doc + registry notes)
scripts/check_inner_state_registry.py       -> OK
scripts/check_env_template_parity.py        -> PASS
other orion-static-gates scripts (stdlib shadow, hostname refs, relative mounts, async routes, chat poachers, control surface parity, system health producers) -> all PASS
```

## Evals run

```text
The digester has no evals/ directory. As a stand-in, I ran a read-only live replay: the new store
readers against live Postgres (127.0.0.1:55432), scored against the persisted EWMA:
  live persisted: 0.114 world_pulse_seed_pending
  new:            4.709 world_pulse_seed_pending:oldest_wait
  subs: seed depth 0.114, seed oldest_wait 4.709, all others 0.0
The historical replays of the durable and GPU-pool age are in the gate doc.
Follow-up: the digester still has no eval harness.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-field-digester/docker-compose.yml config -> renders all three new keys with defaults
Settings() loads the defaults; an env override is honored; a value of 0 is rejected (gt=0)
EXPLAIN on all three queries (pool: live partial index; durable: FIFO partial index; seed: seq scan over 145 rows)
Not built or deployed (per instructions).
```

## Review findings fixed

- Finding (must-fix): the stuck-seed reading would tell Orion to hire Cursor on every turn, and hiring doesn't unstick a stalled reader pipeline.
  - Fix: a stuck seed queue gets a closing line that reports it as stalled with no hire nudge. Durable and GPU-pool oldest-wait keep the nudge.
  - Evidence: `test_stuck_seed_queue_does_not_push_a_hire`, `test_capacity_oldest_wait_drivers_keep_the_hire_nudge`.
- Finding: `min(created_at)` over seeds includes retried and low-priority items that starve by design.
  - Fix: use the head of the line in `CLAIM_SQL` order. The live value dropped from 432 h to 138 h.
  - Evidence: `test_seed_age_order_matches_claim_sql`, live replay.
- Finding: backlogged pool leases may wait up to 24 h by design and would pin a 60 s normal wait.
  - Fix: the age reads queued leases only. Backlogged leases still count toward queue length.
  - Evidence: `test_age_sql_filters_match_the_queue_they_describe`.
- Finding: the new SQL itself was untested, because every test injected lambdas.
  - Fix: a recording-engine test that asserts table, filter and order, plus empty and negative cases.
- Finding: the gate doc overstated its index evidence.
  - Fix: ran EXPLAIN on all three queries and recorded the results.
- Nits:
  - The driver-change log now also fires when the driver clears.
  - `driver_source()` is now used by the disclosure.
  - A test that couldn't fail was fixed.
  - A non-positive expected wait now skips that source instead of crashing the tick.

## Restart required

```bash
scripts/safe_docker_build.sh orion-field-digester up -d --build   # from this worktree after merge, or from an updated main worktree
# optional, so Hub shows the new stuck-seed text (an old Hub falls back to its generic text):
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
  - Concern: while the seed queue stays frozen, its oldest-wait part is the biggest sub-score (4.7 now, about +1.25 per day, reaching 10 around 2026-09-29). Because the score takes the max, smaller durable or GPU-pool readings stay hidden behind it.
  - Mitigation: the stuck-seed text doesn't push a hire. The real fix is the stalled seed pipeline (nothing finished since 2026-09-15). Setting `FIELD_QUEUE_CONTENTION_SEED_EXPECTED_WAIT_SEC` very high mutes only the seed age.
- Severity: low
  - Concern: the seed normal wait (48 h) comes from only 15 finished seeds. The GPU-pool anchor rests on only about 10 h of pool history.
  - Mitigation: both are env knobs. Revisit once the seed queue moves and the pool has a week of data.
- Severity: low
  - Concern: the definition-drift lock doesn't track formula changes, so this meaning change doesn't show up in `_last_change`.
  - Mitigation: recorded in the gate doc and the registry notes.

## PR link

(filled after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
