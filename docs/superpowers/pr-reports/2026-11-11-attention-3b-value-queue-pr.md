## Summary

Attention patch 3b (spec `docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md`, PR #3166; Juniper approved all defaults 2026-11-10), plus a fix for a live regression from 3a that 3b as specified would not have touched.

- **The GPU pool serves its background queue by value.** When two of Orion's own jobs wait for the same seat, the one the market valued higher goes first. Only the background class moves. Juniper's turns, perception, system and urgent work keep their places, and started work is never taken back.
- **Each durable run carries its market net to the pool.** Reading gets a value-only bid at stage 1 (logged, reserves nothing, not obeyed: binding is 3c). Stage 2 reuses stage 1's net.
- **The 5% exploration floor counts acts started, not funded bids** (decided item 4).
- **Fix: Prime was starved of reading concerns.** Since 3a deployed, 37 of 40 after-the-read "look" wants were deferred on agent-lane capacity. A look is not a read: the read already ran and its model time is already charged to the read's act. The look becomes a Prime concern, and Prime thinks on its own lane and bids for itself. A want can now state "this act holds no GPU slot", and reading's looks do.
- Every reorder is logged as a `pool_queue_reordered` event with both leases, both nets and the margin.

## Outcome moved

- **Prime gets reading concerns again.** Live since 3a (2026-11-10 08:42Z to 11-11 01:14Z): 3 funded, 37 deferred, all `capacity_reserved` on `lane:agent` (30) or `lane:agent-gpu2` (7). All 37 had a positive net (min 0.080) and a charge of about 0. The lane check was the only thing deferring them, and a no-slot want skips it. Expected after deploy: about 40 of 40 funded. The week before 3a: 263 of 265 funded.
- **The pool chooses by value.** Replay D (14 days, 2,446 background agent holds): 97 of 872 contested grants change family (11.1%). Reading is served sooner 47 times and later 38 times, so both directions occur. 3a's report had 95 of 863 for the same rule. The numbers now come from the pool's own code (the replay imports `orion/gpu_pool/value_order.py`).
- **The floor measures effort.** Reading holds 46% of Tier 2 acts started this week. Retro leisure holds 0.6% of acts (7 acts), not 73% of bids.

## Current architecture

- `orion/gpu_pool/scheduler.py` `_order`: queued leases sorted by `(priority rank, created_at)`. The pool knew nothing about value.
- Durable runs asked the pool for a hold with no market information. Reading never bid at admission (`run_act_family` returns None for `reading.turn`).
- `ReservationBook.funded_share` counted funded `want_ledger` rows (`FUNDED_SQL`).
- Choice charged every want of a family the family's p90 lane cost and capacity-checked it, including event wants about work already done.

## Architecture touched

- **gpu-pool:**
  - a pure rule, `orion/gpu_pool/value_order.py`;
  - `scheduler._order` (per work class), `queue_position`, and a `Reordered` decision found by counterfactual;
  - runtime `_position`, `_row`, `_view`, `_report_reordered`;
  - 3 lease columns (boot self-heal plus migration v4);
  - 3 settings.
- **Schema:** `GpuLeaseRequestV1` gains `want_id`, `want_net`, `want_family` (optional). `GpuPoolEventV1.event` gains `pool_queue_reordered`.
- **Client:** `orion/gpu_pool/client.py` omits unset value fields, and asks again without them if an old pool refuses them.
- **durable-runs:**
  - `RunBidder` remembers each run's net (`value_for`) and gives reading stage 1 a value-only bid;
  - `register` sends the net on a new hold request;
  - the market seeds the floor from its own act read.
- **physiology:** `choice.acts_started`, `ReservationBook.seed_started` and `appetite.no_slot_cost`/`holds_no_slot`.
- **Hub:** `reading_event_wants.LOOK_COST`.

## Files changed

- `orion/gpu_pool/value_order.py` (new): the value-order rule, shared by the scheduler and replay D.
- `orion/gpu_pool/scheduler.py`:
  - `LeaseView.want_net`/`want_family`;
  - `_order` by value per work class;
  - `queue_position`;
  - `Reordered` (exact: the tick re-scheduled in arrival order).
- `orion/gpu_pool/client.py`: `request_payload` omits unset value fields; `acquire_hold` takes `want_*` and has the old-pool fallback.
- `orion/schemas/gpu_pool.py`: the request fields and the event name.
- `services/orion-gpu-pool/app/{runtime,store,settings,main}.py`, `.env_example`, `docker-compose.yml`, `README.md`: wiring, the columns, the flags (on) and the docs.
- `services/orion-sql-db/manual_migration_gpu_pool_v4_value_order.sql` (new): 3 nullable columns (the pool also adds them at boot).
- `services/orion-gpu-pool/evals/value_queue_replay.py`: imports the pool's rule.
- `services/orion-durable-runs/app/run_bids.py`: `RunValue`, `value_for`, the reading value-only bid (`VALUE_PRODUCER`).
- `services/orion-durable-runs/app/{admission_runtime,pool_hold}.py`: the net rides on a new hold request.
- `services/orion-durable-runs/app/physiology_market.py`: the floor from acts started; the want_ledger floor read is removed.
- `orion/physiology/choice.py`: `acts_started`, `seed_started`, own floor funds kept for the week, and no-slot wants get no lane.
- `orion/physiology/appetite.py`: `NO_SLOT_BASIS`, `no_slot_cost`, `holds_no_slot`.
- `orion/physiology/event_wants.py`: `event_want(cost=...)`.
- `services/orion-hub/scripts/reading_event_wants.py`: `LOOK_COST`.
- `services/orion-durable-runs/evals/{reading_binding_replay,event_wants_replay}.py`: the floor counts acts; reading's looks are no-slot.
- Tests (new):
  - `orion/gpu_pool/tests/test_value_order.py`;
  - `services/orion-gpu-pool/tests/test_runtime_value_order.py`;
  - `services/orion-durable-runs/tests/test_run_value.py`;
  - `orion/physiology/tests/test_floor_acts_started.py`;
  - `orion/physiology/tests/test_no_slot_wants.py`.
- Tests (updated): `test_physiology_bid.py`, `test_prime_charge_market.py`, `test_store_postgres.py`, `test_schema_self_heal_postgres.py`.
- `.github/workflows/orion-durable-runs-tests.yml`: path filters.

## Schema / bus / API changes

- Added:
  - `GpuLeaseRequestV1.want_id`, `want_net`, `want_family` (optional);
  - `GpuPoolEventV1.event = "pool_queue_reordered"`;
  - `gpu_pool_leases.want_id`/`want_net`/`want_family` (nullable).
- Removed: `physiology_market.FUNDED_SQL`, `read_funded`.
- Renamed: none.
- Behavior changed:
  - background grant order;
  - the floor's count;
  - reading's looks are no longer lane-charged or capacity-checked;
  - new `want_ledger` rows from producer `orion-durable-runs.admission.value` (family `hub.world_pulse_read`, one per read).
- Compatibility notes:
  - `GpuLeaseRequestV1` is `extra="forbid"`, but no deploy order is required. The client never sends the new keys unset. A durable-runs that reaches an old pool is refused once (`invalid:`) and asks again without them (tested against the real pydantic text).
  - An old client and a new pool: the fields are optional.
  - sql-writer and mesh-guardian validate `GpuPoolEventV1`. Until they are rebuilt, they drop `pool_queue_reordered` events and nothing else.
  - The `gpu_pool_events.event` column is free text, so no table change is needed.

## Env/config changes

- Added keys (orion-gpu-pool): `GPU_POOL_VALUE_ORDER_ENABLED=true`, `GPU_POOL_VALUE_MARGIN=0.10`, `GPU_POOL_VALUE_AGING_PER_H=0.10`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, `services/orion-gpu-pool/.env_example`, and compose fallbacks too.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. All 3 keys were added to `/mnt/scripts/Orion-Sapienform/services/orion-gpu-pool/.env`.
- Skipped keys requiring operator action: none. (The "diverged" list the script prints is pre-existing and unrelated.)

## Metric quality gate (`want_net` as the pool's queue key)

1. **Provenance:** `orion/physiology/choice.py` `decide` computes `net = weight x value_rank - charge`. It is carried from the decision (`RunBidder._remember`) onto the hold request.
2. **Independence:** this is the market's own existing output, not a new signal. The pool previously read only priority and arrival time.
3. **Theory:** the decided spec (section 1). The market's net is the value of the effort minus its opportunity cost. Serving the higher net first is the queue-order form of the same choice.
4. **Live data:** replay C, 793 real read starts: net p10 0.631, p50 0.798, p90 0.823. It is not flat or saturated, and it varies against other families (self-inquiry 0.54 to 0.73, curiosity 0.65 to 0.68, spec table). Replay D shows changes in both directions.
5. **Existing mechanism:** the market decision. No new metric.
6. **Reversibility:** `GPU_POOL_VALUE_ORDER_ENABLED=false` restores arrival order exactly (tested). The columns are nullable and unused when off.

## Tests run

```text
git diff origin/main --check                                                       clean
pytest orion/gpu_pool/tests                                                        551 passed
pytest orion/physiology/tests                                                      273 passed
services/orion-gpu-pool: pytest tests (scratch Postgres 16, GPU_POOL_TEST_POSTGRES_URI)   216 passed
services/orion-durable-runs: pytest tests (scratch Postgres, ORION_ADMISSION_TEST_DSN)    965 passed, 1 skipped
services/orion-hub: test_reading_event_wants.py test_gpu_pool_routes.py (Postgres)       19 passed
services/orion-field-digester: queue contention + durable waiting (Postgres)              24 passed
pytest orion/grammar/tests                                                         30 passed
Mutation checks:
  value order removed from the grant pass        -> test_value_on_serves_the_higher_want_first_and_says_so fails
  per-class grouping removed                      -> test_an_upkeep_call_of_another_class_... fails
  look without the no-slot cost                   -> test_without_a_stated_cost_... reproduces the live deferral
scripts/check_env_template_parity.py, check_settings_defaults.py orion-gpu-pool --example-drift,
check_metric_lineage.py --gate, check_definition_drift.py --gate, check_env_key_single_source.py,
check_gpu_pool_config.py, check_scripts_dir_no_stdlib_shadow.py, check_migration_required_by.py   all PASS
```

## Evals run

```text
value_queue_replay.py (replay D) fixture: 9/9 PASS
value_queue_replay.py --live --days 14 --aging 0.10: PASS (gate: the granted family changes on > 0 grants)
  2,446 background agent holds; 872 contested grants
  aging 0.10: 97 family changes (11.1% of contested, 4.0% of all)  [3a report: 95 of 863, 11.0%]
  aging 0.00: 126 changes (14.4%)                                  [3a report: 124, 14.4%]
  served sooner: reading stage 1 27, stage 2 20, self-sense 16, curiosity 13, PR notes 12, self-inquiry 9
  wait longer:   self-sense 25, reading stage 1 21, stage 2 17, self-inquiry 15, PR notes 15, curiosity 3
reading_binding_replay.py (replay C) fixture: PASS
reading_binding_replay.py --live --days 14: PASS
  793 read starts (793 with a measured cost); queued_at_pool 506, funded 287; held 0; held > 1 h 0
  reading net p10 0.631 p50 0.798 p90 0.823
  floor's share for reading (acts started, 7 d) 0.461; reading's share of Tier 2 acts started 0.460
  [3a report: 875 starts, held 0, acts-started share 0.459]
run_pool_day_eval.py: PASS
event_wants_replay.py fixture: all checks true
event_wants_replay.py --live --days 14: reading funded 693 of 693 (its looks state no slot again).
  Exits 1 on vision.ask:most_events_low (49.3% of camera asks under 0.5 salience, the check wants
  over 50%). That is about camera asks, not this patch. On main the live lane cannot run at all:
  it asserts "no lanes" and 3a gave reading lanes. This patch fixes the assert.
```

Live counts before the change (want_ledger, producer `orion-hub.reading_events`):

```text
since 3a (11-10 08:42Z .. 11-11 01:14Z): funded 3, deferred 37 (capacity_reserved: lane:agent 30, lane:agent-gpu2 7)
  all 37 net > 0 (min 0.080), mean charge 0.0014
week before 3a (11-03 .. 11-10 08:42Z): funded 263 of 265
```

## Docker/build/smoke checks

```text
No image build run here. No dependency or Dockerfile changes; Python only.
Live path: UNVERIFIED until deploy. Acceptance checks after deploy (read-only):
```

```sql
-- 1. Prime's reading concerns are back: the look split since the deploy (expect ~all funded)
SELECT outcome, reason, count(*) FROM want_ledger
 WHERE producer = 'orion-hub.reading_events' AND decided_at > '<deploy time>' GROUP BY 1, 2;
-- 2. holds carry a net
SELECT want_family, count(*), round(avg(want_net)::numeric, 3) FROM gpu_pool_leases
 WHERE want_net IS NOT NULL GROUP BY 1;
-- 3. the queue really reordered, both directions of family pairs
SELECT detail->>'family' AS served, detail->>'jumped_family' AS waited, count(*) FROM gpu_pool_events
 WHERE event = 'pool_queue_reordered' AND generated_at > '<deploy time>' GROUP BY 1, 2 ORDER BY 3 DESC;
-- 4. reading's value-only bids
SELECT outcome, count(*) FROM want_ledger
 WHERE producer = 'orion-durable-runs.admission.value' AND decided_at > '<deploy time>' GROUP BY 1;
-- 5. no family's p90 background wait rose > 1 h (spec live-week check): compare
--    gpu_pool_events 'granted' waited_ms by holder prefix, 14 days before vs after.
```

```bash
docker logs orion-athena-gpu-pool 2>&1 | grep gpu_pool_queue_reordered | tail
docker logs orion-athena-durable-runs 2>&1 | grep "value_only=True" | tail
```

## Review findings fixed

Code review ran in a subagent on `git diff origin/main...HEAD`. It found 9 issues: 0 blockers, 4 should-fix and 5 nits. All 9 are fixed.

- Finding: one upkeep call of another work class froze the value order for every class, and the queue position a caller saw disagreed with the grant order.
  - Fix: `_order` runs the value sequence per work class; each class keeps its arrival slots.
  - Evidence: `test_an_upkeep_call_of_another_class_does_not_freeze_this_classes_value_order` (it fails with the grouping removed).
- Finding: `pool_queue_reordered` could fire when owner precedence, not value, decided the grant.
  - Fix: the event is now exact. On a tick that granted a background want with a bid, the tick is re-scheduled in arrival order, and a pair is reported only when arrival order would have given that seat to the earlier lease.
  - Evidence: `test_an_owner_granted_first_anyway_is_not_reported_as_a_reorder`.
- Finding: the floor stopped limiting itself for families whose floor funds never become acts (shadow families). Each refresh dropped the funds.
  - Fix: own floor funds are kept for the whole week across seeds.
  - Evidence: `test_own_floor_funds_count_for_the_week_across_seeds_and_funded_bids_do_not`.
- Finding: `__all__` still exported the removed `read_funded`.
  - Fix: removed.
- Finding: reading's value-only bids shared producer and family with Hub's looks in `want_ledger`.
  - Fix: own producer `orion-durable-runs.admission.value`.
  - Evidence: an assert in `test_reading_stage_1_gets_a_value_only_bid_and_stage_2_reuses_its_net`.
- Finding: a run's first hold could wait 5 s for a cold shadow bid.
  - Fix: the wait is at most 1 s (`VALUE_WAIT_SEC`); after that the hold goes out with no net.
- Finding: the old-pool fallback was tested only on hand-written error text.
  - Fix: the test now validates against a real pre-3b schema and returns exactly what the old pool's handler returns.
  - Evidence: the updated `test_an_old_pool_refusing_the_fields_...`.
- Finding: any empty `gpu_slot_s_by_lane` would silently mean "holds no slot". `act_ledger` writes `{}` for acts that held no lease.
  - Fix: the statement now needs the empty map plus `basis.gpu_slot_s_by_lane = "holds_no_slot"` (`appetite.no_slot_cost`).
  - Evidence: a bare `{}` is still estimated (test).
- Finding: `sequence` re-sorted on every pick.
  - Fix: it sorts once.
- Also found and fixed while testing: CI's Postgres store and self-heal tests applied only migrations v1 to v3 and pinned 9 owed columns. They would have failed on v4. Now 13 of 13 pass on Postgres 16.

## Restart required

The post-pull rebuild from the clean primary main checkout covers all of these. No order is required (see compatibility notes).

```bash
scripts/rebuild_services_from_git_diff.sh
```

By hand:

```bash
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

The migration `manual_migration_gpu_pool_v4_value_order.sql` is optional, because the pool adds the 3 columns itself at boot.

## Risks / concerns

- Severity: medium.
  - Concern: reading is about 70% of background agent holds. With value order on, it competes by its net (median 0.80), which is higher than most bidding runs (0.54 to 0.73). Reading may now win more often than it did by arrival.
  - Mitigation:
    - Replay D shows reading both gains (47) and loses (38).
    - The 0.10 margin and 0.10/h aging stop any hold from waiting forever.
    - The spec's live-week check applies: no family's p90 background wait may rise more than 1 h. Query 5 above.
    - Off switch: `GPU_POOL_VALUE_ORDER_ENABLED=false`.
- Severity: low.
  - Concern: a hold that asks the pool before its shadow bid lands (more than 1 s), or after a durable-runs restart, carries no net. It is treated as upkeep and keeps its arrival place.
  - Mitigation: this is conservative. The run is never refused, only not reordered.
- Severity: low.
  - Concern: each read adds one `want_ledger` row under `orion-durable-runs.admission.value`, about 50 a day.
  - Mitigation: it has its own producer. The Prime menu and the event history read only wants with an `event_ref`.
- Severity: low.
  - Concern: the live event-wants replay lane fails on a camera-ask salience check. This is unrelated and pre-existing in spirit; the lane could not run on main at all.
  - Mitigation: reported here and not changed.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
