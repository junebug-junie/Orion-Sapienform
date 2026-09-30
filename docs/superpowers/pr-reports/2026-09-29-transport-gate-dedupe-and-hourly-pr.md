# fix(transport): one trigger per real RPC timeout + durable hourly readings for the baseline gate's log-only week

## Summary

- **Timeouts were counted twice.** One slow LLM-gateway call that timed out produced two transport rows: one from the per-call timeout marker (the `rpc_transport_timeout` grammar atom), and one from a pooled "some call timed out in this 30-second window" count. The pooled one is removed outright. It was a coarser copy of the marker. The marker is now the single owner of timeouts while the per-hop baseline gate is log-only.
- **When the gate starts publishing** (`EQUILIBRIUM_TRANSPORT_BASELINE_EMIT=true`), a timeout the gate saw in a service's snapshot belongs to the gate's episode, and the marker for that same timeout is dropped. The marker does not say which service sent it. So ownership is decided by evidence, count for count, per request channel and time window, not from a list of "covered" services. A timeout nobody's snapshot claims still fires after 75 seconds. Coverage can only fail open.
- **The gate's readings now survive restarts.** Once an hour, each hop gets a summary row (`TransportBaselineHourlyV1`). It goes over the bus to sql-writer and lands in the new `transport_baseline_hourly` table. Before this, the readings lived only in container log lines, which every restart erased, so the log-only week could not be graded.
- **`scripts/analysis/grade_transport_baseline.py`** grades spec acceptance check 1 from that table. For each hop it gives a plain-English verdict, and it prints a per-day table of what the gate would have published.

## Outcome moved

- **Double counting is gone.** On 2026-09-29, timeout-driven transport rows were 289 pooled plus 222 marker, 511 in total. Replaying the same day's 866 real timeouts through the transport cooldown lane, with the pooled source removed, gives **408**, each from a distinct timeout. For 2026-09-28 the drop is 411 to **328**. All transport rows together: 562 to 459 (09-29) and 475 to 392 (09-28).
- **Acceptance check 1 can now be graded.** Before, its evidence was erased on every deploy.

## Live evidence for the dedupe decision (read-only Postgres, 2026-09-29)

- **Code proof.**
  - `RpcHealthAggregator.record_timeout()` is the only thing that bumps the pooled `timeout_count`.
  - `rpc_request()` pairs it with `_emit_rpc_timeout_grammar()` at both call sites (`orion/core/bus/async_service.py:693/696` and `762/765`).
  - Hand-rolled hops (`record_hop_timeout`) never touch the pooled count.
  - So every pooled timeout has a marker.
- **Timing correlation.** Over the last 48 h, the pooled branch fired 528 rows covering 677 timeouts. In each of those windows, `[window_start - 1 s, window_end + 3 s]`, `grammar_atoms` holds a matching `rpc_transport_timeout` for **675 of the 677**.
  - Of the 2 misses, one marker landed 3.1 s after the window closed (writer lag). The other was the newest row, most likely not yet written when the query ran.
  - Channels of the matched markers: LLMGatewayService 746, RecallService 21, gpu_pool lease 7, others 6.
- **Which is better evidence.** The marker is per call, carries the request channel and correlation id, and fires from every service. The pooled count is one number per 30 s window, only from cortex-exec/cortex-orch, with no channel. The marker was kept and the pooled branch killed.
- **7-day volume by day.**
  - Raw markers are only retained from 2026-09-26 23:10, so the replay covers 09-27 to 09-29 only.
  - The replay uses a 30 s lane simulation over raw markers plus the bus_synaptic rows actually published.

  | day | pooled | marker rows | bus_synaptic | total now | raw timeouts | replay: marker | replay: total |
  |---|---|---|---|---|---|---|---|
  | 09-27 | 8 | 25 | 84 | 117 | 35 | 26 | 110 |
  | 09-28 | 239 | 172 | 64 | 475 | 740 | 328 | 392 |
  | 09-29 | 289 | 222 | 51 | 562 | 866 | 408 | 459 |

- **Estimate for when EMIT flips: UNVERIFIED.**
  - Per-hop snapshot history was never stored, so the gate's own claims cannot be replayed. That is what this PR's hourly table fixes going forward.
  - Lower bound: 780 of the 1,606 markers in the last 48 h fell inside cortex-exec windows alone. At least those would collapse into gate episodes (open, escalate, close) instead of one row per timeout.
  - Live sample on 2026-09-29: 14 publisher identities send snapshots, including hub, durable-runs, thought, harness-governor and gpu-pool, so real coverage is wider.

## Current architecture

- **Pooled source.** Equilibrium fired transport from the pooled rpc_health `timeout_count` (cortex-exec/cortex-orch only, while EMIT was off).
- **Marker source.** It also fired from every `rpc_transport_timeout` marker.
- **Shared lane.** Both shared one 30 s transport cooldown lane with bus_synaptic.
- **Gate readings.** The baseline gate wrote per-window readings only to container logs.

## Architecture touched

- **orion-equilibrium-service:**
  - the pooled branch is removed
  - new `app/transport_timeout_owner.py`, a pure reconciler, active only while EMIT is effective
  - new `app/transport_baseline_hourly.py`, a pure accumulator
  - a housekeeping loop (every 10 s) and a best-effort shutdown flush in `service.py`
- **Contract:** `TransportBaselineHourlyV1` and `orion:equilibrium:transport_baseline:hourly`.
- **orion-sql-writer:**
  - new insert-only table `transport_baseline_hourly`, created by `Base.metadata.create_all` at boot, so no manual migration
  - route, plus the subscribe guarantee for stale operator `.env` lists
- **scripts/analysis:** the grader.

## Files changed

- `services/orion-equilibrium-service/app/transport_metacog_gate.py`: pooled builder removed; tombstone comment explains why.
- `services/orion-equilibrium-service/app/transport_timeout_owner.py` (new): marker/gate ownership, count-conserving, fails open.
- `services/orion-equilibrium-service/app/transport_baseline_hourly.py` (new): hourly buckets.
- `services/orion-equilibrium-service/app/service.py`:
  - pooled branch removed
  - `_handle_rpc_timeout_atom`
  - credits and hourly observe on each folded snapshot
  - housekeeping loop and shutdown flush
  - hourly publish with a bounded retry outbox
- `services/orion-equilibrium-service/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three new keys; docs.
- `services/orion-equilibrium-service/tests/`:
  - `test_transport_timeout_dedupe.py` (new, 15 tests)
  - `test_transport_baseline_hourly.py` (new, 8 tests)
  - legacy tests in `test_transport_metacog_gate.py` and `test_transport_baseline_gate.py` rewritten
- `orion/schemas/telemetry/transport_baseline_hourly.py` (new), `orion/schemas/registry.py` (both maps), `orion/bus/channels.yaml`.
- `services/orion-sql-writer/app/models/transport_baseline_hourly.py` (new), `models/__init__.py`, `worker.py`, `settings.py`, `.env_example`, `README.md`, `tests/test_transport_baseline_hourly_sql_shape.py` (new).
- `orion/metacog/tests/test_evidence_map_producer_contract.py`: asserts the pooled builder is gone, and that historical pooled rows still map.
- `scripts/analysis/grade_transport_baseline.py` and `scripts/analysis/tests/test_grade_transport_baseline.py` (new).
- `config/metrics/metric_definitions.lock.json`: re-locked for the one added channel.

## Schema / bus / API changes

- **Added:**
  - `TransportBaselineHourlyV1` (kind `transport_baseline.hourly.v1`, `extra="forbid"`), registered in `_REGISTRY` and `SCHEMA_REGISTRY`, checked with `resolve()`
  - channel `orion:equilibrium:transport_baseline:hourly` (producer equilibrium, consumer sql-writer)
- **Removed:** transport triggers with `upstream.evidence_source="rpc_health_snapshot"` are no longer produced. The mapper still reads historical rows (`orion/metacog/evidence_map.py` unchanged).
- **Behavior changed:**
  - EMIT off: timeouts come only from the marker.
  - EMIT on: a marker claimed by a gate window is dropped (log line `transport_timeout_owner owner=baseline_gate`); an unclaimed marker fires after the grace period (`owner=atom reason=no_gate_window`).
- **Compatibility / rollout order (consumer first):** deploy sql-writer before equilibrium. Until then, hourly rows land in sql-writer's fallback log, not the table.

## Env/config changes

- **Added keys (equilibrium):**
  - `EQUILIBRIUM_TRANSPORT_TIMEOUT_ATOM_GRACE_SEC=75`
  - `EQUILIBRIUM_TRANSPORT_BASELINE_HOURLY_PUBLISH_ENABLE=true`
  - `CHANNEL_TRANSPORT_BASELINE_HOURLY=orion:equilibrium:transport_baseline:hourly`
- **sql-writer:** the route map JSON and the subscribe channel list gained entries.
- **`.env_example` updated:** yes, both services. Compose and README updated too.
- **Local `.env` synced:** yes. I ran `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service orion-sql-writer` from the worktree.
  - Equilibrium: the 3 keys were added.
  - The script also re-added 6 bus_synaptic/FalkorDB keys (`EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_*`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`). They had been removed from the local `.env` during this session, most likely by the parallel bus_synaptic work, so **I reverted that part**. The local `.env` is now its pre-sync content plus my 3 keys; a backup is in the session scratchpad.
- **Diverged, handled by hand:** `orion-sql-writer`'s `SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON`. The local values differ from the template, so the sync script did not touch them (and `--force` would flatten secrets). I added only the new entry to each local JSON value; a backup is in the session scratchpad. The code also merges default routes and forces the subscription, so this was belt and braces: verified in the built image.

## Metric quality gate (hourly summary)

This PR adds no new signal. The rows persist values the gate already computes (z, ratio, floor, calls). The gate itself was audited in PR #2310.

1. **Provenance:** `fold_snapshot` → `KeyObservation` / `TransportConditionEvent` (`orion/metacog/transport_baseline.py:829`) → `TransportBaselineHourly.observe`.
2. **Independence:** these are summaries of one existing sensor, not new signals.
3. **Theory anchor:** spec acceptance check 1, exactly as written.
4. **Live data:** UNVERIFIED until deployed. Nothing is stored yet, which is the point of this PR. The grader says `UNVERIFIED` and exits 3 on an empty table.
5. **Existing mechanism:** none. The only prior home was the log lines.
6. **Reversibility:** set the flag off; drop the table.

## Tests run

```text
pytest services/orion-equilibrium-service/tests services/orion-equilibrium-service/evals   -> 207 passed
pytest orion/metacog/tests orion/schemas/tests/test_metacog_entry.py                      -> 222 passed
python orion/metacog/evals/run_capture_eval.py                                            -> PASS
cortex-exec metacog lanes (4 files from metacog-capture-tests.yml)                        -> 40 passed
pytest scripts/analysis/tests/test_grade_transport_baseline.py                            -> 15 passed
pytest services/orion-sql-writer/tests                                                    -> 682 passed, 12 failed
    (the same 12 fail on untouched origin/main 781f01c12: 677 passed, 12 failed; +5 = new tests)
Static gates (every run step of orion-static-gates.yml, locally)                          -> all pass
    check_definition_drift --gate failed on the new channel (a real change) -> --update -> PASS
Mutation checks on transport_timeout_owner.py:
    never take a credit        -> 6 tests fail
    never fire unclaimed atom  -> 4 tests fail
    credits not consumed       -> 2 tests fail
    no _shutdown override      -> real-shutdown test fails
    no lookback (±2 s only)    -> absorbed-timeout test fails
    grader ignores warm flags  -> 2 grader tests fail
    sql-writer route removed   -> worker round-trip test fails
```

## Evals run

```text
services/orion-equilibrium-service/evals (transport baseline mesh eval) -> pass (in the 201 above)
Live replay (read-only Postgres, see the evidence section) -> 09-29: 511 -> 408 timeout rows; 09-28: 411 -> 328
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer build           -> Image Built
scripts/safe_docker_build.sh orion-equilibrium-service build  -> Image Built
docker run (equilibrium image, operator .env): owner=None (EMIT off), hourly accumulator on,
    channel orion:equilibrium:transport_baseline:hourly, pooled builder absent
docker run (sql-writer image, operator .env): table transport_baseline_hourly, insert-only,
    channel subscribed, route -> TransportBaselineHourlySQL
Rebuilt both after the review fixes: _shutdown override present; warm_at_start column present.
No up/deploy performed.
```

## Review findings fixed

The review ran in a subagent against `origin/main...HEAD`. It returned 15 findings. 14 are fixed, and one (finding 10) is a disclosed follow-up. Every fix below has a test, and the ones marked "mutation-checked" fail when the fix is removed.

1. **Blocker: the shutdown flush never ran.** The chassis cancels `_run` while it is parked in `iter_messages()`, so the flush after the loop was unreachable. On every restart, held markers and partial hours were lost. The old tests called the flush directly, so they passed for the wrong reason.
   - **Fix:** an `EquilibriumService._shutdown` override. It stops housekeeping, flushes, then hands off to the chassis, so the bus is still open when the flush runs.
   - **Evidence:** `test_real_shutdown_path_fires_held_atoms_before_the_bus_closes` asserts the marker row and the hourly row are published before `bus.close()`. Mutation-checked.
2. **Material: a timeout from a short-lived bus could double fire.** Timeouts from short-lived buses are absorbed into a *later* window, because `absorb()` keeps the absorbing window's start. They missed the ±2 s match, so the marker and the gate both fired.
   - **Fix:** the match window now reaches 60 s back (`lookback_s`).
   - **Evidence:** `test_owner_matches_a_timeout_absorbed_into_a_later_window`. Mutation-checked.
3. **Material: over the hourly budget, a gate-owned timeout gives zero rows.**
   - **Resolution:** kept deliberately. The budget is the cap for a mesh-wide outage, and a marker fallback would defeat it. This is now documented in `_handle_rpc_timeout_atom` and the README, and the drop is logged as `transport_baseline_suppressed`.
4. **Material: the grader gave false FAILs during warm-up and after cold starts.** Before a hop is warm its floor simply equals its level, so warm-up hours looked like floor movement.
   - **Fix:** new `warm_at_start` row field. Hours that were not warm at the start are left out of the medians. A warm-up or cold-start row restarts the floor segment.
   - **Evidence:** 2 grader tests plus an accumulator test. Mutation-checked.
5. **Minor: a delayed snapshot reopened an already-flushed hour.**
   - **Fix:** it now becomes its own `flush_reason="late"` row.
   - **Evidence:** a test.
6. **Minor: one invalid row could block the outbox.**
   - **Fix:** a row the bus rejects as invalid is dropped with an error. The pub/sub delivery limit is now documented in the schema and the README.
7. **Minor: housekeeping tasks piled up on `_run` restarts, and the outbox could race.**
   - **Fix:** one housekeeping task per process, and an `asyncio.Lock` on the outbox.
   - **Evidence:** a test.
8. **Minor: marker matching could get slow and grow without bound.**
   - **Fix:** credits and held markers are indexed by channel. Held markers are capped at 5,000; on overflow the oldest fires early (fails open, never dropped).
   - **Evidence:** a test.
9. **Minor: a missing marker timestamp fell back silently, and `bool` parsed as a timestamp.**
   - **Fix:** the fallback to receipt time is now logged at debug level, and `bool` is excluded from timestamp parsing.
10. **Minor, follow-up (not fixed here): the metacog self-loop guard depends on the mode.** With EMIT off, a timeout marker from metacog's own `#log_orion_metacognition` call still fires, because the marker carries no health label. This predates the PR. The fix is to add `health_label` to the `rpc_transport_timeout` marker in `orion/core/bus/async_service.py` and apply the exclusion list on the marker path. That touches every service's bus library, so it is kept out of this patch.
11. **Minor: bucket-cap overflow was invisible.**
    - **Fix:** it is now logged on flush.
12. **Minor: the sql-writer test never went through the real consume path.**
    - **Fix:** a real envelope now goes through `handle_envelope` and lands in a sqlite row.
    - **Evidence:** removing the route makes it fail.
13. **Minor: quiet hours were hard-coded to MDT.**
    - **Fix:** they are computed in America/Denver time, so DST is handled.
    - **Evidence:** a test with dates on both sides of the switch.
14. **Nit: stale "legacy branch" wording** in 3 places.
    - **Fix:** reworded.
15. **Nit: the local sql-writer `.env` lacked the new route and channel.**
    - **Fix:** added by hand to `SQL_WRITER_ROUTE_MAP_JSON` and `SQL_WRITER_SUBSCRIBE_CHANNELS`, and only those two values. The sync script refuses diverged values, and `--force` would flatten secrets.

## Restart required

Order matters: the consumer goes first.

```bash
cd /mnt/scripts/Orion-Sapienform            # after merge + pull, or from a worktree
scripts/safe_docker_build.sh orion-sql-writer up -d --build            # 1. table + route
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "\d transport_baseline_hourly"   # confirm table exists
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build   # 2. publisher
# after the next full hour + 90 s:
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), max(hour_start) from transport_baseline_hourly"
python scripts/analysis/grade_transport_baseline.py --days 7
```

## Merge order with PR #2421

PR #2421 (`fix/bus-synaptic-transport-threshold`, a parallel session) retires the bus_synaptic transport source. It touches the same equilibrium files. A test merge (`git merge-tree`) conflicts in 5 of them:

- `.env_example`
- `README.md`
- `app/service.py`
- `app/settings.py`
- `docker-compose.yml`

Whichever PR merges second resolves them. The conflicts are adjacent edits, not overlapping logic: #2421 removes the bus_synaptic code and keys, and this PR removes the pooled branch and adds the owner/hourly code. With bus_synaptic gone, the replay's "total" column drops by a further 51–84 rows a day.

## Risks / concerns

- **Severity: medium.**
  - **Concern:** the EMIT-on marker/gate matching has not run live yet (EMIT stays off). It is tested with the real payload shapes but UNVERIFIED on live traffic.
  - **Mitigation:** it fails open, so an unmatched marker fires. Every decision is logged as `transport_timeout_owner`.
- **Severity: low.**
  - **Concern:** with EMIT off, a timeout marker from metacog's own background call still fires, because the marker carries no health label (review finding 10, which predates this PR). It is a follow-up.
- **Severity: low.**
  - **Concern:** marker timestamps and window times come from the emitting process's clock. A marker from one host could be matched to a same-channel credit from another host's clock. The matching is count-conserving, so totals stay right; only the attribution can be wrong.
- **Severity: low.**
  - **Concern:** Hub's governor timeout marker (`HUB_GOVERNOR_TIMEOUT_GRAMMAR_ENABLED`, default false) is recorded under a `governor:` hop, which never matches a marker's channel. If that flag is turned on while EMIT is on, a governor timeout would fire twice.
  - **Mitigation:** the flag is off.
- **Severity: low.**
  - **Concern:** each hourly row holds percentiles for that row only. After a restart mid-hour, the grader takes a weighted median of medians. This is stated in the schema and the grader.
- **Severity: low.**
  - **Concern:** `transport_baseline_hourly` has no retention policy (about 1,500 rows a day).
  - **Mitigation:** add one if it is kept past the grading period.
- **Severity: low.**
  - **Concern:** removing the pooled source lets more marker rows through the shared 30 s lane, which could crowd bus_synaptic.
  - **Mitigation:** the replay shows bus_synaptic counts unchanged (64 and 51).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2425
