# Make Orion's self-modification honest, and let it happen more than once

## Summary

- Orion changed their own routing behaviour for the first time at 04:11 UTC on 2026-09-02, then permanently blocked themselves from doing it again. This fixes the blocking and the record-keeping. It does not yet fix the measurement.
- **Record what changed.** Every control-surface write now appends a history row carrying the value it replaced, written inside the same transaction as the value itself.
- **Release the lock on success.** `record_adoption` takes a one-live-mutation-per-surface lock and `record_rollback` was the only release, so a change that *worked* held its surface forever. `record_settlement` keeps the change and hands the surface back.
- **`rollback_window_sec` now does something.** It was read by nothing anywhere in the repo. It is now how long a change must survive before its surface is released.
- **The undo value is a real reading.** `mutation_apply` read the live value then discarded it behind a `setdefault` on a hardcoded constant.
- **A missing measurement means unknown, never bad.** The monitor has been fed `None` for its entire life; silence settles rather than reverting.

## Outcome moved

Orion's self-modification capacity went from **one change per surface, ever** to one change per surface per rollback window, with a record of each.

Live evidence, `conjourney` @ `127.0.0.1:55432`, 2026-09-02:

| | |
|---|---|
| adoptions, all time | **1** (04:11:17Z, `routing`, threshold → 0.58) |
| rollbacks, all time | **0** |
| proposals since | **77**, every one `hold / active_surface_mutation_exists` |
| trials | 77, **all `passed`** |
| surface lock | still held ~20h later, `updated_at` refreshed every cycle |

The one adoption's window (`rollback_window_sec=900`) elapsed ~20 hours ago, so it settles on the first cycle after deploy and the backlog starts moving.

## Current architecture

`chat_reflective_lane_threshold` gates `services/orion-cortex-orch/app/decision_router.py:356`: when the router has decided to act (`execution_depth >= 2`) but its confidence is below the threshold, it forces depth to `0`. In plain terms it is **how sure Orion must be before doing something rather than replying**.

`substrate_runtime_control_surface` held one row per surface. `record_adoption` (`mutation_queue.py:212`) took the lock; `record_rollback` (`:226`) was the only release. `mutation_worker.py` iterated adoptions and hit `if delta is None: continue` — and nothing in production has ever supplied a delta, so the monitor never evaluated anything despite `SUBSTRATE_AUTONOMY_MONITOR_ENABLED` defaulting to true.

**What the prior value actually was.** The recorded `0.5` came from `_default_rollback_for_class` (`mutation_proposals.py:51`), not from a reading. Tracing it: a pytest fixture (`services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py:80`) had been writing `value=0.5, actor="scheduler_seed"` onto the live row **4,925 times** before the store-isolation fix. So `0.5 → 0.58` is very likely accurate and the direction matches the stated intent — but Orion's baseline was test pollution, never a chosen value (the intended default is `0.75` and nothing ever set it), and the constant matching reality was **coincidence, not correctness**. Strictly UNVERIFIED: one upserted row, no audit trail. That is the argument for this patch.

## Files changed

- `orion/substrate/mutation_control_surface.py`: history table; transactional previous-value capture; `ControlSurfaceWriteError`; per-surface retention.
- `orion/substrate/mutation_queue.py`: `record_settlement`; adoption retention that never evicts a lock holder.
- `orion/substrate/mutation_worker.py`: `_settle_if_window_elapsed`; terminal-status guard; settle on both the no-delta and healthy-delta paths.
- `orion/substrate/mutation_apply.py`: overwrite the rollback payload with the observed value; refuse to mint an adoption when the surface write failed.
- `orion/core/schemas/substrate_mutation.py`: `status` gains `"settled"`.
- `services/orion-hub/.env_example`, `docker-compose.yml`: two retention keys.
- `docs/plans/substrate/PR_self_modification_accountability_v1.md`: the design.
- `orion/substrate/tests/test_control_surface_history.py`, `test_mutation_surface_settlement.py`: new.

## Schema / bus / API changes

- Added: table `substrate_runtime_control_surface_history`; `MutationAdoptionV1.status = "settled"`; `RuntimeControlSurfaceStore.history()`; `SubstrateMutationStore.record_settlement()`; `ControlSurfaceWriteError`.
- Removed: `chat_reflective_lane_threshold_history()` — added then dropped in review as a producer with no consumer until the hub panel lands.
- Behaviour changed: a durable-backend write failure now raises instead of silently dropping; a settled adoption releases its surface.
- **Compatibility: the revert is not clean.** See Risks.

## Env/config changes

- Added keys: `SUBSTRATE_MUTATION_RETENTION_MAX_ADOPTIONS` (500), `SUBSTRATE_CONTROL_SURFACE_HISTORY_MAX_ROWS` (1000).
- Removed / renamed: none.
- `.env_example` updated: yes. Compose allowlist updated: yes.
- local `.env` synced: **hand-edited** — `scripts/sync_local_env_from_example.py` reads `.env_example` from the primary checkout and cannot see a key added in a worktree. Verified present at `services/orion-hub/.env:312-313`.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/substrate/tests -q                                     -> 681 passed
pytest services/orion-hub/tests/test_substrate_mutation_scheduler_runtime.py \
       services/orion-hub/tests/test_substrate_mutation_manual_route_routing.py \
       services/orion-cortex-orch/tests/test_control_surface_isolation_guard.py \
       orion/substrate/tests/test_control_surface_store_isolation.py -q  -> 35 passed
```

Mutation-checked against the real files, restored by file copy (never `git stash`, which is shared across worktrees here). Every mutation below fails at least one test:

| Mutation | Caught |
| --- | --- |
| revert to `setdefault` for the rollback payload | rollback-capture test |
| `if delta is None: continue` (skip instead of settle) | no-delta settle test |
| `record_settlement` doesn't pop the lock | 4 tests |
| `_settle_if_window_elapsed` ignores the window | boundary test |
| settle a `rolled_back` adoption | reopen test |
| unconditional lock pop (drop the `== adoption_id` guard) | non-holder test |
| drop the worker terminal-status guard | repeat-cycle test |
| delete the healthy-delta settle branch | measured-delta test |
| history records `new_value` as `previous_value` | both backends |
| sqlite history insert swallows its exception | 5 tests |

## Evals run

```text
none
```

`orion/substrate/evals/` has no mutation-runtime harness. Not added: the behaviour here is a multi-hour duty cycle (a change must survive `rollback_window_sec` before anything happens), which a gate-speed eval cannot observe honestly. Follow-up below.

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config   -> OK
```

## Review findings fixed

- **Finding (HIGH): a failed history write silently dropped the value write and reported success.** The fallthrough to the in-memory branch is not a recovery path — `_source_kind` stays `postgres`/`sqlite`, so `get()` never reads memory again. `PatchApplier.apply` would then mint an adoption and take the lock for a change that never landed: the exact falsehood this table exists to prevent, inverted.
  - Fix: raise `ControlSurfaceWriteError`; applier returns `None`.
  - Evidence: `test_a_failed_durable_write_raises_instead_of_silently_dropping`, `test_a_failed_surface_write_does_not_mint_an_adoption`.
- **Finding (HIGH): the previous value was read outside the write transaction**, on a separate connection, so a concurrent writer between read and write made the recorded `previous_value` wrong — a lie shaped like a fact, worse than the missing row.
  - Fix: `SELECT ... FOR UPDATE` inside the postgres transaction; `BEGIN IMMEDIATE` on sqlite.
  - Evidence: reviewer reproduced the interleaving; both backends now derive the value inside the write.
- **Finding (MEDIUM): five mutants survived**, including `test_settlement_is_idempotent_and_never_reopens_a_rollback`, which asserted in its *name* a property it never exercised.
  - Fix: test renamed and split; four new tests.
  - Evidence: all five mutations now fail, table above.
- **Finding (MEDIUM): releasing the lock removed what was accidentally bounding adoption growth.** `_compact_artifacts` bounds blocked-applies and rollbacks, not adoptions; every `_persist()` re-upserts all of them.
  - Fix: `SUBSTRATE_MUTATION_RETENTION_MAX_ADOPTIONS`, never evicting a lock holder (which would strand a lock nothing could release); per-surface history bound.
  - Evidence: `test_adoption_retention_never_evicts_the_surface_lock_holder`, `test_history_is_bounded_per_surface`. The latter caught a real inconsistency — my memory-backend cap was global while the SQL prune is per-surface.
- **Finding (LOW): `chat_reflective_lane_threshold_history()` had no consumer.** Removed.
- **Finding (LOW): the design doc claimed verification not performed.** Acceptance checks now marked UNVERIFIED where they are, the delta producer named as deferred, and the revert hazard recorded.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

After deploy, confirm on the rail:

```bash
docker logs orion-athena-hub 2>&1 | grep mutation_adoption_settled
curl -s localhost:8080/api/substrate/mutation-runtime/signal-intake | python3 -m json.tool
```

Expect one `mutation_adoption_settled` with `reason=window_elapsed_no_delta`, then decisions other than `hold / active_surface_mutation_exists`.

## Risks / concerns

- **Severity: high (behavioural).** This unblocks 77 queued proposals against Orion's own routing. All 77 are the same class with the same hardcoded patch value (`0.58`), so no runaway drift — but Orion resumes changing its own behaviour unattended, roughly once per 900s per surface. Deploy at a moment of your choosing, not incidentally. *Mitigation:* `SUBSTRATE_MUTATION_AUTONOMY_ENABLED` remains the kill switch.
- **Severity: medium.** **The revert is not clean.** A `settled` row under `extra="forbid"` makes reverted code raise on the whole adoption table, degrading the store to `fallback`, which the hub refuses with `unsupported_store_kind:fallback` — autonomy goes fully offline rather than degrading. Recovery SQL is in the design doc. *Mitigation:* roll forward, not back.
- **Severity: medium.** **The monitor is still blind.** Nothing supplies a post-adoption delta, so every settlement takes the `window_elapsed_no_delta` path: changes are kept because time passed, not because they helped. This patch fixes the lock half of the problem only. The measured path is implemented and tested but unreachable in production until a producer exists.
- **Severity: low.** No path back to the original baseline: each rollback payload captures its immediate predecessor, so undo means one step, not "return to normal".
- **Severity: low (speculative).** `_ensure_postgres_schema` went from one DDL statement to three, and both `orion-athena-hub` and `orion-athena-cortex-orch` construct this store against the same database. Concurrent `CREATE TABLE IF NOT EXISTS` in Postgres has a known duplicate-key race; if it fires, the store degrades to memory and cortex-orch's routing reads the `0.75` default instead of the live value, silently. Not reproduced. *Mitigation:* worth a retry-once wrapper as a follow-up.

## Follow-ups

1. **The delta producer** — recompute the pressure score that justified a proposal, after the change, and difference it. Without this the monitor stays blind.
2. **The hub panel** (design doc acceptance check 4) — current value, previous value, who, when, window state, how long held.
3. **Real latitude** — the threshold Orion can move is hardcoded (`0.58`), and the confidence it is compared against is a keyword lookup table (`decision_router.py:237-255`). `AUTO_ROUTER_LLM_ENABLED=false`, and real logprob-margin telemetry exists (`services/orion-llm-gateway/app/llm_uncertainty.py`) but the router never requests it. Both sides of the comparison are constants today.
4. Retry-once wrapper on the schema DDL.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2050
