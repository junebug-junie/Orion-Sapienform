## Summary

- **L1:** The finalize check (`harness_finalize_reflect`) and response repair (`orion_response_repair`) no longer rebuild the whole stance context. That rebuild was a duplicate that nobody read and took about 9 s. The router and the step-time check now share one skip helper, `brain_reply_context_skipped()`, so they cannot drift apart again. The helper reads the skip flag from both `ctx` and `options`.
- **L2:** The slow synchronous half of the stance build now runs on a dedicated single-worker thread instead of the event loop. That half is the felt-state hydrate, the unified beliefs with their two Falkor snapshots, the dispatch-actions read and the attention frame. While a build runs, cortex-exec keeps answering other requests (reverie, metacog, journal) instead of freezing.
- **L2 race fix:** Moving the build off the loop let a write land in the middle of a Falkor snapshot. The store's cache writes, the generation counter, the snapshot check-and-swap and the region reads now share one in-memory lock (`_cache_lock`). That lock is never held across a Falkor round trip.
- **L3, identity:** The `identity_yaml` producer is now ephemeral (`SNAPSHOT_EPHEMERAL`, `pull_on_cold=False`) in the live registry, `orion/cognition/projection_builder.py`. That registry is the one the shared spine installs. The same change is made in the cortex-exec fallback registry. This stops the `producer_materialize_failed identity_yaml` error and the `degraded` mark it put on the orion anchor. `self_definition` gets the same treatment, in the fallback registry only.
- **L3, probe and reverie:** The current-turn signal probe now tells the gateway how long it will actually wait (`gateway_read_timeout_sec`). The reverie reader drops repeated lines, comparing them by normalized text.

## Outcome moved

- **Finalize leg:** about 9 s of duplicate build per unified turn is removed from the critical path. This is **UNVERIFIED live**; it needs a deploy plus the spec's 20-turn before/after median.
- **cortex-exec freeze:** the process no longer stalls for about 9 s on every stance build. **UNVERIFIED live.**
- **identity_yaml failures:** these were live before the change. In the last 24 h the logs show 36 in cortex-exec, 75 in -background and 5 in -chat. After the change they should go to zero. The new test reproduces the exact error on main and passes on this branch.
- **Duplicate rows:** `cortex_turn` and `chat_stance_belief_log` rows roughly halve, because each turn now writes once instead of once from stance and once from finalize. Downstream readers see more distinct turns per window: self-study row counts, self-inquiry SQL, the recent-attention cue, and the hub surface panel. The `count(*)` baseline in `instruments.yaml` moves; `count(DISTINCT …)` does not. In `substrate_tier_outcomes_events`, the row kept per correlation id is now the stance event, not the finalize event.

## Current architecture

- **Router vs step-time check:** `router.py` skipped `introspect_spark` and `memory_graph_suggest`, and checked the skip flag in `ctx` only. `_should_prepare_brain_reply_context` kept its own list. Neither list skipped the finalize or repair verbs.
- **Stance build:** `build_chat_stance_inputs` ran its Falkor and Postgres reads on the event loop.
- **Falkor store:** writes did not take `_snapshot_lock`, and `_write_generation += 1` had no lock.
- **identity_yaml:** registered as `OPERATOR_STATIC`, which writes through to Falkor. Its `StateSnapshotNodeV1` is not a Falkor durable kind, so every write failed.

## Architecture touched

cortex-exec (executor, router, chat_stance, current_turn_llm_signals), `orion/substrate/falkor_store.py`, `orion/substrate/felt_state_reader.py`, `orion/cognition/projection_builder.py` (also used by cortex-orch's mind runtime), `orion/situational/reverie_reader.py`.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: adds `BRAIN_REPLY_CONTEXT_SKIP_VERBS` and `brain_reply_context_skipped()`.
- `services/orion-cortex-exec/app/router.py`: uses the shared helper.
- `services/orion-cortex-exec/app/chat_stance.py`: adds the stance-build worker (`_run_on_stance_worker`), makes the fallback registry's `identity_yaml` and `self_definition` ephemeral, and documents why autonomy is left alone.
- `services/orion-cortex-exec/app/current_turn_llm_signals.py`: the probe sends `gateway_read_timeout_sec`.
- `orion/substrate/falkor_store.py`: adds `_cache_lock` around cache mutation, the generation bump, the check-and-swap, the snapshot read and the region reads.
- `orion/substrate/felt_state_reader.py`: adds a lock around the reader's lazy init.
- `orion/cognition/projection_builder.py`: makes `identity_yaml` ephemeral in the live registry.
- `orion/substrate/relational/adapters/identity_yaml.py`: docstring update.
- `orion/situational/reverie_reader.py`: over-fetches, then dedupes by normalized text.
- Tests:
  - `services/orion-cortex-exec/tests/test_turn_latency_l1_l3.py` (new)
  - `orion/substrate/tests/test_falkor_store.py` (+3 tests)
  - `orion/situational/tests/test_reverie_reader_dedupe.py` (new)

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: the router log reason is now `skip_verb_or_flag` (was `spark_or_skip_flag`). The probe request carries `options.gateway_read_timeout_sec`, which the gateway already accepts.
- Compatibility notes: no contract change.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- Skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-cortex-exec/tests/test_turn_latency_l1_l3.py  -> 23 passed
pytest orion/substrate/tests/test_falkor_store.py                   -> 51 passed
pytest orion/situational/tests/test_reverie_reader_dedupe.py        -> 3 passed
pytest services/orion-cortex-exec/tests (full, --continue-on-collection-errors)
  -> no new failures vs a clean origin/main worktree; ~129 failures and 15 collection
     errors on main are pre-existing (verb double-registration, env-sensitive tests).
pytest orion/substrate/tests orion/situational/tests orion/cognition/tests orion/substrate/relational/tests
  -> same 3 failed / 3 errors as main, +6 new passes
pytest services/orion-cortex-orch/tests -> same 31 pre-existing failures as main
Regression proof: new L1/L3 tests fail on main (18 of 22); the check-to-swap race test
fails on main ("assert None is not None") and passes here.
python scripts/check_env_template_parity.py -> ok
git diff --check -> clean
```

## Evals run

```text
None. cortex-exec has no eval harness for turn latency. The spec's acceptance checks are
live measurements: the finalize-leg median over 20 turns, the gateway publish ->
cortex-exec receive gap, and zero producer_materialize_failed identity_yaml lines.
They need a deploy. Follow-up: run them after the restart below.
```

## Docker/build/smoke checks

```text
Not run. Per the task, no deploy and no restart. No dependency, compose or env change.
```

## Review findings fixed

- **Finding:** making autonomy non-write-through while keeping `pull_on_cold=True` would make goals appear on cold turns and vanish on warm ones, because the ephemeral store is rebuilt on every call.
  - **Fix:** reverted autonomy to `GRAPHDB_DURABLE` and documented why. Live, the adapter returns None because the gate is off, and logged 0 failures in 24 h. This is a deliberate deviation from the spec's "same treatment"; the correct fix belongs to the autonomy gate.
  - **Evidence:** a registry test asserts the tier.
- **Finding:** the first race test also passed on main.
  - **Fix:** added a deterministic test that fires the write exactly at the generation check before the swap.
  - **Evidence:** it fails on main and passes here.
- **Finding:** the Falkor `query_*`/`read_*` region reads iterate the cache without a lock.
  - **Fix:** they now run under `_cache_lock`.
- **Finding:** the worker comment overstated what the single worker covers.
  - **Fix:** the comment now states that the felt-state reader is still called from the loop (the worst case is a duplicate fetch) and that a hung build now stalls only later stance builds.
- **Finding:** nothing pinned the snapshot-path identity lines to the ctx fallback.
  - **Fix:** a new test asserts they are equal, with a 10-line cap.
- **Nits:** fixed a comment's indentation, updated the identity_yaml docstring, and removed a redundant options argument in the router.
- **Test isolation:** other suites re-import `app.*`, so the new tests patch the globals their imported functions actually resolve names in.

## Restart required

After merge, from the primary checkout on main (`cd /mnt/scripts/Orion-Sapienform && git pull --ff-only`), one line each:

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-exec up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-cortex-orch up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

- cortex-exec rebuilds all four containers (cortex-exec, -chat, -background, -spark) and carries L1, L2 and L3.
- cortex-orch picks up the `projection_builder` registry change, which its mind runtime uses.
- hub is optional: it only picks up the reverie dedupe for its own situation block.

## Risks / concerns

- **Severity: medium.** Concern: a hung synchronous build blocks later stance builds in that container, because the worker cannot be cancelled. Before this change it froze the whole process. Mitigation: nothing new hangs here. Watch for growing stance-build latency.
- **Severity: low.** Concern: real write/scan overlap is now possible, so `falkor_substrate_hydrate_failed ... local mutation during scan` may appear more often, with a stale receipt and a retry on the next call. Mitigation: watch that log line after deploy.
- **Severity: low.** Concern: `_project_identity_from_beliefs` now reads the identity_yaml snapshot instead of the ctx fallback. A test pins the two paths to equal output; the live "11 Orion lines" check is still UNVERIFIED.
- **Severity: low.** Concern: the reverie dedupe changes situation-block content in every service that reads it (cortex-exec, hub). Those services pick it up on their next rebuild.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2508
