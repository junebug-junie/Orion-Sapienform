## Summary

- orion-thought now asks cortex-exec to build the stance context **while orion-mind is running**, instead of only after mind returns. The build never reads mind's output, so the ~9 s build and the ~11 s mind call can overlap.
- New request/reply RPC `stance_context_prepare`. cortex-exec builds the context and keeps what the build changed in an in-process cache, keyed by correlation id (120 s TTL, used once).
- stance_react uses the prepared context. If the prepare is still building, it waits for it instead of building a second time. If no prepare has arrived, it waits up to 2 s. It builds inline only when the prepare failed or never arrived, and a prepare that turns up after that is refused. Net effect: one build per turn, one `chat_stance_belief_log` row, one `cortex_turn` row.
- Same container is guaranteed: the prepare channel is derived from the exec lane channel orion-thought sends stance_react on (`orion:cortex:exec:request[:lane]` -> `orion:cortex:exec:stance_prepare[:lane]`). Each cortex-exec lane container listens on its own.
- Flag `ORION_THOUGHT_STANCE_PREPARE_PARALLEL` ships on. Turning it off restores the old sequential behaviour, with no wait.

Spec: `docs/superpowers/specs/2026-10-06-unified-turn-latency-design.md`, L4. This branch started stacked on #2508, which has since merged, so it now targets `main`.

## Outcome moved

The gap between mind finishing and stance starting should drop from about 9 s to about 0–1 s on turns where mind takes 9 s or more. **UNVERIFIED** until deployed and timed. Read it from the two `stance_prepare_overlap` log lines:
- orion-thought: `mind_ms build_ms wait_ms outcome`
- cortex-exec: `outcome build_ms wait_ms lane`

Known cost: the build's chat-lane probe (the 35B model on gpu0+gpu3) now runs at the same time as mind's 8B model on gpu3. Compare `mind_ms` before and after.

## Current architecture

`run_stance_react` awaited `_maybe_build_mind_coloring` (orion-mind over HTTP), then sent stance_react on `CHANNEL_CORTEX_EXEC_REQUEST`. cortex-exec's router hook then ran `prepare_brain_reply_context` -> `build_chat_stance_inputs`, about 9 s on the dedicated stance worker added in #2508.

## Architecture touched

- **orion-thought:** `bus_listener.py` sends the prepare (on its own bus connection) and marks the stance request with `ctx["stance_prepare_requested"]`.
- **cortex-exec:**
  - new `app/stance_prepare.py`: the cache, the ctx-delta transfer and the RPC handler;
  - new `app/exec_ctx.py`: the exec ctx builder, now shared by `main.handle` and the prepare;
  - a new Rabbit in `main.py` (concurrent handlers);
  - a consume hook in `executor.prepare_brain_reply_context`;
  - `router.py` exports `stance_prepare_overlap` in the result metadata.
- **Contracts:** `orion/schemas/stance_context_prepare.py`, both registries, `orion/bus/channels.yaml`.

## Files changed

- `orion/schemas/stance_context_prepare.py`: request/result models, kinds, channel derivation.
- `orion/schemas/registry.py`: registers both models (`_REGISTRY` and `SCHEMA_REGISTRY`).
- `orion/bus/channels.yaml`: 4 lane request channels plus the `stance_prepare_result:*` wildcard.
- `services/orion-cortex-exec/app/{stance_prepare.py,exec_ctx.py,main.py,executor.py,router.py}`: consumer.
- `services/orion-thought/app/{bus_listener.py,settings.py}`, `.env_example`, `docker-compose.yml`: producer and flag.
- `scripts/sync_local_env_from_example.py`: adds the `ORION_THOUGHT_STANCE_PREPARE_` prefix, so the new key actually syncs.
- Tests: `services/orion-cortex-exec/tests/test_stance_context_prepare.py`, `services/orion-thought/tests/test_stance_prepare_parallel.py`, `tests/test_stance_context_prepare_bus_catalog.py`.

## Schema / bus / API changes

- **Added:**
  - `StanceContextPrepareRequestV1` (`cortex.stance_context_prepare.request.v1`)
  - `StanceContextPrepareResultV1` (`cortex.stance_context_prepare.result.v1`)
  - channels `orion:cortex:exec:stance_prepare{,:chat,:spark,:background}` and `orion:cortex:exec:stance_prepare_result:*`
  - stance_react ctx key `stance_prepare_requested`
  - result metadata `stance_prepare_overlap`
- **Removed / renamed:** none.
- **Behavior changed:** stance_react uses the prepared context when the marker is set. Callers that don't set the marker behave exactly as before.
- **Compatibility:**
  - If orion-thought is restarted before cortex-exec, nothing is listening on the prepare channel. Each turn then waits 2 s and builds inline.
  - Restart cortex-exec first.

## Env/config changes

- Added keys: `ORION_THOUGHT_STANCE_PREPARE_PARALLEL=true` (orion-thought).
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-thought`: yes, `+ORION_THOUGHT_STANCE_PREPARE_PARALLEL='true'`.
- Skipped keys: none.

## Tests run

```text
services/orion-cortex-exec: tests/test_stance_context_prepare.py            20 passed
services/orion-cortex-exec: full suite                                      127 failed / 1078 passed / 15 collection errors
                            failure set identical to the base commit 3599bc3d7 (pre-existing, env-dependent)
services/orion-thought: full suite                                          517 passed, 1 failed
                            (test_mind_enrichment_defaults_off, pre-existing on base)
tests/test_stance_context_prepare_bus_catalog.py                            3 passed
tests/scripts/test_sync_local_env_from_example.py                           19 passed
scripts/check_bus_reply_channels.py                                         0 uncovered
scripts/check_env_template_parity.py                                        PASS
Pre-existing, unrelated failures in tests/test_single_consumer_channels_gate.py, tests/test_channel_prefix_guardrail.py,
tests/test_exec_result_channel_catalog_specificity.py (gpu_pool / context-exec / PadRpc entries, not these channels).
```

The cortex-exec tests cover:
- prepare then stance: one build, rows written once, verb is `stance_react`;
- prepare still in flight: stance waits for it and uses it;
- prepare arrives after stance: stance waits for it;
- prepare failed: stance builds inline;
- prepare absent: inline build after the wait, and a late prepare is refused;
- the cache is used once and expires at its TTL;
- the real router path keeps its own `debug` keys;
- a non-canonical correlation id still finds the cache entry.

The orion-thought tests cover:
- the prepare is sent before mind finishes, on the stance lane;
- flag off: no prepare and no marker;
- the prepare is fail-open;
- the prepare task is cancelled when the turn dies.

## Evals run

```text
No eval harness covers the turn's wall-clock latency. The measurement is the live stance_prepare_overlap log pair after deploy (spec acceptance check 4).
```

## Docker/build/smoke checks

```text
Not run (no deploy/restart in this task). Live path UNVERIFIED.
```

## Review findings fixed

- Finding: applying the delta replaced a dict the build created new (e.g. `debug`), wiping the router's `debug.recall_*` entries.
  - Fix: merge whenever both sides are dicts.
  - Evidence: `test_router_path_keeps_its_own_debug_keys_and_builds_once` drives the real `PlanRunner.run_plan`.
- Finding: the tests built the stance ctx with the prepare's own builder, so they could not detect drift between it and the real router path.
  - Fix: added the real-router test above.
- Finding: the cache was keyed by the raw payload correlation id, while stance looks up the normalized envelope id.
  - Fix: key by `str(env.correlation_id)`.
  - Evidence: `test_non_canonical_payload_correlation_id_still_hits`.
- Finding: the prepare task was orphaned if the mind call raised or was cancelled.
  - Fix: the try/finally now covers the mind call and cancels the task.
  - Evidence: `test_prepare_task_is_cancelled_when_the_turn_fails_before_stance`.
- Finding: the shallow snapshot missed nested in-place mutation.
  - Fix: deep-copy the snapshot, falling back to a shallow copy.
  - Evidence: `test_nested_in_place_mutation_is_transferred`.

## Restart required

Order matters: cortex-exec (all lanes) first, then orion-thought.

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build cortex-exec cortex-exec-chat cortex-exec-spark cortex-exec-background
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: medium.
  - Concern: GPU contention. The chat-lane probe in the build now overlaps mind's call on gpu3, so mind may slow down.
  - Mitigation: watch `mind_ms` in the overlap log, and turn the flag off if mind regresses by more than the time saved.
- Severity: low.
  - Concern: a prepare whose turn never sends stance (mind failed and the turn was dropped) still writes one belief-log row and one cortex_turn row for that turn.
  - Mitigation: cancelling the task does not stop a build that has already started in cortex-exec. This is bounded at one row of each per turn.
- Severity: low.
  - Concern: the ctx transfer assumes the build reads no key that differs between the prepare ctx and the stance ctx. Today the only difference is `mind_coloring` and the router's recall/debug keys, and the build reads none of them.
  - Mitigation: the real-router test pins this.

## PR link

(filled after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
