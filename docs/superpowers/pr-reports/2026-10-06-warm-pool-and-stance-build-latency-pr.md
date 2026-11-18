## Summary

- Juniper's chat turns now take the warm Claude Code pool (PR #2514). None of them did before: every Hub chat turn carries a reading-tool binding, and the motor sent any turn with a binding to a fresh spawn.
- Warm slots now host the reading and introspect tools. Each server reads which turn it belongs to from a small per-slot file. The pool writes that file before the prompt and empties it when the turn ends. A tool call with no turn bound fails instead of acting for the previous turn.
- One log line per turn says why it did or did not try the warm path: `fcc_warm_path_decision`.
- Stance context builds no longer reload the whole substrate graph. They read the 216 nodes and the concept region they actually use, in about 80 ms each, instead of a 14 s full load (17-28 s live).
- One log line per stance build gives the time per phase: `stance_build_phase_timing`.

## Outcome moved

- Warm path: two live chat turns (07:57 and 08:07 UTC) spent 5.7 s and 4.0 s starting a process. The warm acquire is under 0.5 s in tests. Live: UNVERIFIED until deploy.
- Stance build: live `build_ms` was 28,632 and 17,316. The graph load was 26 s and 15 s of that. The new read measures 77 ms (nodes) + 65 ms (concept region) against production, read-only. Live end-to-end: UNVERIFIED until deploy.

## Current architecture

- Hub's `execute_unified_turn` always passes `reading_context="unified_chat"`, so `HarnessRunRequestV1.reading_binding` is always set (`orion/hub/turn_orchestrator.py:1496`). `utterance_origin="juniper"` does reach the governor: `chat_reply` was True.
- `run_fcc_turn` tried the pool only `if chat_reply and not reading_only and reading_binding is None`. That was never true for a Hub chat turn.
- The stance build's unification layer held a `FalkorSubstrateStore`. Its `snapshot()` reloads the whole graph whenever 30 s have passed since the last load, which is always true by the next human turn.

## Evidence (live, read-only)

Issue 1:
- Governor logs: `fcc_turn_start_timing corr=d1c5272c... mode=spawn spawn_or_acquire_ms=5749` and `corr=7059ac4b... mode=spawn ... 3991`, with no `fcc_warm_pool_fallback`. The pool itself was up (`fcc_warm_pool_spawned slot=s0 ... 07:54:35`).
- Transcripts `9c0cc104...` (07:58) and `7445f46c...` (08:07) in `/root/.claude/projects/-mnt-orion-fcc-repo/` list `mcp__orion-reading__recommend_reading` and `mcp__orion-introspect__reading_results`, so a reading binding was attached.
- The same two transcripts have no `"type":"AutoMem"` attachment for `memory/MEMORY.md`. Older transcripts (for example `1d1fbd75...`, 00:37) and a non-chat session from 08:11 (`62bb3271...`) do have it. So auto-memory-off (PR #2501) was firing and `chat_reply` was True.

Issue 2:
- The `orion-athena-cortex-exec` timeline for corr d1c5272c: build start 07:57:20.53, store created 20.80, `falkor_substrate_hydrate_stats` 47.01, `stance_prepare_ready` 49.16. For 7059ac4b: start 08:06:53.43, hydrate done 08:07:08.53, ready 08:07:10.75. The remaining ~1-2 s is the current-turn signal probe's LLM call.
- No other stance build ran in that container during either window. Every container has its own single worker. So it was not queueing behind background builds (PR #2508).
- Full hydrate measured read-only from the host: 14.0 s. That breaks down as 10.6 s in 49 page queries (including reply parsing) and the rest decoding. Graph size: 5,051 nodes, 38,393 edges.
- The layer uses only the non-`world` anchor nodes (216) and no edges. The "~4.5 s per rehydrate" baseline came from before PR #2500 made the load complete; that PR's own doc measures the complete load at 17-25 s.
- The new reads were compared against a full load of the production graph (read-only): node sets were identical (216 = 216, model-equal), and the concept region was identical (64 nodes, 64 edges).

## Architecture touched

- `orion/harness` (motor, warm pool), the MCP servers in `orion/world_pulse_read` and `orion/introspect`, and `orion/fcc/mcp_config.py`.
- `orion/substrate` (new anchor store, layer guard), `orion/cognition/projection_builder.py`, and cortex-exec `chat_stance.py`.

## Files changed

- `orion/fcc/turn_binding_file.py`: new. Atomic per-slot binding file, read on every tool call, fails closed when empty.
- `orion/fcc/mcp_config.py`: `reading_binding_file` / `introspect_binding_file` render mode.
- `orion/world_pulse_read/mcp_server.py`, `orion/introspect/mcp_server.py`: file-mode binding, re-read per call.
- `orion/harness/fcc_warm_pool.py`: slots render the reading/introspect servers in file mode, write the binding at acquire, and clear it at release.
- `orion/harness/fcc_motor.py`: removed the reading-binding exclusion, added the `fcc_warm_path_decision` line, and passes the binding to the pool.
- `orion/harness/tests/test_fcc_warm_pool_chat_binding.py`, `fake_claude_stream.py`: regression tests (Hub request through the runner into the real motor and pool; binding follows the turn).
- `orion/substrate/falkor_anchor_store.py`: new `FalkorAnchorStanceStore` and `build_unification_store_from_env`.
- `orion/substrate/relational/layer.py`: refuses anchors the store does not serve.
- `orion/cognition/projection_builder.py`, `services/orion-cortex-exec/app/chat_stance.py`: use the anchor store; phase timing.
- `orion/substrate/tests/test_falkor_anchor_store.py`: unit lane plus a throwaway-Falkor equivalence lane.
- `.github/workflows/*`: the new tests are wired into the existing lanes.
- READMEs: governor and cortex-exec.

## Schema / bus / API changes

- Added: none on the bus. New MCP server env keys `ORION_READING_BINDING_FILE` and `ORION_INTROSPECT_BINDING_FILE`. These are internal: the harness renders them, they are not operator settings.
- Removed / renamed: none.
- Behavior changed: Hub chat turns use the warm pool. Stance builds read anchor nodes directly.
- Compatibility notes: the fixed-env binding path for spawned turns is unchanged.

## Env/config changes

- Added / removed / renamed keys: none in `.env_example`. Local `.env` sync not needed.

## Tests run

```text
orion/harness warm-pool CI lane (8 files): 98 passed; governor wiring: 4 passed
test_fcc_warm_pool_chat_binding.py: the Hub-path test FAILS on main's fcc_motor ('spawn' == 'warm') and passes here
orion/fcc/tests + introspect + world_pulse_read + harness: 290 passed, 1 failed (test_no_undiscovered_root_callers_of_claude_permission_argv, fails identically on main)
orion/substrate falkor_anchor_store + falkor_direct + relational (ORION_TEST_FALKOR_URI throwaway): 155 passed
cortex-exec stance/latency files: all pass; test_chat_general_stance_plumbing (9) / test_chat_relational_stance (5) fail identically on main (cwd-relative template paths)
```

## Evals run

```text
No eval harness for the stance build or the warm pool. Live equivalence check against production Falkor, read-only:
anchor_snapshot_ms 77.4, concept_region_ms 64.5, anchor_nodes 216 == hydrated 216, nodes_identical true, concept_region_identical true
```

## Docker/build/smoke checks

```text
Not run (read-only live inspection only, per task). No dependency or compose changes.
```

## Review findings fixed

- Finding (should): a warm slot without the reading tools could still serve a turn that has a reading binding.
  - Fix: `acquire` misses with `reading_tools_unavailable` (the slot stays idle), so the turn spawns.
  - Evidence: `test_slot_without_reading_tools_is_not_used_for_a_bound_turn`.
- Finding (should): one Falkor read error dropped every belief for the turn; the old store served its last good cache.
  - Fix: the anchor store serves its last good snapshot, logs `falkor_anchor_snapshot_failed`, and counts it.
  - Evidence: `test_read_error_serves_last_good_snapshot`, `test_read_error_with_no_good_snapshot_raises`.
- Finding (should): a duplicate identity key resolved differently in `snapshot()` (last one wins) and in `get_node_id_by_identity` (resolves to nothing).
  - Fix: both paths now resolve an ambiguous identity to nothing, and `snapshot()` logs it.
  - Evidence: `test_duplicate_identity_resolves_to_nothing_like_the_lookup`.
- Finding (should): the writer's unused in-memory cache grew for the life of the process.
  - Fix: the cache is reset after each write.
  - Evidence: `test_writer_cache_does_not_grow`.
- Finding (should): test gaps for a killed turn and for a failed binding write.
  - Fix: added `test_binding_cleared_after_a_killed_turn` and `test_binding_write_failure_falls_back`.
- Nits fixed: merged the release checks; the timing comment now says the snapshot counters are approximate; the `upsert_edge` annotation matches the writer.
- Nit not fixed: `warm_signature` does not include whether the reading tools are available. Covered by the first fix: a slot without the tools now misses instead of serving.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: medium. Concern: stance beliefs now read Falkor live on every turn instead of from a copy up to 30 s old. Mitigation: these are two bounded queries of about 80 ms each, and they are fresher than before.
- Severity: low. Concern: the materializer's write-through path now looks nodes up by identity with a direct query instead of from the cache. Mitigation: the only write-through producer (autonomy) returns None live, and writes still go through the same single `MERGE`.
- Severity: low. Concern: cortex-orch's warm mind projection (`build_cognitive_projection_for_mind_with_diagnostics`) shares `get_projection_unification_layer`, so it also switches to the anchor store. It gets faster; the anchors it reads are the same.
- Severity: low. Concern: `mind_ms` (14-16 s) was not investigated; that path belongs to orion-mind and is out of scope here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2518

🤖 Generated with [Claude Code](https://claude.com/claude-code)
