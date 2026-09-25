# fix(substrate): route prediction error averages only touched runs; delete topic_coherence

## Summary

- Route's "how surprising was that routing decision" number couldn't show surprise. It averaged over every routing run from the last 24 hours (~600-800), and every run the batch didn't touch scored 0 against itself. A real decision flip read ~0.0003. It now averages only over the runs the batch actually wrote. If the batch wrote none, it reads 0.0. (`route_prediction_error`, `_touched_runs`, `orion/substrate/prediction_error.py`)
- Chat's `topic_coherence` hint was just `1 - repair_pressure`, repair turned upside down, so chat's surprise number counted repair twice. It is deleted, and chat now weights repair and conversation load evenly. (`compute_chat_pressure_hints`, `chat_prediction_error`)
- Every prediction-error receipt now says which formula produced it (`after.definition_version`). The attention runtime restarts a domain's running baseline when that formula changes, and it only learns from receipts made with the current formula. That covers both deploy orders. (`orion/schemas/prediction_error_definitions.py`, `AttentionRuntimeStore.advance_node_prediction_error_baseline`, migration v2)
- New replay script: it runs the old and new definitions over the live projections. (`scripts/analysis/replay_route_chat_prediction_error_definitions.py`)

Juniper approved both definition changes ("hit it").

## Outcome moved

- Route can now read non-calm. Over the last 24h of live data, its maximum went from 0.0065 to 0.5. Ticks at or above 0.1 went from 0% to 1.3%.
- Chat's raw delta no longer double-counts repair. Its score distribution barely moves (mean 0.103 -> 0.115).
- Candidate A's running baselines for route and chat restart on the new formulas. Before, route and chat numbers made with different formulas would have been averaged together in one baseline.

## Current architecture

- `route_prediction_error(prev, curr)` took a categorical mismatch rate over `lane`/`lane_reason`/`output_mode`/`mind_requested` for **every** run in `curr.runs` (up to 2000 runs retained for 24h). New trace_ids were compared against prev's latest run.
- `chat_prediction_error` diffed `conversation_load`/`repair_pressure`/`topic_coherence` and z-scored the raw mean against an EWMA baseline stored on the projection.
- `substrate_node_prediction_error_baseline` (Candidate A precision) folded every receipt value it saw and had no idea which formula produced it.

## Architecture touched

- Producer: `orion-substrate-runtime` (route/chat prediction-error functions, receipt stamp).
- Consumer: `orion-attention-runtime` (baseline store).
- Contract: new `orion/schemas/prediction_error_definitions.py` (no pydantic model, no bus change); new column `substrate_node_prediction_error_baseline.definition_version`.
- Field digester: no code change. It already ignored `topic_coherence`, and the new `after.definition_version` sits outside `pressure_hints`, so no channel is created for it.

## Files changed

- `orion/substrate/prediction_error.py`: route averages touched runs only; chat diffs two keys; docstrings record v2.
- `orion/substrate/chat_loop/grammar_extract.py`: `topic_coherence` removed from `compute_chat_pressure_hints`.
- `orion/schemas/prediction_error_definitions.py`: per-reducer_key definition versions (route_arbitration=2, chat_session=2, rest 1).
- `services/orion-substrate-runtime/app/worker.py`: `_prediction_error_receipt` stamps `after.definition_version`.
- `services/orion-attention-runtime/app/store.py`: version-aware reset and fold; legacy fallback when the column is missing.
- `services/orion-sql-db/manual_migration_node_prediction_error_baseline_v2_definition_version.sql`: adds the column (default 1).
- `scripts/analysis/replay_route_chat_prediction_error_definitions.py`: before/after replay.
- Tests: `orion/substrate/tests/test_prediction_error.py`, `tests/test_chat_substrate_reducer.py`, `tests/test_attention_runtime_store.py`, `services/orion-substrate-runtime/tests/test_prediction_error_receipt_not_gated.py`, `tests/test_replay_route_chat_prediction_error_definitions.py`.
- Docs: `services/orion-substrate-runtime/README.md`, `services/orion-attention-runtime/README.md`, `orion/sentience_striving_program/README.md`.
- `config/metrics/metric_definitions.lock.json`: re-locked. The base moved and no definition delta was recorded (see Risks).

## Schema / bus / API changes

- Added: `after.definition_version` (int) on prediction-error receipts (`StateDeltaV1.after` is an untyped dict). Added the `definition_version` column on `substrate_node_prediction_error_baseline`.
- Removed: `topic_coherence` from `compute_chat_pressure_hints()` output, which is also in the chat reducer receipt's `pressure_hints`. The only other reader, the field digester, already dropped it. No typed schema field held it.
- Behavior changed: route prediction error (v2), chat prediction error (v2).
- Compatibility: the store reads the version with `to_jsonb(row) ->> 'definition_version'` and probes for the column. Before the migration it keeps the old behaviour and logs `node_prediction_error_baseline_definition_version_column_missing`.

## Metric quality gate (CLAUDE.md §0A), for the two changed metrics

**route_prediction_error v2**

1. Provenance: `route_prediction_error()` <- `_route_tick` (worker.py) loads the projection before and after `process_route_grammar_events`. The reducer stamps the tick's clock into `last_updated_at` on every run it creates or merges (`route_loop/grammar_extract.py:52`, `merge.py`). Eviction only deletes runs. So "new or changed" is exactly the set the batch wrote.
2. Independence: only 8 of 605 live ticks in 24h were nonzero. 6 of those are lane flips to or from `chat` (a chat turn being routed), and 2 are a `code_delivery` output mode flipping in and out. So most of its signal coincides with chat turns, which the chat domain also sees. It is **partly redundant with chat's tick timing**, not independent. It does catch non-chat decision changes (code_delivery).
3. Theory anchor: a prediction error is surprise relative to the last observation. For a categorical decision, "did this decision differ from the previous one" is that surprise. v1 answered a different question: what fraction of the last 24h of runs changed this tick. That fraction is bounded near 0 by construction.
4. Live data: see the replay below. v2 can rest at an exact 0.0 because the background decisions really are identical. It can also fire (0.25/0.33/0.5) and return to 0. It is not flat-at-a-floor. It is sparse: 98.7% zeros.
5. Existing mechanism: this is the same function, re-scoped. There is no other route surprise signal.
6. Reversibility: cheap. Revert the function and set `route_arbitration` back to 1 in the versions map, and the baseline resets again.

**chat_prediction_error v2 (topic_coherence removed)**

1. Provenance: `compute_chat_pressure_hints()`. `topic_coherence = max(0, 1 - repair_pressure_level)`.
2. Independence: `topic_coherence` failed. It is an affine transform of `repair_pressure`, and every repair change moved both by the same amount. Removing it takes repair's weight from 2/3 to 1/2: v1 raw = (d_load + 2 d_repair)/3, v2 raw = (d_load + d_repair)/2.
3. Theory anchor: unchanged (hint-change surprise). This only removes a duplicate.
4. Live data: see the replay below. Scores are not degenerate: 34% nonzero, max 1.0, 432 distinct values.
5-6. Removal only. Cheap to revert.

## Before/after replay (real data)

Source: the live `substrate_route_arbitration_projection` (611 runs, 2026-09-24 05:56 to 2026-09-25 05:55 UTC, 605 ticks) and `substrate_chat_session_projection` (1316 turns, 2026-07-24 to 2026-09-25, 1315 scored ticks). Both were dumped at 2026-09-25 ~05:56 UTC. Ticks are recovered from each run's or turn's reducer clock stamp. Attention level for "fraction >= x" is 0.1.

```text
## route_prediction_error
v1: n=605 min=0 mean=3.84e-05 p50=0 max=0.00649 frac_zero=0.987 frac>=0.1=0.000 distinct=9
v2: n=605 min=0 mean=0.00551  p50=0 max=0.5     frac_zero=0.987 frac>=0.1=0.013 distinct=4
v2 nonzero ticks: 8 values={0.25, 0.333, 0.5}
## chat_prediction_error (final 0-1 score)
v1: n=1315 min=0 mean=0.1028 p50=0 max=1 frac_zero=0.615 frac>=0.1=0.253 distinct=488
v2: n=1315 min=0 mean=0.1150 p50=0 max=1 frac_zero=0.658 frac>=0.1=0.249 distinct=432
|v1 - v2| per tick: mean=0.029 max=0.380
chat raw per-tick delta: v1 mean=5.35e-4 max=0.051; v2 mean=6.27e-4 max=0.063
chat variance-floor binding (5e-8): v1 771/1315 ticks, v2 548/1315 ticks
switchover (v2 carrying v1's projection EWMA vs v2's own): max diff 0.27, all later diffs < 0.01 after 13 ticks
```

The v1 route replay matches the live receipts: every `substrate.route_arbitration` receipt in the retained 30-minute window at dump time read 0.0.

**Is route v2 still degenerate?** Not in the §0A sense. It rests at a true 0 and fires on real decision changes. But it is sparse, and it mostly marks "a chat turn was just routed", which chat already sees. **The case for removing route from `ACTIVE_INFERENCE_DOMAINS` and `PREDICTION_ERROR_NATIVE_TARGETS` is not unambiguous, so this PR does not remove it.** Juniper to decide. See Risks.

## Baseline decision

- **Candidate A baseline (`substrate_node_prediction_error_baseline`), route and chat: reset, via code.** Both formulas changed. The route row holds ewma 1.7e-17 and variance 1.9e-20 over 10,254 v1 observations. Chat's replayed variance roughly doubles under v2 (0.0070 -> 0.0149), so carrying the v1 baseline would overstate chat's precision. The reset is automatic. `definition_version` for those two keys is now 2, and on its first tick after the migration the attention runtime sees stored version 1, restarts the row cold (keeping the cursor), and folds only v2-stamped receipts. Cost: route regains Candidate A's 20-observation qualifying count in roughly 15-20 minutes of route ticks. Chat needs about 20 chat turns. Execution, biometrics, and bus_synaptic are untouched (v1 == v1).
- **Chat's own projection EWMA (`ChatSessionProjectionV1.prediction_error_baseline_*`): not reset.** The switchover replay shows a carried v1 baseline converging to v2's own within 13 chat ticks (alpha 0.2). A reset would itself cost a cold start, and it would need a new field on an `extra="forbid"` projection (a consumer-first rollout). Route has no projection-level baseline.

## Tests run

```text
pytest orion/substrate/tests/test_prediction_error.py tests/test_attention_runtime_store.py tests/test_chat_substrate_reducer.py -> 100 passed
pytest services/orion-substrate-runtime/tests/test_prediction_error_receipt_not_gated.py -> 4 passed
pytest tests/test_replay_route_chat_prediction_error_definitions.py -> 2 passed
pytest services/orion-field-digester/tests/test_field_chat_perturbations.py -> 5 passed
pytest orion/substrate/tests orion/attention tests/test_attention_candidate_precision_weighted.py -> 8 failed: the identical failure set on main a65691c4d (mutation-store tests, unrelated)
pytest services/orion-substrate-runtime/tests (--continue-on-collection-errors) -> the identical failure/error set on main (26 lines, only log line numbers differ)
scripts/check_metric_lineage.py --gate PASS; check_definition_drift.py --gate PASS; check_inner_state_registry OK;
check_sentience_instruments --static-only OK; check_env_template_parity PASS; test_grammar_event_producer_catalog + test_agent_trace_schema_registry 5 passed
```

## Evals run

```text
python scripts/analysis/replay_route_chat_prediction_error_definitions.py --route-json route.json --chat-json chat.json
(output above; the dump commands are in the script's docstring)
```

## Docker/build/smoke checks

```text
Not run: no Docker/compose/requirements change. Runtime behaviour changes at restart. See the post-deploy checks.
```

## Review findings fixed

Review: code-review subagent, run on branch `fix/route-pe-and-topic-coherence` against merge base a65691c4d.

- Finding (medium): a route decision can arrive split across two ticks (trace-start first, decision next). The first tick creates an all-"unknown" run, and v2 would have read 0.75 twice for one ordinary decision. Latent: 0 of 613 live runs are `unknown`.
  - Fix: touched runs with no decision are skipped. A run whose previous copy had no decision is compared against the latest *decided* run (`_route_run_has_decision`).
  - Evidence: `test_route_prediction_error_split_decision_does_not_spike`. Replay numbers unchanged.
- Finding (low): the column probe was not scoped to a schema, and after a column drop it was never re-probed.
  - Fix: `table_schema = current_schema()`. Any advance failure clears the cached answer.
  - Evidence: `test_column_probe_is_scoped_to_the_current_schema`, `test_advance_failure_forces_a_column_re_probe`.
- Finding (low): receipts skipped for a version mismatch were dropped with no log. A rolled-back producer would leave a target stuck at 0 observations with no visible cause.
  - Fix: a `node_prediction_error_baseline_version_skipped ... skipped=N` warning.
  - Evidence: `test_version_skipped_receipts_are_logged`.
- Finding (low): the definition-drift lock says "no definition changes".
  - Fix: disclosed under Risks, with Juniper's approval recorded in the Summary. Feeding `PREDICTION_ERROR_DEFINITION_VERSIONS` into the gate is a follow-up.
- Finding (low, test fidelity): the store tests are MagicMock-based, and the versioned upsert has never run against real Postgres.
  - Fix: the post-deploy checks below verify it live. The reviewer ran the new read expression read-only on live Postgres, and it returns null before the migration, as intended.
- Finding (nit): after the reset, chat's fresh v2 baseline folds about 13 scores that were computed against the chat projection's carried-over v1 average (max diff 0.27 in the replay).
  - Fix: accepted and disclosed here. It washes out at alpha 0.2.

## Restart required

Order matters only for how quickly the baselines refill. Every order is safe.

```bash
# 1. migration (additive, metadata-only)
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_node_prediction_error_baseline_v2_definition_version.sql
# 2. producer, then consumer (from a worktree at merged main)
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
```

Post-deploy live checks:

```bash
P="docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc"
# receipts stamped v2
$P "select created_at, receipt_json->'state_deltas'->0->'after'->>'definition_version', receipt_json->'state_deltas'->0->'after'->'pressure_hints'->>'prediction_error' from substrate_reduction_receipts where reducer_name='substrate.route_arbitration' order by created_at desc limit 5"
# route/chat baselines reset to v2 and refilling; others untouched
$P "select target_id, definition_version, observation_count, ewma, variance, last_value from substrate_node_prediction_error_baseline order by target_id"
# expect exactly two reset lines (route, chat), no column_missing, no version_skipped after the substrate runtime is up
docker logs orion-attention-runtime 2>&1 | grep -E "definition_reset|definition_version_column_missing|version_skipped"
# route reads non-calm on the next chat turn (expect 0.25-0.5 once, then 0.0)
```

## Risks / concerns

- Severity: medium. Chat is still diluted by the same disease route had. `chat_prediction_error` averages over every turn ever stored (1316, with no eviction), so its raw delta is ~200x smaller than a touched-only average (mean 6.3e-4 vs 0.138). The 5e-8 variance floor binds on 548/1315 v2 ticks. Mitigation: out of scope, because it is a third definition change Juniper has not approved. The replay's `v2+touched` row shows the fix: floor binding drops to 2 ticks and scores at or above 0.1 go from 25% to 29%. Recommended as the next patch, with a `chat_session` bump to 3.
- Severity: medium. Route v2 is sparse, and most of its signal marks chat turns (6 of 8 pulses). Mitigation: kept in place. The decision to retire it from `ACTIVE_INFERENCE_DOMAINS`/`PREDICTION_ERROR_NATIVE_TARGETS` (and kill its tick) is Juniper's.
- Severity: low. The definition-drift lock does not cover prediction-error formulas. It only reads registry meanings, so this PR's definition changes show up as "no definition changes". Mitigation: the per-domain version map now makes formula changes explicit in code. Registering the formulas in the lock is a follow-up.
- Severity: low. If the migration is skipped, the reset never happens. The store logs a warning every 5 minutes and otherwise behaves as before.
- Severity: low. Behaviour changes for consumers of route's value: the world-model `execution_context` feature, AST/HOT `prediction_error_by_domain`, equilibrium, curiosity, and Candidate A. Route now shows short 0.25-0.5 pulses where it read ~0. This is the intended fix, approved by Juniper.
- Separate, not in scope: chat only ticks on chat turns, so its last value sits in attention for hours.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2332

🤖 Generated with [Claude Code](https://claude.com/claude-code)
