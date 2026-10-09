# fix(attention): stop feeding attention numbers that aren't true (stale readings, diluted chat surprise, unfloored gateway failures)

## Summary

Three places where Orion's attention was being handed a number that did not mean what it looked like. Juniper approved all three ("do 1-3"). Route is left as is (her call).

- **Old readings stop counting as "now" (A).** Each domain's surprise number in the attention competition is the last one its reducer wrote. Chat only writes when a turn lands, so a quiet chat's last number used to stay "current" for hours and win. It now fades to zero over 30 minutes, measured from the receipt that produced it. The running average behind it still learns only from real readings. (`prediction_error_staleness_factor`, `precision_weighted_salience_from_baseline`, `candidate_precision_weighted.py`; clock set in `AttentionRuntimeStore.advance_node_prediction_error_baseline` and persisted in a new `last_value_observed_at` column, so skipped receipts cannot make an old reading look fresh)
- **Chat surprise is no longer divided by its whole history (B).** Chat's surprise averaged a new turn's change over every turn ever stored (~1,700 on 2026-09-29, never evicted). It now averages over only the turns the batch touched, the same fix route got on 2026-09-25. Chat's formula version moves 2 -> 3, so attention's chat baseline restarts by itself. Chat's variance floor was re-derived for the new scale. (`chat_prediction_error`, `prediction_error.py`; `prediction_error_definitions.py`)
- **One gateway timeout no longer reads as total failure (C).** The LLM gateway's failure number was one minute's failures over one minute's calls, with no floor. One timeout on a one-call minute read 1.0 and held until the next busy minute. It now uses the RPC delivery bridge's rule over a rolling 10 minutes: failures / max(calls, 10), and 0 until there are 2 failures. The reading is the worse of the whole node and the worst single worker, and the receipt names which one set it. (`orion/substrate/llm_inference_loop/failure_window.py`, reusing `rpc_delivery.hop_pressure` and `RpcDeliveryConfig` defaults)
- The gateway now reports per-worker call and failure counts, so the substrate can name the failing lane. (`grammar_emit.py`, additive `worker_attempted=`/`worker_failed=` keys)
- There are two new replay scripts and one updated one, all run on live data (numbers below).

## Outcome moved

- **A, last 72 h (2026-09-26 05:24 to 09-29 05:24, 4,321 minutes).** Before the fade, chat was the top target on 713 minutes. On 53 of those its reading was over 30 minutes old, and on 290 it was over 10 minutes old. After the fade, chat is the top target on 498 minutes, with 0 over 30 minutes old and 104 over 10 minutes old. The top target changes on 255 minutes.
- **B, the full stored chat history (1,691 scored turns).** The raw per-turn change goes from a mean of 0.00051 to 0.144, so the new turn is no longer divided by the whole history. The variance floor, not the data, set the score on 922 turns (55%) before and on 2 turns (the cold start) after.
- **C, last 72 h (4,319 minutes).** Old: `node:circe` failure was nonzero on 41 minutes, max 1.0, 8 minutes at 1.0, 12 minutes at or above 0.5. Live field ticks agree: nonzero on 1,167 ticks, max 1.0. New: nonzero on 38 minutes, max 0.057, 0 minutes at or above 0.5.

## Current architecture

- Candidate A (`select_node_targets`) scores `precision x |last_value|` per `node:substrate.*` domain from a persisted EWMA baseline. `last_value` had no age.
- `chat_prediction_error` diffed hints over every turn in `curr.turns`. Turns the batch did not touch contributed 0.0 deltas.
- The `llm_inference` reducer replaced each node's state per 60 s gateway window and wrote `inference_failure_pressure = failed / (served + failed)` for that window. The field digester holds the last value (`mode="replace"`, no decay).

## Architecture touched

- `orion-attention-runtime`: the read-side fade, plus the chat baseline reset via the version bump.
- `orion-substrate-runtime`: chat v3, the rolling failure reading, and the projection field `recent_windows`.
- `orion-llm-gateway`: two extra key=value pairs in the node window atom's summary.
- Contracts: `LlmInferenceWindowCountV1` (new, registered), `LlmInferenceProjectionV1.recent_windows` (new, default empty), `after.failure_window` on llm_inference receipts (untyped dict), `chat_session` definition version 3.
- Codebase node path (checked, as asked): `node:substrate.codebase` is not a Candidate A target. It goes through Candidate B novelty, and the field merge already has its own staleness cut (`MERGE_STALENESS_THRESHOLD_SEC`, `orion/field/pressure.py`), so the fade does not apply there. **Separate finding:** the live attention runtime has been running for 3 days on code from before the 2026-09-25 novelty fix (D1). It still diffs against the prior `salience_score` (confirmed in the container: `scoring.py:46 return found.salience_score`). That locks codebase's novelty at 1.0 while its pressure reads 0.0 (observed 2026-09-28 17:26-17:27, 612 minutes in 72 h). Rebuilding the attention runtime for this PR ships that fix too.

## Files changed

- `orion/attention/field_attention/candidate_precision_weighted.py`: horizon constant, `prediction_error_staleness_factor`, `last_observed_at` on the baseline, and the fade plus result fields (`raw_error`, `staleness_factor`, `reading_age_sec`).
- `orion/attention/field_attention/selectors.py`, `builder.py`: pass `now`, and add a "stale reading" reason.
- `services/orion-attention-runtime/app/store.py`: stamps `last_observed_at` from the last folded receipt and persists it in `last_value_observed_at` (probed like `definition_version`; falls back to the cursor until the column exists).
- `services/orion-sql-db/manual_migration_node_prediction_error_baseline_v3_last_value_observed_at.sql` (new): the nullable column.
- `orion/attention/field_attention/candidate_precision_weighted.py` `normalize_across_targets`: an all-zero set normalizes to 0.0, not a 1.0 tie.
- `orion/substrate/prediction_error.py`: chat touched-only (v3), and the chat floor set to 3e-5.
- `orion/schemas/prediction_error_definitions.py`: `chat_session` 2 -> 3.
- `orion/substrate/llm_inference_loop/failure_window.py` (new), `extract.py`, `reducer.py`: the rolling floored reading. The per-window `inference_failure_pressure()` function is deleted.
- `orion/schemas/llm_inference_projection.py`, `orion/schemas/registry.py`: `LlmInferenceWindowCountV1` and `recent_windows`.
- `services/orion-llm-gateway/app/grammar_emit.py`: per-worker counts.
- `services/orion-sql-db/manual_migration_chat_projection_pe_baseline_v3_reset.sql`: one-shot zeroing of the chat projection's EWMA fields.
- `scripts/analysis/replay_candidate_a_staleness_fade.py`, `replay_llm_inference_failure_window.py` (new), `replay_route_chat_prediction_error_definitions.py` (chat v2 frozen, v3 live, v2->v3 switchover probe).
- Tests: `tests/test_attention_candidate_precision_weighted.py`, `tests/test_attention_field_selectors.py`, `tests/test_attention_runtime_store.py`, `orion/substrate/tests/test_prediction_error.py`, `tests/test_llm_inference_substrate_reducer.py`, `tests/test_replay_attention_input_honesty.py` (new), `tests/test_replay_route_chat_prediction_error_definitions.py`, `services/orion-substrate-runtime/tests/test_worker_llm_inference_tick.py`, `.../test_prediction_error_receipt_not_gated.py`.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: `inference_failure_pressure`'s meaning said "per gateway window", which stopped being true; corrected, and the lock regenerated after merging main so it records the change.
- Docs: READMEs for `orion-attention-runtime`, `orion-substrate-runtime`, and `orion-llm-gateway`, plus a note in `orion-llm-gateway/evals/run_inference_outcome_eval.py`.

## Schema / bus / API changes

- Added: `substrate_node_prediction_error_baseline.last_value_observed_at` (nullable timestamptz, manual migration), `LlmInferenceWindowCountV1`, `LlmInferenceProjectionV1.recent_windows` (default `{}`), receipt `after.failure_window`, and the gateway summary keys `worker_attempted` / `worker_failed`.
- Removed: `orion.substrate.llm_inference_loop.extract.inference_failure_pressure()` (per-window ratio, only its test used it).
- Behavior changed: `inference_failure_pressure` (rolling and floored), `chat_prediction_error` (v3), Candidate A's current error (faded), and `normalize_across_targets` on an all-zero set (0.0, was 1.0).
- Compatibility:
  - `LlmInferenceProjectionV1` is an `extra="forbid"` singleton row, read and written only by `orion-substrate-runtime`. New code reads the old row: `scripts/check_substrate_projection_schema_drift.py` passes against live Postgres.
  - **Rollback hazard:** old code cannot read a row that has `recent_windows`. Rolling back substrate-runtime needs `DELETE FROM substrate_llm_inference_projection;` (the row is a cache and rebuilds from the next window).
  - The gateway keys are ignored by the old reducer (`_KV_RE` just captures them), so deploy order is free.
  - The chat version bump uses the existing reset mechanism, which is safe in either service order.
- No bus channel changes.

## Env/config changes

- Added keys: none. Removed/renamed: none. No `.env_example` changed, so no sync was needed. No flags.

## Metric quality gate (AGENTS.md §0A)

**A. Faded current error (Candidate A).**
1. Provenance: `precision_weighted_salience_from_baseline` <- `baseline.last_value` (receipt `after.pressure_hints.prediction_error`) and `last_observed_at` (the created_at of the last folded receipt, `store.py`).
2. Independence: this is not a new metric. It is the same value with an age weight.
3. Theory anchor: precision-weighting ranks *current* prediction error (Feldman & Friston 2010). A reading from 45 minutes ago is not current error. The linear horizon is the substrate's existing one (`PressureConfig.prediction_error_decay_horizon_seconds`, 1800 s), reused and not re-picked. A test keeps the two equal.
4. Live data: the replay above. Fast domains (biometrics, bus_synaptic, execution, route) are essentially unaffected, since their readings are seconds old. Chat loses exactly its stale wins, and a faded target returns to 0, which is a real rest state.
5. Existing mechanism: `endogenous_curiosity._prediction_error_staleness_decay`, `pressure.prediction_error_pressure`, and `bus_synaptic_surprise.STALENESS_HORIZON_SEC` already do this. This copies their shape.
6. Reversibility: cheap. Read-side only, nothing persisted.

**B. chat_prediction_error v3.**
1. Provenance: `chat_prediction_error` <- `_touched_runs(prev.turns, curr.turns)` <- `compute_chat_pressure_hints`. The chat reducer rewrites a turn only when an event for it arrives (`chat_loop/reducer.py:109`).
2. Independence: unchanged from v2, since the same two hints are diffed.
3. Theory anchor: surprise is the change the new observation brought. v2 answered "average change across every turn ever stored", which is bounded near 0 by construction (the route v1 defect).
4. Live data: see "Before/after replay" below. v3 is not degenerate: raw p50 0.044, max 0.85, 28% exact zeros (consecutive turns with identical hints). It can rest at 0.
5. Existing mechanism: the same `_touched_runs` route uses.
6. Reversibility: revert the function and set the version back. The baseline resets again.

**C. inference_failure_pressure (rolling, floored).**
1. Provenance: gateway `_NodeBucket.summary` -> reducer `extract_llm_inference_windows` -> `fold_window`/`failure_reading` -> `hop_pressure`.
2. Independence: unchanged from before, since it counts the same failures. It remains distinct from `rpc_timeout_pressure` (an error *reply* is a success there).
3. Theory anchor: this is a failure rate, which has a natural zero. The floor and minimum count exist so that one event cannot be the whole rate. This is the RPC bridge's own reasoning, and its constants are reused.
4. Live data: it rests at exact 0.0 almost all the time. It fired during the 2026-09-28 slow periods (max 0.057), and it can reach 1.0 (10 failures in 10 minutes). Checked that it is not stuck: 7 episodes, longest 10 minutes (the window length).
5. Existing mechanism: `orion/substrate/rpc_delivery.py`, reused.
6. Reversibility: moderate, because `recent_windows` is persisted (see the rollback hazard above).

## Before/after replay (real data)

```text
# A: python scripts/analysis/replay_candidate_a_staleness_fade.py --frames frames.tsv   (72 h, 1 frame/min)
span 2026-09-26T05:24:52 .. 2026-09-29T05:24:01  minutes=4321
top-1 changed on 255 minutes
target | before top-1 (on a reading >30 min old) | after top-1 (>30 min old)
node:substrate.biometrics   | 1442 (0)  | 1559 (0)
node:substrate.execution    |  463 (0)  |  488 (0)
node:substrate.chat         |  713 (53) |  498 (0)
node:substrate.route        |  250 (6)  |  231 (0)
node:substrate.bus_synaptic | 1453 (0)  | 1545 (0)
(--stale-min 10: chat 713 (290) -> 498 (104); route 250 (34) -> 231 (15))

# B: python scripts/analysis/replay_route_chat_prediction_error_definitions.py --chat-json chat.json  (1,692 turns, 2026-07-24..09-29)
v2 score: n=1691 mean=0.109 frac_zero=0.652 frac>=0.1=0.257
v2 raw:   mean=0.000513 p50=6.1e-05 max=0.0629 ; variance-floor bound on 922 ticks
v3 score: n=1691 mean=0.141 frac_zero=0.643 frac>=0.1=0.299
v3 raw:   mean=0.144 p50=0.0435 max=0.846 ; variance-floor bound on 2 ticks
switchover v2->v3 without zeroing the projection EWMA: max diff 0.977, 11 turns to settle, 3 false 1.0 readings

# C: python scripts/analysis/replay_llm_inference_failure_window.py --events gw.jsonl  (72 h, 7,317 gateway events)
windows with any upstream failure: 35; max failures in one window: 1
old: nonzero_minutes=41 max=1.000 episodes=34 minutes_at_1.0=8 minutes_>=0.5=12
new: nonzero_minutes=38 max=0.057 episodes=7 longest_hold_min=10 minutes_at_1.0=0 minutes_>=0.5=0
new nonzero writes by scope: {'node': 36}
live field, same 72 h: node:circe inference_failure_pressure nonzero on 1,167 of 122,998 ticks, max 1.0
```

Dump commands are in each script's docstring.

Spot-check against live data (2026-09-29, after the draft, fixed span `created_at` 2026-09-26 05:24 .. 09-29 05:24):
C re-run on a fresh dump gives `new: nonzero_minutes=38 max=0.057 episodes=7 minutes_>=0.5=0` (identical to the table) and
`old: nonzero_minutes=40 max=1.000 minutes_at_1.0=8 minutes_>=0.5=11` (41/12 in the table). The gap is retention: the fresh
dump's first gateway event is 08:06, so the first ~2.7 h of the original span has since been pruned (7,043 events vs 7,317).

Replay limits:
- A uses the field's `node_vector_updated_at` as the reading's age. Live code uses the receipt's `created_at`, which lands a few seconds earlier. Receipts are pruned after 30 minutes, so the proxy could not be cross-checked historically; if anything rewrites the node vector's timestamp without a new receipt, the proxy under-counts age, so the replay's stale counts are lower bounds.
- A, all-faded sets: 0 minutes in the span where every competitor's raw salience was 0, before or after the fade, so the normalization guard below does not change any replay number.
- A cannot combine with B's new chat values, because the chat baseline restarts at deploy.
- C cannot replay per-worker scope, because old summaries have no worker counts. Every failed window in 72 h held a single failure, so per-minute flooring alone would have read 0 always. That is why the window is rolling.

## Baseline decisions

- **Candidate A chat baseline: resets automatically** (version 2 -> 3). Chat needs about 20 turns to regain full confidence.
- **Chat projection's own EWMA: zeroed at deploy** by the one-shot SQL. Without it, the replay shows 3 false 1.0 surprises in the first 11 turns, and those would seed the fresh Candidate A baseline.
- **`inference_failure_pressure`: no baseline.** A new row starts with an empty `recent_windows` and fills within 10 minutes.

## Tests run

```text
pytest tests/test_attention_candidate_precision_weighted.py tests/test_attention_field_selectors.py tests/test_attention_runtime_store.py orion/attention -> pass
pytest orion/substrate/tests/test_prediction_error.py tests/test_llm_inference_substrate_reducer.py tests/test_replay_attention_input_honesty.py tests/test_replay_route_chat_prediction_error_definitions.py -> pass
broad sweep (orion/substrate/tests orion/attention tests/test_*attention* *prediction_error* *llm_inference* test_chat_* test_field_* *substrate*): 10 failed, 1629 passed, 6 errors;
  the same 10 failures / 6 collection errors on main 69dd8902e (felt_state_self_definition_lane, chat prompts, cognitive_substrate phase4/8/9, tests/test_field_* collection)
services (from each service dir): substrate-runtime 13 failed/362 passed/10 errors (identical on main); attention-runtime 39 passed + 1 pre-existing collection error
  (test_heartbeat_chassis, same on main); llm-gateway 322 passed; field-digester 254 passed, 6 skipped, 1 pre-existing error
static gates: check_metric_lineage --gate PASS; check_definition_drift --gate PASS; check_inner_state_registry OK; check_sentience_instruments --static-only OK;
  check_scripts_dir_no_stdlib_shadow clean; check_system_health_producers OK; check_control_surface_store_parity OK;
  test_agent_trace_schema_registry / test_grammar_event_producer_catalog / test_substrate_services_declare_requests / test_field_topology_edges 14 passed
check_substrate_projection_schema_drift (live Postgres): OK, all 8 singleton rows validate against the new schema
```

## Evals run

```text
The three replays above are this PR's evals (live data, before/after).
services/orion-llm-gateway/evals -> pass (part of the 322).
```

## Docker/build/smoke checks

```text
Not run. No compose, requirements, or Dockerfile change. Deploys are from the primary checkout on main after merge (commands below).
```

## Review findings fixed

Review: subagent code review over `git diff origin/main...HEAD` (after merging main).

- Finding (material): the reading's time was re-read from the receipt cursor on every later tick, and the cursor also moves over skipped receipts (malformed, or another definition version). During a substrate-only rollback, which this report's own rollback path allows, old-version chat receipts arrive every tick, so chat's last real reading would look seconds old forever. The in-tick test asserted the right time and never checked the reload.
  - Fix: persist the real time in a new nullable `last_value_observed_at` column (manual migration, probed the same way as `definition_version`, read via `to_jsonb` so the read works before the migration). Written only when a value was actually folded. Cursor fallback only for rows not yet re-folded or before the migration.
  - Evidence: `test_observed_time_survives_a_skipped_receipt_across_ticks` (two ticks, skipped row after the folded one, reload keeps t2), `test_observed_time_is_not_written_without_the_column`, `test_nothing_folded_leaves_the_persisted_observed_time_alone`.
- Finding (minor): once every competitor is faded to exactly 0, `normalize_across_targets`' tie rule read the whole set as salience 1.0.
  - Fix: all-zero set -> 0.0. Nonzero ties still read 1.0.
  - Evidence: `test_normalize_across_targets_all_zero_reads_zero_not_a_tie`, `test_a_fully_faded_set_reads_zero_salience_not_a_tie_at_the_top`; replay A shows 0 affected minutes in 72 h.
- Finding (minor): replay A's age proxy. Fix: disclosed as a lower bound under Replay limits.
- Finding (minor, not changed): `inference_failure_pressure` holds its last value when gateway traffic stops. Kept as disclosed in Risks; a down node reading its last failure share is defensible.
- Finding (nit, not changed): the compatibility wrapper `extract_llm_inference_states_from_events` now returns no failure reading; no caller other than the reducer exists (checked).

## Restart required

After merge, from the primary checkout on `main`, in this order. Order within a service matters. Across services, every order is safe (see Compatibility).

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only

# 1. substrate-runtime: build first, then stop, zero the chat EWMA, start.
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime build
docker stop orion-athena-substrate-runtime
mkdir -p /tmp/chat-pe-v3-reset && docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
  "select projection_json - 'turns' from substrate_chat_session_projection" > /tmp/chat-pe-v3-reset/before.json
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_chat_projection_pe_baseline_v3_reset.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build

# 2. attention-runtime (fade + chat v3 baseline reset; also ships the 2026-09-25 novelty fix it is missing).
#    Column first so the reading time is persisted from the first tick (either order is safe).
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_node_prediction_error_baseline_v3_last_value_observed_at.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-attention-runtime up -d --build

# 3. llm-gateway (per-worker counts; restarts the LLM serving path, pick a quiet moment)
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-llm-gateway up -d --build
```

Rollback of substrate-runtime only:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DELETE FROM substrate_llm_inference_projection;"
```

## Post-deploy checks

```bash
# chat v3 receipts, and the chat projection EWMA restarting from 0
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select created_at, receipt_json->'state_deltas'->0->'after'->>'definition_version', receipt_json->'state_deltas'->0->'after'->'pressure_hints'->>'prediction_error' from substrate_reduction_receipts where reducer_name='substrate.chat_session' and created_at > now()-interval '30 minutes' order by created_at desc limit 5"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select projection_json->>'prediction_error_baseline_ewma_n' from substrate_chat_session_projection"
# attention: chat baseline reset logged, then stale readings faded in frames
docker logs orion-athena-attention-runtime 2>&1 | grep node_prediction_error_baseline_definition_reset | tail -2
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select generated_at, x->'reasons' from substrate_attention_frames f, jsonb_array_elements(f.frame_json->'node_targets' || f.frame_json->'suppressed_targets') x where f.generated_at > now()-interval '10 minutes' and x->>'target_id'='node:substrate.chat' order by 1 desc limit 3"
# attention: codebase novelty no longer locked at 1.0 with pressure 0 (D1 fix now deployed)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*) from substrate_attention_frames f, jsonb_array_elements(f.frame_json->'node_targets') x where f.generated_at > now()-interval '30 minutes' and (x->>'novelty_score')::float = 1 and (x->>'pressure_score')::float = 0"
# gateway: failure_window on receipts, worker counts in summaries
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select created_at, receipt_json->'state_deltas'->0->'after'->'failure_window' from substrate_reduction_receipts where reducer_name='llm_inference_reducer' and created_at > now()-interval '10 minutes' order by created_at desc limit 3"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select event_json->'atom'->>'summary' from grammar_events where source_service='orion-llm-gateway' and created_at > now()-interval '5 minutes' and event_json->'atom'->>'semantic_role'='llm_inference_window_observed' order by created_at desc limit 2"
```

## Risks / concerns

- Severity: medium. Concern: the `LlmInferenceProjectionV1` rollback hazard (old code rejects `recent_windows`). Mitigation: the one-line DELETE above, since the row is a cache.
- Severity: low. Concern: until the `last_value_observed_at` migration is applied (and for each row until its next real reading), the fade falls back to the receipt cursor, which moves over skipped receipts and can make a reading look fresher than it is. Mitigation: migration is in the deploy steps; the service logs `node_prediction_error_baseline_observed_at_column_missing` while it is absent.
- Severity: low. Concern: the definition lock now records C (`high semantics_changed .../inference_failure_pressure`, via the corrected field-channel glossary meaning), but it cannot record B: `chat_prediction_error` is not in any registry the drift gate resolves, and the gate does not track `PREDICTION_ERROR_DEFINITION_VERSIONS` (the same gap disclosed on 2026-09-25). Mitigation: Juniper's approval is recorded in the Summary; B's version bump is the machine-readable record.
- Severity: low. Concern: once the rolling window has no calls, the field digester holds the last written value (unchanged digester contract: "not measured" is not written as 0.0). Live cadence (~1 window with circe calls every 2 minutes) makes this rare.
- Severity: info. Concern: the attention runtime was 3 days stale in production (missing the 09-25 D1 novelty fix). Rebuilding ships everything merged since. Mitigation: post-deploy check above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2400

🤖 Generated with [Claude Code](https://claude.com/claude-code)
