# PE retire step 6: node:substrate.execution, the last fake prediction-error node

## Summary

- Orion no longer reads a fake "execution surprise" number. `node:substrate.execution` was a clipped EWMA z of how much four cortex-exec pressure hints changed between batches, divided by 3. It was not a forecast: 59 % of readings were exact 0 because of the clipping, and spikes to 1.0 followed runs of zeros.
- Killed outright: `execution_prediction_error()`, its helpers and constants, the receipt, and the node write. The execution trajectory reducer and projection stay (`/projections/execution_trajectory` and the Hub chat-turn trace read them).
- The node joins the shared retired set and the field digester prune. It leaves `ACTIVE_INFERENCE_DOMAINS`, `PREDICTION_ERROR_NATIVE_TARGETS`, the self-model domain map, self-readings words, the glossary, the metric inventory, the metacog trend cue, mood-arc enrichment, the Hub domain list and the world-model `execution_context` slot.
- Every set that held only execution is now **empty**, and every reader says "unknown" instead of inventing a number. The self-model's `prediction_error_confidence` and `predicted_shift` read None, with a basis string that says why. The Hub attention organ shows "None live". The world-model step zero-fills `execution_context` and names it in `meta.zero_filled_groups`.
- Successors already exist and are scored forecasts: the "more steps than usual" turn self-bet inside `node:prediction.chat_turn`, and `node:prediction.body_joint` (#3732). Three consumers have no honest successor; they read absent and each has a ticket (below).

## Outcome moved

- **Before:** execution was the top attention target in 6,052 of 122,550 frames in the last 72 h (4.9 %; 14.7 % of the 41,102 frames with a winner). It sourced 17 % of goal ticks (72 h) and 19 % (7 d). It was the only input to the self-model's prediction-error confidence. It was 5.7 % of curiosity focal references.
- **After:** none of that rides on a fake number. Body goals stop (no native target is left); goals come only from attended world sources (decision 1 of the PE suite rebuild). Self-model confidence reads unknown until #3802 decides rebuild or retire.

## Before measurement (live, read-only, 2026-11-24 ~04:00Z; same tables and method as the #3732 report)

| consumer | node:substrate.execution now | after this PR |
|---|---|---|
| Attention frames, 72 h (`substrate_attention_frames`, `node_targets[0]`) | top target in 6,052 of 122,550 frames (4.9 %; 14.7 % of 41,102 frames with a winner); in dominant targets in 8,285 | gone; `node:prediction.chat_turn` was top in 0 frames in the same window (sparse: 61 writes / 7 d) |
| Goals (`field_dominance_run`, ticks past streak 3) | 72 h: 6,100 of 35,368 (17.2 %); 7 d: 14,976 of 78,977 (19.0 %) | 0. `PREDICTION_ERROR_NATIVE_TARGETS` is empty, so no body node authors a goal. World sources (perception, reading, chat, room joint) still do |
| Self-model confidence (`substrate_attention_self_model`, 72 h, 1,132 rows) | execution in 1,112 rows (98.2 %), mean 0.101; mean `prediction_error_confidence` 0.913; `predicted_shift` named execution in 412 rows (36 %) | `prediction_error_confidence` None on every row, basis "unknown: no Active-Inference prediction-error domain is live ..."; `predicted_shift` None |
| Endogenous curiosity focal refs, 7 d (`substrate_endogenous_curiosity_candidates`) | 282 of 4,986 (5.66 %): 214 via attention open loops, 68 via prediction error | gone (retired set + no glossary entry); chat_turn had 21 in the same window |

Queries (copy-paste, read-only):

```sql
-- frames
with f as (select frame_json->'node_targets'->0->>'target_id' t, jsonb_array_length(frame_json->'node_targets')>0 w
           from substrate_attention_frames where generated_at > now()-interval '72 hours')
select count(*), count(*) filter (where w), count(*) filter (where t='node:substrate.execution') from f;
-- goals
with r as (select target_id, greatest(tick_count - min_streak_at_run,0) past from field_dominance_run
           where ended_at > now()-interval '72 hours')
select sum(past) filter (where target_id='node:substrate.execution'), sum(past) from r;
-- self-model
select count(*), count(*) filter (where self_model_json->'prediction_error_by_domain' ? 'execution'),
       avg((self_model_json->>'prediction_error_confidence')::float)
from substrate_attention_self_model where generated_at > now()-interval '72 hours';
-- curiosity
with c as (select jsonb_array_elements(candidates_json) c from substrate_endogenous_curiosity_candidates
           where generated_at > now()-interval '7 days'),
     r as (select jsonb_array_elements_text(c->'focal_node_refs') ref from c)
select count(*) filter (where ref='node:substrate.execution'), count(*) from r;
```

Re-run the same queries after deploy for the after column.

## What each former reader reads now

| reader | before | now |
|---|---|---|
| Attention contest / world-first | execution as a body source | not a candidate (retired set, no glossary entry); `chat_turn` and `body_joint` compete as body sources |
| Goal provenance | execution as a native target | no body goals; attended world sources only |
| Self-model PE confidence, predicted shift | execution only | unknown (None), basis says why. **No successor** (#3802) |
| Equilibrium insight trigger, brain-frame honesty region | self-model PE confidence | absent (they already treat None as absent). **No successor** (#3802) |
| Self-signal episode gate / outcome rollup | execution as a body signal | alarms only (cabinet, strain). The gate observes `node:substrate.*` ids only, so the `node:prediction.*` successors never become episodes. **No successor** (#3801) |
| Self-readings prompt | "surprise in how my task execution is going" | gone; `chat_turn` and `body_joint` lines remain |
| Metacog trend cue | execution PE | biometrics-induction half only |
| Mood-arc live enrichment | `prediction_error_execution` | no PE channel; the v4 scorer 0.0-fills it, as for the other retired channels |
| Endogenous curiosity | execution seeds | `chat_turn` / `body_joint` / `prediction.body` seeds |
| World-model publish | execution scalar in `execution_context[0]` | zero-filled, named in `zero_filled_groups` |
| Non-chat turns (world pulse, curiosity, memory, self) | covered by execution (all cortex-exec runs) | **no surprise node**: the more-steps bet is settled for every caller but beats the base rate only for chat (#3803) |

## Metric quality gate (successor wiring)

This PR wires no new metric. It removes one and points readers at successors that are already live.

1. **Provenance.** `node:prediction.chat_turn`: `services/orion-prediction/app/turn_surprise.py` (bets `turn.self.asks_question`, `turn.self.more_steps_than_usual`, chat caller only), folded by `orion/substrate/prediction_turn_surprise.py`, written by substrate-runtime `_process_prediction_turn_surprise`. `node:prediction.body_joint`: #3732 (`family_a_live.py`, `prediction_joint_surprise.py`).
2. **Independence.** The more-steps bet scores `step_count` of a turn's run artifact against the bet's own stated p. Execution scored the change in four pressure hints between batches. Same harness runs, different quantity; the successor is a forecast and the predecessor was a delta. Body joint shares no input with execution.
3. **Theory anchor.** Both successors are scored forecasts: summed excess surprisal (nats) over the calm expectation, divided by calm spread (z), clip(z/3). Calm point is 0 by construction. Execution had no forecast behind it.
4. **Live sanity.** `node:prediction.chat_turn`, 7 d: 61 writes, 65.6 % exact 0, p50 0, p90 0.65, max 1.0 — it rests and it moves, but it is sparse (chat turns only). The more-steps bet: about 1,240 settled bets in 7 d across all callers, 62 of them chat. `node:prediction.body_joint`: **UNVERIFIED live** — #3732 merged 2026-11-24 03:54Z and has no history rows yet (needs its deploy; #3722).
5. **Existing mechanism.** No new producer, table, schema or channel. Reuses the retired set, digester prune and glossary rules used by steps 3-6.
6. **Reversibility.** A `git revert`. No migration; `ExecutionTrajectoryProjectionV1` keeps its three baseline fields (extra="forbid", stored rows carry them), marked retired.

## Current architecture

- substrate-runtime's `_execution_tick` reduced execution grammar into the trajectory projection, then scored `execution_prediction_error(prev, curr)`, saved a receipt and wrote `node:substrate.execution` every batch, and cached the scalar for the world-model tick.
- After #3732, execution was the last member of `ACTIVE_INFERENCE_DOMAINS`, `PREDICTION_ERROR_NATIVE_TARGETS`, the worker's domain map, the trend cue, mood-arc domains and the Hub domain list.

## Architecture touched

- **orion-substrate-runtime:** tick keeps the reducer only; domain map empty; world-model `execution_context` zero-filled.
- **Shared readers:** retired set, self-model (empty set + basis), selectors, self-readings, self-signal outcomes (`CIRCULAR_PAIRS` emptied), trend cue, mood-arc, semantics + metric lock, inner-state registry notes, glossary, PE readers registry, sentience instruments.
- **orion-field-digester:** prune. **orion-hub:** domain list, "None live" note, text.

## Files changed

- `orion/substrate/prediction_error.py`: function, helpers, constants deleted.
- `services/orion-substrate-runtime/app/{worker,world_model_features}.py`, `README.md`: producer removed, empty map, zero-filled group.
- `orion/substrate/{prediction_error_freshness,attention_self_model,metacog_trend_signals}.py`: retirement, empty set handling.
- `orion/attention/{field_attention/selectors,self_readings,self_signal_outcomes}.py`, `orion/mood_arc/corpus_enrichment.py`, `orion/metrics/semantics.py`, `orion/inner_state_registry.py`, `orion/schemas/{execution_projection,field_goal}.py`, `orion/substrate/attention/top_down.py`, `orion/introspect/tools.py`: readers and comments.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`, `config/substrate/prediction_error_readers.yaml`, `orion/sentience_striving_program/instruments.yaml` (new claim `execution_pe_retired`).
- `services/orion-field-digester/app/tensor/channels.py`: prune.
- `services/orion-hub/scripts/attention_organ_routes.py`, `static/js/{attention-organ,hub-surface}.js`: Hub.
- Scripts/evals: `measure_ast_hot_reducer.py`, `smoke_substrate_motivation_golden_path.py`, `replay_world_first.py`, `run_pe_magnitude_live_sanity.py`, `run_self_readings_live_eval.py`, `run_node_history_live_eval.py`.
- Tests across root, orion and services: vehicles moved off execution. `tests/fixtures/self_readings_history.py` gains a second TEST-ONLY probe, `EVENT_BODY` (execution's old event-written shape), because no live glossary node has that shape any more. Mechanism tests for empty sets (self-model, Hub, AST/HOT harness, mood-arc, outcomes) run on probe domains, plus new tests that pin the real sets empty and the readers honest.

## Schema / bus / API changes

- Added: none.
- Removed: `node:substrate.execution` PE writes and receipts; `execution_prediction_error`, `execution_raw_reading`; metric `node:substrate.execution.prediction_error` (lock re-generated).
- Behavior changed: `ACTIVE_INFERENCE_DOMAINS`, `PREDICTION_ERROR_NATIVE_TARGETS`, `_PREDICTION_ERROR_DOMAIN_NODE_IDS`, `CIRCULAR_PAIRS`, trend-cue node ids, mood-arc PE domains, Hub domain list are all empty. World-model `meta.zero_filled_groups` now includes `execution_context`; `real_execution_context_domains` is always `[]`.
- Compatibility: no migration. Stored execution projections still load (fields kept).

## Env/config changes

- Added / removed / renamed keys: none. `.env_example` unchanged; no local `.env` sync needed.

## Tests run

```text
Repo .venv, PYTHONDONTWRITEBYTECODE=1, -p no:cacheprovider, compared against an origin/main worktree.
- Root/orion/scripts affected set (50 files): see CI; local new failures vs main: 0 (pre-existing on main: test_attention_runtime_store / _worker collection errors in this venv).
- scripts/analysis/tests: 553 passed.
- orion-substrate-runtime: 607 passed; 14 failed + 10 errors, identical set on main (cursor/reducer fixtures).
- orion-equilibrium-service tests+evals: 289 passed.
- orion-thought tests+evals: all affected pass (truth eval + stance self-readings 35 passed).
- orion-attention-runtime: 59 passed (heartbeat_chassis collection error also on main).
- orion-hub affected: 308 passed. orion-field-digester: 297 passed (heartbeat_chassis cwd error also on main).
Static gates: check_definition_drift --gate PASS (re-locked), check_pe_alarm_readers OK, check_integration_seam OK, check_inner_state_registry OK, check_sentience_instruments (execution_pe_retired ok), smoke_substrate_motivation_golden_path GOLDEN_PATH_OK.
```

## Evals run

```text
services/orion-equilibrium-service/evals/test_self_signal_episode_replay_eval.py  PASS (bands re-recorded: 12 opens / 7.03 d with dwell, 21 without; alarms only)
services/orion-thought/evals/test_self_readings_truth_eval.py                      PASS
services/orion-attention-runtime/evals (world-first replay)                        PASS (spike check on chat_turn, storm on body_joint)
```

## Docker/build/smoke checks

```text
No compose or env change. No image built from the worktree. Live path UNVERIFIED until deploy.
```

## Review findings fixed

See the review section appended after the review subagent ran.

## Restart required

After merge, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

substrate-runtime first (stops the writes), digester second (prunes the field node). Verify: `SELECT count(*) FROM substrate_node_prediction_error_history WHERE node_id='node:substrate.execution' AND observed_at > now()-interval '10 minutes'` is 0, and the latest self-model row's `prediction_error_confidence_basis` starts with "unknown: no Active-Inference".

## Risks / concerns

- **Severity medium. Body goals stop entirely.** No native target is left, so every goal comes from an attended world source. This is the spec's direction (decision 1), but it is a step change: execution held 17-19 % of goal ticks. Measure goal rate after deploy.
- **Severity medium. Three consumers lose their input and read absent:** self-model PE confidence + insight trigger + brain-frame honesty (#3802); the self-signal episode gate sees only alarms (#3801); non-chat turns have no surprise node (#3803).
- **Severity low. body_joint is not live yet.** It needs #3732's deploy; until then the body side of the successor is UNVERIFIED (#3722).
- **Issues:** opened #3801, #3802, #3803. Fixes #3576.

## PR link

(filled in after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
