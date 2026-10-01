# PR: Orion's first learned world action -- notice the cabinet warming, hold back background GPU work, check whether it cooled

## Summary

- Orion can now notice its cabinet warming up. When the cabinet is already warm and keeps getting
  warmer, that becomes a "something is surprising here" signal that competes for Orion's attention
  through the same path every other surprise signal already uses. Nothing new was built just to let
  it in. (`orion/autonomy/cabinet_heat.py`, `node:substrate.cabinet`, `SUBSTRATE_CABINET_HEAT_ATTENTION_ENABLED`, OFF)
- When that focus is on Orion's body heat, Orion can propose one action. The cabinet has to be warm
  but not hot, it has to be rising, the AC has to be healthy, and some background GPU work has to be
  running or waiting. The action stops *new* background GPU work from starting for up to 15 minutes.
  Running work finishes, and system, chat and urgent work are never touched. (`shed_background_gpu`, `orion_self_shed`)
- The GPU pool owns the brakes. It allows one shed at a time, at most 15 minutes each, at most 1 hour
  a day, with 15 minutes between sheds, and it remembers this across a restart. It refuses outright
  while the AC-failure reflex is active. If an AC problem appears mid-shed, it drops Orion's shed
  immediately so the reflex takes over. (`orion/gpu_pool/orion_shed.py`, `GPU_POOL_ORION_SHED_ENABLED`, OFF)
- Before acting, Orion writes down what it expects to happen. Half the eligible decisions are held
  back at random as a comparison group. Twenty minutes later both groups are scored against the real
  cabinet temperature sensor. Orion may then record that it "acted" on that attention loop. It can
  never mark the loop resolved or dismissed. (`substrate_world_action_episodes`, `orion/feedback/world_settlement.py`)
- One chain of ids runs from what Orion focused on all the way to the learned outcome. A new eval and
  two new links in the agency-episode audit check that this chain is complete.
  (`orion/autonomy/evals/run_attend_act_loop_eval.py`, `attention_winner` / `loop_outcome` links)

## Outcome moved

Before this patch, attention never reached action. The thing Orion focused on (the workspace
broadcast winner) never fed the proposal builder, nothing on the action menu touched the world, and
nothing wrote a scored outcome after 2026-09-08. After this patch, with the flags on, one complete
loop exists: focus, then proposal, then policy, then the allocator, then a pool shed, then a settled
result, then a sensor-scored outcome against a control group, then a loop verdict. It is UNVERIFIED
on the live rail until Juniper flips the flags (see "Restart required" and "Post-deploy proof").

## Current architecture

- Attention: `services/orion-substrate-runtime` `_attention_broadcast_tick` admits substrate-graph
  nodes whose `metadata.dynamic_pressure >= ORION_ATTENTION_BROADCAST_MIN_SALIENCE` (live 0.05).
  `dynamic_pressure` is written only by `SubstrateDynamicsEngine` (`orion/substrate/dynamics.py`), as
  `0.6 * prediction_error`, seeded from the prediction-error nodes written by
  `_write_prediction_error_node`. Over the last 72 h the cabinet, circe and rpc_delivery never won
  attention, because nothing wrote a prediction error for them.
- Proposals bound only to the field-attention ranking (`attention.dominant_targets[0]`). Nothing
  read the workspace winner.
- The hardware-watch reflex (PR series for `feat/hardware-watch-and-shedding`) is live. Its
  `cooling_incident` named reason shows on the pool's `/health` with `GPU_POOL_SHED_ENABLED=true`, and
  `hardware_watch_incident` has 0 rows. `cabinet_rise_c` already lived as a shared pure function in
  `orion/hardware_watch/rules.py`.

## Architecture touched

- **Contracts:** `orion/schemas/{gpu_pool,proposal_frame,execution_dispatch_frame,policy_decision_frame,action_prediction,attention_salience}.py`,
  `orion/schemas/registry.py`, `orion/bus/channels.yaml`.
- **Attention:** `services/orion-substrate-runtime` gets a cabinet tick that writes
  `node:substrate.cabinet`'s `prediction_error` through the existing writer.
- **Proposals:** a new `workspace.winner` binding, the `shed_background_gpu` template, and
  `orion/autonomy/self_shed.py` (eligibility). `services/orion-proposal-runtime` reads the
  broadcast projection, cabinet rows, hardware-watch `/health`, pool occupancy and the episode ledger.
- **Policy:** the `self_regulate` kind leads to `approved_self_reversible` with scope `self_reversible`.
- **Dispatch:** the `SELF_REVERSIBLE_SCOPE` route plus the master switch, a per-template holdback, a
  precommit episode row, the shed RPC executor, and shed settlement (`orion/execution_dispatch/shed_settlement.py`).
- **GPU pool:** the `orion_self_shed` reason, the shed RPC, caps, the ledger, TTL expiry,
  preempt-on-AC-incident, drain bookkeeping, and `/health`.
- **Feedback:** settle-time scoring, overlap tags, control cells, and the `acted` verdict. The field
  window skips `cabinet_heat_pressure`.
- **Hub:** `_VALID_VERDICTS` accepts `acted`.
- **Eval/audit:** the agency-episode motor lane gains `attention_winner` and `loop_outcome`;
  `run_attend_act_loop_eval.py` (fixture + live); and `scripts/analysis/replay_attention_eligibility.py`.

## Files changed

- `orion/autonomy/cabinet_heat.py`: both cabinet numbers, from one source column. The level (the
  outcome) and the warming error (the attention trigger) are kept separate on purpose, plus the shared SQL read.
- `orion/autonomy/self_shed.py`: winner binding (90 s, dwell 2, `binds_to_nodes`) and the eligibility rules plus snapshot.
- `orion/autonomy/world_episodes.py`: the episode ledger. It holds the precommit and the chain join key. Lazy DDL.
- `orion/autonomy/attend_act_loop.py`: `check_chain` (the D4 rules) and `contrast` (three ways).
- `orion/autonomy/evals/run_attend_act_loop_eval.py` + `fixtures/attend_act_loop_chain.json`: fixture and live eval.
- `orion/autonomy/agency_episode.py`, `agency_episode_reader.py`: the two new motor-lane links and a world-episode precommit.
- `orion/gpu_pool/shed.py`: the `orion_self_shed` reason (precedence 1, background only).
- `orion/gpu_pool/orion_shed.py`: the pool-side controller (caps, ledger, terminal states, manipulation check).
- `services/orion-gpu-pool/app/{runtime,main,settings,orion_shed_store}.py`, `.env_example`,
  `docker-compose.yml`, `README.md`: the RPC, the incident preempt, and the expiry-before-schedule ordering.
- `orion/proposals/{builder,policy,templates}.py`, `config/proposals/proposal_policy.v1.yaml`: the
  `workspace.winner` candidates and the template.
- `orion/policy/{evaluator,builder}.py`, `config/policy/substrate_policy.v1.yaml`: the `self_regulate` kind rule.
- `orion/execution_dispatch/{builder,policy,shed_settlement}.py`,
  `config/execution_dispatch/execution_dispatch_policy.v1.yaml`: the scope, the world gate and the route.
- `services/orion-execution-dispatch-runtime/app/{worker,store,settings}.py`, `.env_example`,
  `docker-compose.yml`, `README.md`: holdback, precommit, the shed executor, and settlement.
- `orion/feedback/{world_settlement,outcome_resolution}.py`,
  `services/orion-feedback-runtime/app/{worker,store,settings}.py`, `.env_example`,
  `docker-compose.yml`, `README.md`: settle-time scoring.
- `services/orion-proposal-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`,
  `README.md`, `tests/conftest.py`: the workspace context.
- `services/orion-substrate-runtime/app/{worker,store,settings}.py`, `.env_example`, `docker-compose.yml`,
  `README.md`: the cabinet tick.
- `services/orion-hub/scripts/attention_loops_store.py`: `acted` is now a valid verdict.
- `services/orion-sql-db/manual_migration_{gpu_pool_orion_shed,world_action_episodes}_v1.sql`: operator migrations.
- `scripts/analysis/replay_attention_eligibility.py`: the 72 h attention replay.
- `scripts/sync_local_env_from_example.py`: sync prefixes for the new keys. Without them the sync
  would have silently skipped them.
- `config/metrics/metric_definitions.lock.json`: re-locked. Two bus channels were added.
- `.github/workflows/attend-act-loop.yml`: the CI lane for the new tests and the eval.
- Tests: `tests/test_shed_background_gpu_pipeline.py`, `tests/test_world_action_dispatch.py`,
  `tests/test_world_settlement_scoring.py`, `services/orion-gpu-pool/tests/test_runtime_orion_shed.py`,
  `services/orion-proposal-runtime/tests/test_workspace_winner_context.py`,
  `services/orion-feedback-runtime/tests/test_world_settlement_pass.py`,
  `services/orion-substrate-runtime/tests/test_cabinet_heat_tick.py`,
  `orion/autonomy/tests/test_agency_episode_world_links.py`, plus a hub verdict test. Two existing
  tests were updated to name the new template (`test_proposal_policy_loader`, `test_proposal_scoring`).

## Schema / bus / API changes

- Added:
  - Bus RPC `orion:gpu_pool:shed:request` (`GpuPoolShedReasonRequestV1`) and reply
    `orion:gpu_pool:shed:reply:*` (`GpuPoolShedResultV1`), both registered.
  - `AttentionWinnerRefV1`.
  - New fields `ProposalCandidateV1.attention_winner` / `.world_eligibility`,
    `ExecutionDispatchCandidateV1.world_action`, `ProposalTemplateV1.binds_to_nodes` / `.holdback_fraction`.
  - New literal values: proposal/dispatch kind `self_regulate`, gate `self_reversible`, decision
    `approved_self_reversible`, tier/scope `self_reversible`, verdict `acted`, PredictableSignal
    `cabinet_heat_pressure`.
  - Tables `gpu_pool_orion_shed` and `substrate_world_action_episodes`.
- Removed: none.
- Renamed: none.
- Behavior changed: the field-window outcome path skips `cabinet_heat_pressure`
  (`settle_time_signal:*`). The pool's incident handler now runs under the runtime lock.
- **Compatibility notes:** every model above is `extra="forbid"`. That makes this a consumer-first
  rollout: every reader of proposal, policy and dispatch frames has to deploy before the writer that
  starts emitting the new fields (deploy order below). Old rows still validate because every field is
  additive and nullable. Reverie prompts will show `outcome: {verdict: "acted", ...}` for a loop Orion
  acted on (`services/orion-thought/app/store.py::load_recent_loop_outcomes`). That is true and
  intended.

## Env/config changes

- Added keys:
  - **orion-gpu-pool:** `GPU_POOL_ORION_SHED_ENABLED` (**false**), `GPU_POOL_ORION_SHED_MAX_TTL_SEC`
    900, `GPU_POOL_ORION_SHED_MAX_SEC_PER_DAY` 3600, `GPU_POOL_ORION_SHED_MIN_GAP_SEC` 900.
  - **orion-execution-dispatch-runtime:** `ORION_WORLD_ACTIONS_ENABLED` (**false**),
    `ORION_WORLD_ACTIONS_ALLOWED` (empty), `ORION_WORLD_ACTION_ELIGIBILITY_MAX_AGE_SEC` 120,
    `ORION_GPU_POOL_SHED_RPC_TIMEOUT_SEC` 5, `ORION_SHED_TTL_SEC` 900.
  - **orion-proposal-runtime:** `ORION_WORKSPACE_WINNER_PROPOSALS_ENABLED` (record-only, **true** in
    `.env_example`), `ORION_HARDWARE_WATCH_HEALTH_URL`, `ORION_SHED_RISE_THRESHOLD_C` 0.5.
  - **orion-feedback-runtime:** `ORION_WORLD_SETTLEMENT_SCORING_ENABLED` (record-only, **true**),
    `ORION_WORLD_SETTLEMENT_INTERVAL_SEC` 30.
  - **orion-substrate-runtime:** `SUBSTRATE_CABINET_HEAT_ATTENTION_ENABLED` (**false**: it changes what
    wins attention), `SUBSTRATE_CABINET_HEAT_TICK_INTERVAL_SEC`, `SUBSTRATE_CABINET_HEAT_RISE_THRESHOLD_C`.
- Removed keys: none. Renamed keys: none.
- Every key above is OFF in code. The `.env_example` files set the two record-only flags ON; every
  flag that lets Orion act or changes attention is OFF there too.
- `.env_example` updated: yes, five services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Every new key is
  verified present in the primary checkout's `services/*/.env`.
- Skipped keys requiring operator action: none.

## Metric quality gate

### `cabinet_heat_pressure` (the outcome level)
1. **Provenance:** `orion/telemetry/cabinet_sensors.py:109` `put("cabinet_temp_c", ...)`, then
   `orion_biometrics_summary.measurements->>'cabinet_temp_c'`, node athena. This is the same column the
   hardware-watch reflex reads.
2. **Independence:** it comes from a physical air sensor, not from GPU/CPU/host telemetry. It is kept
   off the field (no field channel), so it is never maxed or merged with other pressures.
3. **Theory anchor:** cabinet air temperature follows the heat balance (power dissipated minus heat
   the AC removes). The scale sits on the thermal gate's own constants (28.0 re-arm, 29.5 trip, 32.0 hot).
4. **Live sanity** (7 days, 19,356 samples, live eval): 14.6% of readings sit at 0 and 5.9% are
   saturated. Every reading below 28.0 maps to 0. It is not flat and it regularly returns to calm.
5. **Existing mechanism:** reuses the rows and the thermal-gate constants. No field channel already
   carried this, and the scorer reads the same rows through one shared function.
6. **Reversibility:** one `PredictableSignal` value plus this action's posterior cells. No stored
   consumer besides that.

### Cabinet warming error (the attention signal, `node:substrate.cabinet.prediction_error`)
1. **Provenance:** the same rows. `read_cabinet_heat()` folds the thermal gate with hysteresis, plus
   the shared `cabinet_rise_c` (15 min).
2. **Independence:** this is a rise, not the level, so the trigger and the outcome are different
   numbers read from one source. It is not added to `ACTIVE_INFERENCE_DOMAINS`, so the self-model's
   prediction-error aggregates are untouched.
3. **Theory anchor:** with the AC healthy, the heat balance predicts a steady temperature. The 15-min
   change has an SD of 0.42 C with nothing done, so a 0.5 C rise is about 1.2 sigma. The value is
   clamp01(rise / 1.0 C), and is 0 unless the cabinet is elevated and rising by at least 0.5 C.
4. **Live sanity** (7 days on a 1-minute grid): 402 of 9,960 minutes are non-zero (4.0%), across 52
   separate episodes, and the signal returns to exactly 0 between them. That is not degenerate. It
   fires *less* than "elevated" alone, which holds about 65% of the time.
5. **Existing mechanism:** `_write_prediction_error_node` plus the dynamics engine, unchanged.
6. **Reversibility:** one flag. The node then holds its last value: written every tick while on, and
   the dynamics' 30-min decay applies once it stops being refreshed.

## Attention replay (72 h, real broadcast log, `scripts/analysis/replay_attention_eligibility.py`)

Fidelity check: the replay *without* the new signals reproduces the logged winner on **6,091 of 6,091**
bottom-up ticks. Another 1,129 ticks were decided by a goal override; those cannot be replayed and
are counted as "no change".

| Candidate | Ticks non-zero | Would win | Bindable (dwell ≥ 2) | Longest run | Hours it won in | What it displaces |
|---|---|---|---|---|---|---|
| cabinet warming | 214 / 7,220 (3.0%) | 177 (2.45%) | 122 | 32 ticks (~16 min) | 11 | nothing selected 87, harness_closure 41, chat 17, codebase 13, route 10, execution 8, perception 1; **biometrics 0** |
| rpc_delivery timeout pressure | 320 | 15 (0.2%) | 4 | 4 | 7 | nothing selected 6, codebase 4, chat 4 |
| circe inference-failure pressure | 159 | 31 (0.4%) | 15 | 12 | 9 | nothing selected 28, other 3 |

**Decision:** only the cabinet is wired in. It wins about 2.5% of ticks, in bursts, mostly when
nothing else was selected, so it is not a permanent winner. rpc_delivery and circe are **not** wired:
their values are a timeout ratio and a failure ratio, not prediction errors, so labelling them as
prediction errors would be dishonest. No action in this PR consumes them either (A2/A3 are stage 2),
and they would win only 0.2-0.4% of ticks. The replay script is the evidence for whoever picks up stage 2.

## How the action decides, acts and settles

1. **Decide (proposal runtime, every field tick):** if the workspace winner is
   `node:substrate.{biometrics,cabinet}`, at most 90 s old and has held for at least 2 ticks, read the
   world. All of these must hold:
   - the thermal gate reads `elevated` on a fresh reading;
   - the cabinet rose at least 0.5 C in 15 min;
   - hardware-watch is reachable, enabled and ticking, with zero open incidents;
   - at least one background lease is granted or queued;
   - no earlier episode is still unscored.

   Otherwise the frame records `world_action_ineligible:<reasons>` or `winner_unbindable:<reason>`.
2. **Gate:** policy turns `self_regulate` into `approved_self_reversible`. Dispatch blocks it as
   `world_actions_disabled` / `world_action_not_allowed` unless both flags name it. The allocator
   scores it at the **unchanged** 0.02 nats/s floor.
3. **Act (execution dispatch):**
   - If the eligibility snapshot is older than 120 s, the action is blocked (`world_eligibility_stale`).
   - Otherwise a coin is flipped with p = 0.5. Held back: a `control` episode row is written and
     nothing is sent. Treated: a `treated` episode row (with the expected effect) and a `shed_pending`
     result are written first, then one RPC sets `orion_self_shed` for 900 s.
   - The motor cost is the RPC's wall time.
4. **Pool:** the pool re-checks every cap and the reflex. It blocks new background grants and counts
   what it held back and when the running background work drained. It ends the shed on `expired`,
   `cancelled` (the clear verb or the kill switch) or `preempted_by_reflex` (an AC incident opened).
5. **Settle (dispatch, every 30 s):** the terminal state is read from the pool ledger. The outcome is
   emitted once. A row with no terminal state by t0 + TTL + 300 s becomes `settlement_timeout`.
6. **Score (feedback, every 30 s):** at t0 + 20 min, take cabinet minute-means at t0 and at t0 + 20 min.
   - An `expired` treated row gets a ledger row and a posterior update.
   - A control row gets a `randomized_holdback` ledger row and a control cell.
   - `overlap:reflex` is excluded in both arms. `overlap:heat_incident` and `overlap:render_gate` are
     kept as covariates.
   - If the shed actually started, Orion writes `acted` (non-final) on the loop.
7. **Next tick:** the loop keeps competing. `acted` is not in `TERMINAL_VERDICTS`. The verdict
   records the loop's salience on the next broadcast tick.

**Allocator floor, verified (eval check3 + `tests/test_shed_background_gpu_pipeline.py`):**
- Cold, the action is worth ~0.99 nats. At 1-2 s of motor cost that is ~0.5-1 nats/s, and it is
  admitted. Before any cost sample exists it is costed at the 5 s typical, which is still admitted.
- After 12 settled treated rows at 1 s it is still admitted.
- Charging the 900 s TTL as motor cost would give ~0.001 nats/s and the action would never run.
  Both numbers are recorded.
- `ORION_DISPATCH_MIN_NATS_PER_SEC` is untouched.

**Live dry run (read-only, 2026-10-01 02:2xZ):** the eligibility read ran against production data.
The cabinet read 27.07 C (`normal`), rose 0.01 C, hardware-watch was reachable and ticking with 0
incidents, and the pool had 0 background leases. The verdict was ineligible
(`thermal_not_elevated:normal`, `cabinet_not_rising`, `no_background_work`), which is correct for a
cool, idle cabinet.

## Acceptance checks (amended design 0-10)

| # | Check | Covered by | Status |
|---|---|---|---|
| 0 | Reflex live; guard shows `cooling_incident` as a named reason | `test_check0_...` (pool); live `/health` shows `cooling_incident` with the lever enabled | Reflex is live. Its simulated-incident smoke (hardware-watch check 7) was **not re-run here** (production test hook disabled): UNVERIFIED |
| 1 | `cabinet_heat_pressure` 0 below 28.0, 0.375 at 29.5, on real rows | `test_check1_...`, eval `check1_pressure_scale`, live eval `check1_live` (19,356 rows, every reading below 28 maps to 0) | PASS (live rows) |
| 2 | Proposal with `attention_winner` ≤ 90 s and eligibility snapshot | `test_check2_...`, eval winner rules | PASS in tests; live UNVERIFIED (flags off) |
| 3 | Admitted by `cold_start` or ≥ 0.02 nats/s, not via the baseline lane | `test_check3_...`, eval `check3_clears_floor_cold_and_warm` | PASS (real allocator); live UNVERIFIED |
| 4 | Settled `expired` result, `orion_self_shed` record, manipulation check, precommit before dispatch | `test_treated_precommits_before_the_rpc...`, `test_settlement_publishes_once...`, pool tests | PASS in tests (fake pool); live UNVERIFIED |
| 5 | `substrate_action_outcomes` row on `cabinet_heat_pressure`, t0 → t0+20 min | `test_check5_...`, `test_world_settlement_pass` | PASS in tests; live UNVERIFIED |
| 6 | `acted` by orion on the loop; loop not excluded next tick | `test_check6_...`, hub test, `check_chain` | PASS in tests; live UNVERIFIED |
| 7 | A holdback control outcome with the reflex idle | `test_holdback_writes_a_control_episode...`, `test_check7_...` | PASS in tests; live UNVERIFIED |
| 8 | `--live` eval: ≥ 1 closed episode; 0 proposals while an incident was open | `run_attend_act_loop_eval.py --live` | Ran live: 0 episodes (ledger table absent), `check8_proposals_while_incident_open: 0` -> **UNVERIFIED** until the flags are on |
| 9 | Kill-switch drill: flag off, active shed cancelled, next background lease granted | `test_check9_kill_switch_drill...` (fake pool) | PASS in tests; live drill not run (production) |
| 10 | Reflex takes over mid-shed: only `cooling_incident`, system stops, `preempted_by_reflex`, no new proposal, `overlap:reflex` out of the posterior | `test_check10_reflex_takes_over_mid_shed`, `test_ac_incident_open_without_shed_request_still_preempts`, `test_check10_reflex_overlap_excluded_in_both_arms`, eligibility `hardware_watch_incident_open` | PASS in tests (fake pool/watcher); never run against production |

## Tests run

```text
services/orion-gpu-pool/tests                                  147 passed, 15 skipped
tests/test_shed_background_gpu_pipeline.py                     26 passed
tests/test_world_action_dispatch.py                            15 passed
tests/test_world_settlement_scoring.py                         12 passed
orion/autonomy/tests (incl. agency-episode world links)        222 passed
orion/gpu_pool/tests + orion/autonomy/tests                    563 passed
services/orion-proposal-runtime/tests                          7 passed
services/orion-feedback-runtime/tests                          62 passed
services/orion-execution-dispatch-runtime/tests                107 passed
services/orion-substrate-runtime/tests/test_cabinet_heat_tick  4 passed
services/orion-hub/tests/test_attention_loop_closure.py        4 passed
Root proposal/policy/dispatch/feedback/allocator suites: every failing test fails identically on a clean
origin/main worktree (same FAILED set, diffed). They are pre-existing and env-sensitive, not caused here.
substrate-runtime suite: 16 pre-existing failures, the same set on origin/main.
Static gates: env parity PASS, bus reply channels PASS, definition drift PASS (re-locked: 2 bus
channels added), metric lineage PASS, schema/registry tests PASS, gpu_pool config PASS, and every other
orion-static-gates script PASS.
```

## Evals run

```text
python orion/autonomy/evals/run_attend_act_loop_eval.py            20/20 (fixture)
ORION_PG_DSN=... run_attend_act_loop_eval.py --live                 check1_live ok; warming signal 402/9960 min,
                                                                     52 episodes, returns to 0; episodes: none yet;
                                                                     check8: 0; verdict UNVERIFIED (flags off)
python orion/autonomy/evals/run_agency_episode_eval.py             10/10 (report shape unchanged)
scripts/analysis/replay_attention_eligibility.py --hours 72        table above; fidelity 6091/6091
```

## Docker/build/smoke checks

```text
Not built or deployed (Juniper's call). Migrations validated against the live database inside
BEGIN ... ROLLBACK (both tables created and counted, then rolled back; nothing persisted).
Live read-only dry run of the proposal-side eligibility read against production rows + hardware-watch /health.
```

## Review findings fixed

A code-review subagent reviewed `git diff origin/main...HEAD`. It found no blockers. It found three
fix-before-enable items, which would have quietly spoiled the treated-vs-control comparison, and nine
should-fix items. All are fixed. The reviewer's "ledger never retries" finding was already fixed in
b44cb226c before the review finished.

- Finding (fix-before-enable): several queued proposals for one warming event could each reach
  dispatch before the first one's episode row existed, so treated and control rows overlapped.
  - Fix: `_world_admission_refusal` re-checks "is anything in flight" at dispatch, right before the
    draw, and allows at most one world decision per template per tick (`world_action_in_flight`).
  - Evidence: `test_admission_is_checked_before_the_draw...[in_flight]`, `test_two_world_candidates_in_one_tick_only_one_decides`.
- Finding (fix-before-enable): the pool's gap and daily caps refused only treated decisions. Control
  rows kept coming from a different population (inside a previous shed's tail, after the day cap was
  used up, or with the pool flag off).
  - Fix: the same caps are checked *before* the draw, from the episode ledger
    (`pool_gap`, `pool_daily_cap`, `pool_refusing:*`; `ORION_SHED_MIN_GAP_SEC` / `_MAX_SEC_PER_DAY`
    mirror the pool's caps). A decision the pool would refuse becomes neither arm.
  - Evidence: four parametrized cases in `tests/test_world_action_dispatch.py`.
- Finding (fix-before-enable): a crash followed by a replay could redraw the arm. A shed could then
  go out on a row recorded as control.
  - Fix: the draw is now a hash of `dispatch_id`. A treated send whose precommit already exists as
    `control` is refused (`world_episode_recorded_as_control`).
  - Evidence: `test_the_draw_is_stable_per_decision...`, `test_replay_of_a_control_decision_never_sheds`.
- Finding: Orion's `acted` could become the "latest verdict" and re-arm a loop Juniper had dismissed.
  - Fix: `verdicts.load_terminal_verdict_loop_ids` and orion-thought's `load_recent_loop_outcomes` now
    skip `acted`.
  - Evidence: `test_orions_acted_verdict_never_shadows_a_human_dismissal`; the live query was run against production.
- Finding: the pool's `reflex_active` check missed an AC incident that was open but not yet requesting a shed.
  - Fix: the pool tracks open `cooling` incidents and refuses while any are open.
  - Evidence: `test_open_ac_incident_refuses_even_without_a_shed_request`.
- Finding: a ledger failure disabled the feature until the next restart.
  - Fix (b44cb226c): the next set retries `ensure` plus the history reload.
  - Evidence: `test_ledger_down_at_boot_recovers_on_the_next_set`.
- Finding: a failed table setup could leave `lock_timeout=3s` on a pooled connection.
  - Fix: one transaction with `SET LOCAL`.
  - Evidence: code; the pool suite passes (147).
- Finding: the shed settlement query errored forever when `gpu_pool_orion_shed` was absent.
  - Fix: a `to_regclass` check, falling back to `pool_row = NULL`, so orphans still settle; the table
    name is schema-qualified.
  - Evidence: the SQL was run live inside a rolled-back transaction.
- Finding: binding matched any member of the coalition, so `acted` could land on an unrelated loop.
  - Fix: bind only on the selected loop's own node.
  - Evidence: `test_binding_follows_the_selected_loops_node...`.
- Finding: the control arm was scored on a hard-coded clock.
  - Fix: both arms store `settlement.ttl_sec` at precommit.
  - Evidence: a live rolled-back insert reads back `{'ttl_sec': 900, ...}`.
- Finding: world candidates could push field candidates out of the frame even with dispatch off.
  - Fix: they are appended outside `max_candidates`.
  - Evidence: `test_world_candidates_never_push_field_candidates_out_of_the_frame`.
- Finding: the `holdback_fraction` comment contradicted the code.
  - Fix: the comment now says None means no per-template holdback.
- Nits fixed: the pool's `_recent` is pruned to the cap window. The dispatch README documents the
  feedback-scoring dependency and the global-holdback interaction.
- Nits left as stated risks:
  - A tripwire probe whose only candidate is a shed cannot score a success (≤ 4 sheds/day).
  - The broadcast-log lookup matches on exact timestamp equality (UNVERIFIED live). The post-deploy
    proof query for `winner_unbindable:no_broadcast_log_row` would show a mismatch.
  - When the sensor goes stale the cabinet node stops being refreshed and decays (eligibility refuses on unknown).

## Restart required

Nothing is deployed by this PR. Deploy order is **consumer-first** (the additive fields sit on
`extra="forbid"` models). Apply the migrations first:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_gpu_pool_orion_shed_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_world_action_episodes_v1.sql
# readers of proposal/policy/dispatch frames first
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-consolidation-runtime up -d --build
scripts/safe_docker_build.sh orion-dream up -d --build
scripts/safe_docker_build.sh orion-policy-runtime up -d --build
scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
# the writer of the new proposal fields last
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
```

**Flags Juniper flips, in this order** (restart that one service after each; each is independently reversible):

1. `services/orion-substrate-runtime/.env` `SUBSTRATE_CABINET_HEAT_ATTENTION_ENABLED=true`. This lets
   the cabinet compete for attention. Watch it for a day before going further. It changes only what
   Orion attends to.
2. `services/orion-gpu-pool/.env` `GPU_POOL_ORION_SHED_ENABLED=true`. The pool now accepts Orion's
   shed (caps apply). On its own this does nothing, because nothing sends a shed yet.
3. `services/orion-execution-dispatch-runtime/.env` `ORION_WORLD_ACTIONS_ENABLED=true` **and**
   `ORION_WORLD_ACTIONS_ALLOWED=shed_background_gpu`. This lets Orion act.

**Kill:** set (3) back to false and restart dispatch, or set (2) back to false and restart the pool
(an active Orion shed settles `cancelled`). Neither touches the reflex.

## Post-deploy proof queries

```sql
-- the cabinet competing for attention (after flag 1)
SELECT generated_at, projection_json->'attended_node_ids', projection_json->>'selected_action_type',
       projection_json->>'dwell_ticks' FROM substrate_attention_broadcast_log
 WHERE projection_json->'attended_node_ids' ? 'node:substrate.cabinet' ORDER BY generated_at DESC LIMIT 10;
-- proposals and why they were or were not emitted
SELECT created_at, w FROM substrate_proposal_frames, jsonb_array_elements_text(proposal_frame_json->'warnings') w
 WHERE created_at > now() - interval '1 hour' AND (w LIKE 'world_action%' OR w LIKE 'winner_unbindable%') ORDER BY 1 DESC LIMIT 20;
-- the chain (checks 2, 4, 5, 6, 7)
SELECT episode_id, arm, decided_at, open_loop_id, settlement_state, outcome->>'observed_delta',
       outcome->>'excluded_reason', outcome->'overlap', loop_outcome_id FROM substrate_world_action_episodes ORDER BY decided_at DESC;
SELECT * FROM gpu_pool_orion_shed ORDER BY requested_at DESC LIMIT 10;
SELECT dispatch_id, arm, baseline, observed_after, observed_delta, surprise_nats FROM substrate_action_outcomes
 WHERE signal_id = 'cabinet_heat_pressure' ORDER BY observed_at DESC;
SELECT loop_id, verdict, actor, features_at_close FROM attention_loop_outcome WHERE actor = 'orion' ORDER BY created_at DESC;
```
```bash
curl -s localhost:8127/health | jq .shed.orion_self_shed   # caps, 24 h use, active shed
ORION_PG_DSN=... python orion/autonomy/evals/run_attend_act_loop_eval.py --live   # check 8 + contrast
```

## Risks / concerns

- Severity: medium. Concern: the allocator retires the action after about 12-24 treated rows, well
  before the ~47 per arm the design says it needs to fit anything (stated in the design). Stage 1
  proves the loop closes; it cannot finish learning whether shedding helps. Mitigation: stage 3
  (current-bin scoring / `confidently_helpful`) needs Juniper's yes.
- Severity: medium. Concern: world candidates bypass the tick-level `action_warrant` gate, the same way
  the visual baseline does. Their own trigger is stricter and physical, and every other gate still
  applies (policy, allocator floor, pool caps, reflex). Mitigation: it is recorded in the builder
  docstring and visible on every frame.
- Severity: low. Concern: hold-children are inherited. Whether a running background hold's child
  calls count as "new grants" is the reflex's existing rule (U4), unchanged here. Mitigation: the
  manipulation check (`drained_at`, `grants_withheld`) shows what actually happened per shed.
- Severity: low. Concern: if the process crashes between the `shed_pending` write and the RPC, the
  episode never sends and is orphaned as `settlement_timeout` at t0 + 20 min (it never double-sends).
- Severity: low. Concern: `node:substrate.cabinet` is a new prediction-error node. Generic node
  consumers (e.g. endogenous curiosity) will see it like any other. It is excluded from
  `ACTIVE_INFERENCE_DOMAINS` on purpose.
- Collisions: open PRs #2442/#2439 are design-only. #2448 (stage 6.4 lane senders) touches
  llm-gateway, not `orion/gpu_pool/shed.py` or the pool runtime's shed path. No overlap found.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2461

🤖 Generated with [Claude Code](https://claude.com/claude-code)
