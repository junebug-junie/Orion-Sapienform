**DO NOT MERGE before #3309's 168 h calm check passes (on/after 2026-11-22 07:00Z) and #3311 is closed.**

# PE retire step 6: family A joint surprise (body + room) replaces node:substrate.biometrics

## Summary

- Orion gets two real "is my world doing what I expected" signals. One covers the body: every machine, cabinet, closet and AC reading, plus office temperature. The other covers the office room: eCO2/TVOC, PM2.5, light and humidity. Each signal is one joint forecast miss, so coupled readings count once. The nodes are `node:prediction.body_joint` and `node:prediction.room_joint`.
- The room joint is a **world** source, per Juniper's #3580 decision. It can author a goal that permits the read-only fetch/recall acts, and the body-only gates skip it. Office sound stays out of the room group until #3581 fixes the meter: it is still forecast and scored in the blend.
- `node:substrate.biometrics` is retired completely. It was the mean change of pressure hints over a fixed 0.30 scale, not a forecast. Removed: its tick, its function, the silent-node rule (`report_cadence.py`) and two env keys. It joins the shared retired set and the digester prune. It leaves `ACTIVE_INFERENCE_DOMAINS`, native targets, the metacog trend cue, mood-arc enrichment, the Hub domain list, self-readings words, the glossary and the metric inventory.
- orion-prediction scores a body row and a room row beside the blended family A row. Each group has its own covariance, calibration tails and hour-before difficulty. Fresh adaptive group rows are published as `prediction.joint_surprise.v1` on `orion:prediction:joint_surprise`.
- The calm eval gains `--family all` (one check per group) and `--exclude START..END` (outages).

## Outcome moved

- **Before:** attention and goals partly followed a number that was never 0 and never near 1 (`power` pinned at 1.0). It was the top target in 1,488 of 119,160 frames (1.25 %; 3.8 % of frames with a winner) and sourced 5.9 % of goals in the 72 h before this PR.
- **After:** the body slot is a scored forecast whose calm point is 0 by theory. The room becomes a world signal that can give Orion a reason to look outward.

## Gate status (read before merging)

| gate | status | evidence |
|---|---|---|
| #3309 blended 7-day calm check | PRELIMINARY pass on 4.3 days; script verdict NOT YET (days) | below; comment on #3309 |
| #3309 per-group calm check | replay only (72 h, in memory); stored group rows start at deploy | below |
| #3311 coupled load spike = one joint surprise | PASS on stored rows 11-13..11-17 (not closed; Juniper's call) | below; comment on #3311 |
| refusal rate | 2 refused rows outside the excluded 11-14/15 gap (PM2.5 `one_value`) | #3309 run |

**#3311.** The reference event (11-12 09:20Z) predates live scoring, which started 11-13 08:25Z. The same daily athena job runs at about 09:20Z. Stored rows, 5-min horizon:
- At every load-led peak, load carries 0.6 to 1.0 of d², and the other coupled readings carry 0.05 or less each. Example: 11-13 10:30, excess +3.53, load share 1.05.
- The joint p is not repeated per reading. Example: 11-15 09:25, four coupled readings fell outside their own 95 % range. Jointly that is one row (p 0.018). The coupled d² is 88 whitened jointly vs 186 counted per reading.
- Excess returns to about 0 after the job ends.
- Full table and caveats: issue comment.

**#3309, stored blended rows** (excludes 11-14 17:50Z..11-15 06:50Z):

| horizon | n | days | served mean excess (95 % CI) | p ≤ 0.05 | hours at rest (calibrated) | longest h > 0.5 |
|---|---|---|---|---|---|---|
| 5 | 1,248 | 4.33 | +0.023 [-0.059, +0.102] | 0.051 | 0.53 (0.54) | 1 |
| 15 | 1,244 | 4.32 | +0.027 [-0.078, +0.138] | 0.058 | 0.52 (0.54) | 3 |

**Per-group replay**: the loop with groups, run in memory over 11-15 05:36..11-18 05:36. Sound is in no group. 0 refused, 0 groups skipped. Verdict PASS for all three families at `--min-days 2.5`.

| family | h | n | served mean excess (95 % CI) | p ≤ 0.05 | hours at rest (calibrated) | longest h > 0.5 |
|---|---|---|---|---|---|---|
| body | 5 | 862 | +0.040 [-0.069, +0.145] | 0.071 | 0.51 (0.54) | 2 |
| body | 15 | 860 | +0.033 [-0.111, +0.181] | 0.071 | 0.60 (0.54) | 2 |
| room | 5 | 862 | -0.006 [-0.064, +0.057] | 0.060 | 0.49 (0.54) | 1 |
| room | 15 | 860 | -0.011 [-0.099, +0.085] | 0.059 | 0.57 (0.54) | 1 |

## Metric quality gate (CLAUDE.md §0A)

1. **Provenance.**
   - Readings: raw tables only, `services/orion-prediction/app/family_a_load.py`.
   - Forecast and score: `app/family_a_live.py` `FamilyALive.issue` / `score` / `_score_joint_row`.
   - Group calibration: `_group_calibrations` / `_calibrate`.
   - Event: `_event`; published by `app/main.py` `_publish_joint_surprise`.
   - Node value: `orion/substrate/prediction_joint_surprise.py` `JointSurpriseState.reading`, written by substrate-runtime `_process_prediction_joint_surprise`.
2. **Independence.**
   - vs `node:prediction.body` (cabinet temp alone) and `node:prediction.room` (office temp alone): those two readings are also inputs of the body joint. On stored blended rows, cabinet temp is in the top-5 shares on 8.0 % of rows (mean share 0.09 when there), and office temp on 8.4 % (0.10). The overlap is a small slice of the joint, not a duplicate.
   - vs `node:substrate.execution`: no shared input.
   - vs `node:substrate.perception`: camera embeddings; no shared input.
   - Body vs room: they share no target. Their forecasts share covariates, so a weather or people event can move both.
3. **Theory anchor.**
   - Each group's d² is Mahalanobis surprisal under the residual covariance.
   - The p is split-conformal and difficulty-normalised (Lei et al. 2018). Under exchangeability it is uniform, so E[-ln p] has a known calm point (`calm_point`).
   - The node's calm variance is the exact Var(-ln p) for a uniform p on the conformal grid (`calm_excess_variance`, about 1.0).
   - Calm rests at z ~ N(0, 1), so the node reads exact 0 about half the time. Synthetic check in `test_prediction_joint_surprise.py`.
4. **Live sanity.**
   - Stored blended rows (4.3 days): mean excess within CI of 0, p ≤ 0.05 share 0.05-0.06, hours at rest at the calibrated share (tables above).
   - The node fold replayed over stored blended rows (1,506 writes): exact 0 on 57.9 % of writes, p50 0.0, p90 0.62, 0.55 crossed on 11.7 %. It returns to 0. The longest run of writes above 0.55 is 15 (about 75 min, the 2 h athena job).
   - Not degenerate, and it rests.
   - Body and room group rows exist only in replay until deploy (table above).
5. **Existing mechanism.**
   - Reuses the family A loop (#3267/#3334/#3371), the surprise-node fold shape of `node:prediction.body`/`.room` (#3202), the `surprise_excess` value kind, the node writer and the PE history.
   - No new ledger outside orion-prediction.
6. **Reversibility.**
   - Two flags: `PREDICTION_JOINT_SURPRISE_EVENTS_ENABLED`, `SUBSTRATE_PREDICTION_JOINT_SURPRISE_NODES_ENABLED`.
   - Group rows are ordinary rows in the existing table under new `family` values. There is no migration.
   - The retirement half is a `git revert`.

## Before measurement (live, read-only; compare after deploy, as #3576 asks for execution)

| consumer | node:substrate.biometrics now | after this PR |
|---|---|---|
| Attention frames, 11-15 05:07..11-18 05:20 (`substrate_attention_frames`) | top target in 1,488 of 119,160 frames (1.25 %; 3.8 % of the 39,036 frames with a winner); in dominant targets in 1,887 | body_joint competes as a `surprise_excess` body signal; room_joint as a world signal |
| Goals (`field_dominance_run`, ticks past streak 3, same 72 h) | 1,520 of about 25,600 (5.9 %); 7-day: 3,954 of about 65,400 (6.0 %) | biometrics goals gone. body_joint is not goal-eligible (same as `node:prediction.body`). room_joint is goal-eligible as an attended world source |
| Self-model confidence (`substrate_attention_self_model`, 72 h, 4,151 rows) | biometrics in 99.9 % of rows (mean 0.096); mean confidence 0.896 | execution only: 0.887 on the same rows; rows without a fresh execution reading (2.5 %) read unknown |
| Endogenous curiosity focal refs, 7 d | 46 of about 7,400 (0.6 %) | body_joint / room_joint seed like `node:prediction.body` |

## Current architecture

- Family A forecast every raw body and room reading every 5 min and stored one blended joint row per bucket. Nothing consumed it.
- `node:substrate.biometrics` was written by substrate-runtime's biometrics tick from `biometrics_prediction_error()`, with a receipt.
- It was an Active-Inference domain, a native attention target with a precision baseline, a self-reading, a metacog trend cue and a mood-arc enrichment channel.

## Architecture touched

- **orion-prediction:** config group per input; group calibration; group rows; event publish; calm eval.
- **Contracts:** schema, both registries, channel, organs registry.
- **orion-substrate-runtime:** listener plus fold; producer removed; settings, env and compose.
- **Shared readers:** retired set, digester prune, self-model domains, selectors, self-readings, semantics plus lock, metacog trend, mood-arc, Hub, world-first per-node external age, `_WORLD_NODE_IDS`.

## Files changed

- `config/prediction/family_a.v1.yaml`: `group` per input (sound `none` until #3581).
- `services/orion-prediction/app/{family_a,family_a_live,store,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, `evals/run_family_a_calm_live_eval.py`, `tests/test_family_a*.py`: groups, events, eval flags, tests.
- `orion/schemas/prediction.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/integration/organs.yaml`: contract.
- `orion/substrate/prediction_joint_surprise.py` (+ tests): the fold.
- `services/orion-substrate-runtime/app/{worker,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, tests: listener; producer removed; tests moved to live domains.
- `orion/substrate/{prediction_error,prediction_error_freshness,attention_self_model,metacog_trend_signals}.py`; `orion/substrate/biometrics_loop/report_cadence.py` (deleted): retirement.
- `orion/attention/{world_first,self_readings,sustained}.py`, `orion/attention/field_attention/selectors.py`, `orion/schemas/{attention_candidate,prediction_error_definitions,field_goal}.py`, `orion/metrics/semantics.py`, `orion/mood_arc/corpus_enrichment.py`: shared readers.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`, `config/substrate/prediction_error_readers.yaml`: semantics.
- `services/orion-field-digester/app/tensor/channels.py`: prune.
- `services/orion-hub/scripts/attention_organ_routes.py`, `static/js/hub-surface.js`: Hub domain list and text.
- `scripts/sync_local_env_from_example.py`, `scripts/analysis/measure_ast_hot_reducer.py`, `scripts/analysis/replay_biometrics_without_room.py` (deleted), CI workflows: plumbing.
- Tests across root, orion and services: vehicles moved from biometrics to live nodes; sustained-dwell tests now use a test-only probe node, because no glossary node is continuous any more.

## Schema / bus / API changes

- **Added:** `PredictionJointSurpriseV1` (`prediction.joint_surprise.v1`) on `orion:prediction:joint_surprise` (orion-prediction -> orion-substrate-runtime). Nodes `node:prediction.body_joint` and `node:prediction.room_joint`. `prediction_joint_surprise.family` values `family_a_body` and `family_a_room`. Health keys `prediction_joint_surprise` (substrate-runtime) and `family_a.groups` / `events_*` (prediction).
- **Removed:** `node:substrate.biometrics` PE writes and receipts; `biometrics_prediction_error`; `PREDICTION_ERROR_DEFINITION_VERSIONS["node_biometrics"]`.
- **Behavior changed:**
  - `ACTIVE_INFERENCE_DOMAINS` is now `{"execution"}`.
  - World-first keeps `node:prediction.room_joint` current for 600 s, not 180 s.
  - The sustained-body dwell rule has no live subject; no glossary node is continuous.
- **Compatibility:**
  - No migration: the `family` column is varchar(32).
  - Old blended rows keep scoring.
  - A run selected before deploy has no groups, so its rows get no group rows. A group row whose run lacks the group is scored `missing` / `no_group`, never against the blend's yardstick.

## Env/config changes
- **Flag default flipped on (Juniper, 2026-11-18: "turn shit on").** `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES` now defaults to true in `services/orion-substrate-runtime/docker-compose.yml` and in `worker.py`'s unset fallback. Live `.env` already true, so no live change. Test `test_prediction_error_nodes_default_on_when_unset` pins it.


- **Added keys:** `PREDICTION_JOINT_SURPRISE_EVENTS_ENABLED=true` (orion-prediction); `SUBSTRATE_PREDICTION_JOINT_SURPRISE_NODES_ENABLED=true` and `SUBSTRATE_PREDICTION_JOINT_SURPRISE_CHANNEL=orion:prediction:joint_surprise` (orion-substrate-runtime).
- **Removed keys:** `BIOMETRICS_SILENT_NODE_EXCLUSION`, `BIOMETRICS_SILENT_NODE_CADENCE_MULTIPLE` (orion-substrate-runtime).
- **`.env_example` updated:** yes, both services.
- **Local `.env` synced:** keys added to and removed from `/mnt/scripts/Orion-Sapienform/services/{orion-prediction,orion-substrate-runtime}/.env` directly. The env-sync prefix list gained `SUBSTRATE_PREDICTION_JOINT_SURPRISE_`.
- **Skipped keys requiring operator action:** none.

## Tests run

```text
Local runs, repo .venv, PYTHONDONTWRITEBYTECODE=1, -p no:cacheprovider.

Affected-test sweep: 172 files that name the retired node, its domain, or a changed registry, run per service. Results:
- root + orion + scripts: 1,437 passed. 7 failed: 4 were fixed after the sweep (metacog vehicle); 3 also fail on main (test_autonomy_goals_bus_catalog, test_metric_lineage x2).
- orion-equilibrium-service tests + evals: 289 passed.
- orion-thought tests + evals: 922 passed, 32 skipped.
- orion-attention-runtime: 35 passed, 5 skipped.
- orion-hub (affected files): 141 passed.
- orion-field-digester: 58 passed.
- orion-cortex-exec: 170 passed.
- orion-sql-writer: 90 passed, 15 skipped.
- orion-signal-gateway: 24 passed.
- orion-mind: 3 passed.
- orion-substrate-runtime: 590 passed. The 10 failed + 9 errors also fail on main (cursor_reset_auth / quarantine_truth fixtures, cursor_tail_seed, reducer_health, falkor_routed_store, worker_reducer, 2 self_model_tick).
- orion-biometrics: 2 fail, both also on main (hecate / circe expected_offline).
- orion-prediction: tests/test_family_a_live.py + test_family_a.py, 29 + others passed. 27 failures elsewhere in the suite (heat_balance etc.) are a ModuleNotFoundError in the local venv. They are not touched by this PR; CI runs this suite in its own image.

New tests:
- orion/substrate/tests/test_prediction_joint_surprise.py: calm rest point by theory, one spike, shortest horizon wins, groups never mix, refusals.
- services/orion-substrate-runtime/tests/test_prediction_joint_surprise_listener.py: node write, evidence, room goes to its own node, calm writes 0, backlog and overlap refused, invalid payload, write failure, /health.
- tests/test_room_joint_world_source.py: world vs body, 600 s age, ids agree, retired predecessor.
- services/orion-prediction/tests/test_family_a_live.py: 10 group tests (calibration per group, one row per group per bucket, fresh adaptive events only, backlog never published, events off, no-target group skipped, refused rows per group, old run gets no_group, group none, unclassified input refused).

Static gates: check_integration_seam OK, check_pe_alarm_readers OK, check_definition_drift --gate PASS (re-locked), agent_trace schema registry + env sync tests 24 passed, resolve("PredictionJointSurpriseV1") OK.
```

## Evals run

```text
evals/run_family_a_calm_live_eval.py --hours 168 --exclude 2026-11-14T17:50Z..2026-11-15T06:50Z  -> NOT YET (4.33 / 4.32 days; every calm criterion holds)
evals/run_family_a_calm_live_eval.py --replay-hours 72 --family all --min-days 2.5                -> PASS family_a, family_a_body, family_a_room
node-fold replay over stored blended rows (scratch script)                                     -> exact 0 57.9 %, p90 0.62, >0.55 11.7 %
#3311 coupled-event read (scratch script, psql + numpy)                                         -> PASS, see #3311 comment
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-prediction/docker-compose.yml config -q        -> OK (PREDICTION_JOINT_SURPRISE_EVENTS_ENABLED "true")
docker compose ... -f services/orion-substrate-runtime/docker-compose.yml config -q  -> OK (new keys present, BIOMETRICS_SILENT_* gone)
Calm-eval replay ran inside the orion-prediction image (read-only mount of this branch).
No image built from the worktree (would retag the production image). Live path UNVERIFIED until deploy.
```

## Review findings fixed

A review subagent read the committed diff. Its verdict: no high or medium issues. Pairing, fold math, retirement completeness and contract parity were all checked. Its five low findings:

- **One stale input of either group refuses both nodes.** The hygiene gate is all-or-nothing, so an office sensor outage also makes the body joint absent. Absent is safe, but it couples body to world.
  - Fix: filed #3730 (gate per group). Named in Risks.
- **15-min outcomes rarely count in the fold.** The "shortest horizon per target" rule leaves them out whenever a 5-min outcome exists.
  - Fix: said so in the fold's module docstring. The behavior matches `node:prediction.body`.
- **`n_observed` on a missing row changed meaning.** A blended row scored `missing` now stores the count of its own targets with a known standard deviation, not every reading that arrived. Scored rows are unchanged.
  - Fix: noted here. The calm check reads scored rows only.
- **The retired `BIOMETRICS_SILENT_*` keys were back in the local substrate-runtime `.env`.** The env-sync script copies from main's `.env_example`, which still has them until merge.
  - Fix: removed again. They are harmless, because compose and settings no longer read them. After merge they need one more manual delete if a sync re-adds them (the script does not remove keys).
- **The new nodes depend on `SUBSTRATE_WRITE_PREDICTION_ERROR_NODES`.** It is `true` in `.env_example` and in the live `.env`, but compose defaults it to `false`. That gap is not new in this PR.
  - Fix: called out here. The live `.env` sets it, so the nodes write.

## Restart required

After merge, deploy from the clean primary main checkout, in this order:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

- orion-prediction goes first. Its config is baked into the image, and its first select after start fits the groups.
- substrate-runtime goes second, so the listener exists before events matter. Events start with the first adaptive group rows, about 1 h after the prediction restart.
- Verify:
  - `curl -s localhost:8132/health | jq .family_a.groups`
  - substrate-runtime `/health` `.prediction_joint_surprise.counts.node_writes_total` > 0
  - `SELECT family, status, count(*) FROM prediction_joint_surprise WHERE origin_ts > now() - interval '2 hours' GROUP BY 1, 2`

## Risks / concerns

- **Severity medium. The per-group calm checks are replay-only.**
  - Concern: stored body and room rows start at deploy, and the nodes ship ON (Juniper's flags-ON rule). For the first 7 days, attention, curiosity and room-joint goals read a group signal whose calm is shown only in a 72 h replay.
  - Mitigation: the replay passes for both groups. The kill switches above. #3722 for the stored `--family all` check 7 days after deploy.
- **Severity medium. Room-joint goals can authorize fetch/recall.**
  - Concern: this is Juniper's #3580 decision, but it is new authority.
  - Mitigation: world-first needs an "unusual/high" band of its own week before it competes. Measure the room-joint goal share after deploy.
- **Severity low. One stale input refuses the whole bucket.** A refused bucket makes both nodes absent (never calm). This is the same refusal behavior the blend already had. Fix is #3730.
- **Severity low. Execution is the only Active-Inference domain left.**
  - Concern: confidence becomes unknown when execution is stale (2.5 % of rows). #3576 retires execution next.
- **Severity low. The sustained-body dwell rule has no live subject.**
  - Concern: `ATTENTION_SUSTAINED_BODY_ENABLED` now changes nothing. Tests keep the mechanism alive with a probe node. #3723.
- **Severity low. Joint surprise is not a temporal-self experience kind.**
  - Concern: the channel is registered under the prediction organ, which emits other kinds. Big joint spikes as experience items is #3724.
- **Issues:** opened #3722 (per-group calm, verify-live), #3723 (sustained rule no subject), #3724 (joint surprise as experience), #3725 (office temp / sound group revisit), #3730 (gate per group); closed none. Fixes #3575 only on merge after the gates.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3732

🤖 Generated with [Claude Code](https://claude.com/claude-code)
