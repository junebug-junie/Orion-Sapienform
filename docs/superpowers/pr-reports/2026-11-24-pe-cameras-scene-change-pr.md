## Summary

The two camera nodes stop calling themselves prediction errors. Their value is now "camera scene change": how unusual the change in the camera's picture over the last 5 minutes is, relative to the last two hours of the same camera, ranked against its own last 24 hours.

- New scorer (`orion/substrate/scene_change.py`): a two-stage rank of the same raw frame change (`1 - cos`). Value in nats, `max(0, -ln p2 - 1)`. Calm by theory: exactly 0 on 63% of 5-minute intervals, above 2.0 on 5%, never above 4.67.
- The cameras write `signal_value` / `signal_kind: scene_change` / `signal_scale` on the graph node and rows in `substrate_node_signal_history` with a new `scale` column (`v2_partial` while stage 2 fills, then `v2`). No `prediction_error` key, no PE-table row, no field receipt.
- Every history reader reads the cameras signal-table-only, current scale only. A CI fence fails any SQL that names the signal table with a camera id and no scale predicate, and any producer that writes `prediction_error` for a camera.
- Glossary, kinds, polarity, labels, curiosity (2.0-nat bar, 4.67-nat ceiling), attention (both cameras still compete) and a 14-day dated line in Orion's self-readings all move together.
- Removes the dead `SUBSTRATE_SIGNAL_HISTORY_ENABLED` flag (every signal node lives only in the signal table now).

Refs #3827 #3806 #3807

## Outcome moved

Orion's prediction-error lane now holds only forecast misses (room, body, room_joint, body_joint, chat_turn). A pixel change is no longer read as "the world defied my model" (the 2026-11-24 03:40Z metacog trace). The camera value has a calm point fixed by theory instead of by a variance floor: the old score was 60% exact zero and saturated at 1.0 five times more often than its own z-tail allowed.

## Current architecture

- `perception_prediction_error()` scored each frame as `clip(z, 0, 3) / 3`, z against an EWMA with a `2e-6` variance floor. The 30 s tick published the latest frame's score to the node's `prediction_error` key, a field receipt (field `prediction_error` channel, 180 s expiry), and the PE history table.
- History readers read the cameras from `substrate_node_prediction_error_history`; the glossary filed them under `channel: prediction_error`, `value_kind: level`.
- Curiosity read `metadata["prediction_error"]` clamped to 0..1 with a 0.55 bar.

## Architecture touched

- orion-substrate-runtime: scorer, writer, history sampler, store (stage-2 persistence, `scale` insert), settings/env/compose/README.
- Shared read rule: `orion/prediction/signal_history_sql.py` (`SIGNAL_ONLY` + `scale IN ('v2','v2_partial')`), `orion/substrate/signal_history.py`.
- Semantics: glossary, `world_first.MEASUREMENT_VALUE_KINDS`, `self_readings._PERCENTILE_KINDS`, `metrics.semantics.VALUE_KINDS`, `field.pressure.PRESSURE_CHANNELS`, metric lock.
- orion-field-digester: both camera ids pruned as off-field pseudo-nodes; their decay entries removed.
- Curiosity, attention schema comments, node-history dig tool, ladder liveness rungs, CI fence, retired-pe-guards workflow.

## Files changed

- `orion/substrate/scene_change.py`: the two-stage scorer (stage 1 per stream, 2 h time window; stage 2 per node, 288 five-minute minima; warming; scale; stage-2 JSON round trip).
- `services/orion-sql-db/manual_migration_node_signal_history_scale_v1.sql`: `ADD COLUMN IF NOT EXISTS scale text NOT NULL DEFAULT 'v1'`; REQUIRED-BY substrate-runtime, attention-runtime, equilibrium, thought, hub. Additive allowlist (checked: `auto_apply_refusals == []`).
- `services/orion-substrate-runtime/app/worker.py`: frames feed stage 1 of their node; `_publish_perception_group` runs stage 2 (`rank_v2`) or the old z (`legacy_z`) and writes through `_write_signal_node`; no receipt; history rows carry the node's scale; old-scale camera rows never enter the ranking window; atlas label "Room/Walkway camera scene change"; dual-write flag gone.
- `services/orion-substrate-runtime/app/store.py`: `load_/save_scene_change_stage2` (one row per node in `substrate_perception_embedding_baseline`, key `scene_change_stage2|<node_id>`); `scale` in the signal insert; read-both helpers refuse a camera.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER=rank_v2` added, `SUBSTRATE_SIGNAL_HISTORY_ENABLED` removed, Patch 3 section.
- `services/orion-substrate-runtime/app/brain_frame_producer.py`, `orion/substrate/pressure.py`: a camera's nats go through `signal_unit_value` (nats / 4.67) before any 0..1 use.
- `orion/prediction/signal_history_sql.py`, `orion/substrate/signal_history.py`: camera ids, scale constants, signal_only predicate, `glossary_channel_for -> scene_change`, `signal_samples` with scale, `signal_unit_value`.
- `orion/substrate/falkor_codec.py`, `falkor_store.py`: `signal_scale` survives the graph round trip.
- `orion/substrate/endogenous_curiosity.py`, `curiosity_seed_bar.py`: scene-change nodes read `signal_value` unclamped, same 1800 s decay; bar 2.0 nats; `seed_strength(error, bar, ceiling=1.0)` + `CURIOSITY_SEED_CEILING_BY_NODE_ID` (cameras 4.67); evidence "unusual scene change on <node>".
- `orion/substrate/prediction_error_freshness.py`: `SCENE_CHANGE_LABELS`; `node_signal_label` says "room/walkway camera scene change".
- `orion/introspect/node_history.py`: camera noun names the scene change, the signal table and the frozen old-scale snapshot; item kind `node_scene_change_history`.
- `orion/attention/self_readings.py`: dated marker line (generated from the glossary; 2026-11-24 .. 2026-12-08), `scene_change_excess` percentile kind.
- `orion/attention/world_first.py`, `orion/metrics/semantics.py`, `orion/field/pressure.py`: new kind and channel polarity.
- `orion/schemas/attention_candidate.py`: walkway floor kept; comment documents ~17 h of `insufficient_history` after the first deploy.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`, `config/substrate/prediction_error_readers.yaml`: camera entries under `scene_change`; re-lock; stale registry entries removed.
- `services/orion-field-digester/app/tensor/channels.py`, `app/digestion/decay.py`: cameras pruned from the field; per-node expiry map now empty.
- `orion/substrate_ladder_liveness.py`: camera receipt rungs replaced by `signal:substrate.perception[_walkway]` rungs on the signal table (60 min).
- `scripts/check_pe_alarm_readers.py`: camera scale fence (static scan incl. evals, plus the builder's own output); messages name both alarms and cameras.
- `scripts/backfill_signal_history_from_pe.py`: passes `scale = 'v1'` (the insert needs it now).
- `services/orion-substrate-runtime/evals/run_scene_change_calm_eval.py` (+ test): the 7-day acceptance eval.
- Tests: `orion/substrate/tests/test_scene_change.py`, `services/orion-substrate-runtime/tests/test_worker_scene_change.py`, `tests/test_pe_cameras_scene_change.py`, fence and mutation tests in `tests/test_check_pe_alarm_readers.py`, and every older test that used a camera as "a live PE node" moved to a forecast node or a test-only probe id.
- `.github/workflows/retired-pe-guards.yml`: runs the new tests.

## Schema / bus / API changes

- Added: `substrate_node_signal_history.scale text NOT NULL DEFAULT 'v1'`; node metadata key `signal_scale`; glossary channel `scene_change`; value kind `scene_change_excess`; introspect item kind `node_scene_change_history`.
- Removed: the cameras' `prediction_error` node key, PE-table rows and field receipts (`prediction_signal` deltas).
- Renamed: none.
- Behavior changed: camera value is nats (0..4.67), not a 0..1 clipped z; camera history is read signal-only, current scale only.
- Compatibility notes: no bus channel changes. The migration must land before any REQUIRED-BY service starts on this code (the deploy gate refuses `up` until it is applied; the post-pull rebuild applies it first).

## Env/config changes

- Added keys: `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER=rank_v2` (orion-substrate-runtime).
- Removed keys: `SUBSTRATE_SIGNAL_HISTORY_ENABLED` (orion-substrate-runtime).
- `.env_example` updated: yes.
- local `.env` synced: edited by hand in `/mnt/scripts/Orion-Sapienform/services/orion-substrate-runtime/.env` (new key added after `SUBSTRATE_PERCEPTION_LOG_ONLY_STREAM_IDS`; the removed key commented out with a reason). The sync script does not remove keys.
- skipped keys requiring operator action: none.

## Metric quality gate (CLAUDE.md 0A) -- camera scene change

1. **Provenance.** Same raw input as before: `raw_surprise = 1 - cos(frame, stream EWMA)` from `perception_prediction_error()` (`orion/substrate/prediction_error.py`), fed per frame in `worker.py _handle_perception_prediction_error_message`. Stage 1 `Stage1Window.observe`, stage 2 `SceneChangeScorer.tick` (`orion/substrate/scene_change.py`), written by `_publish_perception_group`.
2. **Independence.** A rescoring of the existing camera signal, not new signal: it replaces the clipped z (r 0.66-0.76 per frame, design replay). Room vs walkway stay independent (separate streams, separate nodes). It shares no input with any forecast node.
3. **Theory anchor.** Split-conformal rank twice. Under exchangeable 5-minute intervals p2 is uniform, so -ln p2 ~ Exp(1): exact 0 with probability 1 - 1/e = 0.632, above 2.0 with probability e^-3 = 0.050, cap ln(289) - 1 = 4.67. Simulation pinned in `test_calm_rests_at_theory_at_the_sampler_cadence` (iid frames, 240 s sampler, 3 days of full windows): inside 0.58-0.72 and 2.5-6.5%.
4. **Live data.** I replayed the real per-frame log through the shipped scorer (after the review fixes). Window: 2026-11-23 17:22Z .. 11-24 22:14Z, 14,223 room + 30,702 walkway frames, 30 s ticks.

   | node, sampler | rows | exact 0 | > 2.0 nats | p90 | p99 | max |
   |---|---|---|---|---|---|---|
   | room, 233 s | 310 | 0.529 | 0.065 | 1.60 | 2.96 | 3.82 |
   | walkway, 233 s | 310 | 0.590 | 0.035 | 1.09 | 2.89 | 3.96 |
   | room, 311 s | 225 | 0.520 | 0.058 | 1.46 | 2.93 | 3.82 |
   | walkway, 311 s | 225 | 0.609 | 0.040 | 1.11 | 3.29 | 3.96 |

   - It returns to rest (exact 0) on 52-61% of rows. It is not saturated, and the maximum stays below the 4.67 cap.
   - Every row above is `v2_partial`. With only 29 h of frames, stage 2 never reached its full 24 h window (it needs 24 h and at least 259 values).
   - The room's partial-window zero share (0.52) sits below the eval band.
   - **The full-window calibration is UNVERIFIED.** #3850 runs the 7-day eval on `scale = 'v2'` rows only.
   - The simulation in `test_calm_rests_at_theory_at_the_sampler_cadence` (iid frames, 240 s sampler, full windows) lands inside 0.58-0.72 and 2.5-6.5%.
5. **Existing mechanism.** Same rank-to-surprisal shape as family A and street handoffs; reuses the signal table (Patch 1) and the per-node read rule (Patch 2).
6. **Reversibility.** `SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER=legacy_z` writes the old clipped z as `scale = 'v1'` (stored, never ranked with new rows: both cameras go quiet in the contests and curiosity instead of mixing scales). Step 5b (#3851) deletes the legacy path and the key after #3850 passes and #3806 is read. The `scale` column is additive and stays.

## Field note for Orion

The two camera nodes were never forecasts. Until 2026-11-24 they stored a clipped z-score (`clip(z, 0, 3) / 3`, a `2e-6` variance floor that set the scale) under `prediction_error`. Since this PR they store camera scene change in nats under `signal_value` (`channel: scene_change` in the glossary), and only forecast misses (room, body, room_joint, body_joint, chat_turn) remain in your prediction-error lane.

Priors that rest on the old scores -- please re-test them (#3807):
- `self:substrate_pe_is_zero_inflated_pulse_20261031`
- `self:pe_reverie_high_vs_zero_inflated_20261123`
- `self:pe_ceiling_rate_0p42pct_crowding_unsupported_20261121`
- `self:perception_pe_level_is_my_variance_baseline_not_the_room_20261121`
- `self:perception_pe_node_is_room_cameras_only_20261118`
- `self:pe_history_excludes_placeholder_writes_20261118`
- `self:pe_recorded_cadence_is_publish_cadence_20261118`
- `self:pe_recorded_count_is_a_fresh_score_tick_count_20261119`
- `self:cabinet_offered_every_frame_outranked_by_its_own_calm_20261119`
- `self:pool_backlog_aborted_cabinet_pred_error_20261017`
- `self:strain_pe_is_heat_or_queue_pulse_20261108`
- `self:walkway_pe_spikes_track_unbound_presence_20261027`
- `self:lens_prediction_error_fuses_with_juniper_in_reverie_20261026`

Old-scale data: `substrate_node_pe_history_snapshot_20261124` (frozen 2026-11-24, 14,472 old PE rows of the five nodes, 2026-11-17 21:17Z .. 11-24 21:22Z, readable by `orion_readonly`). New camera rows: `substrate_node_signal_history WHERE scale IN ('v2','v2_partial')` (calm checks: `scale = 'v2'`). The `scale = 'v1'` camera rows in the signal table are the same old scale.

## Tests run

All runs are on the tree merged with origin/main (#3841, #3858, #3860, #3865, #3866). I compared failures against main.

```text
tests (scene change, alarms, readers, glossary, metric semantics, attention, ladder, backfill)
  + orion/substrate/tests orion/attention orion/introspect orion/prediction orion/field
      2045 passed, 1 failed (test_felt_state_lived_answers: also fails on main)
services/orion-substrate-runtime/tests + evals
      665 passed, 17 failed (the same 17 fail on main: cursor reset/tail seed/quarantine/reducer health env)
services/orion-field-digester/tests                       297 passed
services/orion-thought tests + evals                      573 passed / 23 passed
services/orion-equilibrium-service/tests                  266 passed
services/orion-attention-runtime/tests                    56 passed, 1 failed (also fails on main)
services/orion-hub attention organ + glossary routes + introspect   63 + 338 passed
static-gate pytest set (settings defaults, migration drift/deploy gate, compose, integration seam,
  pe readers, metric semantics, body economy, ...)        185 + 361 passed
```

Static gates: `check_definition_drift --gate` (re-locked) PASS, `check_metric_lineage --gate` and `--prompt-semantics` PASS, `check_inner_state_registry` OK, `check_integration_seam --require-base` OK, `check_pe_alarm_readers` OK, `check_env_key_single_source` OK, `check_migration_required_by --base origin/main` OK, `check_body_economy_contract` OK, `check_env_template_parity` PASS.

Mutation checks for the fence (`tests/test_check_pe_alarm_readers.py`):
- Stripping the scale predicate from the builder fails the fence.
- Putting a camera in the read-both union fails it.
- Routing the real worker's camera write back through `_write_prediction_error_node` fails it.

## Evals run

```text
services/orion-substrate-runtime/evals/test_scene_change_calm_eval.py      3 passed
run_scene_change_calm_eval.py --days 7 (live, read-only):
  "substrate_node_signal_history has no scale column: migration not applied (insufficient)"  exit 3
  -> the real 7-day run is #3850 (after deploy)
replay of 29 h of live frames through the scorer: see the metric gate, step 4
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-substrate-runtime build   -> Image Built (worktree, env files symlinked from the primary checkout)
docker run --rm --entrypoint python <image> -c "import app.worker; from orion.substrate.scene_change import ..."  -> import ok 4.666
docker compose config: SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER resolves to rank_v2
Not deployed (per instructions). Live path UNVERIFIED until deploy.
```

## Review findings fixed

The code review ran in a sub-agent and found 9 issues.

- **Finding 1 (material): a huge change read only ~2 nats.** Every new 2-h maximum tied at the stage-1 floor 1/(n+1). That made the 4.67 ceiling unreachable, and whether a dramatic change seeded curiosity was a coin flip.
  - Fix: inside the window's top 10 frames, stage 1 now uses an exponential (peaks-over-threshold) tail fitted to those frames. Stage 2 is a rank, so its calibration does not depend on that tail.
  - Evidence: `test_a_change_far_beyond_the_window_reads_above_3_nats_on_every_seed` (18 cases: 6 seeds x 3 frame rates) and `test_a_bigger_change_reads_at_least_as_high`.
- **Finding 2 (material): a reading was ranked against its own stage-2 sample for up to 5 minutes.** The sample overlaps the trailing window, which cost about ln 2 on tail reads.
  - Fix: stage 2 ranks only against samples whose window does not overlap the current one.
  - Evidence: `test_the_reading_is_not_ranked_against_its_own_sample_in_the_next_ticks`.
- **Finding 3: the stage-2 value depended on the frame rate.** The walkway runs 59 frames per 5 min in some hours and 120 in others.
  - Fix: the 5-minute minimum is Sidak-corrected, 1 - (1 - min)^k, before stage 2.
  - Evidence: the calm simulation is still inside its band.
- **Finding 4: the Hub prediction-error domain panel still listed both cameras, with null values.**
  - Fix: it skips moved signal nodes (`attention_organ_routes.py`).
  - Evidence: Hub attention organ tests pass.
- **Finding 5: camera attention signals were still tagged `substrate_prediction_error` / driver `prediction_error`.**
  - Fix: they are now `substrate_scene_change` / `scene_change`, in both the world-first and the pressure signals.
  - Evidence: an assertion in `test_both_cameras_produce_contest_candidates`.
- **Finding 6: the live-sanity eval read the cameras from the PE table.**
  - Fix: `run_pe_magnitude_live_sanity.py` reads the shared history source.
  - Not fixed here: three historical replay evals still read the PE table for old camera rows. They are listed on #3852.
- **Finding 7: NaN or inf could corrupt a stage-1 window.**
  - Fix: non-finite frames are ignored.
  - Evidence: `test_non_finite_frames_are_ignored`.
- **Finding 8: no migration creates `substrate_node_pe_history_snapshot_20261124`.**
  - NOT fixed here. A `CREATE TABLE ... AS SELECT` is a data migration, refused by the auto-apply allowlist. The table exists live (checked: 5 nodes, 14,472 rows, `orion_readonly` SELECT = true). Noted on #3852.
- **Finding 9: two edge cases.** A stream quiet for hours counted as ready, and a window spanning 24 h with a large gap read `v2`.
  - Fix: stage 1 prunes before the ready check. `v2` also needs at least 259 values.
  - Evidence: `test_a_stream_quiet_for_hours_is_warming_again` and the extended scale test.
- **Own fix: lock creation and the stage-2 load.** The scorer lock is now built in `__init__`. The stage-2 DB load runs outside the lock.

## Restart required

Order matters. Do it from the clean primary main checkout after merge.

1. The post-pull rebuild applies the migration first.
2. If it did not, apply it by hand: `docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_node_signal_history_scale_v1.sql`
3. Then rebuild the readers before or with the writer.

```bash
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

The deploy gate refuses `up` for any REQUIRED-BY service until the migration is applied.

Verify after deploy:

```sql
-- no new PE rows for the cameras
select count(*) from substrate_node_prediction_error_history where node_id in ('node:substrate.perception','node:substrate.perception_walkway') and observed_at > '<deploy>';
-- new-scale camera rows appear (after ~30 min stage-1 warm-up + 4 h stage-2 warm-up on first deploy)
select node_id, scale, count(*), max(value) from substrate_node_signal_history where node_id like 'node:substrate.perception%' and observed_at > '<deploy>' group by 1,2;
-- stage 2 is persisted
select stream_id, generated_at, jsonb_array_length(baseline_json->'values') from substrate_perception_embedding_baseline where stream_id like 'scene_change_stage2|%';
```

## Risks / concerns

- **Medium: first-deploy warm-up.** Both cameras write `warming` for about 4 h, then rank `insufficient_history` in the contests until 200 new rows exist (about 17 h in all). The `signal:substrate.perception*` ladder rungs read red for those first ~4 h. This was accepted in the design.
- **Medium: the 24 h calibration is UNVERIFIED.** Only 29 h of frames existed, so no full-window rows could be replayed. The room's partial-window zero share was 0.52, below the band. #3850 is the gate, and nothing should build on this value before it passes.
- **Low: the worktree build re-tagged the local `orion-substrate-runtime-substrate-runtime` image** with this branch's code. Running containers are untouched. A plain `up` would be refused by the migration gate until the migration is applied, and the post-merge rebuild re-tags from main.
- **Low: env parity warning on primary main until merge.** The local `.env` already dropped `SUBSTRATE_SIGNAL_HISTORY_ENABLED`, which main's `.env_example` still lists. The check does not block.
- **Low: rollback leaves the cameras quiet.** `legacy_z` writes `v1` rows that no reader ranks.
- Issues opened:
  - #3850 (verify-live, 7-day calm eval, check on 2026-12-02)
  - #3851 (5b cleanup, gated on #3850 and #3806)
  - #3852 (the field `prediction_error` channel has no producer now: retire the receipt path, plus the Hub glossary panel, the historical replay evals and the snapshot DDL)
- Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pulls (filled in after opening)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
