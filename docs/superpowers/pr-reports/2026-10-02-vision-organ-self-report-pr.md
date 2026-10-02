# Vision reports on itself: the vision organ lane into capability:vision

## Summary

- The frame router (the one process that sees frames arrive per camera, the tasks it hands the vision host, and every reply or timeout) now publishes one report per 60 s window on `orion:grammar:event` (trace prefix `vision.organ:`): per camera stream, frames received, age of the newest frame, tasks sent, failures by class, detection/caption yield.
- A new substrate reducer lane (`orion/substrate/vision_organ_loop/`, own cursor `vision_organ_grammar_reducer`, `REDUCER_SPECS[6]`) folds each window into `node:substrate.vision_organ`, with a clock-driven silence path: no router window for 180 s writes "can't see" (1.0), never calm.
- Two new field channels feed `capability:vision`: `vision_frame_staleness` -> `pressure`, `vision_processing_failure_pressure` -> `reliability_pressure`.
- Killed the old proxy: substrate-runtime's `_vision_channel_tick` and its artifact listener are deleted, its topology edge removed, and `node:substrate.vision` is pruned from the field (`RETIRED_PSEUDO_NODES`).
- A configured camera that sends nothing is reported as `never_seen` (aged from router start), not omitted and not calm.

## Outcome moved

Before: `capability:vision` was fed only by `node:substrate.vision.prediction_error`, the age of the newest detect artifact pooled across every camera. It read **0.0 on 124,612 of 124,612 field ticks** (2026-09-29 .. 10-02, `substrate_field_state`) while the carbon webcam delivered **zero frames** for that entire span. One live camera hid every dead one.

After: the eye's own report names each stream. Replayed over the router's real 8-day log, carbon reads `never_seen` (staleness 1.0) in 11,567 of 11,567 windows, cam0 reads calm in 11,566 and non-calm once (a real 24 s gap on 2026-09-28 08:58 UTC). A router outage now rises to 1.0 on a clock instead of freezing at the last calm value.

## Current architecture

- `capability:vision` <- `node:substrate.vision` (`prediction_error: pressure`, weight 0.85) <- `_vision_channel_tick` in substrate-runtime, reading the age of the newest `orion:vision:artifacts` message carrying `objects`, any camera.
- `orion/bus/channels.yaml` used to list retina/edge/window as grammar producers; none ever emitted (removed in #2324). No vision service emitted grammar.
- The frame router kept process-lifetime totals in `/healthz` and `orion:system:health` (frames seen, dispatched, host errors, timeouts) with no per-stream breakdown, no window, and no path into the field.

## Architecture touched

- Producer: `services/orion-vision-frame-router` (new `app/grammar_emit.py`, hooks in `app/dispatcher.py`, publisher task in `app/main.py`).
- Contract: `orion/schemas/vision_organ_projection.py` (wire constants + projection models), registry, `orion:grammar:event` producer list.
- Reducer: `orion/substrate/vision_organ_loop/` + substrate-runtime spec/store/tick; sql-writer retention lane table.
- Field: field-digester delta branch, channels, expiry, single-observer, retired pseudo-node; topology edge; glossary.
- Retired: substrate-runtime vision-channel tick + listener + `SUBSTRATE_VISION_CHANNEL_TICK_*`.

## Files changed

- `orion/schemas/vision_organ_projection.py`: wire contract (source `orion-vision-frame-router`, prefix `vision.organ:`, atom roles, failure classes) and `VisionOrganProjectionV1` / `VisionOrganStreamStateV1` / `VisionOrganWindowCountV1`.
- `orion/schemas/registry.py`: registers the projection and stream state.
- `orion/bus/channels.yaml`: `orion-vision-frame-router` added to `orion:grammar:event` producers.
- `services/orion-vision-frame-router/app/grammar_emit.py`: window recorder + trace builder + publisher.
- `services/orion-vision-frame-router/app/dispatcher.py`, `app/state.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: hooks, flag, window, docs.
- `orion/substrate/vision_organ_loop/{constants,extract,reducer,pipeline,__init__}.py`: the reducer lane and the silence receipt.
- `services/orion-substrate-runtime/app/worker.py`: `REDUCER_SPECS[6]`, poll loop, `_vision_organ_tick` + silence check; old tick/listener deleted.
- `services/orion-substrate-runtime/app/{store,settings,grammar_truth}.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-sql-db/manual_migration_vision_organ_substrate_loop.sql`: projection table + cursor seed.
- `services/orion-sql-writer/app/grammar_truth.py` (+ test): retention lane for the new prefix.
- `services/orion-field-digester/app/{ingest/state_deltas,tensor/channels,digestion/decay,settings,worker}.py`, `.env_example`, `docker-compose.yml`.
- `config/field/orion_field_topology.v1.yaml`: edge `node:substrate.vision_organ -> capability:vision` replaces the `node:substrate.vision` edge.
- `config/field/field_channel_glossary.v1.yaml` (+ `tests/test_field_channel_glossary.py`): two channels.
- `orion/substrate_ladder_liveness.py`: rung `receipts:vision_organ_reducer` replaces `substrate.vision_channel`.
- `scripts/check_substrate_projection_schema_drift.py`, `scripts/sync_local_env_from_example.py`: new projection, new prefixes.
- `scripts/eval_vision_organ_replay.py` (+ `tests/scripts/test_eval_vision_organ_replay.py`): end-to-end replay eval.
- `config/metrics/metric_definitions.lock.json`: re-locked (2 field channels added, 1 grammar producer).
- Tests: `services/orion-vision-frame-router/tests/test_grammar_emit.py`, `tests/test_vision_organ_substrate_reducer.py`, `services/orion-substrate-runtime/tests/test_worker_vision_organ_tick.py`, `services/orion-field-digester/tests/test_field_vision_organ_perturbations.py`, two existing substrate-runtime tests updated for the retired tick / new poll task.

## Schema / bus / API changes

- Added: `VisionOrganProjectionV1`, `VisionOrganStreamStateV1`, `VisionOrganWindowCountV1`; grammar trace prefix `vision.organ:` from `orion-vision-frame-router` on `orion:grammar:event`; delta `target_kind="vision_organ"`; node channels `vision_frame_staleness`, `vision_processing_failure_pressure`; table `substrate_vision_organ_projection`; cursor `vision_organ_grammar_reducer`.
- Removed: `node:substrate.vision` writer and topology edge; `SUBSTRATE_VISION_CHANNEL_TICK_ENABLED`, `SUBSTRATE_VISION_CHANNEL_TICK_INTERVAL_SEC`.
- Renamed: none.
- Behavior changed: `capability:vision.pressure` now comes from `vision_frame_staleness`; `capability:vision.reliability_pressure` is newly fed. `node:substrate.vision` is pruned from field state on the next reconcile.
- Compatibility notes: `SUBSTRATE_VISION_ARTIFACTS_CHANNEL` stays (the perception P2 listener reads it). FalkorDB's `node:substrate.vision` concept node is no longer written; endogenous curiosity already decays a stale node's prediction error to zero by age.

## Env/config changes

- Added keys: router `VISION_ORGAN_GRAMMAR_ENABLED` (code false, template true), `VISION_ORGAN_WINDOW_SEC=60.0`; substrate-runtime `ENABLE_VISION_ORGAN_REDUCER` (code false, template true), `VISION_ORGAN_GRAMMAR_BATCH_LIMIT=200`, `VISION_ORGAN_SILENCE_SEC=180.0`; field-digester `ENABLE_VISION_ORGAN_FIELD_DIGESTION` (code false, template true).
- Removed keys: `SUBSTRATE_VISION_CHANNEL_TICK_ENABLED`, `SUBSTRATE_VISION_CHANNEL_TICK_INTERVAL_SEC` (substrate-runtime).
- Renamed keys: none.
- `.env_example` updated: yes (all three services).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, the six new keys landed in the primary checkout's service `.env` files (exit 0).
- skipped keys requiring operator action: the two removed keys are left in the primary `services/orion-substrate-runtime/.env` on purpose until this merges (a pre-merge redeploy of main would otherwise turn the old tick off with nothing replacing it). Delete them after deploy; they are inert once this code runs.

## Metric quality gate (CLAUDE.md 0A)

### vision_frame_staleness

1. **Provenance.** `FrameDispatcher._handle_frame_envelope_inner` -> `OrganWindowRecorder.record_frame(stream)` stamps the newest frame per stream (after `VisionFramePointerPayload` validates); at window close `build_window_events` writes `last_frame_age_sec` (or `none`) and `uptime_sec`; `extract.stream_status_and_staleness` applies `vision_channel_staleness_pressure` (0.0 to 15 s, 1.0 at 60 s; never-seen aged from router uptime); `reducer.reduce_vision_organ_trace_events` takes the `min` over the window's streams.
2. **Independence.** Successor to `node:substrate.vision.prediction_error` (same question, "can Orion see"), which is killed, not kept alongside. Measured upstream of the detector (frame arrival at the router), so it also covers streams the old artifact clock pooled away. Overlaps `node:substrate.perception`'s `embedding_staleness` (same upstream chain: frames stop -> embeddings stop); that channel has no capability edge (shadow), so nothing double counts in `capability:vision`. Independent of `vision_processing_failure_pressure` (frames can flow while the host fails).
3. **Theory anchor.** Sensory availability = recency of the newest sample; the perception design doc's freshness rule that silence must converge toward alarm, never decay toward calm (the `node:substrate.route` decayed-to-zero incident).
4. **Live data.** 8-day replay of the router's own log (2026-09-24 03:50 .. 10-02 04:36 UTC, 11,567 windows): organ calm 11,566, non-calm 1 (0.202 at 2026-09-28 08:58:37, a real cam0 gap); rest point exactly 0.0, genuinely reached. 5-minute live bus replay (2026-10-02 05:06..05:10): 5/5 windows calm, cam0 ~820 frames/window. Per stream: carbon `never_seen` 1.0 in 100% of windows, walkway `never_seen` 1.0 in 100% (camera not deployed). Not saturated, not flat.
5. **Existing mechanism.** `_vision_channel_tick` (replaced, killed); router `/healthz` totals (process lifetime, no stream split, no field path); perception `embedding_staleness` (shadow, different node).
6. **Reversibility.** Three flags; one table; channels removable via `RETIRED_NODE_CHANNELS`; metric lock entries.

### vision_processing_failure_pressure

1. **Provenance.** `dispatcher._organ_failure` on `sweep_timeouts` (class `timeout`), invalid reply payload (`invalid_reply`), `ok=false` (host `error_code` or `host_error`); `_organ_reply_ok` on ok replies; reducer `failure_reading` = `hop_pressure(failed, ok, 10, 2)` over the last 600 s of windows, worse of pooled and worst single stream.
2. **Independence.** Not derived from frame arrival; vision tasks are plain bus publish + reply, not `rpc_request`, so the RPC delivery bridge does not see them. No existing field channel measures it.
3. **Theory anchor.** Delivery reliability: share of handed-off work with no usable answer; same counting rule as the RPC delivery bridge, which this lane reuses rather than re-picking. The borrowed floor (10 attempts, 2 failures) only matters at low volume; cam0 alone dispatches ~120 tasks per 10 min.
4. **Live data.** Router `/healthz` 2026-10-02: 141,092 dispatched, 143,304 replies, `host_errors_total=0`, `host_timeouts_total=1` over 8 days. The reading is 0.0 in every replayed window because the host genuinely did not fail; a non-calm value was **not observed live** and is shown only by tests (`test_host_down_reads_full_failure`, `test_a_failing_stream_is_not_diluted_by_a_busy_healthy_one`). Absent (no hint) when nothing was dispatched, never a fabricated 0.0.
5. **Existing mechanism.** Router `host_errors_total`/`host_timeouts_total` (cumulative, not windowed, not in the field).
6. **Reversibility.** Same as above.

### Retired: node:substrate.vision prediction_error

Pooled detect-artifact age across all cameras. 0 non-zero values in 124,612 field ticks 2026-09-29..10-02 (query below), during which carbon sent nothing. Killed outright: tick, listener, settings, compose keys, topology edge, ladder rung; field residue pruned via `RETIRED_PSEUDO_NODES`.

## Tests run

```text
frame-router:     PYTHONPATH=<wt>:. pytest services/orion-vision-frame-router/tests -q        81 passed
field-digester:   PYTHONPATH=<wt>:. pytest tests -q --ignore=tests/test_heartbeat_chassis.py 264 passed, 6 skipped
                  (test_heartbeat_chassis needs repo-root cwd; 3 passed from root)
substrate-runtime: pytest tests -q --ignore=tests/test_grammar_consumer_integration.py
                  same 13 failures + 9 errors as origin/main (env/DB-dependent), 0 new
sql-writer:       pytest tests/test_grammar_retention_periodic.py -q                         67 passed
orion/substrate:  pytest orion/substrate/tests -q   3 failed (test_felt_state_self_definition_lane, same on main), 861 passed
root tests/:      full run diffed against origin/main: 0 new failures after the metric re-lock
new:              tests/test_vision_organ_substrate_reducer.py 15 passed;
                  services/orion-substrate-runtime/tests/test_worker_vision_organ_tick.py 4 passed;
                  services/orion-field-digester/tests/test_field_vision_organ_perturbations.py 10 passed
static gates:     producer catalog, topology edges, substrate requests, ladder liveness, sql migration drift,
                  metric lineage, definition drift (after --update), inner-state registry, env-template parity,
                  env single source, compose parity -- all PASS
```

## Evals run

```text
PYTHONPATH=. python scripts/eval_vision_organ_replay.py --router-log <docker logs of the router>
  windows 11567 (2026-09-24 03:50 .. 2026-10-02 04:36 UTC)
  organ staleness: calm 11566, non_calm 1 (max 0.202, 2026-09-28T08:58:37Z)
  organ failure: calm 11567
  per stream: cam0 calm 11566 / stale 1; carbon never_seen 11567; walkway never_seen 11567
PYTHONPATH=. python scripts/eval_vision_organ_replay.py --live 310   (read-only bus subscription)
  windows 5: organ calm 5; cam0 calm 5 (~820 frames, ~12 dispatches per window); carbon never_seen 5
pytest tests/scripts/test_eval_vision_organ_replay.py   2 passed (calm AND non-calm on a fixture log)
```

## Docker/build/smoke checks

```text
Not deployed (by instruction). No docker build run; runtime proof is the deploy-time queries below.
```

## Review findings fixed

Review: code-review subagent on `git diff origin/main...HEAD`. No blockers.

- Finding: silence rewrite cadence was `VISION_ORGAN_SILENCE_SEC/3`, so a setting above 900 s let the digester's 300 s expiry flip "can't see" to "unmeasured" between writes; a router window above 300 s did the same to live readings.
  - Fix: cadence capped at `min(silence/3, 60 s)`; `VISION_ORGAN_WINDOW_SEC` bounded 5..100 s in router settings.
  - Evidence: `test_silence_writes_stay_under_the_digester_expiry_for_long_silence_settings`, `test_window_length_is_bounded_below_the_digester_expiry`.
- Finding: the "never a reading over half a window's streams" claim relied on stream atoms arriving before the closing atom.
  - Fix: the reducer now checks the closing atom's `streams=N` against the stream atoms it holds; on a mismatch it skips the reading and warns on the receipt.
  - Evidence: `test_window_missing_a_stream_atom_gives_no_reading`.
- Finding: `DRY_RUN` recorded dispatches that then all timed out, reading as a dead host.
  - Fix: under `DRY_RUN` the organ counts frames only (no dispatches, no timeouts).
  - Evidence: `test_dry_run_counts_frames_but_not_tasks`.
- Finding: metric lock conflicted with main.
  - Fix: merged main, took main's lock, re-ran `--update`.
  - Evidence: definition drift gate PASS (690 definitions).
- Finding: dead `window_sec` field on the stream state; `VisionOrganWindowCountV1` not registered; the replies-vs-dispatched denominator was undocumented; the single-router assumption was undocumented.
  - Fix: field dropped, model registered, both documented in the schema.
- Finding: stale comments about the retired tick.
  - Fix: retirement notes in `prediction_error.py` (yield helpers kept, uncalled, for the future day-shape prior), `settings.py`, and the substrate README P2 section.
- Finding: an idle poll re-read the projection every second.
  - Fix: the silence check now runs every 10 s.
- Documented, not fixed in code (below): expiry falls back to the digester's derived "perfect vision" vector; rollback order; FalkorDB residue; deploy order.

## Restart required

Apply the migration first, then deploy from the primary checkout on main after merge:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_organ_substrate_loop.sql
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-field-digester up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
```

Use this order: migration first, then the field digester, then substrate-runtime, then the router.

- **Field digester before substrate-runtime.** Otherwise an old digester drops `vision_organ` deltas, and the retired node's value decays toward a calm-looking 0.
- **Expect a short 1.0 reading.** Between the substrate-runtime and router restarts, `capability:vision` may read 1.0 for a few minutes because no router windows exist yet. That is the silence path working.

Optional cleanup, a production write, so it was not run here: the old FalkorDB concept node `node:substrate.vision` (graph `orion_substrate`) still holds `prediction_error=0` and is no longer written. Endogenous curiosity already ages it to zero. To delete it:
`docker exec orion-athena-falkordb redis-cli GRAPH.QUERY orion_substrate "MATCH (n {node_id:'node:substrate.vision'}) DETACH DELETE n"`

Rollback: turn the router flag (`VISION_ORGAN_GRAMMAR_ENABLED=false`) off first, then the reducer. If the reducer is off while the router keeps emitting, unconsumed `vision.organ:` rows pin sql-writer's grammar retention floor, and pruning of `grammar_events` stops.

After deploy, delete `SUBSTRATE_VISION_CHANNEL_TICK_ENABLED` and `SUBSTRATE_VISION_CHANNEL_TICK_INTERVAL_SEC` from the primary `services/orion-substrate-runtime/.env`. They are inert at that point.

Proof queries after deploy:

```sql
-- router reports landing
SELECT trace_id, count(*) FROM grammar_events WHERE source_service='orion-vision-frame-router'
  AND created_at > now()-interval '10 minutes' GROUP BY 1 ORDER BY 1 DESC LIMIT 5;
-- reducer receipts
SELECT count(*), max(created_at) FROM substrate_reduction_receipts
  WHERE reducer_name='vision_organ_reducer' AND created_at > now()-interval '10 minutes';
-- projection: per-stream status
SELECT projection_json->'status', projection_json->'vision_frame_staleness',
       projection_json->'vision_processing_failure_pressure',
       jsonb_object_keys(projection_json->'streams')
  FROM substrate_vision_organ_projection;
-- field: organ node and capability
SELECT field_json->'node_vectors'->'node:substrate.vision_organ',
       field_json->'capability_vectors'->'capability:vision',
       field_json->'capability_provenance'->'capability:vision',
       field_json->'node_vectors' ? 'node:substrate.vision' AS old_node_still_there
  FROM substrate_field_state ORDER BY generated_at DESC LIMIT 1;
-- the retired metric's history (the 124,612-tick finding)
SELECT count(*) FILTER (WHERE (field_json->'node_vectors'->'node:substrate.vision'->>'prediction_error')::float > 0), count(*)
  FROM substrate_field_state WHERE generated_at < '2026-10-02';
```

## Risks / concerns

- Severity: medium. Concern: `capability:vision` now takes the freshest stream, so carbon (or walkway) being dark does not raise capability pressure; it shows only on the projection and in each receipt. Mitigation: deliberate -- a laptop webcam is off whenever the laptop is, and there is no day-shape prior yet; a worst-stream reading would pin capability:vision at alarm every evening. Revisit when a per-stream expectation exists.
- Severity: medium. Concern: `vision_processing_failure_pressure` has never been non-zero on live data (the host failed once in 141k tasks), so its alarm side is verified by tests only. Mitigation: it is an absent-not-zero reading with a floor; first live failure burst will be in receipts (`failure_window.scope`).
- Severity: medium. Concern: once both organ channels expire (substrate-runtime or the lane dead for more than 300 s), the field digester's existing derived fallback gives `capability:vision` `pressure=0.0` and `confidence=1.0`. Only the empty `capability_provenance` shows it is unmeasured. This is existing digester behaviour, shared with `rpc_delivery`. Mitigation: the ladder rung `receipts:vision_organ_reducer` goes stale in that case. A digester-wide fix, treating an unmeasured edge source as non-calm, is a follow-up.
- Severity: low. Concern: if sql-writer stops persisting grammar events, the silence path reports vision as dark (1.0) though the cameras may be fine -- "can't confirm" reads as alarm, not calm. Intended direction, but it can name the wrong organ.
- Severity: low. Concern: walkway is enabled in `config/vision_frame_router.yaml` but its camera is not deployed (PR #2287), so it reports `never_seen` forever. Mitigation: true; disable it in the policy until the camera ships, or leave it as a visible reminder.
- Severity: low. Concern: no CI workflow runs the frame-router / field-digester / reducer unit tests; only the static gates run in CI. Mitigation: tests listed above were run locally; same situation as the llm_inference lane.

## Also checked (not changed here)

- **Hub situation brief, perception hardcoded off** (`orion/situational/context.py:523`): deliberate and documented twice (the adapter docstring and `services/orion-hub/app/settings.py`), originally pending a vetted DB dependency for Hub's event loop. That concern looks resolved (the perception reads already run under `asyncio.to_thread`), but turning it on puts camera-derived private-home narrative into Hub chat prompts, which is a privacy call, and Hub has no `ORION_SITUATION_PERCEPTION_ENABLED` key at all. Not a contained bug fix; recommend a separate PR adding the Hub key (default off) for Juniper to flip.
- **Lab context is always a stub** (`context.py:1092`): there is no lab provider anywhere (`stub` is the only value in code and templates); even enabled it returns `available=False`. Honest "unknown", not fake data. Needs a real provider or removal.
- **Carbon silence**: carbon-x1's `docker.service` was stopped gracefully on 2026-09-30 20:10 UTC and not restarted (enabled at boot, but the host has not rebooted since 2026-08-27). Before that, the retina container was running (814 h) yet the router saw zero carbon frames since its own start on 09-24, and the percept store received no JPEG since at least 09-21. Why the earlier silence happened is UNVERIFIED: the container logs are unreachable with the daemon down. Fix is an operator step on carbon (start docker / the retina), not code.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2483

🤖 Generated with [Claude Code](https://claude.com/claude-code)
