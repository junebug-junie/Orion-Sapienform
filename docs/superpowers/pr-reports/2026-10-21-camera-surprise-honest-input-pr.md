# fix(perception): camera surprise reads its real input

## Summary

- When a camera stops sending frames, its surprise node no longer records calm 0.0 readings. The tick still writes a 0.0 placeholder (the field digester needs a write every tick), but it marks it `prediction_error_missing_reason` (`stale`, `warming`, or `no_scored_frame_since_last_tick`). A marked reading never enters `substrate_node_prediction_error_history`, has no magnitude in the broadcast (absent), and the self-model omits it.
- Each tick publishes the latest scored frame that arrived since the previous tick, and clears it. The same frame is never recorded twice. **Peak-hold was built, replayed, and removed: it failed the metric gate** (record below).
- The ladder watch (host cron, every 10 min, Hub card) has two new rungs, `camera_input:cam0` and `camera_input:walkway`. They go red when a 24/7 camera has sent no embedding for 15 min. On replay they fire 20 min into the 49 h outage of 2026-10-18.
- The walkway camera is now a world source. It competes in both world-first attention contests, but only at its own p99+ band, judged by mid-rank (`ATTENTION_MIN_BAND_BY_NODE_ID`). The broadcast's salience signals still fence it.
- The equilibrium self-signal gate now observes a tracked episode for a node that stopped being a body signal as absent, so the episode can close `went_silent` instead of hanging open.

## Outcome moved

- **49 h of silence no longer looks like 49 h of calm.** From 2026-10-18 04:06 to 10-20 05:51 no embeddings arrived. The room node wrote 8,306 zero rows (both nodes together) into the 7-day window, 31% of it. The room camera's p90 then read 0.137, against 0.205 over live hours only. After this patch those ticks write no rows.
- **A dead camera now raises a card.** It used to raise nothing for 49 h. The receipts rung stayed green the whole time, because the tick keeps writing receipts.
- **The walkway can now win attention, but only for its rare moments.** Replay 10-17..21 (live hours): 9.9 win-ticks/day, in 7.9 distinct episodes/day. 45% of walkway wins fall within 2.5 min of a person/dog/bike/stroller motion story, against 7.2% for a random live minute.

## Current architecture (before)

- `services/orion-substrate-runtime/app/worker.py` `_handle_perception_prediction_error_message` scores every embedding (`perception_prediction_error`, z-score of cosine distance from the stream EWMA, 3 sigma = 1.0) and caches the latest score per node group.
- `_publish_perception_group` runs every 30 s. It writes the cached score, or 0.0 when stale or warming, to the graph node and a receipt. `_prediction_error_magnitudes` samples every `node:substrate.*` node into `substrate_node_prediction_error_history` whenever `observed_at` moves, so stale 0.0s became history rows.
- Every attention contest ranks against that 7-day window (`orion/attention/pe_history_cache.py`, the broadcast's own cache, the equilibrium gates).
- The walkway node was fenced from all attention (`NO_ATTENTION_CONSUMER_NODE_IDS`) and classed as body (internal). That meant the equilibrium self-signal gates read it, including its stale zeros.
- The ladder watch checked perception receipts, which keep arriving while the camera is dead.

## Architecture touched

- orion-substrate-runtime: perception tick and history sampler.
- Shared attention seam: `orion/schemas/attention_candidate.py`, `orion/attention/world_first.py`, `orion/substrate/attention_broadcast.py`, `orion/substrate/prediction_error_freshness.py`.
- orion-attention-runtime: candidate fence.
- orion-equilibrium-service: episode gate.
- Host cron ladder watch: `orion/substrate_ladder_liveness.py`. It runs from the primary checkout, so it goes live on `git pull` with no restart.
- Semantic layer: glossary and metric lock.

## Files changed

- `services/orion-substrate-runtime/app/worker.py`: take-and-clear latest score under a lock (listener thread vs tick thread); missing reasons; node metadata marker; sampler skips marked nodes.
- `orion/substrate/prediction_error_freshness.py`: `PREDICTION_ERROR_MISSING_REASON_KEY`, `missing_input_reason`, `STATUS_MISSING_INPUT` in `classify_reading`.
- `orion/schemas/attention_candidate.py`: walkway is world; `ATTENTION_MIN_BAND_BY_NODE_ID`; `world_first_fenced`.
- `orion/attention/world_first.py`: band floor in `judge_candidate` (mid-rank); docstrings.
- `orion/substrate/attention_broadcast.py`: world-first builder uses `world_first_fenced`; camera nodes absent on missing marker / walkway embedding staleness.
- `services/orion-attention-runtime/app/worker.py`: uses `world_first_fenced`.
- `services/orion-equilibrium-service/app/self_signal_episode_gate.py`: tracked no-longer-body signals observed absent.
- `orion/substrate_ladder_liveness.py`: `camera_input:*` rungs, `CAMERA_INPUT_MAX_AGE` (15 min), `CAMERA_INPUT_STREAMS_EXPECTED_ONLINE` (cam0, walkway).
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`: meaning/rest/absent_means for both camera nodes; re-locked after merging main.
- `services/orion-substrate-runtime/README.md`, `orion/substrate/endogenous_curiosity.py` (comment): describe the new behaviour.
- `services/orion-substrate-runtime/evals/run_camera_surprise_input_eval.py` (+ test): read-only gate eval.
- Tests: `services/orion-substrate-runtime/tests/test_worker_perception_honest_input.py` (new), `test_worker_perception_walkway_split.py`, `orion/substrate/tests/test_attention_broadcast_world_first.py`, `services/orion-attention-runtime/tests/test_world_first_worker.py`, `services/orion-equilibrium-service/tests/test_self_signal_episode_gate.py`, `tests/test_attention_world_first.py`, `tests/scripts/test_substrate_ladder_liveness.py`.
- `.github/workflows/substrate-neighborhood.yml`: runs the new worker tests and the eval test.

## Schema / bus / API changes

- Added: node metadata key `prediction_error_missing_reason` (string, in-process; not in the Falkor `DYNAMICS_METADATA_KEYS` allowlist, see Risks). Ladder rungs `camera_input:cam0`, `camera_input:walkway`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - Camera nodes' history holds real readings only.
  - Each reading is a frame that arrived since the previous tick.
  - The walkway is `external` and competes in world-first attention at mid-rank p99+ only.
- Compatibility notes:
  - No bus channel or schema model changed.
  - Field digester receipts are unchanged (still 0.0 while missing).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Metric gate record (both camera nodes' `prediction_error`)

1. **Provenance.**
   - Per-frame score: `orion/substrate/prediction_error.py` `perception_prediction_error` (cosine distance, `_domain_zscore` with the 2e-6 variance floor, `min(1, z/3)`).
   - Published value: `worker.py` `_take_perception_score` -> `_publish_perception_group`.
   - Missing marker: `_publish_perception_group` -> `_write_prediction_error_node(missing_reason=...)`.
2. **Independence.**
   - Same chain as before: no new metric, the meaning of an existing one changed.
   - The walkway node is the same formula on a different stream with its own baseline.
   - The new alarm reads `substrate_perception_embedding_baseline` row times. That is the scorer's input, so it measures availability, not surprise.
3. **Theory anchor.**
   - Surprise = deviation of the current frame embedding from the stream's running expectation (prediction error, P2 spec 2026-08-12).
   - "No input" is not "no surprise": an absent measurement must read absent (CLAUDE.md §0A).
4. **Live-data sanity.** `run_camera_surprise_input_eval.py --days 7` on live Postgres (2026-10-21), summarised:

   | | rest (exact 0.0) | p50 | p90 | at 1.0 | >= 0.55 (curiosity) |
   |---|---|---|---|---|---|
   | cam0 per-frame (90,255) | 61.3% | 0.000 | 0.201 | 1.27% | 2.89% |
   | room, **latest (ships)**, 9,578 ticks | 61.0% | 0.000 | 0.208 | 1.24% | 2.82% |
   | room, peak-hold (rejected) | **4.5%** | 0.218 | **1.000** | **10.2%** | **20.1%** |
   | walkway per-frame (63,818) | 62.1% | 0.000 | 0.207 | 1.41% | 3.08% |
   | walkway, **latest (ships)**, 7,130 ticks | 62.9% | 0.000 | 0.203 | 1.51% | 3.11% |
   | walkway, peak-hold (rejected) | **7.5%** | 0.211 | **1.000** | **11.2%** | **20.3%** |

   - Missing ticks over 7 days: room 4,302 stale + 2 no-new-frame; walkway 4,711 stale + 4 no-new-frame. Those are the ticks that used to be 0.0 rows.
   - The shipped series can rest: 61% exact 0.0, the same as the per-frame score. Rest is reachable and is not a decay artifact, because it comes from real frames only.
   - **Why peak-hold failed.** The score saturates at 1.0 for 1.3% of frames. The max of a 30 s window's ~4 frames was therefore 1.0 in 10% of windows. Exact-zero ticks fell from 61% to 4.5%. Curiosity's fixed 0.55 threshold would fire 7x as often.
   - Peak-hold also hurt attention. In the contest replay below, room-camera wins fell 422 -> 44, and the camera's above-chance catch of arrivals vanished. The walkway's p99+ band disappeared (no reading reaches mid-rank p99 when 11% tie at the ceiling).
   - Peak-hold would only make sense after the 3-sigma saturation itself is revisited. That is a separate metric change with its own gate (follow-up).
5. **Existing mechanism.**
   - The alarm reuses the ladder watch (cron, debounced orion-notify Hub card). No new service.
   - Missing-ness reuses the self-model's omit-when-not-fresh rule (`classify_reading`).
6. **Reversibility.**
   - The marker is in-process metadata plus one sampler check. Removing it restores old writes.
   - The band floor is one dict entry.
   - No schema, migration, or training default bakes any of it in.

## Replay evidence (stored data, read-only)

Contest replay: the worktree's real `node_candidate` / `rank_candidates` code on a 60 s grid, 2026-10-15 06:00 .. 10-21 03:00 (8,460 ticks).

- BEFORE = stored history (stale zeros included), walkway fenced.
- AFTER = this patch: latest frame per tick, no rows while missing, walkway admitted at its mid-rank p99+.

Script: scratchpad `replay_honest.py`. The committed eval reproduces the gate numbers.

**Winners, live hours (5,476 ticks):**

| | no winner | room camera | walkway | world:chat | cabinet | bus_synaptic | execution |
|---|---|---|---|---|---|---|---|
| before | 70.7% | 493 (9.0%) | fenced | 5.0% | 4.0% | 2.9% | 2.7% |
| after | 70.8% | 422 (7.7%) | 68 (1.2%) | 5.0% | 3.9% | 2.9% | 2.8% |
| peak-hold variant | 77.8% | 44 (0.8%) | 4 (0.1%) | 5.1% | 4.2% | 3.1% | 3.0% |

- During the 49 h gap all three are identical. The stale zeros could never win: they read quiet, so they only ever distorted the window.
- **Room-camera wins by band:**
  - before: high 472, unusual 21;
  - after: high 422, unusual 0.
  - The 21 "unusual" (p99) wins before existed only because the stale zeros pushed 1.0 readings above p99.
- **Room arrivals caught** (cam0 presence -> present, 66 in live hours; a room-camera win within -30 s..+150 s):

  | | caught | room win-ticks | random placement of the same number of wins |
  |---|---|---|---|
  | before | 26 | 493 | median 16, p95 22 |
  | after | 23 | 422 | median 14, p95 20 |
  | peak-hold | 2 | 44 | median 1, p95 4 |

  - The room camera's "high" band is above chance both before and after. The task's working claim that "221/234 camera wins were noise" did not hold up on this replay. Above chance, but weak: see the board offer to the vision agents below.
- **Walkway (live, since motion stories begin 10-17 00:36, 48.6 h):**
  - 20 win-ticks = 9.9/day, in 16 episodes (7.9/day).
  - 45.0% of win-ticks (37.5% of episodes) fall within 2.5 min of a person/dog/bicycle/stroller story; the base rate is 7.2%.
  - Its own readings by mid-rank band: below p90 7.2% near an event, p90-p99 6.2% (chance), p99+ 35.9% (n=39).
  - The p90 band would have leaked chance-level wins; the floor keeps it out.
- **Alarm fire times** (cron every 10 min, `camera_input` rung, real embedding timestamps):
  - cam0: red 10-15 19:10 (newest 18:54:00), green 20:50; **red 10-18 04:30 (newest 04:06:45, 23 min), green 10-20 06:00**.
  - walkway: red 10-15 17:30, 10-15 19:10, 10-16 02:30, 10-17 15:50, **10-18 04:30 -> green 10-20 06:00**.
  - Every red is a real outage. The next-longest gap in those 7 days was 3 min 14 s.
- Live now: `scripts/check_substrate_ladder_liveness.py --verbose` (read-only) shows `ok camera_input:cam0 (3s old, limit 15m)` and `ok camera_input:walkway (2s old)`.

## Tests run

```text
services/orion-substrate-runtime: tests/test_worker_perception_honest_input.py tests/test_worker_perception_prediction_error.py tests/test_worker_perception_walkway_split.py tests/test_worker_pe_history.py -> 52 passed
services/orion-substrate-runtime full tests (minus 2 files needing a local Postgres) -> 474 passed, 14 failed, 9 errors; the same 14 failed / 9 errors on a clean origin/main worktree (pre-existing: cursor reset/quarantine/reducer health)
mutation checks: latest-frame take-and-clear removed -> republish test fails; sampler marker check removed -> 2 history tests fail
tests/test_attention_world_first.py tests/scripts/test_substrate_ladder_liveness.py tests/test_metric_definition_drift.py tests/test_metric_rest_range.py orion/substrate/tests/test_attention_broadcast_world_first.py services/orion-attention-runtime/tests -> 271 passed, 5 skipped
orion/metacog/tests/test_self_signal_episode.py tests/test_attention*.py orion/substrate/tests/test_attention_broadcast*.py test_attention_self_model*.py test_endogenous_curiosity*.py ... -> 592 passed
services/orion-equilibrium-service: test_self_signal_episode_gate.py evals/test_self_signal_episode_replay_eval.py test_self_signal_outcome_rollup.py -> 38 passed
scripts/check_metric_lineage.py --gate PASS; --prompt-semantics PASS; check_definition_drift.py --gate PASS; check_inner_state_registry.py OK
```

`check_metric_lineage.py --metric node:substrate.perception` reports UNREGISTERED because the token is wrong, not because the metric is missing. The registry keys on `<node>.<channel>`; `--metric node:substrate.perception.prediction_error` resolves to its URN with the updated card. Nothing was hand-registered.

## Evals run

```text
python services/orion-substrate-runtime/evals/run_camera_surprise_input_eval.py --dsn postgresql://...:55432/conjourney --days 7
  -> GATE PASS (shipped mode: latest); table above
pytest services/orion-substrate-runtime/evals/test_camera_surprise_input_eval.py -> 4 passed
  (reconstruction checked on 5 verbatim live baseline rows vs the logged scores 0.151, 0, 0, 0)
```

## Docker/build/smoke checks

```text
No image built: no dependency, Dockerfile, compose, or env change. Live path UNVERIFIED until deploy (see post-deploy checks).
Ladder rungs verified live, read-only (camera_input:cam0/walkway ok).
```

## Review findings fixed

(filled in after the review subagent)

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-substrate-runtime orion-attention-runtime orion-equilibrium-service; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build || break; done
```

- The ladder watch needs no restart: host cron runs the primary checkout's script.
- Order: substrate-runtime first. It stops writing stale rows and owns the broadcast contest. attention-runtime and equilibrium only change in how they judge the walkway.

## Optional operator cleanup (NOT run)

The 9,012 stale 0.0 rows already in the window age out by 2026-10-27 on their own. To drop them now: snapshot them, then delete. These are production writes; Juniper's call. The predicate: a camera-node row with no embedding for its stream in the 60 s before it. Both steps are one line each:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "\copy (SELECT h.* FROM substrate_node_prediction_error_history h WHERE h.node_id IN ('node:substrate.perception','node:substrate.perception_walkway') AND NOT EXISTS (SELECT 1 FROM substrate_perception_embedding_baseline b WHERE b.stream_id = CASE h.node_id WHEN 'node:substrate.perception_walkway' THEN 'walkway' ELSE 'cam0' END AND b.generated_at BETWEEN h.observed_at - interval '60 seconds' AND h.observed_at)) TO '/tmp/camera_stale_rows.csv' CSV HEADER"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "DELETE FROM substrate_node_prediction_error_history h WHERE h.node_id IN ('node:substrate.perception','node:substrate.perception_walkway') AND NOT EXISTS (SELECT 1 FROM substrate_perception_embedding_baseline b WHERE b.stream_id = CASE h.node_id WHEN 'node:substrate.perception_walkway' THEN 'walkway' ELSE 'cam0' END AND b.generated_at BETWEEN h.observed_at - interval '60 seconds' AND h.observed_at)"
```

## Risks / concerns

- Severity: medium.
  - Concern: the field digester still receives 0.0 for a camera whose input is missing (receipt path), and so do `node_vectors` readers.
  - Mitigation: the audit found none of them ranks perception from the field value. attention-runtime uses the field only to list node ids. `collect_field_channel_pressures` max-merges, and `prediction_error` has no dimension mapping. A true "unmeasured" for that channel needs field-digester expiry support (follow-up).
- Severity: low.
  - Concern: `prediction_error_missing_reason` is not in the Falkor durable allowlist (`DYNAMICS_METADATA_KEYS`). Every reader that acts on it (sampler, broadcast, self-model) runs in substrate-runtime against the in-process cache. After a restart, the next tick (30 s) rewrites it.
  - Mitigation: none needed today. Promote it if a reader in another process ever needs it.
- Severity: low.
  - Concern: the walkway changes from body to world. Any walkway self-signal episode open at deploy is observed absent and closes `went_silent` after the gate's silence window.
  - Mitigation: covered by a test.
- Severity: low.
  - Concern: the 3-sigma saturation makes the top 1.3% of frames indistinguishable, which is the root of the peak-hold failure.
  - Mitigation: follow-up, gated separately.
- Severity: low (merge order).
  - Concern: PR #2826 (street attention node) edits the same `_WORLD_NODE_IDS` lines in `attention_candidate.py`.
  - Mitigation: resolve by taking the union. A board item names this.

## Post-deploy read-only checks

- `docker logs --since 5m orion-athena-substrate-runtime 2>&1 | grep perception_prediction_error_tick_completed | tail -4` shows `missing=no` with real scores.
- `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select node_id, count(*), max(observed_at) from substrate_node_prediction_error_history where node_id like 'node:substrate.perception%' and observed_at > now() - interval '10 minutes' group by 1"` shows about 20 rows per node.
- `make substrate-ladder-check` shows `camera_input:*` ok.

## PR link

(filled in after `gh pr create`)
