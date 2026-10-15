## Summary

- The walkway camera now runs on its own `orion-vision-retina` instance, `orion-vision-retina-walkway`. The retina is the canonical capture service, so this replaces the `orion-vision-edge` walkway profile from #2287, which is removed. It reads the camera's full-resolution 2560x1920 main stream instead of the 640x480 substream, which puts about 16x the pixels on people across the street, and samples 2 frames a second.
- `RtspFrameSource` now drains the live stream on a reader thread and serves only the newest frame, once. Calling `cap.read()` once per tick on a 30 fps stream returned the oldest buffered frame, so the picture drifted further behind the live one the longer it ran.

- The walkway camera is mounted (Reolink, 192.168.1.87, edge reads the 640x480 substream `h264Preview_01_sub`).
- Replaces the placeholder zone shapes in `config/vision_zones.yaml` with ones traced from a live 2560x1920 frame:
  - **patio:** the porch slab, its plant bed and the furniture, lower right. Never embedded and never thumbnailed.
  - **mailbox:** the community mailbox cluster, left-middle.
  - **walkway:** everything public below the sky line.
- Tests that hardcoded the placeholder coordinates now use boxes on the real shapes.
- The walkway camera's surprise score gets its own node, `node:substrate.perception_walkway`. The room cameras (cam0, carbon) keep `node:substrate.perception` to themselves. Before, one shared slot held whichever camera scored last, so the busy street would have taken over the room reading.

## Outcome moved

Before this, the patio rule protected the wrong part of the picture. The placeholder put the "patio" on the lower-left lawn, while the real porch is lower right. Starting the camera on the placeholder would have embedded crops of people on the porch.

## Current architecture

The walkway pipeline is merged and deployed (PR #2287). The zones file is baked into the vision-host, vision-council and sql-writer images, and the walkway edge instance (`vision-edge-walkway` compose profile) is not started.

## Architecture touched

Config only: `config/vision_zones.yaml`. No code changes.

## Files changed

Walkway surprise split (this update):

- `services/orion-substrate-runtime/app/worker.py`: walkway streams go to their own cache (`_handle_perception_prediction_error_message`, ~1617); the tick writes both nodes through `_publish_perception_group` (~1709-1760), each isolated so one failing never skips the other.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: `SUBSTRATE_PERCEPTION_WALKWAY_STREAM_IDS` (default `walkway`).
- `orion/schemas/attention_candidate.py`: `PERCEPTION_WALKWAY_NODE_ID` and `NO_COGNITION_CONSUMER_NODE_IDS`.
- `orion/substrate/endogenous_curiosity.py:287`, `orion/substrate/attention_broadcast.py:140,349`: skip fenced nodes.
- `orion/substrate_ladder_liveness.py`: rung `receipts:substrate.perception_walkway`.
- `config/field/orion_field_topology.v1.yaml`: comment noting the node (no `nodes:` entry, no edge, same as the room node).
- Tests: `services/orion-substrate-runtime/tests/test_worker_perception_walkway_split.py` (new), `test_prediction_error_receipt_not_gated.py` (walkway in the every-node receipt check), `orion/substrate/tests/test_endogenous_curiosity_walkway_fence.py` (new), `orion/substrate/tests/test_attention_broadcast_world_first.py` (fence test).

Zones and retina:

- `config/vision_zones.yaml`: real polygons traced from a frame.
- `orion/vision/tests/test_zones.py`: boxes on the real patio and walkway.
- `services/orion-vision-council/tests/test_unresolved.py`: same.

## Schema / bus / API changes

None.

## Env/config changes

- Added to the substrate-runtime `.env_example`: `SUBSTRATE_PERCEPTION_WALKWAY_STREAM_IDS=walkway`. Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-substrate-runtime` (added that one key). No skipped keys.
- Added to the retina `.env_example`: `WALKWAY_RTSP_URL` (left empty; never-sync), `WALKWAY_RETINA_FPS` (2.0), `WALKWAY_RETINA_JPEG_QUALITY` (88), `WALKWAY_RETINA_HTTP_PORT` (8028).
- Removed from the edge `.env_example`: `WALKWAY_RTSP_URL` and `WALKWAY_EDGE_PORT`.
- Local `.env` files: the walkway URL now lives in the retina `.env` and points at the main stream. Same credentials as cam0, verified with ffprobe. Both keys were removed from the edge `.env`. Neither file is committed.

## Tests run

```text
services/orion-substrate-runtime/tests: 454 passed, 17 failed -- the same 17 fail on origin/main (DB-backed cursor/quarantine/truth tests); test_grammar_consumer_integration.py needs a local Postgres and errors on both
orion/substrate/tests: 1129 passed, 3 failed -- the same 3 fail on origin/main (test_felt_state_self_definition_lane.py)
tests/test_attention_world_first_parity.py, test_attention_self_readings.py, test_attention_event_decay.py: 47 passed
services/orion-attention-runtime/tests/test_world_first_worker.py: 6 passed
every command in .github/workflows/orion-static-gates.yml: pass (glossary entry withheld, see the metric gate section)
orion/vision/tests: 25 passed
services/orion-vision-council/tests: 110 passed
orion-vision-host test_crop_embeddings_patio + test_crop_thumbnails (inside the existing image): 22 passed
services/orion-sql-writer/tests: no new failures (failure list identical to origin/main)
```

## Evals run

```text
services/orion-substrate-runtime/evals: 12 passed
None. This is a config change. The shapes were checked visually by drawing them over the live frame.
```

## Docker/build/smoke checks

```text
ffprobe on rtsp://<cam0 creds>@192.168.1.87:554/h264Preview_01_sub -> h264 640x480
```

## Review findings fixed

- Finding: a `stop()` that timed out on a blocked read left the old reader thread alive, and the next `start()` revived it through the shared stop flag. The result was two RTSP sessions, each decoding the 5MP stream. Reproduced by the reviewer.
  - Fix: each reader thread has its own stop event. A reader that is still stuck is abandoned with its flag already set, so it exits on its next read.
  - Evidence: `test_a_stop_that_times_out_never_leaves_two_live_readers`.
- Finding: a decoder exception killed the reader silently.
  - Fix: the reader logs the error, releases the capture, waits, and reopens. The capture is also released in a `finally`.
  - Evidence: `test_a_decoder_exception_reconnects_instead_of_killing_the_reader`.
- Finding: after every boot or reconnect, health showed degraded until the first frame arrived.
  - Fix: before the first frame, an empty read is no longer counted as a failure.
- Finding: shutdown could hit Docker's 10s default grace before the bus closed.
  - Fix: `stop_grace_period: 15s`; the join timeout is 5s.
- Finding: docs and messages still described an edge walkway instance.
  - Fix: corrected in the router YAML and README, the edge settings and stream filter, and the zones comment.

- Finding: the zone overlay drawn on the live frame was the review.
  - Fix: none needed. The patio covers the slab, plant bed and furniture; the mailbox box covers the mailbox cluster plus the ground in front of it.
  - Evidence: the overlay image was checked in-session.

## Metric quality gate: `node:substrate.perception_walkway.prediction_error`

1. **Provenance.** Same scorer as the room node: `orion/substrate/prediction_error.py:1354` `perception_prediction_error()`, stage 1 `1 - cos(frame_embedding, stream EWMA)`, stage 2 z-score of that against its own EWMA, negatives clipped, `min(1, z/3)`. Routed by `services/orion-substrate-runtime/app/worker.py:1617` (walkway streams only), cached in `_PerceptionGroupState`, written by `_publish_perception_group` (`worker.py:1741`) every tick through `_prediction_error_receipt` and `_write_prediction_error_node`.
2. **Independence.** Shares the formula and the SigLIP embedding model with `node:substrate.perception`, but not a single input: its baseline row is the walkway stream's own (`substrate_perception_embedding_baseline`, keyed by stream), and after this patch no frame feeds both nodes. The two read different scenes (street vs rooms), so they are separate signals, not a transform of one another. It is redundant with nothing else: no other metric reads walkway frame embeddings.
3. **Theory anchor.** Predictive coding: a running expectation of the scene's embedding, and surprise as the distance of the new frame from it, scaled by how much surprise is usual for that scene. Same anchor the room node was admitted on.
4. **Live sanity: PENDING.** The walkway camera has not produced data yet. Before removing the node from `NO_COGNITION_CONSUMER_NODE_IDS`, pull a day of real history and check it can rest at 0 on a quiet street and fires on real arrivals, and that it is not a decayed-to-zero artifact (it is written every tick, so the `embedding_staleness` channel should sit at 0.0 while the camera delivers):

   ```sql
   SELECT date_trunc('hour', observed_at) AS hour, count(*) AS n,
          avg((value = 0)::int) AS frac_zero,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS p50,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95,
          max(value) AS max
   FROM substrate_node_prediction_error_history
   WHERE node_id = 'node:substrate.perception_walkway'
     AND observed_at > now() - interval '24 hours'
   GROUP BY 1 ORDER BY 1;
   ```

   Also confirm the room node stopped seeing the walkway: `grep substrate_perception_prediction_error_scored` in the substrate-runtime journal should show `stream_id=walkway node_id=node:substrate.perception_walkway`, never `node_id=node:substrate.perception`.
5. **Existing mechanism.** Checked: per-stream baselines already existed (`substrate_perception_embedding_baseline`); only the published node was shared. This reuses the scorer, the baseline table, the receipt writer and the node writer. Nothing new was built except the routing and the second write.
6. **Reversibility.** Cheap. Set `SUBSTRATE_PERCEPTION_WALKWAY_STREAM_IDS=none` to fold the walkway back into the room node, or delete the walkway branch and tick call. No schema, table or migration was added. The node id is in no lock file (the glossary entry was withheld) and in no consumer list.

**Gate refusal (exact output, not bypassed).** With the drafted glossary entry added, `python scripts/check_metric_lineage.py --gate` printed:

```text
  orphans by surface: {'bus_channel': 17, 'field_channel': 1, 'inner_state': 12, 'organ_signal': 146}
  746 URNs, 5771 files scanned
metric lineage gate: FAIL (1)
  - orphan ratchet: field_channel grew 0 -> 1. A registered metric with no consumer feeds nothing (CLAUDE.md 0A). Wire it to a real consumer, or retire the registry entry.
```

The entry is withheld rather than forcing the gate. Options for Juniper: (a) register it with a declared non-cognition reader, such as the prediction-error history recorder that will feed the live check above; (b) add it when the walkway node gets its first real consumer, after the live check; (c) accept a deliberate ratchet bump.

## Restart required

Walkway surprise split: after merge, from the primary checkout on main, before starting the walkway retina:

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Zones and retina:

Rebuild the three images that bake the zones file, then start the walkway edge. All of this runs from the primary checkout on main after merge:

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-retina --profile walkway up -d --build orion-vision-retina-walkway
```

## Risks / concerns

- **Resolved: walkway frames no longer feed the room perception-surprise node.** Juniper chose a separate node. Streams in `SUBSTRATE_PERCEPTION_WALKWAY_STREAM_IDS` now feed `node:substrate.perception_walkway` from their own cache; `node:substrate.perception` only ever holds cam0/carbon scores, and the world-model embedding cache stays room-only. Both nodes are written every tick. The walkway node is fenced out of endogenous curiosity and attention until its live-data check runs. Proven by `test_worker_perception_walkway_split.py`.
- **Severity: medium. The walkway node's metric registry entry is not in this PR.** The metric-lineage orphan ratchet refuses a field channel with no consumer (`field_channel grew 0 -> 1`), and the node has no consumer on purpose. The drafted glossary entry is ready; it needs Juniper's call (see the metric gate section).
- **Severity: low. Other generic readers still see the walkway node.** The substrate dynamics engine seeds pressure from any node's `prediction_error` (`orion/substrate/pressure.py:48`), the field digester puts it in the field's node vectors, and the attention frame lists it as observed, not attended. None of these can make it win attention or seed curiosity, but it is not invisible.
- **Severity: low. The background reader decodes the full 5MP stream continuously** to keep 2 frames a second. That costs about one CPU core.
- **Severity: low. Walkway frames are not watchable in Hub yet.** Hub's live view proxies percept-store streams only, and walkway frames are saved to local disk.

- **Severity: medium. The Halloween skeletons on the porch may read as "people on the patio".**
  - Two skeleton props sit at the patio table. The detector may count them as `person`, so the patio presence count, and the street line stance reads, could say people are on the patio all night.
  - They are never embedded or named, because the patio rule holds.
  - Mitigation: watch the patio presence row after go-live.
- **Severity: low. The shapes were traced from a night frame.** Recheck the mailbox box against a daytime frame.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
