# Orion Vision Window

Aggregates vision artifacts into rolling time windows and publishes `VisionWindowPayload` summaries for Council.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `service.bus` above.

## Role in the pipeline

```text
orion:vision:artifacts  ──►  orion-vision-window  ──►  orion:vision:windows
   (edge + host artifacts)         (aggregate)              (VisionWindowPayload)
```

Also publishes, per host artifact that carries tracked-label boxes (walkway
camera, `docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`
idea 1), one `VisionCropObservationV1` to `orion:vision:crops:sql-write`
(`CHANNEL_CROP_OBSERVATIONS_PUB`, toggle `WINDOW_CROP_OBSERVATIONS_ENABLED`) for
orion-sql-writer's individuals reducer. A box is "tracked" when the host set a
`zone` or an `embedding` on it (`want_crop_embeddings`). Patio boxes carry
vectors and, on person crops, `face_embedding`/`face_thumb_ref` like the rest
(2026-10-15). Frame size
comes from the host's `frame_width`/`frame_height` output. Code: `app/crops.py`.
Counters: `vision_window_crop_observations_{published,failed}_total` on
`/api/vision-window/metrics`.

## Evidence tiers

Window summaries include an `evidence` block used by Council for grounded interpretation:

| Field | Meaning |
|-------|---------|
| `hard_labels` | Detection labels above score threshold (factual tier) |
| `believed_hard_labels` | Habituated labels (vote debounce over last N flushes; council gate input) |
| `belief` | Metadata: `schema`, `vote_n`, `enter_votes`, `exit_votes`, `observation_count` |
| `soft_labels` | Tokens from captions matching stoplist (YouTube, google, …) |
| `edge_person_hits` | Always 0 (edge artifacts excluded from pipeline evidence) |
| `host_person_hits` | Person detections from host `retina_fast` artifacts |
| `caption_count` | Number of captioned artifacts in the window |

Council treats `hard_labels` as admissible evidence; captions are soft hints only. Council gates on `believed_hard_labels` when `belief.schema == scene_belief.v1`.

**Empty flicker:** Enter votes treat empty observations as carrying forward the last non-empty labels; exit votes count raw observations only. Tune `WINDOW_BELIEF_EXIT_VOTES` to control how quickly removed objects leave belief.

**Production defaults (2026-07-04 tuning):** `WINDOW_BELIEF_ENTER_VOTES=3`, `WINDOW_BELIEF_EXIT_VOTES=0` — blocks intermittent marginal detections (e.g. package) from belief ping-pong while keeping door/screen stable.

## Configuration

```bash
cd services/orion-vision-window
cp .env_example .env
```

Key channels: `CHANNEL_WINDOW_INTAKE` (artifacts in), `CHANNEL_WINDOW_PUB` (windows out).

## Tests

From repo root:

```bash
PYTHONPATH=services/orion-vision-window:. pytest services/orion-vision-window/tests -q
```

See also: `docs/vision_services.md` for cross-service contracts.

## Presence history (Temporal Self seam S1)

Besides the single-row `substrate_embodied_presence` upsert, the presence state machine
(`app/presence.py`) appends one row to `vision_presence_transition` per state change
(`present` / `recent` / `absent`), `unknown` when a stream sends no window for
`STALE_AFTER_MS` (120 s) or the service stops, and a same-state keepalive row every
`VISION_PRESENCE_KEEPALIVE_SEC` (600 s) while someone is present. The chronicle closes a company
arc with no row for 1500 s as stale, so a crash cannot leave "someone is here" asserted. The enrolled identity label is stored only when
`identity_confirmed` is true. orion-durable-runs folds these rows into Temporal Self `company`
arcs. Flag: `VISION_PRESENCE_TRANSITIONS_ENABLED` (default true). Needs
`services/orion-sql-db/manual_migration_vision_presence_transition_v1.sql`; without it each write
logs `presence_transition_write_failed` and nothing else changes.

## Stationary fixtures (presence)

On streams in `WINDOW_STATIONARY_STREAMS` (default `walkway`) a `person` box
that keeps IoU >= `VISION_STATIONARY_IOU` (0.5) with its first box, gaps <=
`VISION_STATIONARY_MAX_GAP_SEC` (300), for `VISION_STATIONARY_MIN_SEC` (1800)
is a prop. A window whose person boxes are all props does not make the stream
`present` (`app/fixtures.py`); any moving person box does. Found live
2026-10-15: a Halloween skeleton at the patio table, detected as `person` in
the same box all night. Room cameras are not listed: a person at a desk sits
still for hours. Tests: `tests/test_stationary_fixtures.py`.

## Caption gate (detector agreement)

On streams in `WINDOW_CAPTION_GATE_STREAMS` (default `walkway`) a caption
reaches the window summary -- and so the council narrative, `vision_events`,
the chat situation brief's street line -- only if the detections of its own
frame (`inputs.source_objects`, attached by vision-frame-router's
`caption_dispatch`) include a `WINDOW_CAPTION_GATE_LABELS` object at
`>= WINDOW_CAPTION_GATE_MIN_SCORE` that is not a stationary prop (same
`VISION_STATIONARY_*` rule as above, tracked per label), and it arrived within
`WINDOW_CAPTION_MAX_AGE_SEC` of its frame (`app/caption_gate.py`). A rejected
caption stays on the artifact as `text=""`, `confidence=0.0`,
`rejected_reason="gate:<reason>"` (`no_source_detections`,
`no_gate_detection`, `stationary_only`, `stale`, `no_frame_ts`); host-side no-captions are
counted as `host:<reason>` (e.g. `host:null_answer`). Counts land in
`summary.caption_gate` and `/api/vision-window/metrics`
(`vision_window_caption_gate_*`); each rejection is logged
(`[WINDOW] caption_rejected ...`). The rejected text never enters the summary.
Found live 2026-10-15: BLIP described a dark empty walkway frame as "A camera
is placed on a wooden bench in a garden" and Orion repeated it as the street
line. Known gap: prop runs live in memory, so for the first
`VISION_STATIONARY_MIN_SEC` (30 min) after a window restart a prop (the patio
skeleton) still counts as moving; a non-null Qwen caption of it would pass then
(the router's `same_scene` skip limits that to one ask per 5 min, and the eval
saw Qwen answer "nothing notable" for the skeleton frames). A caption-only
artifact (`task_type=caption_frame`, no objects) is not a frame: it does not
count toward `item_count`/scene-inventory `frame_count`, and a window holding
only such an artifact does not vote into belief or presence. Tests:
`tests/test_caption_gate.py`.
