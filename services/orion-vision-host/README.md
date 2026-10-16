# Orion Vision Host

GPU-backed vision inference service (FastAPI + Redis bus). Tasks arrive on `CHANNEL_VISIONHOST_INTAKE` with payload schema **`VisionTaskRequestPayload`** (`orion/schemas/registry.py`, `orion/bus/channels.yaml`). Replies are published to `reply_to` (channel pattern `orion:vision:reply:*`) as envelope kind **`vision.task.result`** with payload **`VisionTaskResultPayload`**, including **`error_code`**, **`timings`**, and optional **`meta`** (e.g. `warnings`) on failures. Optional **`vision.artifact`** broadcast on `CHANNEL_VISIONHOST_PUB`.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `service.bus` above.

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | **Liveness:** process up; static scheduler limits and bus flag. |
| `GET /ready` | **Readiness:** HTTP **503** when not ready — profiles loaded, bus connected if enabled, at least one CUDA device passes VRAM hard floor, and **warm-on-start** profiles loaded without error. Body lists `degraded_reasons` and `warm_failed_profiles`. |
| `GET /profiles` | Loaded YAML version, enabled profiles, routing. |

## Operator checklist

1. **Caches:** Set `MODEL_CACHE_DIR`, `HF_HOME`, and `TRANSFORMERS_CACHE` to fast, persistent disk (see `.env_example` / deployment env).
2. **GPU visibility:** `CUDA_VISIBLE_DEVICES` and `VISION_DEVICES` must list indices the container can see. NVML (`nvidia-ml-py`) must work for VRAM-aware scheduling.
3. **Bus:** When `ORION_BUS_ENABLED=true`, Redis must be reachable before `/ready` goes green.
4. **VRAM floors:** Tune `VISION_VRAM_RESERVE_MB`, `VISION_VRAM_SOFT_FLOOR_MB`, `VISION_VRAM_HARD_FLOOR_MB` to match co-hosted workloads.
   The admission check is `nvml_free + own_reusable - reserve >= hard_floor`, where `own_reusable` is this process's own PyTorch cache (`memory_reserved - memory_allocated`) on that device. NVML counts that cache as used; without it the service refuses its own tasks once the cache grows (2026-10-15). Memory held by other processes still counts as used. Every refusal carries the breakdown in `error` / `meta.refusal_reason`, and `GET /health` → `scheduler` reports `gpu_hard_floor_refusals_total`, `gpu_hard_floor_refusal_streak`, `last_refusal_reason`, `last_refusal_at`, `last_admit_at`.
5. **Concurrency:** `VISION_MAX_INFLIGHT`, `VISION_MAX_INFLIGHT_PER_GPU`, `VISION_QUEUE_WHEN_BUSY`, `VISION_MAX_QUEUE` — queue full returns `error_code=queue_full` on the bus reply and in structured logs.
6. **Timeouts:** `VISION_TIMEOUT_S` wraps the threaded `VisionRunner.execute` path (wall-clock); logs include `scheduler_total_s`, estimated `queue_wait_est_s`, and `inference_s` when available.
7. **Models:** Override `VISION_VLM_MODEL_ID` per node for your VRAM budget — default (`Salesforce/blip-image-captioning-base`) is sized for a shared/small card (e.g. athena's P4). `model_manager.py`'s `load_vlm_captioner` also supports BLIP2, Qwen2-VL, and Qwen2.5-VL model_ids (selected by substring match — see `.env_example` comment); Qwen2-VL-class models need real headroom (~4-5GB fp16) and route through a chat-template prompt, unlike BLIP's plain image+text call. Enable only profiles you need via `VISION_ENABLED_PROFILES`.
8. **Caption quality:** the caption prompt is chosen by model family (`vlm_family.caption_prompt_for`): chat-template VLMs (Qwen2-VL/Qwen2.5-VL) get the factual `CAPTION_PROMPT`; BLIP/BLIP2 and unrecognized models get no prompt, because they continue text rather than follow it (handed the prompt, BLIP-base echoes it back -- every cam0 caption was blank until 2026-10-08 for this reason). `caption_sanitize` then rejects prompt-echo, too-short, stoplist and repetition garbage. A rejected caption publishes `{"text": "", "confidence": 0.0, "rejected_reason": "<reason>"}` and appends `caption_rejected:{reason}` to task warnings; empty text is never published at confidence 1.0.

## GPU placement on athena (2026-10-15)

The shared vision host now runs on athena's Tesla T10 (16GB, PCI bus id
`00000000:D8:00.0`), next to the qwen lane below. Whisper-tts and
orion-vision-edge moved to the Tesla P4 (`00000000:86:00.0`). Text below
that says the shared instance is "on athena's P4" is history.

The card is pinned by `docker-compose.yml`'s `device_ids` from
`VISION_HOST_GPU_DEVICE_ID` (host PCI bus id, no default). The container
then sees one card: NVML index 0 and torch `cuda:0` are the same card, so
`CUDA_VISIBLE_DEVICES=0` and `VISION_DEVICES=cuda:0` stay container-relative.
The scheduler maps each `cuda:N` through `CUDA_VISIBLE_DEVICES` to its NVML
index (`gpu.torch_to_nvml_index`), so the VRAM gate checks the card the work
runs on even in a container that sees several cards.

Check the pin from the host:

```bash
docker inspect orion-athena-vision-host --format '{{json .HostConfig.DeviceRequests}}'
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv
```

## "Circe" Qwen2-VL lane (`docker-compose.circe-qwen.yml`)

**CORRECTION 2026-09-21: this lane now runs on athena's Tesla T10 (16GB,
full-rate fp16, nvidia-smi index 1), not circe.** It moved off circe's P100
(pulled for this T10 upgrade) and off the temporary V100 it briefly shared
after that. The "circe" name in this section's title, the compose filename,
the `CIRCE_QWEN_` env prefix, and the `circe-vl` bus channel suffix are all
historical labels now, kept deliberately -- see
`docker-compose.circe-qwen.yml`'s own header comment for why renaming a live
bus channel and env prefix is a contract change, out of scope for a location
move. Read every "circe" reference below as historical unless it says
otherwise.

**2026-10-15: walkway captions.** The lane now has a second caller: the frame
router's walkway `caption_dispatch` (`config/vision_frame_router.yaml`). Those
`caption_frame` tasks carry `request.caption_prompt` (honored only on
instruction-tuned VLMs; on BLIP the task returns no caption with
`rejected_reason="caller_prompt_unsupported"` instead of BLIP's own caption,
see `vlm_family.resolve_caption_prompt`)
and `request.caption_null_answers` (an exact "nothing notable" reply becomes
`caption = {text: "", confidence: 0.0, rejected_reason: "null_answer"}`).
Since 2026-10-16 they also carry `request.caption_crop_boxes` (frame-pixel
`[x1, y1, x2, y2]` detector boxes) and `request.caption_crop_pad`: the host
captions the union of the boxes, padded and grown to at least 448 px a side
(`app/caption_crop.py`), instead of the whole 2560x1920 frame shrunk to
`VISION_VLM_QWEN_MAX_PIXELS`. The crop used is returned as `crop_box`, and a
rejected caption logs the model's raw reply
(`caption_frame rejected reason=... raw=...`).
They address the frame by `image_path`, so the compose file mounts
`/mnt/telemetry/vision/frames` read-only, and the lane's artifact broadcast is
opened for `stream_id=walkway` only (`VISION_ARTIFACT_BROADCAST_STREAMS` /
`CIRCE_QWEN_VISION_ARTIFACT_BROADCAST_STREAMS`, `main.artifact_broadcast_allowed`)
so orion-vision-window receives them; dream re-observations carry no
stream_id and still never broadcast. The scheduler now queues up to 2 tasks
(`VISION_QUEUE_WHEN_BUSY=true`, `VISION_MAX_QUEUE=2`) so the two callers wait
for each other instead of being refused.

A second, independent instance of this same service running
`Qwen/Qwen2-VL-2B-Instruct` instead of the shared athena instance's
BLIP-base. Not a replacement for the athena instance above — that one keeps
doing retina/detection/embedding work on athena's P4; this lane is a
second, separate container on athena's other card. This lane exists for
exactly one job: re-observing a generated image for `orion-thought`'s
reverie visual chain (`services/orion-thought/app/visual_chain.py`), which
needs a real, non-degenerate caption that BLIP-base's quality ceiling
cannot reliably produce (live-evidenced 3/3 ticks on 2026-08-25/26 — every
real GPU call rejected by `sanitize_caption` as too-short/empty) and that
athena's P4 has no VRAM headroom to fix in place (2.4GB free measured live;
Qwen2-VL-2B needs ~4-5GB fp16) -- the T10 does, with room to spare
alongside whisper-tts, which also moved onto this card the same day.

**Why a second instance, not just a bigger model on athena:** checked and
ruled out — see the paragraph above. **Why its own channel, not the shared
one:** two vision-host instances racing the same shared
`orion:exec:request:VisionHostService` channel already caused a real
incident (PR #1859/#1860, 2m13s of dropped presence updates) — every
dedicated instance gets its own isolated channel, no exceptions
(`orion/bus/channels.yaml`'s `orion:exec:request:VisionHostService:*`
convention).

**No shared filesystem assumption** (`/mnt/telemetry` is local ext4, not
NFS-exported) — this lane never reads a frame by path, by design, since it
used to run on a different host than percept-store. The caller
(`orion-thought`) uploads the generated image's bytes to
`orion-percept-store` first and hands this lane a `percept_sha256`; this
service's own `runner.py::_load_image_from_percept_store` is what fetches
those bytes back, server-side, to run inference on. Now that this lane runs
on athena, colocated with percept-store on the same docker network,
`VISION_PERCEPT_STORE_URL` (set via `CIRCE_QWEN_VISION_PERCEPT_STORE_URL`
— see below) uses the docker-internal service name
(`http://orion-athena-percept-store:8000/percepts`) -- when this container
ran on circe, a different host, it had to use athena's real tailscale IP
instead (`http://100.92.216.81:8021/percepts`); docker-internal names only
resolve on their own host's docker network. Keep that distinction in mind
if this lane ever moves to a third host.

**Every lane-specific override reads a `CIRCE_QWEN_`-prefixed key, never
the bare shared name — this is load-bearing, not a style choice, no
exceptions.** Real regression, 2026-08-26, caught in two rounds. Round 1
(live-caught): this file read `VISION_PERCEPT_STORE_URL`, the three
`VISION_VRAM_*` floors, `VISION_TIMEOUT_S`, and `ORION_BUS_ENFORCE_CATALOG`
as bare `${VAR:-<circe-correct-default>}` hooks. The shared athena
instance's own `.env_example` sets every one of those same bare names to a
*different* value for athena's own purposes. The moment
`services/orion-vision-host/.env` exists at all — which this file's own
bring-up instructions below require passing as a second `--env-file` —
the athena-only values silently won over this lane's real defaults, no
error. Percept-store broke functionally (re-observation tick came back
`image_not_found`, "Temporary failure in name resolution" on the
athena-only hostname, live-caught in `orion-thought`'s own tick log); the
VRAM floors and catalog-enforcement flag collided the same way but stayed
silently harmless (the P100 has headroom to spare; no catalog rejections
observed). Round 2 (code-review finding on the round-1 fix itself, same
session): `LOG_LEVEL`, `ORION_BUS_ENABLED`, `HEARTBEAT_INTERVAL_SEC`, and
`TORCH_CUDA_ALLOC_CONF` were still bare, same collision mechanism, silently
harmless only because the values happened to coincide — fixed for the same
reason as round 1, not left as a known-benign exception now that the
principle above is stated as a blanket "no exceptions."

**Bring up (from a worktree ON ATHENA, never the shared checkout):**

Not via `scripts/safe_docker_build.sh` -- that wrapper hardcodes
`-f services/$SERVICE/docker-compose.yml` and has no hook for a second
compose file for the same service; passing `-f` yourself would merge both
files and risk touching the main vision-host service too. Run docker
compose directly:

```bash
# Confirm the GPU is actually free RIGHT NOW -- do not trust an earlier
# session's snapshot, this host's GPU assignments have moved before:
nvidia-smi --query-gpu=index,name,memory.free --format=csv

docker compose \
  --env-file .env \
  --env-file services/orion-vision-host/.env \
  -f services/orion-vision-host/docker-compose.circe-qwen.yml \
  up -d --build

curl -fsS http://localhost:${CIRCE_QWEN_HOST_PORT:-6602}/health
curl -fsS http://localhost:${CIRCE_QWEN_HOST_PORT:-6602}/ready
```

Every `CIRCE_QWEN_*` key has a safe default baked into the compose file
itself, so the second `--env-file` above is optional for a first bring-up
-- but include it anyway (AGENTS.md section 8's dual-env-file convention):
an override made through the normal `sync_local_env_from_example.py`
workflow lands in `services/orion-vision-host/.env`, and a bring-up that
only passes the root `.env` would silently ignore it (review finding).

Env keys: see `.env_example`'s "Circe Qwen2-VL lane" section.

## Caption sanitizer

`retina_fast` caption path (`_run_caption_frame`):

- Prompt: list visible objects/people only — no activity guesses.
- Post-decode: `sanitize_caption()` rejects `prompt_echo`, `too_short`, and high stoplist ratio (YouTube/google/video slop).
- Rejected captions are cleared; warnings surface in artifact meta.

Env: `VISION_VLM_MODEL_ID`, `VISION_VLM_TEMPERATURE` (default `0.2`).

## Identity hypothesis (`identity_face`)

`docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-situated-observation-design.md`
section 4. Real face detection + embedding (MTCNN + InceptionResnetV1,
VGGFace2-pretrained, `facenet-pytorch`) matched against **one enrolled
subject** -- not a growing gallery, not a stranger tracker.

**Non-negotiables, enforced in code, not just here:**

- **One enrolled subject. Gallery does not grow at runtime.**
  `app/identity_gallery.py::save_gallery_embedding` is only ever called
  from `scripts/enroll_identity_face.py` (a human-run CLI) --
  `tests/test_identity_gallery_never_grows_at_runtime.py` structurally
  pins that no file under `app/` references it.
- **Non-matches are never stored.** A query embedding is compared in
  memory and discarded; it is never written to disk or returned to the
  caller, matched or not.
- **`unsure` is a real, reachable third state.** `match_embedding` returns
  `{"subject", "similarity", "state"}` with `state` one of
  `probable` / `possible` / `unsure` -- never a binary match/no-match. A
  `subject` other than `"unknown"` is returned only for `probable`/`possible`.
- **Never reaches the general artifact broadcast.** `CHANNEL_VISIONHOST_PUB`
  (`orion:vision:artifacts`) is consumed by `orion-vision-window`,
  `orion-vision-council`, and `orion-substrate-runtime` (`orion/bus/
  channels.yaml`) -- none identity-aware or retention-gated. `app/main.py`'s
  `should_broadcast_artifact()` excludes `identity_face` from both real
  broadcast call sites (found live, 2026-08-26: the bus-first path and the
  `/v1/vision/task` HTTP endpoint each had their own, and the second one
  was still unguarded after the first fix landed --
  `test_every_publish_artifact_broadcast_call_site_is_guarded` pins both
  now). Only the direct RPC/HTTP reply to whoever explicitly requested
  `task_type=identity_face` carries the real result.

**Enrollment** (real photos required -- ships with zero enrolled by
default; `scripts/` is copied into the image, confirmed live 2026-08-26
after an earlier version of this Dockerfile omitted it entirely). The
`/mnt/telemetry/orion-vision-host` host mount is not reliably writable by
an operator's own shell user (confirmed live: `athena`'s own user got
`Permission denied` there) -- `docker cp` into the container's own
filesystem is the tested, working path:

```bash
# Locally, against a dev checkout with facenet-pytorch installed:
cd services/orion-vision-host
python3 scripts/enroll_identity_face.py --subject juniper photo1.jpg photo2.jpg photo3.jpg

# Against the real running container -- docker cp, not the host mount:
docker exec orion-athena-vision-host mkdir -p /tmp/enrollment_photos
docker cp photo1.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker cp photo2.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker cp photo3.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker exec orion-athena-vision-host python3 scripts/enroll_identity_face.py \
  --subject juniper \
  /tmp/enrollment_photos/photo1.jpg \
  /tmp/enrollment_photos/photo2.jpg \
  /tmp/enrollment_photos/photo3.jpg
docker exec orion-athena-vision-host rm -rf /tmp/enrollment_photos  # cleanup
```

Writes one JSON file (mean embedding across all usable photos) to
`IDENTITY_GALLERY_DIR` (default `/mnt/telemetry/orion-vision-host/identity_gallery`,
the existing bind mount -- no new volume). Re-running with the same
`--subject` overwrites the entry (re-enrollment, not accumulation). The
script only ever reads its source images, never copies or retains them
itself -- the `rm -rf` above is on the operator, matching the real
enrollment this repo shipped with (see below).

**Where the real enrollment's photos came from:** no photos of Juniper
were sourced or requested directly. `orion-juniper-affective-state`
already runs a real, already-approved, already-live capture path (Hub's
own "Check now" button) -- `POST /v1/juniper/affect/capture_and_assess`
triggers a real ~8s clip from Juniper's own carbon webcam and returns
`capture.video_sha256`. That video is fetchable from `orion-percept-store`
(`GET {PERCEPT_STORE_BASE_URL}/{sha256}`, hash re-verified on receipt) the
same way `orion-juniper-affective-state`'s own `_fetch_percept` does.
`ffmpeg -i clip.mp4 -vf "select='not(mod(n\,40))'" -vsync vfr frame_%02d.jpg`
extracted 6 evenly-spaced frames, enrolled via `docker cp` + the script
above, then deleted from the container (`rm -rf /tmp/enrollment_photos`).
No new capture infrastructure -- this reuses a capability that already
exists and is already consented to for a different purpose.

**Unenrolled behavior:** `task_type=identity_face` still runs face
detection; every candidate comes back `{"subject": "unknown", "similarity":
null, "state": "unsure", "reason": "not_enrolled"}` and the artifact's
`gallery_enrolled` flag is `false` -- not an error.

**Thresholds** (`config/vision_profiles.yaml`'s `identity_face.params`):
`match_threshold` (below = `unsure`) and `probable_threshold` (at/above =
`probable`; between the two = `possible`). Live-validated 2026-08-26 against
real enrolled photos (a real `orion-juniper-affective-state` capture,
6 frames enrolled, a held-out 7th frame from the same clip queried):
`similarity=0.9585, state="probable"` -- comfortably clears
`probable_threshold` (0.55) with real margin. Negative control (the empty
room's own ceiling-camera frame) correctly returned zero candidates
(`no_face_detected`), not a false positive. Both directions checked, not
just the positive case.

**Deviation from the design doc's literal prose, noted honestly:** the doc
describes running face detection on the person crop GroundingDINO already
produces. `_run_identity_face` runs MTCNN directly on the full frame
instead -- `runner.py`'s pipeline steps do not currently pass one step's
artifacts into the next step's request (`_run_pipeline` hands every step
the same original request dict), and building that plumbing is separate,
larger, riskier work than this patch's scope. MTCNN already does its own
face localization on a full image without needing a person crop first, so
this is a real simplification, not a silently dropped requirement.

**Wildcard reply-channel fan-out -- narrowed 2026-08-26, not fully closed:**
`orion-vision-frame-router`'s `dispatcher.py::_handle_reply_envelope_inner`
was found to fully Pydantic-deserialize every `identity_face` reply --
including the real subject/similarity/state hypothesis -- before checking
whether the correlation id belonged to a request it made, then discard it.
Fixed by reordering: `RouterState.clear_pending()` only needs
`env.correlation_id` (already on the envelope, no payload parsing
required), so it now runs before `VisionTaskResultPayload.model_validate`
instead of after -- typed deserialization and trigger-label extraction now
only happen for corr_ids frame-router actually dispatched.

**What this does NOT close, verified live:** `main.py`'s `_reply_loop`
calls `self.bus.codec.decode(data)` -- a full JSON decode into
`BaseEnvelope.payload: Dict[str, Any]` -- for every message on the
wildcard pattern *before* `dispatcher.handle_reply_envelope` is ever
invoked. That raw dict, containing the same subject/similarity/state
fields as plain values, is already resident in process memory by the time
this fix's ownership check runs; the fix narrows *further* processing
(the typed model, label extraction) for unowned replies, it does not
prevent the bus client itself from decoding an unowned reply's payload in
the first place. Closing that fully would mean either a non-wildcarded
reply lane for `identity_face` or changing what the bus codec decodes
before per-message dispatch -- both out of scope here. `orion/bus/
channels.yaml` documents `orion:vision:reply:*` as an intentional
multi-subscriber fan-out channel (`consumer_services: ["*"]`), and nothing
bus-level enforces "check ownership before parsing payload" as a property
of that channel -- a future wildcard subscriber implementing its own
reply handler could reintroduce the narrower (typed-deserialize) version
of this exposure without violating any existing gate. Pre-existing
property of every task_type's reply channel, not identity_face-specific --
addressed here because identity_face is the first task_type where the
content is privacy-sensitive. See
`services/orion-vision-frame-router/app/dispatcher.py` and its
`test_reply_for_unowned_correlation_id_never_deserializes_payload` /
`test_owned_reply_with_malformed_payload_is_not_counted_as_a_valid_reply`
tests.

**Not built here:** wiring the resulting hypothesis into
`orion-vision-council`'s evidence-grounding context as the design doc's
section 4 describes ("passed to the gateway call as grounding context,
exactly like hard labels") -- a real next integration step, in a
different service. Also not built: a `vision_events` table-level retention
policy (design doc section 6.5: "identity-bearing rows are the most
sensitive this system will hold. Ships WITH section 4, not after"). The
immediate leak path that requirement was guarding against -- identity data
reaching every consumer of the general artifact broadcast unfiltered -- is
closed; what remains open is retention for identity data that reaches
`vision_events` through some *future* real integration (e.g. the
council-grounding wiring above), which does not exist yet either. Flagged,
not silently skipped.

### VLM model families

`app/vlm_family.py` is the single source of truth both `model_manager.py`
(which transformers class to instantiate) and `runner.py` (how to build the
prompt / decode the reply) read for "which family does this model_id
belong to":

| Family | `model_id` match | Prompt / decode |
|--------|------------------|------------------|
| BLIP2 | contains `blip2` | `processor(images=, text=)`, full-sequence decode; captions sent with no text |
| BLIP | contains `blip` (not `blip2`) | same as above |
| Qwen2-VL | contains `qwen2-vl`/`qwen2_vl` | `apply_chat_template` + image, decode sliced by input token length |
| Qwen2.5-VL | contains `qwen2.5-vl`/`qwen2_5_vl` | same chat-template path as Qwen2-VL |
| anything else | — | generic `AutoModelForVision2Seq`, BLIP-style call shape; captions sent with no text |

The chat-template path exists because a chat-tuned VLM echoes its whole
templated prompt back through `generate()` — decoding the full sequence (as
BLIP correctly can) would hand the caller the prompt glued onto the real
answer. `_generate_vlm_text` slices by real input token length instead of
string-matching a prefix off the decoded text, which cannot reliably strip
chat special tokens.

## Tiled detection (walkway, 2026-10-15)

The GroundingDINO processor resizes every image to 800 px on the short side
(1333 max long side). The walkway frame is 2560x1920, so a person on the far
sidewalk (~50 px tall at full resolution) reaches the model at ~21 px and is
mostly missed. When a detect request sets `detect_tiles: true` (only the
router's walkway `tiled_detect` block does, at most one frame per
`min_seconds_between`), `_run_detect_grounding_dino` also runs the detector on
overlapping crops of the full-resolution frame (`app/tiling.py`):

1. Tile boxes are shifted into full-frame pixels. A box touching an interior
   tile edge is "cut".
2. A cut box mostly inside a same-label whole box is dropped (fragment).
3. Cut boxes of one label from two tiles that overlap are unioned (an object
   split by a seam).
4. Per-label NMS over whole-frame + tile boxes, then `max_detections`.

Everything downstream (zones, crop embeddings, faces, thumbnails, stationary
rule, crop observations) sees ordinary full-frame boxes. The artifact gains a
`detect_tiles` block (`tiles`, `tiles_s`, `full_frame_s`, raw/merged counts),
and each tiled task logs one `[DETECT_TILES]` line with the per-pass timing.
Frames whose short side is below `tile_min_short_side_px` are never tiled.

Knobs (`retina_detect_open_vocab` params in `config/vision_profiles.yaml`;
a dict `detect_tiles: {grid: [3, 2], ...}` in the request overrides them):
`tile_grid`, `tile_overlap`, `tile_region` (normalized sub-rectangle to tile),
`tile_include_full_frame`, `tile_batch` (one batched forward instead of
sequential passes; off: no gain on CPU, and ~4x activation memory on the P4),
`tile_min_short_side_px`, `tile_contain_frac`, `tile_seam_merge_frac`.

Eval on real frames (frames are never committed -- they show neighbours):
`evals/walkway_tiled_detection_eval.py <frames_dir> --out <dir> [--layout
name:cols,rows,overlap[,x0,y0,x1,y1]] [--draw]`; see its docstring for the
read-only docker command. Results: `docs/superpowers/pr-reports/
2026-10-15-walkway-tiled-detection-pr.md`.

## Crop embeddings (walkway individuals)

Deploy steps, rebuild set, and the rule that editing `config/vision_zones.yaml`
means rebuilding vision-host, vision-council AND sql-writer:
`services/orion-sql-writer/README.md` "Deploy checklist (walkway camera)".
Any box that merely overlaps a no-embed zone gets no embedding and no thumbnail.

Spec: `docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`
idea 1. When a detect request sets `want_crop_embeddings: true` (only the
router's `walkway` stream policy does), `_run_detect_grounding_dino` zones
every box whose label is in `crop_embedding_labels` and embeds the allowed
crops in one batched pass on the already-warm `embed_image` SigLIP tower
(`app/crop_embeddings.py`). Each object gains `zone`, `embedding_ref`, and an
L2-normalized inline `embedding`; the detect artifact also carries
`frame_width`/`frame_height` and a `crop_embeddings` counter block.

- The camera name comes from task meta `stream_id` (set by the frame router),
  copied into the request by `VisionRunner._request_with_stream_id`.
- **No-embed zones (generic):** a box in a no-embed zone (`config/vision_zones.yaml`,
  `embed: false`) is never cropped; it keeps its zone and no vector. Since
  2026-10-15 no zone uses this -- Juniper lifted the patio rule, so patio
  boxes get vectors, thumbnails and faces like the rest of the walkway.
- **Faces (walkway, 2026-10-15):** when the request also sets
  `want_crop_faces: true` (router `walkway` policy only), each embedded crop in
  `crop_face_labels` (default `person`) is run through the `identity_face`
  profile's already-loaded MTCNN + InceptionResnetV1 (`crop_faces`). The most
  confident face at or above `crop_face_min_prob` (0.95) and
  `crop_face_min_side_px` (40) adds `face_embedding` (512-d, unit),
  `face_score` and `face_thumb_ref` (own rate limiter, triggered tier only) to
  the object. A face failure costs the face, never the crop vector. cam0/carbon
  have no zones on file, never reach the embed list, and so never get a face
  here; Juniper's enrolled gallery is not read or written by this path. If the
  `identity_face` profile is disabled, crops get no face.
- **Fails closed:** if the zones file is missing or unreadable, no crop is
  embedded at all. The Dockerfile copies `config/vision_zones.yaml` to
  `/app/config/`; override with `VISION_ZONES_PATH` only for local runs.
- Knobs live in the `retina_detect_open_vocab` profile params
  (`config/vision_profiles.yaml`): `crop_embed_profile`,
  `crop_embedding_labels`, `crop_embedding_max_per_frame`, `crop_min_side_px`.
- **Thumbnails (ask card):** each EMBEDDED crop -- the exact crop list the
  embedder receives, so a no-embed-zone box cannot get one -- is also saved as
  a <=160 px JPEG (quality 70) named by its sha256 under
  `VISION_CROP_THUMB_DIR` (default `/mnt/telemetry/orion-vision-host/crop_thumbs`),
  and the object gains `thumb_ref = "thumb:<sha256>"`. The host prunes files
  not rewritten for `VISION_CROP_THUMB_RETENTION_DAYS` (10, longer than an
  ask's 7-day life). orion-hub mounts the directory read-only and serves one
  thumbnail by hash at `/api/vision/crop-thumbs/<sha256>`. Empty dir disables.
- Test: `tests/test_crop_embeddings_patio.py`, `tests/test_crop_thumbnails.py`,
  `tests/test_crop_faces.py` (embedder and face models mocked).

### OSNet shadow body vector (2026-10-16)

Juniper approved storing a person re-identification vector next to the SigLIP
body vector for about a week, so the two can be scored against each other
(`services/orion-sql-writer/evals/run_osnet_vs_siglip_eval.py`). **Shadow
only:** the individuals matcher in orion-sql-writer never reads it, so
nothing about who a crop is assigned to changes.

- **What:** every embedded crop whose label is `person` also gets
  `osnet_embedding` (512-d, unit) and `osnet_model_id`
  (`osnet_ain_x1_0:ms_d_c:<sha12>`), from the same crop pixels SigLIP saw.
  Torchreid's test transform: resize to 256 x 128, ImageNet mean/std, fp32.
- **Model:** OSNet-AIN x1.0, Torchreid model zoo multi-source checkpoint
  (MSMT17 + DukeMTMC-reID + CUHK03), classifier head stripped, as published at
  a pinned revision of `huggingface.co/LibreYOLO/LibreReID-osnet` (MIT). The
  network is vendored in `app/osnet_ain.py` (Torchreid source, MIT); the
  `torchreid` 0.2.5 sdist on PyPI is a third-party repackaging that needs a
  Cython build and pulls opencv-python/matplotlib/tensorboard.
- **Weights:** downloaded at image build to `/app/models/osnet/osnet_ain_x1_0.pt`
  (`VISION_OSNET_WEIGHTS_PATH`), sha256-checked by the Dockerfile and again by
  `app/osnet_shadow.py`, loaded with `weights_only=True`. Never in git.
- **Kill switch:** `VISION_OSNET_SHADOW_ENABLED` (default `true`). Loaded at
  warm-up only on a host with `embed_image` enabled (not the circe Qwen lane).
- **Failure is contained:** a missing or wrong weights file, a load error or
  an inference error costs the OSNet vector only. The SigLIP vector, faces and
  detections are unchanged. The load is retried at most every 10 minutes.
- **Measured on the T10 (2026-10-16, built image, throwaway container):**
  weights 8.4 MB on the GPU; peak allocated 188 MB, reserved 292 MB at a batch
  of 8 (nvidia-smi delta 310 MiB including allocator slack); 15.5 ms for one
  crop, 30 ms for a batch of 8 (3.8 ms per crop), while the live captioner
  shared the card.
- **Which body model actually runs:** the `embed_image` profile asks for
  SigLIP2, which does not load under transformers 4.49.0, so the loader falls
  back to SigLIP v1. That fallback is now visible: an ERROR log line at load,
  `GET /health` -> `models.body_embedder` (requested vs loaded, the error), a
  `body_embedder_fallback:<id>` entry in `/ready`'s `degraded_reasons` (does
  not change `ready`), and `embedding_model_id` on every embedded crop.
  `/health` -> `models.osnet_shadow` shows OSNet's state, VRAM, vector count
  and last error; `osnet_shadow_failed` joins `degraded_reasons` when it
  cannot load.
- **Deploy last:** migration -> orion-sql-writer -> orion-vision-window ->
  orion-vision-host. An older window drops the new fields silently (its
  `VisionObject` ignores unknown keys). The kill switch does not remove the
  order: `embedding_model_id` is sent on every embedded crop either way.
- Test: `tests/test_osnet_shadow.py` (the real-weights test runs inside the
  built image; it skips where the weights are absent).

## Observability (logs-first)

Each finished task emits one structured line prefix `[VISION_TASK]` with JSON containing at least: `correlation_id`, `task_type`, `ok`, `device`, `error`, `error_code`, `queue_depth_at_submit`, `scheduler_total_s`, `inference_s`, `queue_wait_est_s`.

**Dashboard hints:** aggregate `error_code`, p95 `scheduler_total_s` / `inference_s` by `task_type`, and `queue_depth_at_submit` at submit time.

## Config drift: `adaptive_degrade`

The repository root `config/vision_profiles.yaml` may describe `runtime.adaptive_degrade`. **That block is not implemented in `orion-vision-host`** — scheduling uses env-backed floors and `VisionScheduler` only. Do not assume resolution drops or profile disables happen automatically.

## Smoke scripts

- `scripts/publish_test_task.py` — publish a task over the bus.
- `scripts/tap_artifacts.py` — subscribe to artifact channel.

## Tests

```bash
cd services/orion-vision-host
PYTHONPATH=. python3 -m pytest tests/ -q --tb=short
```

Scheduler tests do not require CUDA or model weights.
