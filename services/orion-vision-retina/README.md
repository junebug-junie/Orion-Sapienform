# Orion Vision Retina

Canonical **visual intake** for Orion: sample frames from a camera or folder, persist JPEGs to shared storage, publish `VisionFramePointerPayload` on the Redis bus, and emit `SystemHealthV1` telemetry.

Retina is the **eye** — it does not run YOLO, face detection, motion, captions, or GPU inference. Those live downstream (`orion-vision-host`, `orion-vision-window`, etc.).

## Relationship to `orion-vision-host`

**Retina does not run inside or on top of vision-host.** They are separate services.

| | **orion-vision-retina** (this service) | **orion-vision-host** |
|---|----------------------------------------|------------------------|
| Role | Continuous **capture** + frame evidence | On-demand **GPU inference** (embed, detect, caption) |
| Trigger | Internal loop (`RETINA_FPS`) | Bus RPC: `VisionTaskRequestPayload` on `CHANNEL_VISIONHOST_INTAKE` |
| Output | `orion:vision:frames` (`vision.frame.pointer`) | `orion:vision:artifacts`, task replies |
| Needs GPU | No | Yes |
| “retina” naming | Service name `vision-retina` | Task profile `retina_fast` / `pipeline_retina_fast` (inference pipeline, not this container) |

```text
  [camera / folder / rtsp]
           │
           ▼
   orion-vision-retina  ──publish──►  orion:vision:frames
           │                              (VisionFramePointerPayload)
           │                                      │
           │                                      ├──► orion-vision-edge (detector worker; legacy)
           │                                      └──► (future bridges / window / host subscribers)
           │
           ▼
   /mnt/telemetry/vision/frames/*.jpg   ← shared disk path in image_path

   Separate path (on-demand GPU):
   Client / cortex ──► orion-vision-host  (task_type=retina_fast, image_path=...)
                              └──► artifacts, captions, detections
```

**Integration today:** Retina writes `image_path` on disk and publishes pointers. **Vision-host** runs when something sends it a **task** with an `image_path` (see `services/orion-vision-host/scripts/publish_test_task.py`). Nothing in host currently auto-subscribes to `orion:vision:frames`; wiring “every new frame → host task” would be a follow-up bridge. The bus catalog lists host as a consumer of `orion:vision:frames` for contract alignment.

**Shared dependencies:** Redis (`ORION_BUS_URL`), `orion` Python package, and typically the same frame directory layout as edge (`/mnt/telemetry/vision/frames`).

## What this service does

```text
source (folder | mock | rtsp | webcam)
  → sample at RETINA_FPS
  → save JPEG → FRAME_STORAGE_DIR
  → VisionFramePointerPayload
  → BaseEnvelope (kind=vision.frame.pointer)
  → CHANNEL_RETINA_PUB (default orion:vision:frames)
  → periodic orion:system:health
```

**Non-goals:** detectors, substrate emitters, SQL/RDF/vector writes, vision council/scribe.

## Configuration

Copy env and edit for your mesh:

```bash
cp .env_example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORION_BUS_URL` | `redis://100.92.216.81:6379/0` | Redis pub/sub — shared mesh bus (same as sibling vision services) |
| `RETINA_SOURCE_TYPE` | `folder` | `folder`, `mock`, `rtsp`, `webcam`, `ring` |
| `RETINA_SOURCE` | `/mnt/telemetry/vision/intake` | Folder path, RTSP URL, webcam index, or Ring device name/id |
| `RETINA_SOURCE_PATH` | — | Legacy alias for `RETINA_SOURCE` |
| `RETINA_CAMERA_ID` | `retina-cam-01` | Pointer metadata |
| `RETINA_STREAM_ID` | `retina-stream-01` | Pointer metadata |
| `RETINA_FPS` | `1.0` | Capture rate |
| `FRAME_STORAGE_DIR` | `/mnt/telemetry/vision/frames` | JPEG output |
| `FRAME_RETENTION_SECONDS` | `300` | Cleanup age for this camera/stream's own frames only |
| `RETINA_PUBLISH_STALL_SECONDS` | `120.0` | Publisher watchdog: degraded health, then exit for restart. `0` disables |
| `CHANNEL_RETINA_PUB` | `orion:vision:frames` | Publish channel |
| `RETINA_RTSP_TRANSPORT` | `tcp` | RTSP only: `tcp`, `udp`, `http`. TCP stops UDP packet loss (smeared frames, `RTP: bad cseq`) |
| `RETINA_RTSP_IO_TIMEOUT_SECONDS` | `5.0` | RTSP only: FFmpeg socket `timeout` + OpenCV read timeout, so a quiet stream cannot block `cap.read()` forever |
| `RETINA_RTSP_STALL_SECONDS` | `30.0` | RTSP only: no frame this long → WARNING; if the reader thread is also wedged, abandon it and open a fresh capture. `0` disables. Must be > 3×IO timeout + reconnect |
| `RETINA_RTSP_RECONNECT_QUIET_SECONDS` | `90.0` | RTSP only: no (re)connect until this long after the last camera session (every path, incl. process start). `0` disables |
| `RETINA_RTSP_SESSION_STATE_DIR` | `/var/lib/orion-vision-retina` | RTSP only: persisted last-session time (`rtsp_session_<stream_id>.json`) so restarts/deploys wait too. Compose mounts host `/mnt/telemetry/vision/retina_state` |
| `RETINA_RTSP_HEAL_ENABLED` | `true` | RTSP only: live corruption self-heal |
| `RETINA_RTSP_HEAL_CHECK_EVERY` | `10` | Score every Nth served frame (smear + decoder gray, `app/frame_corruption.py`) |
| `RETINA_RTSP_HEAL_WINDOW_SECONDS` | `120.0` | Corruption must persist this long before a heal |
| `RETINA_RTSP_HEAL_MIN_SHARE` | `0.8` | Share of checks in the window that must be corrupt |

### Ring doorbell source (2026-10-17)

`RETINA_SOURCE_TYPE=ring` (`app/ring_source.py`), deployed as
`orion-vision-retina-doorbell` (no compose profile, stream/camera id
`doorbell`). The doorbell is battery-powered with no Ring Protect plan, so it
has no RTSP, no saved recordings, and cannot take a snapshot while it records
([Ring snapshot limitations](https://github.com/dgreif/ring/wiki/Snapshot-Limitations)).
The only picture source is a live view. So:

- The source listens for Ring push events (`RETINA_RING_EVENT_KINDS`, default
  `motion,ding`) for the device named by `RETINA_SOURCE` (Ring app name or
  device id; empty = first camera on the account).
- Each event opens one WebRTC live view (`aiortc`) for
  `RETINA_RING_BURST_SECONDS` (15). `capture_loop` samples the newest frame
  at `RETINA_FPS`, exactly as for RTSP. Downstream sees the same
  `vision.frame.pointer` stream as the walkway, only in bursts.
- **Never continuous.** One live view at a time, none without an event or a
  self-look (below), and `RETINA_RING_COOLDOWN_SECONDS` (30) from the end of
  one to the next start. Each live view wakes the camera and costs battery.
- **Every push is recorded (2026-10-20).** A ding that arrived while the
  motion-triggered live view was open, or in its cooldown, used to be dropped
  ("ignored: burst running or in cooldown") -- the usual case: walk up =
  motion, then press. Now every push of a configured kind is counted
  (`dings_total`, `motions_total`, `last_ding_ts`), logged with a correlation
  id (`ring ding recorded corr=... live_view=started|extended|already_open|cooldown`),
  and published (`app/ring_events.py`):
  - one `vision.event.bundle` on `orion:vision:events` -> orion-vision-scribe
    -> `vision_events` row, `event_type` `doorbell_ring` (salience 1.0) or
    `doorbell_motion` (0.2), `stream_id` `doorbell`, correlation id = the
    push's. Same rail as the window's `camera_caption` rows.
  - for a ding, one `StreetEventV1` on `orion:vision:street:events`
    (`doorbell_ring`, salience 1.0).
  A ding during an open live view extends it by `RETINA_RING_BURST_SECONDS`
  (at most 4 bursts long in total); a ding in the cooldown opens a new one.
  Burst frames keep their own envelope correlation ids (the frame router
  keys pending tasks by them); the push's id is in each frame's
  `Published frame pointer ... ring_corr=` log line.
- **ding spellings.** ring_doorbell 0.9.14 `listen/eventlistener.py`
  `_get_ding_event` compares `action.lower == PUSH_ACTION_DING.lower()` (the
  method, never called), so a legacy-format ding push carries kind
  `com.ring.push.HANDLE_NEW_DING`. `ring_source.KIND_ALIASES` maps it, and the
  new-format `com.ring.pn.live-event.ding`, to `ding`.
- **Active looking** (spec 2026-10-20 street continuity, section 5). The
  retina polls Redis key `RETINA_RING_SELF_LOOK_KEY`
  (`orion:vision:look:doorbell`, on the bus Redis) every 2 s. The handoff
  reducer (later PR) SETs it, value = handoff id, with a TTL, when it expects
  someone at the door. A new value opens a live view if none is open and the
  cooldown has passed, at most `RETINA_RING_SELF_LOOK_MAX_PER_DAY` (8) per
  UTC day and `RETINA_RING_SELF_LOOK_MIN_SPACING_SECONDS` (300) apart. Each
  is logged (`ring self_look handoff=... outcome=...`) and counted
  (`self_looks_today`, `self_looks_total`, `last_self_look_reason`). The
  live-view call needs no event: `generate_async_webrtc_stream` takes only
  the device and an SDP offer. Opening one with no preceding event is
  UNVERIFIED against Ring's servers. Off: `RETINA_RING_SELF_LOOK_ENABLED=false`.
- Between events an empty read is not a failure (`waiting_for_event`). A live
  view that delivers no frame sets `last_error` (health `degraded`) and counts
  in `ring_bursts_failed`. Health `details` also carry `ring_events_seen`,
  `ring_events_ignored` (pushes that opened or extended no view),
  `ring_bursts_started`, `ring_bursts_extended`, `ring_last_event_kind`,
  `ring_last_event_ts`, `ring_vision_events_published`,
  `ring_street_events_published`, `ring_events_publish_failed`.
  `GET /health` (doorbell: `127.0.0.1:8029`) returns the same payload.
  `last_frame_age_sec` growing between visits is normal for this camera.

No compose profile: the normal `up` (and the post-pull rebuild) builds and
starts it. Until the Ring login it waits and reports `ring not authenticated`
in health. Login, once:

```bash
docker exec -it orion-vision-retina-doorbell python -m app.ring_login
```

It asks for the Ring email, password and 2FA code, and stores only the
refresh token, owner-only, in `/mnt/telemetry/vision/ring_state/` (mounted
into the doorbell container only),
next to the FCM push credentials and a stable hardware id (a changing one
would add a new "authorized device" in the Ring app on every recreate). The
source checks for the token every 60 s, so no restart is needed after login.

The push link is supervised every 60 s: firebase-messaging stops for good
after 3 connection errors, and ring_doorbell refreshes its session only once.
A stopped listener shows as `last_error = "ring push listener stopped"` and is
rebuilt, and the whole connection is rebuilt every 12 h. Each live view is
registered with Ring before it starts, so every exit path (no answer, error,
shutdown) closes it -- an orphaned session keeps a battery camera streaming.

### RTSP stall watchdog (2026-10-15)

The walkway retina once published nothing for ~34 minutes while the container
stayed up and logged nothing; only a restart fixed it. Now:

- RTSP runs over TCP with an FFmpeg socket timeout (`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|timeout;<µs>`,
  set by `RtspFrameSource` only) plus OpenCV `CAP_PROP_OPEN/READ_TIMEOUT_MSEC`. The FFmpeg bundled
  with OpenCV 4.10 (FFmpeg 5.1) honours `timeout`; `stimeout`/`rw_timeout` are ignored by it.
- If no frame arrives for `RETINA_RTSP_STALL_SECONDS` after the camera has connected once, it
  logs one WARNING. If the reader thread is still finishing steps (opens, reads, waits), the
  camera is down or sending garbage and the reader reconnects on its own. If the reader made
  no progress for the whole window it is wedged in FFmpeg: it is abandoned (never joined) and
  a fresh one opens a new capture. At most two abandoned readers may be alive, so camera
  sessions do not pile up; after 4 windows stuck at that cap the process exits so the
  container restart policy brings the eye back.
- `evals/frame_smear.py` checks saved frames for H.264 decode corruption (see its docstring and
  the post-deploy check below).
- System health `details` carry `last_frame_age_sec` and `source_stall_restarts`; a stall logs
  `[RETINA] rtsp stalled: ...` at WARNING and `rtsp frames resumed ...` on recovery.

### Capture task supervision + publisher watchdog (2026-10-17)

The walkway published nothing for ~3 h while health said `ok`. The frame
pruner raced another pruner on the shared frames dir (a file deleted between
`isfile` and `getmtime`), the exception escaped `capture_loop`, the task
ended, and nothing logged it. Now:

- Each writer prunes only its own files: this service only
  `<camera_id>_<stream_id>_<ms>_<hex8>.jpg` (`frame_store.owned_frame_pattern`),
  the edge only `frame_*`/`still_*`. Walkway frames now live
  `FRAME_RETENTION_SECONDS`, not the edge's 60 s. A file vanishing mid-sweep is skipped.
- Cleanup runs inside `capture_loop`'s protective try: housekeeping never stops capture.
- `capture_loop`, the health loop and the clip consumer must run forever. If one
  ends any other way than `stop()`, it logs CRITICAL (with the traceback) and the
  process exits; the container restart policy brings it back (the RTSP quiet
  period keeps the camera clean on the way up).
- Nothing published for `RETINA_PUBLISH_STALL_SECONDS` (outside the RTSP quiet
  wait and a Ring camera between events): health `status` is `degraded`,
  `details.capture_stalled` is true with `capture_stalled_reason`. If
  `capture_loop` has not ticked for that long, or the RTSP reader still delivers
  frames while nothing is published with no save/upload/bus error in that window,
  the process exits for restart. Before exiting it stops the source (bounded),
  so an RTSP reader sends TEARDOWN. A camera that is down, or a failure a
  restart cannot fix (disk full, bus down, upload refused), reads degraded
  with `last_error` and does not exit. The loop clock is monotonic: an NTP
  step or a laptop resume is not a hang.
- Known gaps: a Ring camera whose every publish fails never reads
  `capture_stalled` (it is "between events" once the frame is taken); its
  `last_error` and `source_ok` report it. Files in the shared dir that no live
  writer owns (a retired camera_id/stream_id) are pruned by nobody.

### RTSP reconnect quiet period + corruption self-heal (2026-10-16)

Root cause, proven live: the walkway Reolink keeps a dropped RTSP session
alive. A client that reconnects inside that window gets the stale session
muxed into the new one -- ~6,000 `RTP: PT=60/61: bad cseq` lines a minute and
every frame smeared, for hours (2026-10-15 after a stall + restart, fixed only
by a camera reboot; 2026-10-16 for 10 h after a deploy restart). `docker stop`,
wait 180 s, `docker start` gave 0 bad cseq and 20/20 clean frames with no
camera reboot. Matches https://community.reolink.com/topic/563/.

- **Quiet period.** `RtspSessionGate` blocks every RTSP connect until
  `RETINA_RTSP_RECONNECT_QUIET_SECONDS` have passed since the last moment a
  session may have existed (open attempt, decoded frame, release, abandoned
  wedged reader). One gate in the reader thread covers every path: reader
  reconnect, watchdog replacement, `pause_device` reopen, heal, process start.
  It logs `[RETINA] rtsp quiet period: waiting Ns before connecting` (INFO).
  The stall watchdog ignores the wait.
- **Restart-proof.** The time is persisted (atomic write, every 5 s while
  streaming, immediately on release). After a deploy the new container waits
  out the remainder; if the file is missing it waits one full period. **A
  ~90 s frame gap after every deploy/restart is expected -- that is the fix.**
- **Clean teardown.** Every release goes through `cap.release()`, which sends
  RTSP `TEARDOWN` (verified in the built image, OpenCV 4.10 / FFmpeg 5.1,
  against a fake server that logs methods). `stop()` waits up to 12 s for a
  blocked reader to release it itself; compose `stop_grace_period: 30s` gives
  that time before SIGKILL.
- **Self-heal.** Every `RETINA_RTSP_HEAL_CHECK_EVERY` served frames are scored
  off the event loop with the eval's own math. When at least
  `RETINA_RTSP_HEAL_MIN_SHARE` of the checks over
  `RETINA_RTSP_HEAL_WINDOW_SECONDS` are corrupt: WARNING, drop the session,
  wait the quiet period (doubled per consecutive heal, max 8x), reconnect.
  Never reboots the camera. Dark/flat frames are not corrupt by that math.
  After 3 heals in a row with no clean window between them it stops
  dropping sessions (`source_corruption_heal_suppressed=true`, WARNING) and
  keeps streaming until a clean window resets it -- this caps the blind time
  if the scorer misreads a scene (fog / infrared night are not yet measured).
  Settings refuse an RTSP config with < 5 checks per window (silently dead heal).
- Health `details`: `source_quiet_wait_remaining_sec`,
  `source_last_session_end_ts`, `source_corrupt_frame_rate`,
  `source_corruption_checks`, `source_corrupt_frames`, `source_corruption_heals`,
  `source_corruption_heal_suppressed`.

### Post-deploy check: corrupted frames (2026-10-15)

A container that is up and publishing can still publish broken pictures. Run this after every
retina deploy or camera change. It reads the newest saved frames; it does not open the camera.

```bash
.venv/bin/python services/orion-vision-retina/evals/frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20
```

(Use `--glob 'frame_*.jpg'` for cam0. The host needs OpenCV; the repo `.venv` has it.)

It prints one line per frame and exits 1 if any frame is corrupt (`--max-corrupt-fraction 0.1`
allows 10%). A frame is corrupt when either:

- `smear_fraction` > 0.05: rows copied down the picture, seen as vertical streaks.
- `gray_fraction` > 0.4: macroblocks filled with flat mid-gray, which FFmpeg paints when it
  lost a reference frame.

Clear blue sky and dark night do not count as gray: the check needs neutral colour, one exact
brightness, and no texture. Overcast sky, fog, or infrared (black and white) night mode can be
neutral mid-gray; a synthetic overcast top 60% reads 0.20, under the 0.4 limit. The gray limit
was set from FFmpeg's own concealment output (0.68-0.91), **not from real corrupted camera
frames**: treat a gray-only failure as a strong hint and look at the frame before acting.
Half-written newest frames (no JPEG end marker) are skipped and counted as
`skipped_unreadable`, because a cut-off JPEG decodes with a flat gray bottom.

If it fails, look at the retina logs for `RTP: ... bad cseq` and `error while decoding MB`.
On 2026-10-15 the walkway camera sent streaked frames over TCP with ~2000 `bad cseq` lines per
minute. The cause was the camera's own encoder state, not the network: rebooting the camera
fixed it at once (0 `bad cseq`, 0 decode errors, 131 frames clean). **When this check fails
and the transport is already TCP, reboot the camera first**; restarting the retina container
does not reset the camera's encoder.

Settings live in `app/settings.py` (Pydantic). Schemas: `orion/schemas/vision.py` (`VisionFramePointerPayload`) — already in `orion/schemas/registry.py`; this service does not register new schema types.

## Run locally

```bash
cd services/orion-vision-retina
cp .env_example .env   # set ORION_BUS_URL to your Redis
docker compose build
docker compose up -d
docker logs -f orion-vision-retina
```

Build context is the **repo root** (Dockerfile copies `orion/` + `services/orion-vision-retina/app`).

### Folder smoke test

```bash
mkdir -p /mnt/telemetry/vision/intake /mnt/telemetry/vision/frames
cp /path/to/test.jpg /mnt/telemetry/vision/intake/

export RETINA_SOURCE_TYPE=folder
export RETINA_SOURCE=/mnt/telemetry/vision/intake
export RETINA_FPS=1
```

Tap the bus:

```bash
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:vision:frames
```

Expect envelopes with `kind=vision.frame.pointer` and payload fields `image_path`, `camera_id`, `stream_id`, `width`, `height`, `frame_ts`.

### Optional: drive vision-host from a saved frame

After retina has written a JPEG:

```bash
cd services/orion-vision-host
python scripts/publish_test_task.py --image /mnt/telemetry/vision/frames/<file>.jpg --task retina_fast
```

That exercises **host inference**, not retina itself.

## On-demand video+audio clip capture (AffectGPT)

`POST /capture/clip` (new 2026-08-22, `RETINA_CLIP_ENABLED=false` by
default). Records `RETINA_CLIP_DURATION_SEC` of video (v4l2) and audio
(pulse) **concurrently** via ffmpeg subprocess, uploads both to
`orion-percept-store`, returns their sha256 refs. Nothing is written to
carbon's disk beyond a `TemporaryDirectory` that's always cleaned up --
same privacy discipline as `upload_frame`'s percept-store path. Gated by
`RETINA_CLIP_TOKEN` (header `X-Orion-Retina-Token`) once enabled -- unlike
every other route on this service, a POST here triggers a live recording,
so set this. **The HTTP route's refs alone are still not consumable
end-to-end from a bare curl**: `orion-affectgpt-worker` requires local file
paths, not a percept-store ref directly. The bus RPC path below (via
`orion-juniper-affective-state`) IS the built fetch-by-hash bridge -- see
that service's README.

**Live-verified against real hardware, 2026-08-22** (real device names, real
PulseAudio backend, real timing) -- the disclaimer that used to sit here was
written without access to carbon and has been superseded by an actual run.
`tests/test_vision_retina_clip_capture.py` (fake ffmpeg) still covers the
subprocess construction/error-handling layer in CI, where no camera exists.
To re-verify after a change:

```bash
# on carbon (Docker path), after deploying -- RETINA_HTTP_PORT defaults to
# 8027, not 8022, to avoid colliding with other services on a shared host
# (see docker-compose.yml comment); check your actual .env value.
curl -X POST "http://localhost:${RETINA_HTTP_PORT:-8027}/capture/clip?target_stream_id=${RETINA_STREAM_ID}" \
  -H "X-Orion-Retina-Token: ${RETINA_CLIP_TOKEN}"
# expect: {"ok": true, "video_sha256": "...", "audio_sha256": "...", ...}
# target_stream_id is REQUIRED and must equal THIS instance's own
# RETINA_STREAM_ID (2026-08-22) -- omitted or wrong returns
# {"ok": false, "error_code": "wrong_camera"}, same guarantee as the bus
# RPC path (see "Camera-identity check" below).
# then fetch both back from percept-store and confirm they're a real,
# audible/viewable clip -- sha256 round-tripping correctly does not by
# itself prove the *content* is a valid recording of anything.
```

If `PULSE_SERVER`/the audio socket mount is wrong for carbon's actual setup
(see docker-compose.yml comments), the symptom will be an ffmpeg audio
failure specifically -- video capture (v4l2) doesn't depend on that
mount and should work independently, which narrows down which half broke.

**Live-verified 2026-08-22** (real hardware, no longer just "should work") --
see `docs/operations/carbon-webcam.md`'s "Live-verified on carbon" section
for the actual evidence (sha256s, ffprobe output, byte counts), not just the
claim. The `/dev/video0` device-contention bug (`pause_device()` /
`_device_lock`, see below) was found and fixed from this first real run.

### Bus-reachable twin: `orion:exec:request:RetinaClipCaptureService`

Same capture, triggered over the bus instead of HTTP -- for a caller with no
network path to this node at all. carbon accepts no inbound HTTP whatsoever
(see "Nothing needs to reach carbon inbound" above); Hub's "Affect check"
button reaches it through `orion-juniper-affective-state` (circe), which
does the bus RPC. Request payload is `RetinaClipCaptureRequestPayload`
(empty -- no caller-tunable fields, see that schema's docstring in
`orion/schemas/vision.py`), reply is `RetinaClipCaptureResultPayload` on
`orion:retina:clip:reply:<corr_id>`. Gated the same way as the HTTP route
(`RETINA_CLIP_ENABLED`), minus the HTTP token check -- the bus itself is the
trust boundary here, same as every other channel this service already
publishes/consumes with `ORION_BUS_ENFORCE_CATALOG`. Both entry points share
one implementation (`RetinaService.capture_and_upload_clip()`) and one
`_clip_capture_lock`, so a capture started through either path excludes the
other.

No toggle, no scheduling logic lives here -- that's Hub's job
(`services/orion-hub/scripts/api_routes.py`'s `/api/vision/affect-capture`,
the Vision panel's "Affect check" button). See
`services/orion-juniper-affective-state/README.md` for the rest of the
chain (percept-store fetch, worker hand-off).

**Known, accepted risk (disclosed, not silently accepted, review finding
2026-08-22):** trusting the bus as the sole boundary means ANY bus-connected
service can trigger a live webcam+mic recording of Juniper via this channel
-- there is no per-caller identity or credential check, only "reachable the
bus." This is not unique to this channel (every RPC channel in this codebase
works the same way), but it is qualitatively more sensitive here: the action
IS a live recording, not a data query. `RETINA_CLIP_MIN_INTERVAL_SEC`
(default 30s, `ClipCaptureCooldownError`) bounds the worst case to a known
rate rather than de facto continuous recording -- it does not add real
authentication. Real per-caller auth on this channel (or a signed capability
token) is legitimate follow-up work, not done here; note this in any future
threat-model pass on the bus.

**Camera-identity check (Juniper's explicit instruction, 2026-08-22): "I want
this to only run on my carbon webcam."** This channel has no built-in
per-instance routing at all -- confirmed live 2026-08-22: no second retina
deployment is actually configured anywhere in this repo today (the office/
room camera, "Eye-Ball-1"/`cam0` in Hub's Vision panel, is a completely
separate service, `orion-vision-edge`, sharing zero code path with this
one), but the docs for THIS service already anticipate a future room-camera
retina deployment. Without a check, that future deployment (or any retina
instance with `RETINA_CLIP_ENABLED=true`) would silently race this one for
every request on the shared channel. `RetinaClipCaptureRequestPayload.target_stream_id`
(required, no default -- see `orion/schemas/vision.py`) closes that: every
instance compares the incoming request's `target_stream_id` against its own
`RETINA_STREAM_ID` BEFORE checking `RETINA_CLIP_ENABLED` or anything else,
and refuses (`error_code="wrong_camera"`) on a mismatch. Both entry points
enforce this the same way (`_handle_clip_request` for the bus RPC path,
`capture_clip_endpoint`'s required `?target_stream_id=` query param for
HTTP -- review finding, 2026-08-22: the HTTP route shipped this check
LATER than the bus path did, so the guarantee was fully bypassable via a
plain curl for one commit).

**Precise about what this actually guards against (review finding,
2026-08-22 -- an earlier version of this note overclaimed):** it's a
value-equality check on an unauthenticated channel/route, not
authentication. It closes the *accidental-misconfiguration* failure mode
this section opens with -- a second retina instance quietly left with
`RETINA_CLIP_ENABLED=true` no longer silently races this one for every
request. It does NOT stop a caller who deliberately knows or guesses the
right `target_stream_id` (the documented default, `"carbon"`), and it does
NOT stop two instances that are BOTH misconfigured with the same
`RETINA_STREAM_ID` from racing each other -- that's the same
unauthenticated-channel risk the "Known, accepted risk" paragraph above
already discloses, not a new gap this check claims to close.

**Deploy order:** `target_stream_id` is required with no default on
`RetinaClipCaptureRequestPayload` (`extra="forbid"`), so a version mismatch
between this service and `orion-juniper-affective-state` fails safely in
either direction -- a request missing the field, or an old retina that
doesn't recognize it, both land on `error_code="invalid_request"` (payload
parse failure), never a silent wrong-camera capture. No coordinated
rollout is required; a stale request just gets a clean rejection until both
sides are updated.

## Tests

From repo root (worktree or main):

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_*.py -v
```

## Modules

| File | Role |
|------|------|
| `app/settings.py` | Env contract |
| `app/sources.py` | Frame source adapters |
| `app/ring_source.py` | Ring doorbell: push events -> short WebRTC live view |
| `app/ring_login.py` | One-time Ring login (token to `RETINA_RING_STATE_DIR`) |
| `app/frame_store.py` | Save + retention |
| `app/clip_capture.py` | On-demand video+audio clip capture (AffectGPT, live-verified 2026-08-22) |
| `app/envelopes.py` | `BaseEnvelope` builder |
| `app/health.py` | `SystemHealthV1` helper |
| `app/main.py` | `RetinaService`, FastAPI lifespan |

## Docs

- Pipeline overview: `docs/vision_services.md`
- Bus channel: `orion/bus/channels.yaml` → `orion:vision:frames`
