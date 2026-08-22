# Running Orion's eye on carbon

carbon's built-in webcam, feeding the normal vision pipeline.

**You have to run this.** Tailnet policy refuses SSH as `athena`, so it can't be
deployed for you — same block as circe.

## What made it possible

Capture used to publish a **local file path**, and the frame router enforces
`require_image_path_exists`. carbon shares no filesystem with athena, so it
physically could not feed the pipeline. Frames now carry a `sha256`: carbon
encodes in memory, POSTs to `orion-percept-store` on athena, and publishes the
content address. Everything downstream — router, host, window, council, census,
blindness alerting — then treats carbon as a normal camera.

## You do not need Docker

Retina is a plain uvicorn app: the Dockerfile only pip-installs
`requirements.txt`, copies `app/` and `orion/`, and runs uvicorn. Nothing about
it is container-specific, the deps are light (no torch, no CUDA), and a plain
process opens `/dev/video0` directly instead of needing device passthrough.

**Nothing needs to reach carbon inbound.** Capture runs inside the app's
`lifespan`, so the HTTP surface is only a health shell. carbon needs *outbound*
to athena and nothing else:

- `100.92.216.81:6379` — the bus
- `100.92.216.81:8021` — the percept store

## Setup (venv — recommended)

```bash
git clone <repo> ~/Orion-Sapienform && cd ~/Orion-Sapienform
python3 -m venv .venv && . .venv/bin/activate
pip install -r services/orion-vision-retina/requirements.txt

cp services/orion-vision-retina/.env_example services/orion-vision-retina/.env
```

Edit `services/orion-vision-retina/.env`:

```ini
ORION_BUS_URL=redis://100.92.216.81:6379/0

RETINA_SOURCE_TYPE=webcam
RETINA_SOURCE=/dev/video0            # ls /dev/video* to confirm
RETINA_CAMERA_ID=carbon-webcam
RETINA_STREAM_ID=carbon              # keeps it separate from cam0 everywhere

# The bit that makes this work from a machine with no shared disk.
RETINA_FRAME_MODE=percept_store
RETINA_PERCEPT_STORE_URL=http://100.92.216.81:8021/percepts

# A laptop webcam is a close, continuous view of one person. Far slower than a
# room camera on purpose: this is a presence sensor, not a scene sensor.
RETINA_FPS=0.2                       # one frame per 5s
JPEG_QUALITY=75
```

Run it:

```bash
cd ~/Orion-Sapienform
set -a && . services/orion-vision-retina/.env && set +a
PYTHONPATH=.:services/orion-vision-retina \
  .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8022
```

`127.0.0.1` on purpose — nothing needs to reach it.

### Keep it running across reboots

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/orion-retina.service <<'UNIT'
[Unit]
Description=Orion vision retina (carbon webcam)
After=network-online.target

[Service]
WorkingDirectory=%h/Orion-Sapienform
EnvironmentFile=%h/Orion-Sapienform/services/orion-vision-retina/.env
Environment=PYTHONPATH=%h/Orion-Sapienform:%h/Orion-Sapienform/services/orion-vision-retina
ExecStart=%h/Orion-Sapienform/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8022
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now orion-retina
journalctl --user -u orion-retina -f
```

`Restart=on-failure` covers a sleep/resume that drops the bus connection.

## Setup (Docker — only if you prefer it)

Same `.env`, plus `RETINA_VIDEO_DEVICE=/dev/video0` for the device passthrough
the compose file wires up:

```bash
docker compose --env-file .env --env-file services/orion-vision-retina/.env \
  -f services/orion-vision-retina/docker-compose.yml up -d --build
```

## Verify from athena

```bash
curl -s localhost:8021/stats          # count climbing

psql -h localhost -p 55432 -U postgres -d conjourney -c \
 "SELECT stream_id, count(*) FROM vision_scene_inventory
   WHERE observed_at > now()-interval '10 min' GROUP BY 1;"
```

`carbon` should appear alongside `cam0`.

## Turning it off

Stop the process. That is the entire off switch, it lives on the machine being
watched, and it needs nothing from athena or Hub:

```bash
systemctl --user stop orion-retina     # or Ctrl-C, or: docker stop orion-vision-retina
```

## On-demand affect clip capture (AffectGPT, 2026-08-22)

Port 8022 below matches carbon's own venv/systemd port above -- fine on a
dedicated laptop. **If deploying the Docker path on a shared host instead**
(e.g. athena, which can also run retina for the room camera), check
`.env_example`'s `RETINA_HTTP_PORT` comment first -- confirmed live
2026-08-22 that `orion-athena-cortex-gateway` already owns 8022 there;
the Docker compose default is 8027 for exactly that reason.

A second, separate HTTP surface on this same service:
`POST http://127.0.0.1:8022/capture/clip` records a few seconds of video
(v4l2) and audio (pulse/PipeWire mic) **concurrently**, uploads both to
`orion-percept-store` (now accepts `audio/wav`/`video/mp4`, not just
images), and returns their sha256 refs. See
`services/orion-vision-retina/README.md` for the full contract.

**Live-verified on carbon, 2026-08-22** -- the actual evidence, not just the
claim (CLAUDE.md's own "runtime truth beats config truth" bar, review
finding 2026-08-22: an earlier version of this note asserted "confirmed"
with no artifact attached):

```text
video_sha256=051a73480d23e38076188bf363b2a62dfe2cbff8f0c328810a25541db1d5e991
  video_bytes=115757  ffprobe: h264, 640x480, nb_frames=94, duration=7.966667s
audio_sha256=d7c4b1e48d81549b0eeca5b748c6833f30aad9027818bcabbc18506b45ec446b
  audio_bytes=256412  ffprobe: pcm_s16le, 16000Hz, mono, duration=8.010438s
```

Both blobs independently re-fetched from percept-store with `curl` and
re-hashed with `sha256sum` -- exact match against the sha256 the capture
response reported, both directions (chain-of-custody, not just "the API said
ok"). First real run also surfaced and fixed a genuine `/dev/video0`
contention bug against the continuous presence loop (`pause_device()`, see
the vision-retina README) -- not something a mock or fixture could have
caught. A same-session dark/quiet capture was checked frame-by-frame
(5 frames spanning the full clip, `cv2.imread(..., IMREAD_GRAYSCALE).mean()`)
and confirmed a real room condition (camera covered), not a pipeline bug.

**Bus-reachable twin, same day**: `orion:exec:request:RetinaClipCaptureService`
(reply on `orion:retina:clip:reply:<corr_id>`) does the identical capture
over the bus instead of HTTP, for a caller with no network path to carbon at
all -- which is every caller except one on this exact tailnet segment, since
"nothing needs to reach carbon inbound" (above) is the whole security
posture. This is how Hub's "Affect check" button reaches carbon: Hub calls
`orion-juniper-affective-state` (circe), which bus-RPCs retina, fetches the
resulting blobs from percept-store, and hands them to `orion-affectgpt-worker`
-- see that service's README for the full chain. **This IS now consumable
end-to-end** via that path; the HTTP route's sha256 refs alone are still not
(no local worker fetch-by-hash from a bare curl).

This is a materially different, more sensitive capability than the presence
frames above (see the amended "No facial affect" bullet below) — off by
default (`RETINA_CLIP_ENABLED=false`) and gated by a shared-secret token
(`RETINA_CLIP_TOKEN` / header `X-Orion-Retina-Token`) once enabled, because
unlike everything else on this service, a POST here triggers a live
recording. Extra setup beyond the base install above:

```bash
sudo apt install ffmpeg   # or your distro's equivalent -- not pulled in by
                           # requirements.txt, which is Python-only
pactl list sources short   # confirm the mic source name (RETINA_CLIP_AUDIO_INPUT)
ls /dev/video*             # confirm the camera device (RETINA_CLIP_VIDEO_DEVICE)
```

Add to `services/orion-vision-retina/.env` (see `.env_example` for the
full set): `RETINA_CLIP_ENABLED=true`, `RETINA_CLIP_TOKEN=<pick one>`, and
adjust `RETINA_CLIP_VIDEO_DEVICE`/`RETINA_CLIP_AUDIO_INPUT` if the defaults
above don't match. Verify:

```bash
curl -X POST "http://127.0.0.1:8022/capture/clip?target_stream_id=carbon" \
  -H "X-Orion-Retina-Token: <same value>"
# expect: {"ok": true, "video_sha256": "...", "audio_sha256": "...", ...}
# target_stream_id is REQUIRED and must equal this instance's own
# RETINA_STREAM_ID (2026-08-22, camera-identity guarantee) -- omitting it
# or getting it wrong returns {"ok": false, "error_code": "wrong_camera"}.
```

## What this deliberately does not do

- **Nothing is written to carbon's disk.** The upload path uses `imencode`,
  never `imwrite`. If the store is unreachable the frame is dropped and the next
  one attempted — no spooling. A backlog of webcam images of your own face on
  your own laptop is a worse failure than a gap in the record. Clip capture
  above follows the same rule: nothing survives past an always-cleaned-up
  temp directory.
- **Frames expire in an hour.** `orion-percept-store` sweeps on a timer. The
  interpretation is the durable artifact; the picture is not. Clips follow
  the same retention.
- **No *continuous* facial affect on this presence stream.** carbon's normal
  low-fps stream reports presence, not emotion — reading emotion off it
  would have to clear the bar `typo_rate` failed (built, tested, then
  deliberately not wired because it never reached a genuine rest state
  across 111 real sessions; see `orion/schemas/affective_state.py`). The
  on-demand AffectGPT capture above is a deliberate, explicit exception to
  that stance, not a reversal of it — separate model, separate trigger,
  off by default, and its own accuracy questions are tracked in
  `services/orion-affectgpt-worker/README.md` rather than assumed solved.
- **No object inventory on this stream.** There is no scene to remember at 40 cm,
  and inventorying a person's desk is a different act from inventorying a room.

## Screens

Screen redaction is decided but **not built**: blank `screen`/`laptop` regions at
the sensor before bytes leave the node, using boxes GroundingDINO already
returns and currently discards. carbon is lower risk than the room camera — a
laptop webcam faces you, not your display — but screens *behind* you are in
frame. Worth knowing before pointing it at the room.
