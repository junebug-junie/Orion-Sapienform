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

## What this deliberately does not do

- **Nothing is written to carbon's disk.** The upload path uses `imencode`,
  never `imwrite`. If the store is unreachable the frame is dropped and the next
  one attempted — no spooling. A backlog of webcam images of your own face on
  your own laptop is a worse failure than a gap in the record.
- **Frames expire in an hour.** `orion-percept-store` sweeps on a timer. The
  interpretation is the durable artifact; the picture is not.
- **No facial affect.** carbon reports presence. Reading emotion off a webcam has
  to clear the bar `typo_rate` failed — built, tested, then deliberately not
  wired because it never reached a genuine rest state across 111 real sessions.
  See `orion/schemas/affective_state.py`.
- **No object inventory on this stream.** There is no scene to remember at 40 cm,
  and inventorying a person's desk is a different act from inventorying a room.

## Screens

Screen redaction is decided but **not built**: blank `screen`/`laptop` regions at
the sensor before bytes leave the node, using boxes GroundingDINO already
returns and currently discards. carbon is lower risk than the room camera — a
laptop webcam faces you, not your display — but screens *behind* you are in
frame. Worth knowing before pointing it at the room.
