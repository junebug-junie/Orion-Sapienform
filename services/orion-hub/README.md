Orion Hub

Orion Hub is the central interaction gateway for the Orion Mesh — the bridge between humans and Orion’s cognition layer.It handles voice input/output, chat visualization, and telemetry publishing to the shared Orion Bus.All cognition is delegated to Orion’s Brain service.

🧠 Core Architecture

Service

Role

hub-app

FastAPI + WebSocket gateway (real-time voice I/O, ASR via Faster-Whisper).

coqui-tts

Coqui TTS container for natural speech output.

caddy

Reverse proxy / HTTPS gateway for local and public exposure.

⚠️ The LLM itself (Mistral, Mixtral, etc.) now lives in the Orion Brain container.Orion Hub must connect to a running Brain instance before startup succeeds.

✨ Features

🎤 Real-time push-to-talk voice interaction

🧩 External Brain integration (Mistral / Mixtral / others)

📡 Telemetry publishing to the Orion Bus (via Redis)

👤 Coqui TTS for expressive voice output

🩶 Dynamic web UI with speech visualization

🔄 Built-in Caddy proxy for both local dev and production HTTPS

⚙️ Prerequisites

Linux host with NVIDIA GPU (CUDA + drivers installed)

Docker + Docker Compose v2

NVIDIA Container Toolkit for GPU passthrough

Orion Bus running (see ../orion-bus)

Orion Brain running (see ../orion-brain)

Mesh-wide root .env defined at /mnt/services/Orion-Sapienform/.env

📁 Project Layout

caddy/
  Caddyfile.dev
  Caddyfile.prod
docker-compose.yml
Dockerfile
Makefile
.env
README.md
templates/
static/

🚀 Quickstart

1. Run the “bullshit pre-step” (once per terminal session)

Docker Compose does not automatically read environment files from parent directories.You must manually export the mesh-wide .env before Compose runs:

set -a
. /mnt/services/Orion-Sapienform/.env
set +a

This makes variables like PROJECT, NET, and TELEMETRY_ROOT visible to all service stacks.

(Yes, this is the “bullshit” step — but it ensures node-aware naming like orion-janus-hub and proper networking.)

2. Bring up the Hub

🧑‍💻 Development

make up

Then tunnel from your laptop:

ssh -N -L 18080:127.0.0.1:80 user@<server-ip>

Open → http://localhost:18080

🌐 Production

make up MODE=prod

Then open → https://yourdomain.com

🧰 Makefile Targets

Command

Description

make up

Start Orion Hub (defaults to dev mode)

make down

Stop services and remove orphans

make restart

Restart cleanly

make logs

Follow logs from all services

make log SVC=hub-app

Follow logs for a specific service

make shell

Enter the main hub container

make env

Print resolved environment context

🧩 Telemetry & Bus

All voice transcripts, LLM responses, and TTS events are published to the shared Orion Bus,forming the event substrate for Orion’s memory and reflection systems.No data is stored locally — the Bus acts as the ephemeral nervous system.

ORION_BUS_URL=redis://${PROJECT}-bus-core:6379/0

🔁 Service Dependencies

Orion Hub requires the following services to be running first:

orion-brain (LLM + inference API)

orion-bus (event routing and pub/sub)

app-net Docker network (created once globally)

Startup order (in your rebuild script):

bus → brain → hub → mirror → others

🦯 Roadmap



