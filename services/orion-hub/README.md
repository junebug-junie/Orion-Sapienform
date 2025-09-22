# Orion Hub

Orion Hub is the **central interaction gateway** for the Orion Mesh. It provides voice input/output, chat history, and future text + vision input, routing all interactions to Orion's Brain service and publishing telemetry to the shared Orion Bus.

---

## Core Architecture

**Services (Docker Compose):**

* **voice-app** — FastAPI app (UI + WebSocket audio). Uses **Faster-Whisper** on GPU for STT.
* **coqui-tts** — **Coqui TTS** service for high-quality speech synthesis.
* **caddy** — **Caddy** reverse proxy for dev/prod routing and HTTPS in production.

> The LLM service is no longer bundled; Orion Hub connects to an external `ai-brain-service`.

---

## Features

* Push-to-talk **real-time voice** interaction
* **External Brain Service** integration (Mistral/Mixtral/etc.)
* **Telemetry publishing** to Orion Bus (transcripts, LLM tokens, TTS events)
* **Natural TTS** via Coqui
* **Dynamic UI** with speech visualizer
* **Conversation controls**: copy transcript, clear history
* Future: typed input, vision integration, context persistence

---

## Prerequisites

* Linux host with an **NVIDIA GPU** (driver installed)
* **Docker** + **Docker Compose v2**
* **NVIDIA Container Toolkit** (GPU passthrough to containers)
* **Dev (headless)**: SSH access from your laptop to the server
* **Prod**: public domain, DNS A record to your server’s public IP, router/NAT forwards for **80/443**, firewall open for **80/443**

---

## Project Layout

```
caddy/
  Caddyfile.dev
  Caddyfile.prod
docker-compose.yml
Dockerfile
Makefile
README.md
requirements.txt
scripts/           # app logic (ASR, TTS, routing)
static/
templates/
```

---

## Quickstart

### Development

```bash
MODE=dev docker compose up --build
```

Create a tunnel from your laptop:

```bash
ssh -N -L 18080:127.0.0.1:80 user@<server-ip>
```

Open: [http://localhost:18080](http://localhost:18080)

### Production

Configure DNS + firewall, then:

```bash
MODE=prod docker compose up -d --build
```

Open: `https://yourdomain.com`

---

## Makefile Targets

* `make start` — start services (`MODE=dev` by default)
* `make start MODE=prod` — start in production mode
* `make logs` — follow logs for all services
* `make log-voice` — follow logs for voice-app only
* `make shell` — open a bash shell inside the voice-app container
* `make down` — stop services and remove volumes/orphans
* `make nuke` — stop, remove volumes, prune network

---

## Telemetry & Bus

All transcripts, LLM responses, and TTS events are published to the shared **Orion Bus** (Redis). This allows Orion’s memory and analytics systems to evolve without modifying the user-facing interface.

---

## Roadmap

* [ ] Add typed input to the UI
* [ ] Integrate vision events
* [ ] Conversation persistence across sessions
* [ ] Streaming token output from brain service

---

## License

MIT — see [LICENSE](LICENSE).

