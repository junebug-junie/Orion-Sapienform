# 🌀 Orion Hub — Dumb UI & Bus Client

**Version:** 0.4.x  
**Stack:** Python · FastAPI · WebSocket · Tailwind · Orion Bus (Async)

---

## 📖 Overview

**Orion Hub** is the browser gateway into the mesh.

It is a **"Dumb UI"**:
- **No Cognition**: It does not run LLMs, Agents, or ASR/TTS engines.
- **No Routing Logic**: It delegates all decisions to upstream services.
- **Bus Only**: It communicates exclusively via **Titanium Contracts** over the Orion Bus.

From Hub’s perspective:
> “I accept text/audio from the user, wrap it in a typed envelope, and ship it to the Gateway. I display whatever comes back.”

---

## 🚀 Key Integrations

Hub talks to two main services via Bus RPC:

### 1. Cortex Gateway (`orion-cortex-gateway`)
- **Channel**: `orion:cortex-gateway:request`
- **Contract**: `CortexChatRequest` -> `CortexClientResult`
- **Responsibility**: Handles all chat modes (`brain`, `agent`, `council`), recall injection, and tool execution.

### 2. TTS Service (`orion-whisper-tts`)
- **Channel**: `orion:tts:intake`
- **Contract**: `TTSRequestPayload` -> `TTSResultPayload`
- **Responsibility**: Synthesizes speech from text.

---

## 🛠️ Configuration

Hub is configured via environment variables.

### Required Environment Variables

```env
# Service Identity
SERVICE_NAME=hub
SERVICE_VERSION=0.4.0
HUB_PORT=8080

# Bus Configuration
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://localhost:6379/0

# Cortex Gateway Integration
CORTEX_GATEWAY_REQUEST_CHANNEL=orion:cortex-gateway:request
CORTEX_GATEWAY_RESULT_PREFIX=orion:cortex-gateway:reply

# TTS Integration
TTS_REQUEST_CHANNEL=orion:tts:intake
TTS_RESULT_PREFIX=orion:tts:reply

# Legacy / Collapse Channels
CHANNEL_COLLAPSE_INTAKE=orion:collapse:intake
CHANNEL_COLLAPSE_TRIAGE=orion:collapse:triage
CHANNEL_CHAT_HISTORY_LOG=orion:chat:history:log

# Voice Channels (Legacy references)
CHANNEL_VOICE_TRANSCRIPT=orion:voice:transcript
CHANNEL_VOICE_LLM=orion:voice:llm
CHANNEL_VOICE_TTS=orion:voice:tts
```

---

## 🏃 Running Hub

### Via Docker Compose

```bash
docker-compose up --build hub-app
```

### Via Python (Local Dev)

1. **Install Dependencies** (if not already in env):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   cd services/orion-hub/scripts
   uvicorn main:app --reload --port 8080
   ```

---

## 🧪 Smoke Tests

### 1. Check Health
```bash
curl http://localhost:8080/health
# {"status": "ok", "service": "hub"}
```

### 2. Verify Session
```bash
curl http://localhost:8080/api/session
# {"session_id": "..."}
```

### 3. Test Chat (End-to-End)
*Requires `orion-cortex-gateway` and `orion-cortex-orch` to be running.*

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello World"}],
    "mode": "brain"
  }'
```

---

## 🏗️ Architecture Notes

- **`scripts/bus_clients/`**: Contains typed clients for `Cortex` and `TTS`.
- **`scripts/websocket_handler.py`**: Manages real-time loop.
- **`scripts/api_routes.py`**: Stateless HTTP wrappers for the same bus clients.
- **`orion/schemas/*`**: Source of truth for all payloads.

Hub **does not** import `orion.core.bus.service` (legacy). It uses `orion.core.bus.async_service.OrionBusAsync`.
