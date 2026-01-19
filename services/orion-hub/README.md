# 🌀 Orion Hub — Titanium Edition

**Version:** 0.4.x  
**Stack:** Python · FastAPI · WebSocket · Tailwind · Orion Bus (Async/Redis)

---

## 📖 Overview

**Orion Hub** is the browser gateway into the mesh.

It is a **"Dumb" UI** that:

- Captures **voice** and **text** from the browser
- Maintains lightweight UI state (history, mode, visualizers)
- Publishes strictly typed **Titanium Contracts** onto the **Orion Bus**
- Waits for answers from downstream workers:
  - **Cortex Gateway** (for all chat/cognition)
  - **TTS Service** (for speech synthesis)

From Hub’s perspective:

> “I don’t know about LLMs, Agents, or RAG. I just send a `CortexChatRequest` to the Gateway.”

---

## 🏗️ Architecture

Hub communicates exclusively via the **Orion Bus** using `BaseEnvelope` and strict Pydantic schemas.

### 1. Chat & Cognition

*   **Intake**: `orion-cortex-gateway:request`
*   **Schema**: `CortexChatRequest` (from `orion.schemas.cortex.contracts`)
*   **Flow**: Hub -> Bus -> Cortex Gateway -> Orchestrator/Agents -> Cortex Gateway -> Hub

Hub supports three modes via the `mode` field in the request:
1.  **Brain**: Direct chat (formerly "chat_general").
2.  **Agent**: Goal-oriented reasoning with packs.
3.  **Council**: Multi-agent deliberation.

### 2. Text-to-Speech (TTS)

*   **Intake**: `orion:tts:intake`
*   **Schema**: `TTSRequestPayload` (from `orion.schemas.tts`)
*   **Flow**: Hub -> Bus -> TTS Service -> Hub (returns audio blob)

### 3. Speech-to-Text (ASR)

*   **Note**: Hub no longer performs local ASR.
*   **Flow**: Browser sends text (preferred) or downstream services handle raw audio (future). Currently, Hub expects text input from the UI (which may use browser WebSpeech API or similar, or the user types).

---

## 🚀 Running Hub

### Requirements

*   Redis (Orion Bus)
*   `orion-cortex-gateway` (for chat)
*   `orion-whisper-tts` (for voice, optional)

### Docker Compose

```bash
docker-compose up -d
```

### Environment Variables

Key variables in `.env`:

```env
# Bus
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://localhost:6379/0

# Titanium Channels
CORTEX_GATEWAY_REQUEST_CHANNEL=orion-cortex-gateway:request
TTS_REQUEST_CHANNEL=orion:tts:intake
```

---

## 🧪 Verification & Smoke Tests

### 1. Check Health
```bash
curl http://localhost:8080/health
# {"status": "ok", "service": "hub"}
```

### 2. Verify Bus Connection
Check logs on startup:
```
INFO:orion-hub:Connecting OrionBus → redis://...
INFO:orion-hub:OrionBusAsync connection established successfully.
INFO:orion-hub:Bus Clients initialized.
```

### 3. Test Chat (Simulated)
If you have access to the bus (e.g., via `redis-cli` or a python script), monitor `orion-cortex-gateway:request`.
When you chat in the UI, you should see a JSON envelope with kind `cortex.gateway.request`.

### 4. Verify Chat History Bus Traffic
Use the bus probe to watch chat history events while sending a UI message:

```bash
python scripts/bus_probe.py --pattern orion:chat:history:* --pattern orion:chat:history:turn
```

Expected lines include:

```
{"channel":"orion:chat:history:turn","kind":"chat.history", ...}
```

---

## 🧵 Philosophy

Hub is intentionally thin:

> UI + WebSocket + Bus, nothing else.

All real cognition, memory, and embodiment live elsewhere in the mesh. Hub just gives you a clean window into Oríon’s head.
