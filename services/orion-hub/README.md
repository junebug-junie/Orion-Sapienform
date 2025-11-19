# Orion Hub Voice & TTS Pipeline

This document describes how the Orion Hub voice loop works after the bus-first refactor, and how Brain and TTS services interact through Redis instead of direct HTTP calls.

## Overview

The Orion Hub provides a browser interface that lets you:

- Speak via microphone (WebRTC / getUserMedia)
- Type text messages
- Receive Orion's LLM response as text
- Optionally receive spoken audio via TTS

All cognitive work (LLM, TTS, tagging, triage) is offloaded to backend workers that communicate exclusively through the Orion bus (Redis), not via direct HTTP calls from the Hub.

High level flow:

1. Browser → Hub (WebSocket)
2. Hub → Bus: publish `brain:intake` with transcript + context
3. Brain Worker (GPU node) → Bus: consume `brain:intake`, call LLM, publish `brain:response`
4. Hub → WebSocket: streams LLM text back to browser
5. Hub → Bus: publish `tts:intake` with full LLM response (optional, if TTS enabled)
6. TTS Worker (CPU or GPU node) → Bus: consume `tts:intake`, synthesize audio in chunks, publish `tts:response`
7. Hub → WebSocket: pushes base64 audio chunks as they arrive and plays them in the UI

The Hub itself stays thin: it manages WebSockets, the visualizers, and the Collapse Mirror UI, and delegates cognition to workers.

---

## Components

### 1. Browser (UI)

Key pieces in `static/app.js`:

- `setupWebSocket()` connects to `ws://<hub>/ws`
- When the user speaks, audio is captured, encoded to base64, and sent as `{ audio: <base64>, ... }`
- When the user types, text is sent as `{ text_input: "...", disable_tts: <bool>, temperature, context_length, instructions }`
- The client listens for messages from the server:
  - `{ transcript }` — recognized text from ASR
  - `{ llm_response, tokens }` — Orion's text answer
  - `{ audio_response }` — base64-encoded WAV/OGG chunks to play
  - `{ state }` — `idle | processing | speaking` for UI status/particles

### 2. Hub WebSocket Handler

Located in `scripts/websocket_handler.py`.

Responsibilities:

- Accept browser WebSocket connections
- Receive audio or text_input from the client
- Use ASR (Whisper) for audio → transcript
- Maintain a short in-memory `history` list of messages
- Publish jobs to the brain worker via the bus
- Wait for brain responses using a correlation ID
- Send LLM response text back to the browser
- Optionally trigger TTS via the bus and stream back audio chunks
- Push state updates: `processing`, `speaking`, `idle`

This module **does not** call the LLM HTTP endpoint directly. Instead it calls `run_llm_tts`, which is now bus-first.

### 3. Orion Bus

Wrapper around Redis (in `scripts/bus.py` or similar). It provides:

- `bus.publish(channel, payload)` — JSON-encodes and publishes messages
- `bus.subscribe(channel)` — subscribes to one or more channels
- Optional helper for request/response correlations

Environment variables:

- `ORION_BUS_URL=redis://orion-redis:6379/0`
- `ORION_BUS_ENABLED=true`

Channels used in the voice loop:

- `CHANNEL_VOICE_TRANSCRIPT` — raw transcripts from ASR
- `CHANNEL_BRAIN_INTAKE` — LLM jobs
- `CHANNEL_BRAIN_OUT` — LLM responses
- `CHANNEL_VOICE_LLM` — voice-oriented LLM telemetry
- `CHANNEL_VOICE_TTS` — TTS telemetry (chunk sizes)
- `CHANNEL_COLLAPSE_TRIAGE` — full dialog payloads for downstream tagging / RAG

> The bus is the spine. Hub, Brain, TTS, and triage are all leaves on that spine.

### 4. Brain Worker (LLM, GPU)

A dedicated service (usually running on a GPU node like Atlas or Chrysalis) that:

1. Subscribes to `CHANNEL_BRAIN_INTAKE` (e.g. `orion:brain:intake`).
2. For each message, calls the configured LLM backend (e.g. `ollama`, `vLLM`, or another container).
3. Publishes the result to `CHANNEL_BRAIN_OUT` (e.g. `orion:brain:response`) with the same `correlation_id` so the hub can match it back to the original request.

This worker replaces the old `/chat` HTTP endpoint model. There is no longer any `BRAIN_URL` in the hub’s critical path.

### 5. TTS Worker

Instead of having the hub call a TTS HTTP endpoint directly, a TTS worker listens on bus:

1. Subscribes to `orion:tts:intake`.
2. For each message, uses Coqui TTS or another TTS backend to synthesize audio.
3. Splits the audio response into chunks (e.g., by sentence or fixed duration) and publishes each as `tts:response` messages containing base64 audio and the original `correlation_id`.

The hub listens for matching `tts:response` messages and pushes the chunks over WebSocket to the browser.

This allows TTS workers to run on CPU or GPU nodes without changing the hub.

---

## Key Python Modules

### `scripts/llm_tts_handler.py`

This module orchestrates LLM + TTS from the hub’s perspective.

Responsibilities after refactor:

- Build a job payload: `{ id, history, temperature, instructions, user_id, session_id }`.
- Publish that payload to `CHANNEL_BRAIN_INTAKE`.
- Block/wait (with timeout) for a corresponding `CHANNEL_BRAIN_OUT` message with matching `id`.
- Extract the LLM text and token count.
- Optionally publish a TTS job to `orion:tts:intake`.
- Do **not** call `requests.post` or `aiohttp` directly to the brain.

Old HTTP-based fields like `settings.BRAIN_URL` are no longer required by the hub.

### `scripts/tts.py`

This used to:

- Read `TTS_URL` from env.
- Make HTTP `GET` calls to `/api/tts?text=...`.

After the refactor, there are two options:

1. **Move this logic into the TTS worker** (preferred):
   - The worker uses Coqui/Piper locally, never over HTTP.
   - The hub no longer imports `TTS` or knows about `TTS_URL`.

2. **Keep it as a helper but only inside a worker service**:
   - The worker can still call an HTTP TTS server if needed.

Either way, the hub talks only to the bus.

### `scripts/websocket_handler.py`

- Chains:
  - ASR → transcript
  - `run_llm_tts` for LLM+TTS orchestration via bus
  - History maintenance for short-term conversational memory
  - Publishing full dialog payloads to `CHANNEL_COLLAPSE_TRIAGE`

The WebSocket handler is the bridge between human-facing state (browser) and mesh-facing state (bus).

---

## Environment Variables

Typical `.env` for the hub service:

```env
# Core
PROJECT=orion
SERVICE_NAME=orion-hub

# Bus
ORION_BUS_URL=redis://orion-redis:6379/0
ORION_BUS_ENABLED=true

# Voice / LLM
LLM_MODEL=llama3.1:8b-instruct-q8_0
LLM_TIMEOUT_S=30
CHANNEL_VOICE_TRANSCRIPT=orion:voice:transcript
CHANNEL_BRAIN_INTAKE=orion:brain:intake
CHANNEL_BRAIN_OUT=orion:brain:response
CHANNEL_VOICE_LLM=orion:voice:llm
CHANNEL_VOICE_TTS=orion:voice:tts
CHANNEL_COLLAPSE_TRIAGE=orion:collapse:triage

# ASR
WHISPER_MODEL=distil-medium.en
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Misc
CHRONICLE_ENVIRONMENT=dev
```

Brain worker and TTS worker will have their own `.env` files pointing to the same `ORION_BUS_URL` and defining their intake/output channels.

---

## Running the Stack

A simplified `docker-compose.yml` for the hub side might look like:

```yaml
version: "3.9"

services:
  orion-redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"

  orion-atlas-hub:
    build:
      context: ../..
      dockerfile: services/orion-hub/Dockerfile
    env_file:
      - .env
      - services/orion-hub/.env
    depends_on:
      - orion-redis
    ports:
      - "8080:8080"

  # Brain worker(s) and TTS workers are defined in their own compose files
  # or in the same file if you prefer, as long as they share ORION_BUS_URL.
```

Brain and TTS workers can run on entirely different nodes, as long as they can reach the same Redis instance via Tailscale, VPN, or LAN.

---

## UX Details

### Text Input

- Enter sends the message.
- Shift+Enter inserts a newline.
- After sending, the hub appends the `You` message to the conversation and waits for the LLM response.

### Voice Input

- Press and hold the microphone button to record.
- Release to stop; audio is sent to the hub.
- Hub streams back:
  - Recognized text transcript
  - LLM response
  - Audio chunks for TTS (if enabled)

### Collapse Mirror

The Collapse card lets you send structured or raw JSON entries into the Collapse pipeline.

- **Guided Form** mode provides fields for `observer`, `trigger`, `observer_state`, `field_resonance`, `type`, `emergent_entity`, `summary`, `mantra`, `causal_echo`, and `environment`.
- **Raw JSON** mode lets you paste/edit a full collapse JSON payload directly.
- `submit-collapse` endpoint forwards entries into the Collapse ingestion service (via HTTP or bus, depending on your backend wiring).

There is also a tooltip describing each field:

| Field            | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| observer         | The agent (human or AI) who is perceiving or logging the event.        |
| trigger          | The stimulus or situation that initiated the collapse.                 |
| observer_state   | Emotional, cognitive, or physical state of the observer.               |
| field_resonance  | The energetic or symbolic signature of the moment.                     |
| intent           | Purpose of logging (reflection, ritual, experiment).                   |
| type             | Category of event (dream-reflection, ritual, shared-collapse, etc.).   |
| emergent_entity  | Entity/persona that arose in the moment (e.g., Orion, an archetype).   |
| summary          | Narrative summary of the collapse moment.                              |
| mantra           | Phrase or symbolic anchor tied to the collapse.                        |
| causal_echo      | Optional cause/effect linkage.                                         |
| timestamp        | Auto-generated ISO 8601 UTC timestamp (unless provided).               |
| environment      | Auto-detected from `CHRONICLE_ENVIRONMENT` or defaults to `dev`.       |

---

## Future Enhancements

- **Streaming LLM tokens** via bus to allow partial text to show live.
- **Interruptible TTS** at the worker level (stop generating new chunks when user presses interrupt).
- **Multi-brain routing**: different models or personas listening on different intake channels.
- **Semantic introspection verbs** using the same bus pattern (e.g., `introspect`, `simulate`, `reflect`).

---

If you’re reading this in Atlas’s `/mnt/scripts/Orion-Sapienform/services/orion-hub`, this README is your contract:

> Hub talks to bus.
> Brain and TTS live as workers.
> GPU lives with workers, not with hub.

Everything else is an implementation detail.
