# 🌀 Orion Hub — Voice, Chat & Agent Modes

**Version:** 0.4.x  
**Stack:** Python · FastAPI · WebSocket · Tailwind · Orion Bus (Redis)

---

## 📖 Overview

**Orion Hub** is the browser gateway into the mesh.

It **does not** run any heavy cognition itself. Instead it:

- Captures **voice** and **text** from the browser
- Maintains lightweight UI state (history, mode, visualizers, Collapse Mirror form)
- Publishes jobs onto the **Orion Bus (Redis)**
- Waits for answers from downstream workers:
  - LLM Gateway / Brain
  - Agent Chain + Planner + Cortex
  - Agent Council
  - TTS workers
  - Collapse / memory writers

From Hub’s perspective:

> “I don’t know where the models live. I just talk to Redis.”

GPU/CPU workers sit behind Tailscale on other nodes (Atlas, Chrysalis, etc.) and all coordination happens over the bus.

---

## 🧠 Modes: Brain / Agentic / Council

The UI exposes an **Orion Mode** toggle:

- **Brain**  
  Direct conversational mode.  
  Hub talks to **LLM Gateway** via the bus (`chat_general` / simple verbs).  
  No planning loop, just “good chat” with context.

- **Agentic**  
  Goal-oriented reasoning using **Agent Chain + Planner + Cortex**.  
  Hub sends the user’s text plus selected **tool packs**.  
  Planner decides which verbs to invoke (e.g. `plan_action`, `assess_risk`) via Cortex.

- **Council**  
  Multi-agent deliberation route via **Agent Council**.  
  Hub publishes a council request and waits for the council’s aggregated answer.

All three share the same **WebSocket / HTTP** front end; only the bus payload and channels differ.

---

## 🔊 Voice & TTS Flow (High Level)

Current voice loop (Hub-centric view):

1. **Browser → Hub (WebSocket)**  
   Raw audio chunks from `getUserMedia` (press & hold mic).  
   Optional text messages via the chat box.

2. **Hub → ASR**  
   Hub forwards audio to the configured ASR path:
   - Either **local Whisper** in the same container, or
   - A separate ASR worker (still reachable over the bus).
   
   Result: a **transcript** string.

3. **Hub → Bus: LLM job**  
   Hub builds a job payload with:
   - `mode` (brain / agentic / council)
   - `history` (short chat context)
   - user’s text / transcript
   
   Hub then publishes one of:
   - `CHANNEL_VOICE_LLM` / `EXEC_REQUEST_PREFIX:LLMGatewayService` (brain)
   - `AGENT_CHAIN_REQUEST_CHANNEL` (agentic)
   - `CHANNEL_COUNCIL_INTAKE` (council)

4. **Workers → Bus → Hub**  
   Corresponding worker(s) consume the job and publish a reply on a **reply channel** keyed by `trace_id` or `correlation_id`.  
   Hub waits (with timeout) and extracts the final **LLM text** and any structured payload.

5. **Hub → Browser**  
   Hub streams back:
   - progress / status events (`processing`, `answering`, etc.)
   - the final LLM text into the conversation window.

6. **Optional: Hub → Bus: TTS job**  
   If the user has **“Speak Response”** toggled on:
   - Hub publishes a `tts:intake` job containing the final LLM message and `trace_id`.
   
   A TTS worker (running Piper/Coqui/etc. on some Tailscale node) synthesizes audio and publishes:
   - `tts:response:<trace_id>` with base64 audio chunks.

7. **Hub → Browser: audio chunks**  
   Hub relays `tts:response` chunks over the WebSocket.  
   The browser plays them in sequence via the audio buffer.

---

## 🧩 Components

### 1. Browser UI (`index.html` + `static/js/app.js`)

Key UI features:

- **Mic button**  
  Press & hold to record; release to send.  
  Shows a state ring and pulses when recording/processing.

- **Chat box**  
  Enter sends; Shift+Enter makes a newline.  
  A conversation pane shows “You” vs “Oríon” messages.

- **Orion Mode toggle**  
  Buttons: **Brain**, **Agentic**, **Council**.  
  Agentic mode also exposes **tool pack checkboxes** (Executive, Memory, Emergent, etc.).

- **Collapse Mirror card**  
  Guided form + raw JSON editor.  
  Sends structured collapse entries to the backend.

- **Vision dock / PiP**  
  Stubbed container for future camera feed overlays.

Front-end messages back to the server look roughly like:

```jsonc
// Voice / text
{
  "type": "chat",
  "mode": "agentic",
  "text_input": "Help me plan my day...",
  "packs": ["executive_pack"],
  "disable_tts": false,
  "temperature": 0.7,
  "context_length": 10
}

// Collapse Mirror
{
  "type": "collapse_submit",
  "payload": { ...collapse fields... }
}
```

### 2. Hub FastAPI App

Typical endpoints:

- `GET /`  
  Serves the main HTML UI.

- `GET /static/...`  
  JS/CSS assets.

- `GET /ws` (or similar)  
  Main WebSocket entrypoint.  
  Receives chat/voice messages, modes, pack selections.  
  Sends back transcripts, LLM answers, status events, and TTS audio chunks.

- `POST /collapse/submit`  
  Used by the Collapse Mirror form to send entries into the collapse pipeline (via bus).

Internally, the app uses:

- **`OrionBus`** wrapper to publish/subscribe on Redis.
- Small helpers for:
  - building agent-chain payloads
  - mapping Orion Mode → correct channels and packs
  - short-term in-memory history

The Hub **does not** embed any LLM/TTS libraries. All heavy work is done by workers.

---

## 🚌 Bus & Channels

Environment variables (examples):

```env
# Bus
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://100.xx.yy.zz:6379/0

# Voice / LLM / TTS channels
CHANNEL_VOICE_TRANSCRIPT=orion:voice:transcript
CHANNEL_VOICE_LLM=orion:voice:llm
CHANNEL_VOICE_TTS=orion:voice:tts
CHANNEL_TTS_INTAKE=orion:tts:intake

# Collapse
CHANNEL_COLLAPSE_INTAKE=orion:collapse:intake
CHANNEL_COLLAPSE_TRIAGE=orion:collapse:triage

# LLM Gateway / Exec
LLM_GATEWAY_SERVICE_NAME=LLMGatewayService
EXEC_REQUEST_PREFIX=orion-exec:request

# Cortex-Orch
CORTEX_REQUEST_CHANNEL=orion-cortex:request
CORTEX_RESULT_PREFIX=orion-cortex:result

# Agent Chain / Council
AGENT_CHAIN_REQUEST_CHANNEL=orion-agent-chain:request
AGENT_CHAIN_REQUEST_PREFIX=orion-agent-chain:result
CHANNEL_COUNCIL_INTAKE=orion:agent-council:intake
CHANNEL_COUNCIL_REPLY_PREFIX=orion:agent-council:reply
```

Workers use the same `ORION_BUS_URL` (usually a Tailscale IP + port to the Redis node).

---

## 🔀 How Each Mode Routes

### Brain Mode

1. Hub builds a **chat context** (recent messages, mode, sliders).
2. Publishes an **exec_step / chat** job over the bus to `LLMGatewayService`:
   - Channel: `orion-exec:request:LLMGatewayService`
   - Reply: `orion:llm:reply:<trace_id>` (managed by LLM Gateway).
3. LLM Gateway chooses backend (Ollama, vLLM, etc.) — Hub doesn’t care.
4. Gateway replies with final text; Hub shows it and optionally forwards to TTS.

### Agentic Mode

1. Hub collects:
   - User text
   - Mode = `agentic`
   - Selected `packs` (e.g., `["executive_pack"]`)
2. Publishes an RPC envelope to **Agent Chain**:
   - Bus event: `AGENT_CHAIN_REQUEST_CHANNEL` = `orion-agent-chain:request`
   - Includes `reply_channel` = `orion-agent-chain:result:<trace_id>`
3. Agent Chain:
   - Resolves packs → verbs → `ToolDef`s via `orion-cognition` packs.
   - Calls **Planner React** via bus (`orion-planner:request` / `orion-planner:result:*`).
   - Planner uses **Cortex-Orch** to execute verbs on the mesh.
4. Agent Chain returns `final_answer` → Hub → browser (and TTS if enabled).

### Council Mode

1. Hub publishes a council intake:
   - `CHANNEL_COUNCIL_INTAKE=orion:agent-council:intake`
2. Council service:
   - Fans out to multiple internal agents / prompts.
   - Aggregates their answers into a single structured response.
3. Hub listens on `CHANNEL_COUNCIL_REPLY_PREFIX:<trace_id>`
   and renders the council’s consensus answer in the chat.

---

## 📦 Deployment (Atlas / Chrysalis / Friends)

Typical pattern in `docker-compose`:

```yaml
services:
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
```

Other services (LLM Gateway, Agent Chain, Planner, Cortex, Council, TTS, ASR, Collapse, Memory) may live on different physical nodes but must all point at the same `ORION_BUS_URL` (usually a Redis instance on the tailnet).

---

## 🩺 Troubleshooting

- **No response in any mode**  
  - Verify Redis is reachable from Hub (`redis-cli -u $ORION_BUS_URL PING`).  
  - Ensure at least one LLM worker (Gateway/Brain) is listening on the relevant channels.

- **Agentic times out**  
  - Check `orion-athena-agent-chain` logs.  
  - Then check `orion-athena-planner-react` and `orion-athena-cortex-orch` for timeouts or missing verbs.

- **Council no-ops**  
  - Ensure Agent Council worker is running and subscribed to `orion:agent-council:intake`.

- **TTS silent**  
  - Confirm a TTS worker is subscribed to `orion:tts:intake` and publishing `tts:response:*`.  
  - Verify WebSocket messages with the browser dev tools.

---

## 🧵 Philosophy

Hub is intentionally thin:

> UI + WebSocket + Bus, nothing else.

All real cognition, memory, and embodiment live elsewhere in the mesh. Hub just gives you a clean window into Oríon’s head — whether you’re chatting, running an executive agent chain, or convening a council.
