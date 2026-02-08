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

### 4. In-app Notifications (Hub UI)

*   **Channel**: `orion:notify:in_app`
*   **Schema**: `HubNotificationEvent` (from `orion.schemas.notify`)
*   **Flow**: orion-notify -> Bus -> Hub (WebSocket broadcast)
*   **WebSocket payload**: `{ "kind": "notification", "notification": { ... } }`
*   **HTTP history**: `GET /api/notifications?limit=50`

#### Chat Attention

`event_kind="orion.chat.attention"` renders as a special toast/card with:

- **Open chat** (focuses the chat input)
- **Dismiss** (`ack_type="dismissed"`)
- **Snooze 30m** (`ack_type="snooze"`)

The hub proxies acknowledgements to `orion-notify`:

- `POST /api/attention/{attention_id}/ack`
- `GET /api/attention?status=pending`

#### Chat Messages

`event_kind="orion.chat.message"` renders as a message card + toast with:

- **Open chat** (focuses chat input + switches session_id)
- **Dismiss** (`receipt_type="dismissed"`)

Hub proxies receipts to `orion-notify`:

- `POST /api/chat/message/{message_id}/receipt`
- `GET /api/chat/messages?status=unread|seen`

Presence endpoint (for notify presence checks):

- `GET /api/presence` → `{ "active": true|false, "last_seen": ... }`

#### Notification Settings UI

Hub exposes a Notification Settings panel (gear icon) that loads and updates:

- Recipient profile (quiet hours, timezone)
- Event/severity preferences (channels, escalation delay)

Preference rows are provided for:

- `orion.chat.attention`
- `orion.chat.message`
- severity `error`, `warning`, `info`

The panel calls:

- `GET /api/notify/recipients/{recipient_group}`
- `PUT /api/notify/recipients/{recipient_group}`
- `GET /api/notify/recipients/{recipient_group}/preferences`
- `PUT /api/notify/recipients/{recipient_group}/preferences`

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

# Landing Pad (Topic Rail)
LANDING_PAD_URL=http://orion-landing-pad:8370
LANDING_PAD_TIMEOUT_SEC=5

# Topic Studio (Topic Foundry proxy)
TOPIC_FOUNDRY_BASE_URL=http://orion-topic-foundry:8615

# Optional overrides (serve-mode edge cases)
HUB_API_BASE_OVERRIDE=
HUB_WS_BASE_OVERRIDE=
```

Tailscale Serve / TLS works out of the box: Hub derives API and WS bases from the current origin (same-origin `/api/...` and `ws://`/`wss://`). For edge cases (reverse proxies or nonstandard routing), set `HUB_API_BASE_OVERRIDE` and/or `HUB_WS_BASE_OVERRIDE` to override the base URLs; otherwise leave them blank. 

Topic Rail endpoints are proxied through Landing Pad. Ensure `POSTGRES_URI` is set on the Landing Pad service so it can read the `chat_topic_summary` and `chat_topic_session_drift` tables.

Topic Studio relies on the Topic Foundry `/capabilities` endpoint to configure supported segmentation modes and defaults, uses `/runs?limit=20` to populate the recent run picker, and the segments list uses `include_snippet=true&include_bounds=true` with `limit/offset` for faster previews and paging.

### Manual UI checklist
- Navigate between **Hub** and **Topic Studio** tabs; ensure no overlays block pointer events on Hub.
- In Topic Studio, run **Preview** with `turn_pairs`, then switch to `conversation_bound` after setting a `boundary_column`.
- Train a run, poll for completion, then load segments and click a segment to confirm full text renders in the detail pane.

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

### 5. Topic Rail Summary + Drift
```bash
curl "http://localhost:8080/api/topics/summary?window_minutes=1440&max_topics=20"
curl "http://localhost:8080/api/topics/drift?window_minutes=1440&min_turns=10&max_sessions=50"
```

### 6. Topic Foundry smokes (via Hub proxy)
Hub proxies Topic Foundry under `/api/topic-foundry`, so smoke scripts can target the Hub host.

**Via Hub proxy (recommended):**
```bash
scripts/smoke_topic_foundry_all.sh http://localhost:8080/api/topic-foundry
```
or:
```bash
HUB_BASE_URL=https://tailscale-host.example.com scripts/smoke_topic_foundry_introspect.sh
```

**Direct service port (optional):**
```bash
TOPIC_FOUNDRY_BASE_URL=http://127.0.0.1:8615 scripts/smoke_topic_foundry_preview.sh
```

**Inside Docker network (optional):**
```bash
TOPIC_FOUNDRY_BASE_URL=http://orion-topic-foundry:8615 scripts/smoke_topic_foundry_facets.sh
```

### 7. No-Write Debug Mode (skip memory publishing)
Use the header + JSON flag to avoid publishing `orion:chat:history:*` events while still running recall/LLM:

```bash
curl -sS http://localhost:8080/api/chat \
  -H "content-type: application/json" \
  -H "X-Orion-No-Write: 1" \
  -d '{ "mode":"brain","use_recall":true,"recall_profile":"reflect.v1","no_write":true,
        "messages":[{"role":"user","content":"GrowthSynthesis23"}] }'
```

Expected:
- Response includes `memory_digest` (when recall is enabled).
- No events appear on bus patterns `orion:chat:history:*` for that request.

---

## 🧵 Philosophy

Hub is intentionally thin:

> UI + WebSocket + Bus, nothing else.

All real cognition, memory, and embodiment live elsewhere in the mesh. Hub just gives you a clean window into Oríon’s head.

The UI now includes a **Topic Rail** panel (below Vision/Collapse) that shows the top topics and most drifting sessions for a selected window. Use the window/max-topic/min-turn controls and the optional model version field to query the Hub proxy endpoints. Auto-refresh is enabled by default (60s) and can be toggled off.
