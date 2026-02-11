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


## Topic Studio Integration Contract

Topic Studio does **not** call Topic Foundry directly from browser; it always goes through Hub proxy:
- `GET /api/topic-foundry/ready`
- `GET /api/topic-foundry/capabilities`

Proxy target is controlled by `TOPIC_FOUNDRY_BASE_URL` in Hub settings/env.

### Expected capability keys used by active Topic Studio UI
- `segmentation_modes_supported` (array): drives segmentation mode select options.
- `supported_metrics` (array): drives model metric select options.
- `default_metric` (string): preferred metric if available in supported list.
- `defaults.embedding_source_url` (string): embedding URL default/hint.
- `defaults.metric`, `defaults.min_cluster_size`: form prefill defaults.
- `default_embedding_url` (string): fallback for embedding default.
- `llm_enabled` (boolean): disables LLM segmentation options + enrich button when false.

### UI behavior when keys are missing
- Missing `segmentation_modes_supported` / `supported_metrics`: selector may appear empty.
- Capability fetch failure: UI applies hardcoded fallback modes/metrics and marks endpoint warning.
- Missing `llm_enabled`: treated as false (`Boolean(undefined)`), so UI shows effectively disabled LLM controls.
- `/ready` fetch failure: status badge becomes **Unreachable**.
- `/ready` success with degraded checks: status badge stays reachable but check-level badges can show fail.

## Topic Studio Troubleshooting

### `REACHABLE` but capability parse appears broken
- `REACHABLE` is computed from successful `/ready` fetch, not `/capabilities` parse.
- Check `Topic Foundry /capabilities` payload for arrays/keys listed above.
- Inspect `#tsCapabilitiesWarning` and browser console for endpoint parse/fetch errors.

### `LLM disabled` shown unexpectedly
- Hub uses `/capabilities.llm_enabled` directly.
- Verify Foundry env `TOPIC_FOUNDRY_LLM_ENABLE=true` and confirm payload returns `"llm_enabled": true`.
- For bus mode, also ensure `TOPIC_FOUNDRY_LLM_USE_BUS=true` + `ORION_BUS_ENABLED=true` in Foundry.

### Static JS cache/version notes
- Template includes an explicit cache-busting query string on app bundle, e.g. `/static/js/app.js?v=1.0.56`.
- If UI behavior does not match source, hard-refresh or bump the `v=` string in `templates/index.html` when deploying.

