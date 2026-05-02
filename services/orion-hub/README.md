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
*   **HTTP history**: `GET /api/notifications?limit=50` — returns a **snapshot of the Hub process in-memory deque** (`NotificationCache`) of recent bus-fed events, not a full historical inbox; a full page refresh reloads that cache. The UI shows both upstream `created_at` (when the producer ran) and Hub `received_at` (when the event was ingested) where present.
*   **First-load batching**: On initial hydrate, if the notification count meets or exceeds the threshold, the Hub shows one summary toast instead of per-row toasts. Override the threshold by setting `data-notification-batch-threshold` on `#hubNotificationsPanel` in `templates/index.html` (default `5`).

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

### 5. Substrate Review Runtime Debug Surface (Hub convenience panel)

Hub now includes a compact **Substrate Review** debug row in the main runtime debug area, with a separate high-z modal for bounded operator actions.

- Inline row: compact queue/due/outcome/source posture summary.
- Modal actions:
  - Refresh status
  - Execute one bounded `operator_review` cycle
  - Execute one bounded cycle with explicit frontier follow-up allowed
  - Run a lightweight smoke check
- Safety posture:
  - single-cycle only
  - operator surface only
  - no hidden recursion
  - strict-zone guardrails remain in runtime
- `/substrate` remains the primary standalone inspection page; Hub modal is a convenience control surface.
- In-shell navigation now includes a `#substrate` tab that embeds `/substrate` via iframe so switching tabs preserves Hub shell/session context.

### 5.1 Substrate Mutation V2.1 Lineage Inspection

Read-only admin endpoints for mutation lifecycle inspection (manual route only, no scheduler loop):

- `GET /api/substrate/mutation-runtime/lineage?limit=20`
- `GET /api/substrate/mutation-runtime/lineage?proposal_id=<proposal-id>`
- `GET /api/substrate/mutation-runtime/active-surfaces`
- `GET /api/substrate/mutation-runtime/blocked-applies?limit=20`
- `GET /api/substrate/mutation-runtime/rollbacks?limit=20`
- `GET /api/substrate/mutation-runtime/routing-replay-inspect?limit=50`
- `GET /api/substrate/mutation-runtime/routing-live-ramp-posture`
- `GET /api/substrate/mutation-runtime/cognition-context`
- `GET /api/substrate/mutation-runtime/routing-pressure-sources?limit=50`
- `GET /api/substrate/mutation-runtime/producer-pressure-events?limit=50`
- `GET /api/substrate/mutation-runtime/cognitive-pressure?limit=50`
- `GET /api/substrate/mutation-runtime/cognitive-proposals?limit=20`
- `GET /api/substrate/mutation-runtime/cognitive-proposals/<proposal-id>/lineage`

Structured lifecycle logs are emitted with prefix `substrate_mutation_lifecycle` and include stable lineage keys (`lineage_id`, `proposal_id`, `queue_item_id`, `trial_id`, `decision`, `surface_key`, `blocked_reason`).

#### SQL lineage queries (developer examples)

One proposal lifecycle (swap `<proposal-id>`):

```sql
SELECT 'proposal' AS stage, payload_json
FROM substrate_mutation_proposal
WHERE proposal_id = '<proposal-id>'
UNION ALL
SELECT 'queue' AS stage, payload_json
FROM substrate_mutation_queue
WHERE payload_json::text LIKE '%' || '<proposal-id>' || '%'
UNION ALL
SELECT 'trial' AS stage, payload_json
FROM substrate_mutation_trial
WHERE payload_json::text LIKE '%' || '<proposal-id>' || '%'
UNION ALL
SELECT 'decision' AS stage, payload_json
FROM substrate_mutation_decision
WHERE payload_json::text LIKE '%' || '<proposal-id>' || '%'
UNION ALL
SELECT 'adoption' AS stage, payload_json
FROM substrate_mutation_adoption
WHERE payload_json::text LIKE '%' || '<proposal-id>' || '%'
UNION ALL
SELECT 'rollback' AS stage, payload_json
FROM substrate_mutation_rollback
WHERE payload_json::text LIKE '%' || '<proposal-id>' || '%';
```

Active live mutations by target surface:

```sql
SELECT target_surface, adoption_id, updated_at
FROM substrate_mutation_active_surface
ORDER BY updated_at DESC;
```

Recent blocked applies (auto-promote decisions that did not reach adoption):

```sql
SELECT d.created_at, d.decision_id, d.payload_json
FROM substrate_mutation_decision d
LEFT JOIN substrate_mutation_adoption a
  ON a.payload_json::text LIKE '%' || (d.payload_json->>'proposal_id') || '%'
WHERE d.payload_json->>'action' = 'auto_promote'
  AND a.adoption_id IS NULL
ORDER BY d.created_at DESC
LIMIT 50;
```

Recent rollbacks:

```sql
SELECT rollback_id, created_at, payload_json
FROM substrate_mutation_rollback
ORDER BY created_at DESC
LIMIT 50;
```

### 5.2 Scheduled Autonomy Safety Posture (single-leader)

Scheduled mutation autonomy is intentionally fail-closed for non-shared control-plane persistence:

- `SUBSTRATE_AUTONOMY_ENABLED=true` requires mutation store posture backed by shared Postgres.
- If unsupported/degraded (for example memory/sqlite fallback), scheduler ticks no-op with structured `substrate_mutation_scheduler` log status `unsafe_mode_noop`.
- Hub startup logs an explicit warning when autonomy is enabled but runtime posture is unsafe.

Live control-surface inspection:

- `GET /api/substrate/mutation-runtime/live-routing-surface`
  - Returns current live value for `routing.chat_reflective_lane_threshold`, including control-surface store source/degraded metadata.
- `GET /api/substrate/mutation-runtime/routing-replay-inspect`
  - Returns routing replay corpus sample, corpus composition (rich-signal coverage), and replay-derived evaluator confidence/metrics for `routing_threshold_patch`.
- `GET /api/substrate/mutation-runtime/routing-live-ramp-posture`
  - Returns current ramp posture for `routing_threshold_patch` (proposals/apply gates, last decision/adoption/rollback, and live routing threshold).
- `GET /api/substrate/mutation-runtime/cognition-context`
  - Returns the mutation-derived context injected into cognition surfaces (routing live threshold, ramp active flags, latest routing proposal/decision/adoption/rollback, evaluator confidence/coverage).
- `GET /api/substrate/mutation-runtime/routing-pressure-sources`
  - Returns recent routing-lane mutation pressure inputs with provenance (`source_kind`, `evidence_refs`, `derived_signal_kind`, confidence) from runtime/social telemetry hints.
- `GET /api/substrate/mutation-runtime/producer-pressure-events`
  - Returns first-class producer pressure events (`source_service`, `source_event_id`, `correlation_id`, category, confidence, evidence refs) grouped by source/category and linked to generated routing mutation signals.
- `GET /api/substrate/mutation-runtime/cognitive-pressure`
  - Returns recent cognitive-lane pressure signals (`contradiction_pressure`, `identity_continuity_pressure`, `stance_drift_pressure`, `social_continuity_pressure`) with provenance/evidence.
- `GET /api/substrate/mutation-runtime/cognitive-proposals`
  - Returns recent cognitive lane proposals (proposal-only / operator-gated).
- `GET /api/substrate/mutation-runtime/cognitive-proposals/<proposal-id>/lineage`
  - Returns full lineage/evidence for a single cognitive proposal.

Routing-only live ramp gates:

- `SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED` (default `true`)
- `SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED` (default `false`)
- `SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED` (default `false`)
- `SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD` (default `-0.05`)

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
```

### Manual UI checklist
- Navigate between **Hub** and **Topic Studio** tabs; ensure no overlays block pointer events on Hub.
- In Topic Studio, run **Preview** with `turn_pairs`, then switch to `conversation_bound` after setting a `boundary_column`.
- Train a run, poll for completion, then load segments and click a segment to confirm full text renders in the detail pane.

Topic Studio relies on the Topic Foundry `/capabilities` endpoint to configure supported segmentation modes and defaults, uses `/runs?limit=20` to populate the recent run picker, and the segments list uses `include_snippet=true&include_bounds=true` with `limit/offset` for faster previews and paging.

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

### 8. Memory cards: how rows get created (Hub vs recall vs auto-extractor)

**Three different paths:**

1. **Recall / RAG (`memory_used`, `memory_digest`, recall canary)** — Context injected into the model at chat time. This does **not** write `memory_cards` rows. Seeing `memory_used=true` in logs or the UI only proves the recall lane ran.
2. **Operator memory cards** — Rows in Postgres (`memory_cards`, usually `pending_review` until approved). Created when an operator uses the Hub Memory tab or calls `POST /api/memory/cards` with a valid body (same `X-Orion-Session-Id` header as the UI). Default `status` is `pending_review`, so new cards show in **Review queue**.
3. **Stage 1 auto-extractor (optional)** — `orion-cortex-orch` subscribes to `orion:chat:history:turn` and, when `ORION_AUTO_EXTRACTOR_ENABLED=true` and `RECALL_PG_DSN` is set on **cortex-orch**, may insert `pending_review` cards from regex candidates (`orion/core/storage/memory_extraction.py`). Default is **off**; Stage 2 LLM extraction remains disabled (`NotImplementedError` if forced). Chat alone with the extractor off does **not** populate the queue — an empty Review queue with `GET /api/memory/cards` returning `200` and `items: []` is expected, not evidence Postgres is broken.

**Quick end-to-end check (seed one card via Hub API):**

```bash
curl -sS -X POST "${HUB_BASE_URL:-http://localhost:8080}/api/memory/cards" \
  -H "Content-Type: application/json" \
  -H "X-Orion-Session-Id: ${ORION_SESSION_ID}" \
  -d '{"types":["fact"],"title":"Seed fact","summary":"Proves memory HTTP + DB path","provenance":"operator_highlight"}'
```

Then open the Hub Memory tab → **Review queue**, or `GET /api/memory/cards?status=pending_review` with the same session header.

**Automated smoke:** `scripts/smoke_memory_cards_e2e.sh` (requires `ORION_HUB_URL`, `ORION_HUB_SESSION_ID`, and `RECALL_PG_DSN` set to the same memory-store contract as Hub).

---

## Recall Strategy Staging + Shadow Ramp (Operator-Only)

Safety invariants:
- Production recall mode remains `v1`.
- Recall strategy changes stay proposal/staging/shadow-only unless an operator explicitly acts.
- `recall_weighting_patch` and `recall_*_candidate` live apply remain blocked by `PatchApplier`.

### Stage a recall proposal into a profile

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/mutation-runtime/recall-strategy-proposals/<proposal_id>/promote-to-staged-profile" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"override": false, "created_by": "operator"}'
```

### Activate staged profile for shadow compare/eval only

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/mutation-runtime/recall-strategy-profiles/<profile_id>/activate-shadow" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"operator_rationale":"advance shadow ramp"}'
```

### Evaluate active shadow profile (dry-run / recording)

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/mutation-runtime/recall-shadow-profile/evaluate" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"dry_run":true,"record_pressure_events":true,"corpus_limit":24}'
```

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/mutation-runtime/recall-shadow-profile/evaluate" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"dry_run":false,"record_pressure_events":true,"operator_rationale":"record latest eval telemetry"}'
```

### Review shadow eval run history

```bash
curl -sS "http://localhost:8080/api/substrate/mutation-runtime/recall-shadow-eval-runs?limit=20"
```

```bash
curl -sS "http://localhost:8080/api/substrate/mutation-runtime/recall-shadow-eval-runs/<run_id>"
```

### Create production-candidate review artifact (operator-only, no production switch)

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/mutation-runtime/recall-strategy-profiles/<profile_id>/create-production-candidate-review" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"override":false,"created_by":"operator","operator_checklist":{"eval_history_checked":true}}'
```

```bash
curl -sS "http://localhost:8080/api/substrate/mutation-runtime/recall-production-candidate-reviews?limit=20"
```

### Inspect profile + posture endpoints

- `GET /api/substrate/mutation-runtime/recall-strategy-profiles`
- `GET /api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}`
- `GET /api/substrate/mutation-runtime/recall-strategy-profiles/{profile_id}/lineage`
- `GET /api/substrate/mutation-runtime/recall-shadow-profile-posture`
- `GET /api/substrate/mutation-runtime/recall-shadow-eval-runs`
- `GET /api/substrate/mutation-runtime/recall-shadow-eval-runs/{run_id}`
- `GET /api/substrate/mutation-runtime/recall-production-candidate-reviews`
- `GET /api/substrate/mutation-runtime/recall-production-candidate-reviews/{review_id}`
- `GET /api/substrate/mutation-runtime/recall-strategy-readiness`
- `GET /api/substrate/mutation-runtime/cognition-context`
- `GET /api/substrate/autonomy-readiness`

### Unified autonomy readiness smoke + interpretation

```bash
curl -sS "http://localhost:8080/api/substrate/autonomy-readiness" | jq .
```

How to interpret quickly:
- `surfaces.live` should only represent routing threshold live surface(s).
- `recall.production_mode` should remain `v1` and `recall.live_apply_enabled` should remain `false`.
- `cognitive.live_apply_enabled` should remain `false` with proposal/draft-only posture.
- `warnings` can be non-empty during partial subsystem outages; endpoint should still return `200`.
- `safe_next_actions` provides bounded operator-safe next steps; it never triggers mutation/apply.

### Manual Recall Canary + Operator Judgment (evidence-only)

Run manual canary query (operator-token guarded, no production mutation):

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/recall-canary/query" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"query_text":"what changed in recall shadow posture today?"}'
```

Inspect canary status rollups:

```bash
curl -sS "http://localhost:8080/api/substrate/recall-canary/status?limit=20" | jq .
```

Record operator judgment for a canary run:

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/recall-canary/runs/<canary_run_id>/judgment" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"judgment":"v2_better","failure_modes":["missing_exact_anchor"],"operator_note":"v2 surfaced anchored card","should_emit_pressure":true,"should_mark_review_candidate":false}'
```

Create review artifact from canary run (evidence/review only):

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/recall-canary/runs/<canary_run_id>/create-review-artifact" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"review_type":"production_candidate_evidence","include_comparison_summary":true,"include_operator_judgment":true}'
```

Safety guarantees for canary workflows:
- No endpoint in this path promotes Recall V2 to production.
- No endpoint in this path switches production recall default away from `v1`.
- No endpoint in this path applies recall mutation patches.
- Canary artifact creation is evidence-only and operator-bounded.

### Selecting a Recall Profile for Canary Testing

Where to select:
- Open Hub `Debug Panel` and find the `Recall Canary` card.
- Use the `Recall profile` dropdown above the manual canary query box.

How profiles load:
- Hub fetches `GET /api/substrate/recall-canary/status` and reads:
  - `data.available_profiles`
  - `data.default_canary_profile_id`
  - `data.production_recall_mode`
  - `data.recall_live_apply_enabled`
- Dropdown options render as `{label} — {status}`.
- Last selected profile is stored in localStorage and reused only when still present in `available_profiles`.

How to run with a selected profile:

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/recall-canary/query" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"query_text":"what changed in recall shadow posture today?","profile_id":"<recall_profile_id>"}'
```

What to inspect in response:
- `data.canary_run_id`
- `data.selected_profile` (selected profile metadata)
- `data.production_recall_mode` (must remain `v1`)
- `data.safety.production_default_unchanged` (must be `true`)

Judgment + review artifact lineage:
- Record judgment as usual with `POST /api/substrate/recall-canary/runs/<canary_run_id>/judgment`.
- Create evidence artifact with `POST /api/substrate/recall-canary/runs/<canary_run_id>/create-review-artifact`.
- Artifact summaries include selected profile metadata so review lineage remains tied to the tested profile.

Confirm production safety posture:
- `production_recall_mode` remains `v1`.
- `recall_live_apply_enabled` remains `false`.
- No production promotion path exists in this UI lane.

Troubleshoot empty dropdown:
- If Hub shows `No recall profiles available for canary testing.`, stage at least one recall strategy profile first.
- Re-check status payload:

```bash
curl -sS "http://localhost:8080/api/substrate/recall-canary/status" | jq '.data.available_profiles, .data.default_canary_profile_id, .data.production_recall_mode, .data.recall_live_apply_enabled'
```

Invalid profile check:

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/recall-canary/query" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"query_text":"what changed in recall shadow posture today?","profile_id":"definitely_not_a_real_profile"}' | jq .
```

### Cognitive Proposal Review Ritual (operator-only, draft/stance only)

Hub UI:
- Open `Debug Panel` and use the visible `Cognitive Proposal Review` modal button.
- The modal is the primary operator surface and is explicitly bounded to review/draft/context actions.
- Safety labels in modal: review/draft/context only, no live cognitive apply, no identity/policy/prompt rewrite.

Inspect cognitive posture:

```bash
curl -sS "http://localhost:8080/api/substrate/cognitive-proposals/status" | jq .
```

Review a proposal as draft-only evidence (no apply):

```bash
curl -sS -X POST "http://localhost:8080/api/substrate/cognitive-proposals/<proposal_id>/review" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"decision":"accept_as_draft","rationale":"bounded operator review","review_labels":["safety_ok"]}'
```

Inspect drafts and create bounded stance note context:

```bash
curl -sS "http://localhost:8080/api/substrate/cognitive-drafts?limit=20" | jq .
curl -sS -X POST "http://localhost:8080/api/substrate/cognitive-drafts/<draft_id>/create-stance-note" \
  -H "content-type: application/json" \
  -H "X-Orion-Operator-Token: $SUBSTRATE_MUTATION_OPERATOR_TOKEN" \
  -d '{"summary":"bounded stance context","note":"non-authoritative operator-approved context","visibility":"metacog_only","ttl_turns":20}'
```

Safety guarantees for cognitive review workflow:
- No endpoint in this path rewrites identity kernel, policy constraints, or production self-model.
- No endpoint in this path invokes mutation execute-once.
- No endpoint in this path performs cognitive live apply.
- Stance notes are non-authoritative bounded context and can be archived.

### Autonomy Constitution / Policy Matrix

The autonomy constitution is now a typed read-only artifact in:
- `services/orion-hub/scripts/autonomy_constitution.py`

It defines policy surfaces, invariants, and safety posture consumed by autonomy readiness.

Hub UI:
- Use `Policy Matrix` button in the `Autonomy Readiness` card to open the read-only constitution modal.

Smoke commands:

```bash
curl -s "http://localhost:8080/api/substrate/autonomy-constitution" | jq .
curl -s "http://localhost:8080/api/substrate/autonomy-readiness" | jq '.policy_matrix'
curl -s "http://localhost:8080/api/substrate/cognitive-proposals/status" | jq .
curl -s "http://localhost:8080/api/substrate/cognitive-proposals" | jq .
curl -s "http://localhost:8080/api/substrate/cognitive-drafts" | jq .
curl -s "http://localhost:8080/api/substrate/cognitive-stance-notes" | jq .
```

Operational refresh:
- `docker compose restart orion-hub`
- or `docker compose up -d --build orion-hub`
- hard refresh browser after template/static JS updates.

### Operator Runbook: Substrate Control Plane

1) Inspect autonomy posture:
- `curl -s "http://localhost:8080/api/substrate/autonomy-readiness" | jq .`
- Confirm `surfaces.live` contains only `routing_threshold_patch`.

2) View policy matrix / constitution:
- `curl -s "http://localhost:8080/api/substrate/autonomy-constitution" | jq .`
- In Hub, use the visible `Policy Matrix` modal launcher in `Autonomy Readiness`.

3) Run recall canary:
- `POST /api/substrate/recall-canary/query` with operator token.

4) Judge recall canary output:
- `POST /api/substrate/recall-canary/runs/<canary_run_id>/judgment`.

5) Create recall review artifact (evidence only):
- `POST /api/substrate/recall-canary/runs/<canary_run_id>/create-review-artifact`.

6) Review cognitive proposals:
- In Hub, use `Cognitive Proposal Review` modal.
- API path: `POST /api/substrate/cognitive-proposals/<proposal_id>/review`.

7) Create bounded stance notes:
- `POST /api/substrate/cognitive-drafts/<draft_id>/create-stance-note`.

8) Forbidden actions (never permitted by this lane):
- cognitive live apply
- recall production promotion/apply
- identity kernel rewrite
- production self-model rewrite
- policy override
- freeform production prompt rewrite

9) Live mutable surface:
- only `routing_threshold_patch` (gated).

10) Routing rollback check:
- inspect `GET /api/substrate/autonomy-readiness` routing + recent activity blocks/rollbacks.
- inspect `GET /api/substrate/mutation-runtime/cognition-context` and `GET /api/substrate/mutation-runtime/routing-live-ramp-posture`.

11) Restart/rebuild Hub after UI changes:
- `PROJECT=orion-athena docker compose -f services/orion-hub/docker-compose.yml restart hub-app`
- if backend script changes: `PROJECT=orion-athena docker compose -f services/orion-hub/docker-compose.yml up -d --build hub-app`

12) Verify modal visibility/reachability:
- `Autonomy Readiness`, `Policy Matrix`, `Cognitive Proposal Review`, `Recall Canary`, `Substrate Review`, and memory/recall debug modal launchers are visible in Debug Panel.

13) Smoke commands:
- `curl -s "http://localhost:8080/api/substrate/autonomy-constitution" | jq .`
- `curl -s "http://localhost:8080/api/substrate/autonomy-readiness" | jq '.policy_matrix'`
- `curl -s "http://localhost:8080/api/substrate/cognitive-proposals/status" | jq .`
- `curl -s "http://localhost:8080/api/substrate/cognitive-proposals" | jq .`
- `curl -s "http://localhost:8080/api/substrate/cognitive-drafts" | jq .`
- `curl -s "http://localhost:8080/api/substrate/cognitive-stance-notes" | jq .`

14) Test-runner troubleshooting:
- local host may not provide `python`/`pytest`; prefer container runner:
- `PROJECT=orion-athena docker compose -f services/orion-hub/docker-compose.yml exec hub-app python3 -m pytest --version`
- run targeted suites with `python3 -m pytest` inside `hub-app`.

15) Reproducible pytest bootstrap (no ad-hoc runtime pip installs):
- Bootstrap local envs (if present): `./scripts/bootstrap_test_envs.sh --service orion-hub`
- This checks `venv` and `orion_dev`, upgrades pip, installs service requirements + root dev deps, and prints python/pytest paths + versions.
- If an env is missing, bootstrap prints a create command and continues.

16) Reproducible Hub pytest in container:
- Hub image now supports gated dev deps with `INSTALL_DEV_DEPS`.
- Dev compose defaults to `INSTALL_DEV_DEPS=true` for `hub-app`.
- Rebuild and verify:
  - `PROJECT=orion-athena docker compose -f services/orion-hub/docker-compose.yml up -d --build hub-app`
  - `PROJECT=orion-athena docker compose -f services/orion-hub/docker-compose.yml exec hub-app sh -lc 'cd /repo && python3 -m pytest --version'`

17) Stable test commands:
- Default known-good suite:
  - `./scripts/test_hub.sh`
- Targeted passthrough usage:
  - `./scripts/test_hub.sh services/orion-hub/tests/test_recall_canary_profile_dropdown.py -q --tb=short`
- Local shared-runner mode (same pytest args, no container exec):
  - `HUB_TEST_RUNNER_MODE=local ./scripts/test_hub.sh services/orion-hub/tests/test_recall_canary_profile_dropdown.py -q --tb=short`
- Global runner directly:
  - `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_recall_canary_profile_dropdown.py -q --tb=short`
- Makefile helpers:
  - `make -C services/orion-hub bootstrap-test-envs`
  - `make -C services/orion-hub test-hub`
  - `make -C services/orion-hub test-hub-substrate ARGS='services/orion-hub/tests/test_recall_canary_profile_dropdown.py -q --tb=short'`
  - `make test SERVICE=orion-hub ARGS='services/orion-hub/tests/test_recall_canary_profile_dropdown.py -q --tb=short'`

Rule:
- Do not use ad-hoc `python3 -m pip install pytest` inside a running container except emergency debugging. Prefer Dockerfile/compose-managed dev deps and the scripts above.

### Recall V2 Battle Test Harness

Purpose:
- Run repeatable canary-only Recall V2 battle queries against real Orion memory pain points using the existing canary query lane.
- Produce operator-review evidence without creating judgments or artifacts automatically.

Safety boundaries:
- Production recall remains `v1`.
- No Recall V2 promotion is performed.
- No default/global profile mutation is performed.
- No mutation execute-once path is invoked.
- No live apply policy changes are performed.

Before running:
- Select a recall profile in Hub `Recall Canary` dropdown, or fetch available profiles via API:

```bash
curl -s "http://127.0.0.1:8080/api/substrate/recall-canary/status" \
  | jq '.data.available_profiles, .data.default_canary_profile_id, .data.production_recall_mode, .data.recall_live_apply_enabled'
```

Battle fixture:
- `services/orion-hub/tests/fixtures/recall_canary/orion_memory_battle_cases.json`

Run with default canary profile:

```bash
python3 services/orion-hub/scripts/run_recall_canary_battle.py \
  --base-url http://127.0.0.1:8080 \
  --fixture services/orion-hub/tests/fixtures/recall_canary/orion_memory_battle_cases.json \
  --output /tmp/recall_battle_runs.jsonl
```

Run with explicit profile:

```bash
python3 services/orion-hub/scripts/run_recall_canary_battle.py \
  --base-url http://127.0.0.1:8080 \
  --profile-id <profile_id> \
  --fixture services/orion-hub/tests/fixtures/recall_canary/orion_memory_battle_cases.json \
  --output /tmp/recall_battle_runs_explicit.jsonl
```

Behavior:
- Loads fixture cases.
- Fetches `/api/substrate/recall-canary/status`.
- Uses `--profile-id` if provided, else `default_canary_profile_id`.
- Validates selected profile before posting cases.
- Posts only to `/api/substrate/recall-canary/query`.
- Prints case-level rollup table and overall summary.
- Writes JSONL output when `--output` is supplied.

Operator review workflow:
- Runner does not auto-judge or auto-create review artifacts.
- Use existing operator workflow after reviewing output:
  - `POST /api/substrate/recall-canary/runs/<canary_run_id>/judgment`
  - `POST /api/substrate/recall-canary/runs/<canary_run_id>/create-review-artifact`

Common failure modes to watch:
- `missing_exact_anchor`
- `irrelevant_semantic_neighbor`
- `stale_memory`
- `unsupported_memory_claim`
- `wrong_project`
- `wrong_timeframe`
- `insufficient_context`

### Staging a Recall Canary Profile

Why profile catalog can be empty:
- `available_profiles` is sourced from staged/shadow recall strategy profiles in mutation store.
- Fresh runtime state (or cleared sqlite/postgres control-plane store) can yield no canary profiles.

Seed a bounded default canary profile (idempotent):

```bash
python3 services/orion-hub/scripts/seed_recall_canary_profile.py --profile-id recall_v2_shadow_default
```

Verify canary status after seeding:

```bash
curl -s "http://127.0.0.1:8080/api/substrate/recall-canary/status" \
  | jq '.data.available_profiles, .data.default_canary_profile_id, .data.production_recall_mode, .data.recall_live_apply_enabled'
```

Using Hub dropdown after seeding:
- Open Hub `Debug Panel` -> `Recall Canary`.
- `Recall profile` dropdown should now contain seeded profile option(s).

Run battle harness with default profile:

```bash
python3 services/orion-hub/scripts/run_recall_canary_battle.py \
  --base-url http://127.0.0.1:8080 \
  --fixture services/orion-hub/tests/fixtures/recall_canary/orion_memory_battle_cases.json \
  --output /tmp/recall_battle_runs.jsonl
```

Run battle harness with explicit profile:

```bash
python3 services/orion-hub/scripts/run_recall_canary_battle.py \
  --base-url http://127.0.0.1:8080 \
  --profile-id recall_v2_shadow_default \
  --fixture services/orion-hub/tests/fixtures/recall_canary/orion_memory_battle_cases.json \
  --output /tmp/recall_battle_runs_explicit.jsonl
```

Safety checks:
- Confirm `production_recall_mode` remains `v1`.
- Confirm `recall_live_apply_enabled` remains `false`.
- Seeding is canary/review-only and does not perform promotion or live apply.

Safety guarantees for this flow:
- Production recall remains `v1`; no endpoint in this workflow switches production default.
- Candidate review creation only persists operator review artifacts.
- Recall live apply stays blocked by mutation apply guardrails.

### Manual Recall Canary Console in Hub

Where to open:
- Open Hub `Debug Panel` and click `Recall Canary` -> `Modal`.

Operator workflow:
- Select a `Recall profile` from canary/shadow-only options loaded from `GET /api/substrate/recall-canary/status`.
- Hub uses the server-configured operator token automatically for modal actions.
- Enter a manual query and click `Run Canary Query`.
- Review `canary_run_id`, selected profile metadata, `production_recall_mode`, `recall_live_apply_enabled`, and V1/V2/comparison summaries.
- Set judgment (`v2_better`, `v1_better`, `tie`, `both_bad`, `inconclusive`), select failure modes, add notes, and click `Submit Judgment`.
- Click `Create Review Artifact (Evidence Only)` only when explicitly needed.

Troubleshooting:
- Missing controls: hard refresh browser and reopen the Recall Canary modal.
- Missing profile options: seed/stage canary profile and verify status payload has `available_profiles`.
- Missing/invalid token: ensure `SUBSTRATE_MUTATION_OPERATOR_TOKEN` is configured in Hub runtime.
- `mutation_operator_token_not_configured`: configure backend runtime token and restart `hub-app`.

Safety posture:
- Production recall remains `v1`.
- Selected profiles remain shadow/canary review-only.
- No auto-judgment, no auto artifact creation, no promotion path, and no live apply path are added by this UI.

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
