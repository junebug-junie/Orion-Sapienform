# 🌀 Orion Hub — Titanium Edition

**Version:** 0.4.x  
**Stack:** Python · FastAPI · WebSocket · Tailwind · Orion Bus (Async/Redis)

---

## 📖 Overview

**Orion Hub** is the browser gateway into the mesh.

It is a **"Dumb" UI** that:

- Captures **voice** and **text** from the browser (PCM → WAV over WebSocket; see **Voice debugging** below)
- Maintains lightweight UI state (history, mode, visualizers)
- Publishes strictly typed **Titanium Contracts** onto the **Orion Bus**
- Waits for answers from downstream workers:
  - **Cortex Gateway** (for all chat/cognition)
  - **TTS Service** (for speech synthesis)

From Hub’s perspective:

> “I don’t know about LLMs, Agents, or RAG. I just send a `CortexChatRequest` to the Gateway.”

Hub also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of its main chat/cognition bus connection.

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

#### Kids story lane (verb override)

In the main chat UI, use the **Story** mode button (bounded fast lane, `chat_kids_story`) next to **Quick**, or send a single explicit verb: `verbs: ["chat_kids_story"]` with normal `messages` / recall options.
Verb must be active on the hub node (`orion/cognition/verbs/active.yaml`). Default recall profile is
`chat.story.kids.v1` (vector-off; SQL + timeline + optional cards) unless the client sets `profile_explicit`.

#### Repair pressure — pre-turn appraisal (v2)

When `ENABLE_PRE_TURN_APPRAISAL=true`, Hub appraises repair pressure **before** the cortex chat call using logprob probes on cortex-exec (not the legacy post-turn `phrase_match` substrate effect pipeline).

**Flow**

1. HTTP (`handle_chat_request`) or WebSocket chat handler calls `run_pre_turn_appraisal_wiring()`.
2. Hub builds a paired turn window (`build_turn_window`) and RPCs cortex-exec via `PreTurnAppraisalClient`.
3. Bus request: `orion:cortex:pre_turn_appraisal:request` → reply `orion:cortex:pre_turn_appraisal:result:{correlation_id}`.
4. Hub applies `TurnAppraisalBundleV1`: attaches `repair_pressure_contract` metadata when mode changes and returns `substrate_effect_summary` for the UI chip.
5. When v2 is enabled, `run_substrate_effect_pipeline()` is **skipped** (legacy phrase_match does not run).

**Operator gate (single flag on Hub)**

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_PRE_TURN_APPRAISAL` | `false` | Master enable for pre-turn appraisal v2. |
| `PRE_TURN_APPRAISAL_PARADIGMS` | `repair_pressure` | Comma-separated paradigm names (resolved on cortex-exec via `PARADIGM_REGISTRY`). |
| `PRE_TURN_APPRAISAL_TIMEOUT_MS` | `60000` | RPC timeout (ms); logprob probe needs LLM headroom. |
| `CHANNEL_PRE_TURN_APPRAISAL_REQUEST` | `orion:cortex:pre_turn_appraisal:request` | Bus request channel. |
| `CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX` | `orion:cortex:pre_turn_appraisal:result` | Reply channel prefix (`:{correlation_id}` appended). |

Cortex-exec always listens on the pre-turn channels; there is no separate cortex-exec enable flag.

**Speech contract overlay (separate flag, default on)**

`ENABLE_REPAIR_PRESSURE_SPEECH_WIRING=true` (Hub + cortex-exec) merges `repair_pressure_contract` metadata into the TURN CONTRACT on the same turn. Pre-turn v2 can attach that metadata; speech wiring controls whether cortex-exec compiles it into `chat_general.j2`.

**Enable**

```bash
# After syncing from .env_example (repo root):
python scripts/sync_local_env_from_example.py orion-hub orion-cortex-exec

# services/orion-hub/.env
ENABLE_PRE_TURN_APPRAISAL=true
PRE_TURN_APPRAISAL_PARADIGMS=repair_pressure
PRE_TURN_APPRAISAL_TIMEOUT_MS=60000
```

Restart Hub and cortex-exec after changing env.

**Rollback**

Set `ENABLE_PRE_TURN_APPRAISAL=false` on Hub. Legacy phrase_match substrate effect pipeline resumes on the next chat turn.

**Tests**

```bash
cd services/orion-hub
pytest tests/test_pre_turn_appraisal_wiring.py tests/test_handle_chat_request_substrate_effect.py -q
```

**Code**

- `scripts/pre_turn_appraisal_client.py` — bus RPC client
- `scripts/pre_turn_appraisal_wiring.py` — turn window + bundle apply
- `scripts/api_routes.py`, `scripts/websocket_handler.py` — chat integration
- `scripts/substrate_effect_pipeline.py` — legacy skip when v2 enabled

#### Unified-turn chat grammar trace (stance disposition)

`orion/hub/turn_orchestrator.py::execute_unified_turn` (the pipeline `run_unified_turn` routes into when `client_mode == "orion" and ORION_UNIFIED_TURN_ENABLED=true`) publishes the same `hub.chat:` `GrammarEventV1` trace the classic `websocket_handler.py` chat path already produces (session context, utterance word count, repair signal) -- extended with the Thought stance decision (`proceed`/`defer`/`refuse`, `disposition_reasons`, `boundary_register`), a fact unified turn has that the classic path never did. Both paths feed the same `chat_grammar_consumer` reducer on substrate-runtime; whichever pipeline actually produced a given turn, its facts land in the same `active_chat_session` projection.

Gated by the existing `PUBLISH_HUB_CHAT_GRAMMAR` flag (default `true`) -- no new env key. Fires once per turn, right after the stance decision resolves, whether or not the turn goes on to the harness governor. Fail-open: a publish failure is logged and swallowed, never raised into the chat response.

**Note:** as of the turn-incompletion liveness marker below, `PUBLISH_HUB_CHAT_GRAMMAR` also gates that unrelated operational signal (no user content, just `correlation_id` + timeout status) -- flipping this flag off for chat-privacy reasons also silences turn-incompletion telemetry, and there is currently no way to control them independently. Flagged in code review as a real (non-blocking) coupling, not yet split into a separate flag.

The stance fields land in `active_chat_session` and go no further today -- `compute_chat_pressure_hints` doesn't read them, so they never reach `SelfStateV1` or phi. Registered `REHEARSAL` in `orion/self_state/inner_state_registry.py` (`chat_stance_disposition`) rather than left implicit; see `docs/superpowers/specs/2026-07-13-stance-disposition-inner-state-path.md` for why the obvious composition route (into `SelfStateV1.social_pressure`) was rejected and what the real paths forward look like.

**Code**

- `orion/hub/turn_orchestrator.py::_publish_unified_turn_chat_grammar` — call site, builds the stance-aware event set
- `scripts/grammar_emit.py::build_chat_turn_grammar_events` — pure builder, `stance_disposition`/`stance_disposition_reasons`/`stance_boundary_register` params
- `orion/substrate/chat_loop/grammar_extract.py::extract_chat_turn_state` — reducer-side parsing into `ChatTurnStateV1.stance_disposition` etc.

#### Turn-incompletion liveness marker

`orion/hub/turn_orchestrator.py::execute_unified_turn`'s `run is None` branch fires when
`HarnessGovernorClient.run()` doesn't return a decoded `HarnessRunV1` -- most commonly a true
harness-governor RPC timeout, but this branch also covers a codec decode failure on an
otherwise-received reply (rare; in that sub-case the governor's own reply did arrive and may
already have flushed real lifecycle grammar on its own trace lane, so treat the marker as "Hub
gave up / couldn't use the reply," not strictly "nothing happened on the governor side"). This
is otherwise the one unified-turn failure mode where no governor-side grammar event exists at
all -- `HarnessGrammarCollector` only flushes its buffered lifecycle atoms once, at the end of a
run, and a true timeout never reaches that point.

Publishes a single `GrammarEventV1` (`semantic_role="exec_turn_timeout"`) under its own trace
lane (`cortex_exec_trace_id(NODE_NAME, correlation_id, lane="hub_turn_timeout")`) rather than the
governor's `harness_motor` lane -- Hub can't reliably know which physical node the governor's own
`HarnessGrammarCollector` would have used when its RPC never returned. `correlation_id` is set
explicitly on the event rather than left to trace-id parsing, avoiding the lane-suffix-pollution
bug class already fixed once for the harness/cortex-exec producers. Required widening
`EXECUTION_SOURCE_SERVICES` (`orion/substrate/execution_loop/constants.py`) to include
`orion-hub`. Consumed by `orion-field-digester`'s `turn_incompletion` channel -- see that
service's README glossary. Same `PUBLISH_HUB_CHAT_GRAMMAR` gating and fail-open shape as the
chat-narrative trace above (see the Note in that section).

**Code**

- `orion/hub/turn_orchestrator.py::_publish_turn_timeout_grammar` — call site
- `scripts/grammar_emit.py::build_turn_timeout_grammar_events` — pure builder
- `orion/substrate/execution_loop/grammar_extract.py` — reducer-side `exec_turn_timeout` role parsing into `ExecutionRunStateV1.turn_timed_out`

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

### 4.1 Endogenous outreach — Orion speaks first

`scripts/endogenous_outreach.py`. The only path by which Hub emits chat text
nobody asked for. **Enabled** (`HUB_ENDOGENOUS_OUTREACH_ENABLED=true` in
`.env_example` and the live `.env`). The `settings.py` Field default is still
`False`, so a deploy that loses the key fails closed rather than silently
reaching out.

**The trigger is real** (2026-08-16, replacing a randomized-timer stub).
Every `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` (default **10s**, see caveat below)
the loop asks `scripts/tension_outreach_trigger.py::current_run()`: has the
same node been winning `orion.attention.tension`'s live Borda competition for
a sustained, unbroken run of real ticks right now — not a single blip. See
that module's own docstring for the full account, including why the
persistence bar (`HUB_ENDOGENOUS_OUTREACH_MIN_RUN_LENGTH`, default **6** as
of a 2026-08-22 recalibration; was 8) is derived from a real replay of live
history rather than guessed — it stays operator-tunable from real
post-deploy firing-rate data, unlike the trigger's other internals. The message itself is generated from live substrate signals
and real chat history, and lands on the same rails a normal turn uses — that
part never changed.

**Poll cadence root-caused live, 2026-08-19** — Orion had never once reached
out since this shipped. Two stacked causes: (1) the trigger's own query was
broken 2026-08-16→2026-08-18 (the `make_interval` bug PR #1715 fixed — see
that module's docstring); (2) even after that fix, the original 300s poll
interval almost never observed a qualifying run — a real run lasts only
~18-27s wall-clock, and replaying 6h of real history against the actual poll
loop caught 0 of 9 real qualifying episodes at 300s. `HUB_ENDOGENOUS_
OUTREACH_TICK_SEC` was lowered to 10s (data-derived episode catch rate ~33%
on that sample) as a deliberate middle ground, not the class's own 5.0s
floor (~56% catch) — the query is cheap, but this shares one Postgres
instance with every other service. **Disclosed, not silently accepted: this
remains a partial fix** — even the floor tops out around ~56% catch on the
measured sample, because several real episodes' catchable window is under 5
seconds. Full closure needs a wall-clock-persistence redesign ("was there a
qualifying run since last checked", not "is one happening right now"), not
done here — see `docs/superpowers/specs/2026-08-16-tension-driven-outreach-
design.md`'s "Poll-cadence root cause" section for the full replay numbers.

**Level-aware, not just change-aware** (2026-08-19). The trigger's reason
object now also carries `sustained_load_pressure`
(`orion.field.significance`, PR #1718) — the LATEST tick's read of whether
*something, somewhere* in the field is genuinely `loaded_steady` right now
(high level, low dispersion, no adaptive baseline), globally, not scoped to
the same node the deviation run names. When it's nonzero,
`build_outreach_prompt` states it as a second, separate real fact alongside
the deviation-run fact — it does not put a feeling ("worried"/"concerned")
in Orion's mouth; that judgment is left to generation, grounded in both real
numbers. `GET .../status` reports both `peak_deviation_pressure` and
`sustained_load_pressure` on `last_tension_reason` so an operator can see
which fact(s) actually drove a given outreach.

**A daydream, not only telemetry (2026-08-28).** Every other grounding lane
above is an instrument reading, so an unprompted message could only ever be
Orion narrating its own dials. `_fetch_current_daydream` adds the one lane
that is not: the caption orion-thought's *reverie visual chain* writes to
`reverie_visual_chain` (~1 row/600s — it generates an image from whatever
Orion is currently thinking/noticing/remembering, then looks at what came
out). The newest **usable** caption inside a **12h** window goes into the
prompt with a coarse relative age, explicitly framed as Orion's own so the
generated text cannot thank Juniper for a picture she never sent.

**One caption, not a list — and that is a measured retraction, not caution.**
The first version of this lane shipped the last 3 *distinct* captions,
de-duplicated by Jaccard token overlap at 0.2. Measured against all 328 live
rows, that mechanism does not work and no threshold fixes it:

* Consecutive captions re-describe **one** image, so they differ mostly in
  *length*. Jaccard divides by the union, which penalises exactly that: two
  17th-century celestial maps measured **0.150**, under the threshold, so
  both rendered.
* The containment coefficient corrects that length bias and is *worse* — at
  0.4 it surfaced three map captions.
* Eyeballing sampled `(newest, next-distinct)` pairs across the corpus, both
  variants returned obvious duplicates (two Roman aqueducts; two 17th-century
  star charts).

The producer already knows this. `visual_chain.py`'s Patch 4 exists because
Juniper reported *"still doing the same images of Roman aqueducts, no
change"* on 2026-08-27: `prior_description` continuity locks onto a visual
attractor for 10+ runs. Presenting "your last 3 daydreams" over a corpus that
lands on attractors is a claim the data cannot support (AGENTS.md §0A). The
newest usable caption needs no such claim, and it cut the lane from 38% to
23% of the prompt — it was previously the largest block in it, for the lane
that is explicitly the least load-bearing.

`chain_json.continuity_streak`/`continuity_reset` were evaluated as a
ready-made theme boundary and **rejected**: live they run a rigid `3 2 1 0`
period-4 cycle, because `resolve_visual_chain_continuity` forces a reset
every `visual_chain_continuity_max_runs` runs unconditionally. That is a
mechanical cap, not a signal that the imagery changed. Showing genuine
*drift* ("celestial maps for two hours; Roman aqueducts before that") would
need a real theme detector — embeddings, not bag-of-words — and is left as
follow-up rather than faked.

**Caption validity is load-bearing, because only one caption ships.**
`_looks_like_daydream_prose` rejects three live failure modes of the vision
model, and `_strip_appended_list` repairs a fourth. All four are **producer
bugs worth fixing in orion-thought**; these are consumer-side guards, not the
fix. To re-measure any count below, run the eval — do not trust a number
written here, the table grows ~6 rows/hour:

| shape | example | handling |
|---|---|---|
| raw grounding output | `objects(103,419),(554,604)`, `bridge(269,261),(879,661)` | reject |
| bare tag dump | `1. Sun 2. Mercury 3. Venus …`, `two trees, lake, reflection, purple sky` | reject |
| second-person address | `The graph you provided is a phase diagram…` | reject |
| appended instruction echo | `a spiral galaxy. Directly visible objects and people include: 1. **Galaxy**: …` | **truncate** |

The last one was 12 of 290 rendered captions (4.1%) — the vision prompt's own
instruction text echoed back with literal markdown and a dangling enumerator.
The prose *before* it is genuinely good, so the tail is cut rather than the
caption dropped; when nothing usable precedes the list, the length and prose
checks reject what remains. `_DAYDREAM_LIST_START_RE` matches the structural
markers (a literal `**`, or a colon followed by an enumerator) rather than the
instruction wording, which is the captioner's to change.

The rejects have no false positives on the live corpus, including a short
18-word real caption that a naive alphabetic-character-ratio test wrongly
drops, and a real caption ending *"…are directly visible."* that a naive
instruction-echo test wrongly drops — the latter caught by the eval on its
first run. `_DAYDREAM_SCAN_LIMIT` is 12 rather than 1 for the same reason: ~3%
of rows are debris and ~10% have a NULL caption, so "newest row" is often not
"newest usable row".

Two more deliberate choices worth not undoing. The whitespace collapse in
`_clean_daydream` is an **injection guard**, not prompt-shape hygiene — this
is model-generated text interpolated into a prompt, and flattening newlines
is what stops a caption forging its own prompt line. And the lane reads
`chain_json->>'description'`, **not** the typed `prior_description` column
that looks like the obvious simplification: `visual_chain.py:527` sets
`prior_description = description or continuity_fallback`, so on a
caption-failure row it carries the *previous* run's caption forward and would
silently re-surface a stale daydream as current.

Three columns on that table were checked and **deliberately not used** —
since the chain went live on 2026-08-25, `theme_key` is NULL on every row
(`count(theme_key) = 0`; the producer never sets it), `ema_salience` is
exactly `0.000`, and `terminal_reason` is always `'max_steps'`. They carry no information today. Like `embodied_presence`, the
daydream is enrichment only and is **not** part of `is_empty()`: having been
daydreaming is never on its own a reason to interrupt Juniper.

#### Evals

`services/orion-hub/evals/` is this service's first eval directory. It exists
because of a specific mistake: the daydream lane's original de-duplication
shipped a calibration claim that live data falsified the same day. A unit test
could not have caught it — unit tests run on fixtures the same reasoning
invented. This runs the real caption pipeline over the real rows.

```bash
scripts/test_service.sh orion-hub --with-evals
# or directly:
pytest services/orion-hub/evals -q
DAYDREAM_EVAL_DATABASE_URL=postgresql+psycopg2://... pytest services/orion-hub/evals -q
```

Read-only, and it **skips** cleanly with no reachable database — but a missing
or renamed `reverie_visual_chain` **fails** rather than skipping, since that is
the loudest thing that can go wrong with this lane.

It measures the caption usability rate over a **24h window** (not the lifetime
of the table: a lifetime rate cannot detect a later failure — at ~6 rows/hour a
0.85 floor trips after ~7h of total collapse on today's corpus but would need
~10 days once the table reaches 10k rows), asserts that real debris present in
the raw captions does not survive cleaning, asserts no rendered caption echoes
the captioner's instructions, checks the window is not empty (liveness), and
pins its own two thresholds so they cannot be quietly set to values that make
everything pass.

**Repo-wide gap this exposed:** 11 services carry an `evals/` directory and
none were reachable through `scripts/test_service.sh` or any Makefile target.
The `--with-evals` flag above is new and opt-in; wiring the other ten is
follow-up, not this change.

**Through the real unified turn, not a lookalike (2026-08-19).** Generation
used to call `CortexGatewayClient.chat()` directly — a bare bus RPC to
`orion-cortex-gateway` that never reached `orion-harness-governor` at all:
no fcc motor, no substrate appraisal/reflect/voice finalize beats, no
post-turn learning closure, no audit artifact, and (root-caused the same
day) a verb-less default (`chat_general`) whose hidden internal
`synthesize_chat_stance_brief` pre-step sent a ~15.7k-char system prompt
while requesting 8000 completion tokens on the `quick` route with no larger
fallback lane — overflowing context on every single attempt. Juniper's call
once this was traced: "if orion is going to reach out to me, it needs to be
real and not bullshit." Generation now calls
`orion.hub.turn_orchestrator.execute_unified_turn` — the SAME function
`websocket_handler.py` calls for a real `client_mode == "orion"` turn, not
a cheaper substitute. This is a bigger change than swapping which client
generates text: `execute_unified_turn` records the outreach prompt into
Orion's real observation stream (`emit_observation()`) and runs it through a
real `ThoughtClient.react()` stance evaluation that can `defer`/`refuse` the
turn — accepted deliberately, since there is no shallower entry point
(`HarnessRunRequestV1` requires a `thought_event`), and Thought's own
defer/refuse is now the honest "something else is happening" signal for an
unsolicited turn, not just an infrastructure-availability check. Permissions
are no longer this module's decision either: `execute_unified_turn`
hard-codes every unified turn's `ContextExecPermissionV1` to read-only
(every write/mutate/network/shell flag stays `False`), stricter than the
old direct-call path's own options ever were. `payload={"no_write": True}`
suppresses only the governor's OWN chat-history persistence — this module's
three delivery rails below (unchanged) remain the sole persistence path, so
an outreach turn keeps its `endogenous_outreach` tag instead of landing as
an untagged duplicate. `HUB_ENDOGENOUS_OUTREACH_LLM_ROUTE` is gone (killed,
not deprecated) — route selection is the harness governor's call now,
identically for outreach and real chat.

Pipeline:

```text
tick -> gates -> grounding read -> execute_unified_turn (real observation +
  Thought stance + harness governor + fcc motor + finalize chain) -> 3 delivery rails
```

Grounding (a tick with none of this is skipped, never filled with placeholder
text — AGENTS.md §0A):

- fresh endogenous-curiosity candidates from
  `substrate_endogenous_curiosity_candidates` (via `curiosity_hint._fetch_fresh_candidates`,
  widened to a 1h window for this slower cadence)
- `hub_presence.presence_snapshot()` — how long since the last real turn
- the last few turns from `chat_history_log`, scoped to the live session

Orion may also decline: the prompt permits a literal `PASS` reply, which is
dropped without consuming the daily budget.

Delivery — three rails, all pre-existing:

| Rail | Mechanism | What it's for |
| --- | --- | --- |
| Live chat bubble | in-process push to each socket's `tts_q`, `{"kind":"orion_outreach"}` | a browser that is open right now |
| Chat history | `chat.history.message.v1` on the bus, `role=assistant`, tag `endogenous_outreach` | durability as data (see caveat) |
| Notification | `HubNotificationEvent` on `NOTIFY_IN_APP_CHANNEL` | a browser opened afterwards |

**Reload caveat:** rail 2 makes the outreach durable *as data* only. Hub's
frontend has no conversation-restore fetch at all — the only history-shaped
endpoint, `/api/chat/messages`, has zero callers in `app.js` — so a reload does
not bring the bubble back. Rail 3 (the notification list) is what a returning
browser actually sees. Giving the UI a real restore path is separate work.

The frontend handles `orion_outreach` in its own early-return branch
(`static/js/app.js`) rather than falling through to the generic assistant
branch — that branch also runs `updateMemoryPanelFromResponse()`, which would
blank the recall panel still showing the last real turn. It also costs the
frame's piggybacked biometrics snapshot, which is harmless: `biometrics_heartbeat`
re-pushes every `BIOMETRICS_PUSH_INTERVAL_SEC` (default 5s). `addNotification`
suppresses the toast for `notification_type == endogenous_outreach`, since the
same `tts_q` carries both rails and the bubble already showed the text.

**Single-process assumption:** the live-bubble rail is in-process. Hub runs one
uvicorn worker (`Dockerfile` CMD has no `--workers`). If that changes, this rail
must move onto the bus like the other two already are.

Safety gates, checked in this order (first hit wins, reported by the status
endpoint): `disabled`, `turn_in_flight`, `quiet_hours`, `daily_cap`, `cooldown`.
Three details that are easy to get wrong and are enforced by tests:

- **`turn_in_flight` reads a per-connection `busy` flag, not just
  `active_turn`.** The ws handler only populates `active_turn["correlation_id"]`
  for the unified-`orion` and `agent-claude` lanes; the UI's **Quick, Story, and
  Agent** modes all fall through to the general cortex path and never touch it.
  `note_busy()`/`note_idle()` are called for every inbound message regardless of
  mode.
- **The gate is re-checked immediately before delivery.** Generation is a bus
  RPC bounded by `HUB_ENDOGENOUS_OUTREACH_TIMEOUT_SEC` (default 300s — raised
  from 60s 2026-08-19, see "Through the real unified turn" above), and a turn
  can start inside that window. A tick that becomes blocked mid-generation is
  dropped with reason `<gate>_after_generation`.
- **Quiet hours and the daily-cap reset use `HUB_ENDOGENOUS_OUTREACH_TZ`,** not
  the container clock. Hub's compose and Dockerfile set no `TZ`, so the process
  timezone is UTC; this key must name the operator's real zone or the window
  silences the wrong nine hours. Set to `America/Denver` in `.env_example`, so
  the 23→08 window is 23:00–08:00 Mountain. Getting this wrong is not cosmetic:
  under the old `UTC` value the same window silenced 17:00–02:00 Mountain — the
  whole evening — while leaving outreach open across the working day.

Generation is read-only by construction (2026-08-19), not by an option this
module sets: `execute_unified_turn` hard-codes every unified turn's
`ContextExecPermissionV1` with every write/mutate/network/shell flag at its
safe `False` default — true for real turns too, not a special carve-out for
outreach. `payload={"no_write": True}` still gets set explicitly, but only to
suppress the governor's own chat-history persistence step so this module's
own tagged rails stay the sole persistence path (see "Through the real
unified turn" above).

Every failure is swallowed and logged; no chat turn, websocket, or bus consumer
is affected.

Operator surfaces:

```bash
curl -fsS http://localhost:8080/api/debug/endogenous-outreach/status | jq
curl -fsS -XPOST http://localhost:8080/api/debug/endogenous-outreach/trigger | jq
```

`trigger` skips **only** the random roll. Every safety gate still applies —
including `disabled`, deliberately: this router carries no auth dependency, so a
`force` carve-out would let one unauthenticated POST undo "off by default". To
test, flip `HUB_ENDOGENOUS_OUTREACH_ENABLED` and restart.

### 4.2 Curiosity investigation — Orion's own time, and its own graph

`scripts/curiosity_investigation.py`. Code decides only **when** Orion gets
time; Orion decides what to do with it, inside a real
`execute_unified_turn`. Nothing in the loop names a subject.

> This section covers the loop **from Hub's side** — the gates, the wiring, the
> addresses. The program itself (what a prior is, why Orion gets a graph nobody
> curates, where priors come from, and what this does not establish) lives in
> `orion/curiosity/README.md`. `orion/sentience_striving_program/README.md` §15
> evaluates it against that program's own outcomes.

#### When it runs

`HUB_CURIOSITY_INVESTIGATION_DAILY_CAP` is a **budget**, not a pace. It is
keyed on the operator's local date, so it frees at local midnight and — with
nothing else in the way — the loop spends the whole day as fast as the cooldown
allows. Measured live on 2026-08-28: all six of that day's runs fired between
00:48 and 02:57 MDT, followed by 240 consecutive ticks logging
`blocked reason=daily_cap` through the entire day Juniper was awake to watch
them.

`HUB_CURIOSITY_INVESTIGATION_WINDOW_START_HOUR` / `_END_HOUR` (in
`HUB_ENDOGENOUS_OUTREACH_TZ`, range `-1..23`) spread the budget across the
hours Orion is allowed to think. The gap between runs is **derived** —
`window / cap`, floored by `MIN_COOLDOWN_SEC` — so one knob sets both how much
Orion thinks and how often, and raising the cap cannot rebuild the 3am cluster.
At `8`/`22` with a cap of 6 that is one run every 2h20m: 08:00, 10:20, 12:40,
15:00, 17:20, 19:40, with the seventh landing exactly on the window close.

Equal values or `-1` disable the window, the same convention as
`HUB_ENDOGENOUS_OUTREACH_QUIET_*`, and restore the previous behaviour exactly.
A manual run (`POST /curiosity/api/run-now`) overrides the window along with
the cooldown and the cap.

Two limits, both deliberate:

- **A disabled cap disables the pacing.** With `DAILY_CAP=-1` there is no
  budget to spread, so spacing falls back to `MIN_COOLDOWN_SEC` alone. The
  window still keeps runs out of the small hours; it is the cap that makes them
  rare.
- **A day that starts late ends early.** Spacing is measured from the last run,
  not anchored to window open, so a first run at 15:00 with a cap of 6 fires
  three times before 22:00 and strands the other three. Anchoring instead would
  mean either a catch-up burst at 08:00 — the clustering this removes — or a
  schedule that ignores how long turns actually take.

If `HUB_ENDOGENOUS_OUTREACH_TZ` fails to load, the window is **disabled** rather
than evaluated in UTC: a guessed local hour would run Orion at the wrong hours
while the config insisted it was bounded. Look for `curiosity_bad_timezone` in
the logs. The startup line reports the pace it actually resolved:

```text
curiosity_investigation started tick=300.0s cooldown=8400s(floor=1800.0s) cap=6 window=08-22 America/Denver ...
```

**What it shows Orion.** Its own open priors (ordered by how uncertain *it*
said it was, and the prompt says the ordering is not neutral), a random sample
of Juniper-approved crystallizations and concept-induction judgements, what it
has recently settled, and — if the previous run left one — the note that run
wrote to itself.

**What Orion can reach inside the turn.** Real credentials against real stores,
named in the prompt as possible and never as required:

| store | credential | access |
|---|---|---|
| `memory_crystallizations`, `memory_concept_relation_decisions`, `chat_history_log`, `journal_entries` | Postgres role `orion_readonly` | `SELECT` only, those four tables only |
| `orion_substrate` (the Concept Atlas) | FalkorDB ACL user `orion_curiosity` | `GRAPH.RO_QUERY` only |
| `orion_worldview` (**Orion's own graph**) | same ACL user, via a selector | `GRAPH.QUERY` — read **and** write |

The boundary is enforced by the databases, not by a wrapper: as
`orion_curiosity`, a write to the Atlas is refused twice over (the key ACL, and
`GRAPH.RO_QUERY` refusing a write command). The credentials reach the
`claude -p` sandbox through an allowlist out of `~/.fcc/.env`
(`orion/curiosity/sandbox_env.py`); removing a key from that file is the kill
switch.

**What accumulates.** Orion writes `:Prior`, `:Concept`, `:Finding`, `:Hop` and
`:TurnOutcome` nodes into `orion_worldview` itself, in Cypher, in-turn. Hub only
ever **reads** it back — every Hub query goes out as `GRAPH.RO_QUERY`. Nothing
Orion puts there needs approval.

- **Priors** are claims with a confidence and a status. A prior tested
  `HUB_CURIOSITY_STALE_PRIOR_TESTS` times without its status moving leaves the
  main list and is offered separately, with retiring it named as a real
  outcome — so one claim cannot be re-litigated forever.
- **Hops** are up to `HUB_CURIOSITY_MAX_HOPS` stopping points, recorded as they
  happen rather than reconstructed at the end, so the journal entry can recount
  the path actually taken.
- **`:TurnOutcome`** is how a decision made *inside* the turn crosses back out:
  `continue_line`/`continue_note` open the next run, `reach_out` asks for a
  message to Juniper. **Absence is the safe default** — no node means no
  continuation and no message. Nothing is inferred from the prose.

**Gates, in order.** `disabled` → `daily_cap` → `cooldown` → `pg_role_missing`
→ `graph_unavailable` → `stores_not_ready` / `stores_unavailable` /
`no_approved_material` → `empty_generation` / `no_lookup`. `stores_not_ready`
(the pool has not finished starting) is deliberately separate from
`stores_unavailable` (it could not be read): the first is a sub-second race at
Hub startup and logs at INFO, escalating to WARNING if it outlives one tick. The last one is load-bearing: a turn with
fewer than `MIN_HARNESS_STEPS` harness steps did not look anything up, and its
fluent prose is refused rather than journalled.

**The ACL is re-asserted before every run**, not just at startup. `aclfile` is
unset *and* immutable on this FalkorDB (`CONFIG SET aclfile` → "can't set
immutable config"), so the grant lives only in the running process's memory and
does not survive a restart. See `orion/curiosity/acl.py` — including why
`clearselectors` is load-bearing there (without it, every Hub start appends a
duplicate selector, measured live).

**Outreach is a second turn** (`HUB_CURIOSITY_OUTREACH_ENABLED`, off by
default). If Orion sets `reach_out`, a separate `execute_unified_turn` composes
the message, so it gets its own `ThoughtClient.react()` stance check — Orion can
find something worth saying and the system can still decide *not now*. Delivery
goes through `EndogenousOutreach.offer_message`, which applies that module's own
gates: quiet hours, daily cap and cooldown are **shared** with tension-triggered
outreach, because from Juniper's end they are the same interruption.

**Note on addresses.** Hub runs `network_mode: host`, so it reaches FalkorDB at
`127.0.0.1:6380`; Orion's sandbox is on `app-net` and reaches the same server at
`orion-athena-falkordb:6379`. Likewise `HUB_CURIOSITY_SANDBOX_HUB_URL` is Hub's
address **as seen from the sandbox** (`host.docker.internal:8080`), because that
value is only ever rendered into the prompt.

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

### 5.3 Self-Observability panel (`Self` tab)

Self-observability v2: the Hub surfaces Orion's own self-model in a dedicated
**Self** tab (four cards: attention schema, coalition focus, curiosity gaps,
hub presence) backed by `GET /api/substrate/observability/summary`. Every
section degrades to `null` independently (missing table, unset
`POSTGRES_URI`), so the panel renders partial truth instead of erroring.

- Hub records chat-turn timestamps and mirrors a liveness snapshot to
  `substrate_hub_presence` (`HUB_PRESENCE_WRITER_ENABLED`, default on;
  apply `services/orion-sql-db/manual_migration_hub_presence_v1.sql`).
- The Agent lane can prepend a one-line curiosity focus hint from fresh
  endogenous candidates (`HUB_AGENT_CURIOSITY_HINT_ENABLED`, default off;
  advisory only, structural gate, no keyword classification).

### 5.4 Drives Analytics panel — REMOVED 2026-08-13

The `Drives` tab (standalone `/drives-analytics` page + Hub shell iframe embed, 6
`/api/drives-analytics/*` endpoints, `scripts/drives_analytics.py`/
`drives_analytics_queries.py`, and the `Postgres drive_audits` table it read from) has been
removed outright, not just hidden. It was already historical-only as of 2026-07-30 — the
`DriveEngine` producer it visualized was retired that day (drive-pressure/goal-generation
deletion sprint, `orion/sentience_striving_program/README.md` sec8) — kept alive for a
few weeks afterward as a "kill the producer, not the reader" deliberate historical-forensics
view. The `drive_audits` table itself (346,066 frozen rows, 261MB, snapshotted to
`/tmp/drive_audits_drop_2026-08-13/` before dropping) has also been dropped — nothing left
to read even if the page had stayed. Full removal PR:
`docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md`.

If you're looking for the old design context: `orion/autonomy/README.md`'s (now-removed)
"§ Hub Drives Analytics" section and
[docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md](../../docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md)
(kept as historical record, not updated) describe what this page used to do.

### 5.5 Self-Brain: Substrate State Visualization (`Self-Brain` tab)

**What it is:** real-time visual display of substrate state at 5-second granularity, organized
by four independent signal dimensions emitted from `orion-substrate-runtime`:

| Dimension | Signal | Source | Meaning |
|-----------|--------|--------|---------|
| Node kinds | Activation | graph | Peak activation per node category (tension, concept, etc.) |
| Lanes | Health | reducers | Lane freshness (lag) + backlog status |
| Self-state | Field signals | field digester | 13-dim projection from active inference |
| Prediction Confidence | Error confidence | active inference | Model's own certainty: 0.0–1.0, **no transform** |

**Display:** Two canvas views per dimension:
1. **Brain map** (top): regions as colored circles, size/color = intensity + state (firing/steady/starving)
2. **EKG sparkline** (middle): intensity history over loaded window (120 frames, ~10 minutes)

**Prediction Confidence dimension (new):** Displays the model's own prediction-error confidence
from active inference, clamped to [0.0, 1.0]. Green sparkline when selected. **No transformation
or aggregation** — this is a direct read from the active inference pipeline's unconditional output,
not a synthetic signal.

**Endpoints (read-only, degrade instead of 500):**

- `GET /api/self-brain/frames/tail?limit=N` — last N frames (ascending order)
- `GET /api/self-brain/frames/range?from=ISO&to=ISO&max=N` — frames in timestamp range
- `GET /api/self-brain/window` — earliest/latest timestamps + frame count + server time

**Data dependency:** reads Postgres `substrate_brain_frame_log` (24-hour retention, 5s cadence).
Requires `POSTGRES_URI` from `orion-substrate-runtime` service. Degrades to empty frame list if
Postgres unavailable.

**Frontend:** `templates/self-brain.html` + `static/js/self-brain.js` (embedded in Hub shell
via `#self-brain` tab button). Scrubber allows playback through historical window; LIVE mode
polls every 3s.

See also: `docs/superpowers/specs/2026-07-28-spark-introspector-retirement-and-honest-substrate-convergence.md`
for signal legitimacy audit and architecture decisions.

### 5.6 Concept Atlas: golden concepts, decay, typed relations, autonomous ingestion

**What it is:** the concept-graph-pipeline design's live substrate. A shared FalkorDB-backed
concept graph (`SUBSTRATE_STORE_BACKEND=falkor`, graph `orion_substrate` — same instance
`orion-cortex-exec` and `orion-recall` read from) seeded with four golden concepts (Orion,
Juniper, Claude, the Orion↔Juniper relationship, see `orion/substrate/seed.py`) plus concepts that
grow organically from real conversation via topic-foundry clustering
(`orion/substrate/adapters/topic_foundry.py`). Inspect it live at the **Concept Atlas** Hub
tab (`GET /concept-atlas`, backed by `GET /api/substrate/concepts/summary` and `.../network`).

**Pipeline stages, each independently gated by its own env flag:**

| Stage | Where | Flag (default) |
|---|---|---|
| Seed golden concepts at startup | `api_routes.py::seed_golden_concepts_at_startup()` | `SUBSTRATE_CONCEPT_SEED_ENABLED` (`true`) |
| Live activation decay | `api_routes.py::decay_concept_activations()`, ticked by `main.py`'s `substrate_decay_task` | `SUBSTRATE_DECAY_SCHEDULER_ENABLED` (`true`), interval `SUBSTRATE_DECAY_SCHEDULER_INTERVAL_SEC` (`120`) |
| Manual topic-foundry ingestion | `POST /api/substrate/concepts/ingest-topic-foundry` (`concept_atlas_routes.py`) | operator-triggered, no flag |
| Typed relation classification (supports/contradicts/refines) | `concept_atlas_routes.py::_classify_typed_concept_relations()`, called from the ingestion route above | runs automatically as part of ingestion, capped at `_RELATION_CLASSIFICATION_PAIR_CAP=10` pairs/call — see `services/orion-hub/scripts/concept_relation_classifier.py` for the real LLM classifier |
| Autonomous scheduled training + ingestion | `main.py`'s `substrate_topic_foundry_scheduler_task`, calling `concept_atlas_routes.py::trigger_topic_foundry_training_run()` then the ingestion route above | `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED` (**`true`** — flipped on live 2026-07-17; shipped disabled by default, real compute cost), interval `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_INTERVAL_SEC` (`86400`), window `SUBSTRATE_TOPIC_FOUNDRY_WINDOW_DAYS` (`30`) |
| Mention-edge → entity ingestion (added 2026-07-28) | Same ingestion route above, additionally calls `topic_foundry_client.py::fetch_mention_edges_for_run()` (`GET /kg/edges?predicate=mentions`) and passes the result into `map_topic_foundry_run_to_substrate`'s `mention_edges`/`segment_topic_id_map` params | no flag, runs unconditionally as part of ingestion; degrades to zero entities on fetch failure, same as the segments fetch |
| Scheduled enrichment (added 2026-07-28) | Same scheduler tick, new step: `concept_atlas_routes.py::trigger_topic_foundry_enrichment()`, calls `POST /runs/{run_id}/enrich` for whatever the latest completed run is | `SUBSTRATE_TOPIC_FOUNDRY_ENRICH_ENABLE` (own gate, real LLM compute cost per un-enriched segment -- **`true`**, flipped on live 2026-07-28 per explicit operator go-ahead), cap `SUBSTRATE_TOPIC_FOUNDRY_ENRICH_LIMIT` (`200`) |

**Mention edges — real, LLM-enriched entity data that used to go nowhere (2026-07-28):**
topic-foundry's `kg_edges.py::generate_edges_for_run()` computes real typed edges
(`mentions`/`asks_about`/`claims_about`/`next_step`) from each segment's LLM-enriched
`meaning` field, but used to publish them on `orion:kg:edge:ingest.v1`, a bus channel with
zero live consumers (`orion-rdf-writer` never actually subscribed; `orion-graphdb` never
existed as a real service — see `orion/bus/channels.yaml`'s former comment on that channel).
That publish is now retired outright. Only the `mentions` predicate is wired into the
substrate graph so far (as `EntityNodeV1` + `associated_with` edges from the mentioning
topic's concept node) — `asks_about`/`claims_about`/`next_step` have no corresponding
substrate node kind yet and are out of scope until/unless that's designed. The edges
themselves still persist in topic-foundry's own Postgres and remain queryable via its
`GET /edges`/`GET /kg/edges` API regardless of this wiring.

**Why enrichment needed its own scheduler step, same day:** confirmed live 2026-07-28,
`topic_foundry_segments` had 0 of 22 rows ever enriched — nothing in this codebase had ever
called `POST /runs/{run_id}/enrich`, meaning the mention edges described above had no real
data to work with regardless of the ingestion wiring being correct. That endpoint also
triggers topic-foundry's typed KG edge generation as a same-request side effect on its side
(`app/services/enrichment.py::_run_enrichment`'s trailing `_generate_edges` call). Watch
`substrate_topic_foundry_scheduler_enrich_tick` log lines for `enriched_count`/`failed_count`,
and the underlying `topic_foundry_segments`/`topic_foundry_edges` tables directly, before
trusting this produces non-degenerate data at real volume.

**Known limitation:** `_concept_nodes()` and the other Concept Atlas Hub-tab helpers in
`concept_atlas_routes.py` explicitly filter to `node_kind == "concept"`. The `EntityNodeV1`
records mention-edge ingestion writes are real and generically visible to any consumer that
iterates the full substrate node set (e.g. `orion/substrate/endogenous_curiosity.py`), but
they will not appear in the Concept Atlas Hub UI's network/god-node views until those routes
are widened to include entity nodes — a separate follow-up, not done here.

**AI Town's own concept graph (added 2026-08-20):** a second, fully parallel
instance of the same pipeline above, reading `aitown_chat_history_log` (the
AI-Town-only table post-cutover, PR #1734) instead of `chat_history_log`,
writing into a second FalkorDB graph (`FALKORDB_AITOWN_SUBSTRATE_GRAPH`,
default `orion_substrate_aitown`, same instance) instead of `orion_substrate`.
Interpretability-only — never feeds `orion-cortex-exec`'s chat-stance producer
or any other Orion cognition consumer
(`docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-
readability-design.md`, "AI Town's own concept graph" / Non-goals).

- **Scheduler**: same tick as the Orion pipeline above (`main.py`'s
  `substrate_topic_foundry_scheduler_task`, same
  `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED`/`_INTERVAL_SEC`), gains a second
  trigger/enrich/ingest step-group gated by its own
  `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED` (default `true`) — set
  `false` to pause AI Town ingestion without touching Orion's own pipeline.
- **Manual ingestion**: `POST /api/substrate/concepts/ingest-topic-foundry-aitown`
  (`concept_atlas_ingest_topic_foundry_aitown()`), same operator-triggered,
  no-flag shape as the Orion route above.
- **Reading it**: `GET /api/substrate/concepts/summary` and `.../network`
  both take an optional `?graph=aitown` query param (default/unrecognized
  values resolve to Orion's graph) — the "first cut" read path the design
  spec's own "Missing questions" named as sufficient before a dedicated AI
  Town Concept Atlas UI page is worth building. No such page exists yet;
  the Hub tab's Cytoscape view still only ever renders Orion's graph.
- **Dataset/model**: `orion-hub-aitown-dataset-v1`/`orion-hub-aitown-v1-<fingerprint>`
  in topic-foundry, distinct from Orion's `orion-hub-autonomous-dataset-v2`/
  `orion-hub-autonomous-v4-<fingerprint>` — no `where_sql` filter (the source
  table is already AI-Town-only by construction, unlike Orion's dataset which
  excludes AI Town rows via `where_sql`).

**Surfaced into live cognition two ways**, not as permanent context bloat — only when
salient:

- `orion-cortex-exec`'s always-on chat-stance producer lane reads golden/relationship
  concepts via `orion/substrate/relational/adapters/concept_induction_ctx.py` (repointed at
  this store in PR #1128 — it previously read a dead spaCy pipeline that never returned
  anything).
- `orion-recall`'s purposeful-phase belief lane reads a turn-scoped neighborhood via the
  `concept_region` collector (see `services/orion-recall/README.md` § 15, PR #1133) — a
  cheap label-substring match against the current turn's text, empty when nothing matches.

**Decay math, if you're debugging why an activation value looks wrong:**
`decay_concept_activations()` takes an explicit `elapsed_seconds` parameter from its caller
(the scheduler passes true wall-clock time since the previous tick, tracked via
`time.monotonic()`) rather than deriving elapsed time from `node.temporal.observed_at`
internally — the latter is only a documented one-shot fallback for ad-hoc/manual invocation.
A function called repeatedly on a loop that re-derives elapsed time from a never-advancing
`observed_at` on every call compounds: each tick re-decays an already-shrunk value against
an ever-growing elapsed-since-creation window, collapsing activation to `decay_floor` within
roughly one configured half-life regardless of the half-life value (a real bug caught in
review during PR #1131 — see that PR's description for the numeric trace).

**Activation was seeded at 0.0 with no half-life until 2026-07-17 (fixed):** decay math
being correct is meaningless if there's nothing to decay. No `ConceptNodeV1` producer ever
set `signals.activation` when constructing a node — every concept was born at the schema
default (`activation=0.0`, `decay_half_life_seconds=None`), and `decay_activation()` treats
a falsy half-life as "clamp to floor, don't decay." So the live scheduler above was decaying
an input that was permanently `(0.0, None)` — 120s ticks that correctly computed nothing,
forever. This was not limited to the two organic-growth adapters (`topic_foundry.py`,
`concept_induction.py`) — a code-review pass on the first version of this fix found 16+ live
`ConceptNodeV1` construction sites still missing it, including the three golden/seed
concepts (`orion/substrate/seed.py`) and every `orion/substrate/relational/adapters/*.py`
producer. Rather than patch each call site by hand (and risk missing the next one), the fix
lives at the schema boundary: `ConceptNodeV1` now has a `model_validator(mode="after")`
(`orion/core/schemas/cognitive_substrate.py::_seed_activation_if_unset`) that auto-seeds
`activation = salience` and a 30-day default half-life whenever a producer leaves
`signals.activation` at its pure schema default — covering every current and future
producer, not just the ones that remember to opt in. `orion/substrate/adapters/_common.py::make_activation()`
remains available for a producer that wants an explicit, non-default initial value.

**Golden concepts still needed a second fix, same day:** the schema-boundary validator above
seeds `activation = salience`, but `orion/substrate/seed.py`'s three golden concepts
(Orion, Juniper, the relationship) never had a `salience` in `seed_concepts.yaml` at all —
so even after the validator fix, they seeded to `activation = 0.0` (a real number now, but
still flat zero). Fixed in the same PR (#1173): `seed.py` now constructs them with
`signals=SubstrateSignalBundleV1(confidence=1.0, salience=1.0)` — defensible for
`promotion_state="canonical"`/`authority="human_verified"` concepts, not a magic number.

Reinforcement-on-recall (bumping activation when a concept is actually retrieved in a live
turn, mirroring `orion/memory/crystallization/dynamics.py::recall_boost()`'s existing
precedent for the separate crystallization system) shipped in the same follow-up, PR #1173
— see the concept_region collector at §15 of `services/orion-recall/README.md` for how it's
wired, and `services/orion-recall/app/collectors/CONCEPT_REINFORCEMENT_DESIGN.md` for the
full design conversation (sync-write-vs-bus, which fields move and why).

**Side effect caught in review:** `GET /api/substrate/concepts/summary`'s `_at_risk_concepts()`
used to treat "every concept node's activation is identical" as a proxy for "no live decay
signal exists yet" and returned an empty, explained `at_risk` list in that case. Once
activation is seeded at construction, a brand-new low-salience concept would otherwise show
up as "at risk of decaying toward its floor" on its very first tick — it hasn't decayed at
all, it just started low. Fixed by replacing the variance proxy with an explicit age gate
(`_AT_RISK_MIN_AGE_SECONDS`, one hour): a concept must have existed long enough for real
decay ticks to plausibly have run before it's eligible for `at_risk` at all.

**A shipped fix that silently reverted itself within one restart, if you're ever debugging
"why did my Falkor write not stick" (PR #1175, fixed):** live-verifying the golden-concept
salience fix above turned up a separate, previously-undiscovered bug in
`FalkorSubstrateStore._migrate_legacy_payload_nodes()` (`orion/substrate/falkor_store.py`) —
the legacy-payload rewrite path that runs on every hydrate. It read a node still carrying an
old `payload_json` blob, parsed it, and rewrote it via the normal `upsert_node()` path — but
that write's `MERGE (n:SubstrateNode:<type-label> {node_id})` can never match the legacy row
itself (labeled `SubstrateNode` only, no type label yet), so the write always landed on a
*different* node and the legacy row's `payload_json` was never removed. It persisted forever,
got re-parsed on every future hydrate, and re-clobbered the canonical node's real data —
this is exactly what reverted the golden-concept salience fix within one Hub restart cycle,
live, the same day it shipped. It also cascaded into duplicate relationships (one edge existed
as 4 near-identical copies) via the same bare-label MERGE ambiguity on edge source/target
matching. Fixed with an explicit cleanup delete after a successful migration write
(`_delete_orphaned_legacy_node_duplicate()`), using `DETACH DELETE` so the cascaded duplicate
edges get cleaned up as a side effect with no separate edge-dedup logic needed. Self-healing —
no manual data migration was required, the fix took effect for every affected node the next
time that node's hydrate ran.

**Edges silently lost their real source/target linkage on every hydrate, if
`edge_counts_by_predicate` on the summary route ever reads empty despite real edges existing
(PR #1179, fixed):** found immediately after live-verifying the #1175 fix above — the edge
hydration query read `source_id`/`target_id` as edge *properties*
(`e.source_id`/`e.target_id`), but `upsert_edge()` deliberately never writes those two fields
onto the edge (the real linkage already lives in the graph topology, the
`(source)-[e]->(target)` pattern itself). So they always read back `NULL`, and
`decode_edge()` `str()`-coerced that into the literal string `"None"` for every edge's
source/target node_id, forever, since before this session. Fixed by deriving `source_id`/
`target_id` from the already-MATCH-bound `source`/`target` node variables instead
(`orion/substrate/falkor_store.py::_edge_hydrate_return_clause()`). Same self-healing
property — no stored data was wrong, this was purely a read-path bug.

**Autonomous scheduler window math, if debugging why a training run didn't fire:**
`trigger_topic_foundry_training_run()` floors its rolling window's `end_at` to a UTC day
boundary before computing `start_at` from it — NOT `datetime.now()` verbatim. This is load-
bearing: topic-foundry's own `POST /runs/train` dedups by a `spec_hash` computed over the
exact `start_at`/`end_at` it receives, so a microsecond-unique window on every tick would
mean the dedup never fires and every tick trains a brand-new HDBSCAN model from scratch
regardless of interval (also a real bug caught in review, PR #1136). Practically: repeated
ticks within the same UTC day resolve to the same already-queued/running/complete run;
you'll see a new training run at most once per day at the default interval.

**Restart required after enabling the autonomous scheduler:** none beyond the normal Hub
restart — the flag defaults off, so turning it on is a deliberate `.env` edit + restart, not
a silent behavior change on an existing deploy.

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

See also: PRs #1128 (stance repoint), #1131 (decay scheduler), #1132 (relation classifier),
#1133 (recall wiring), #1136 (autonomous scheduler), #1166 (decay was structurally inert --
schema-boundary activation seed), #1173 (golden-concept salience + reinforcement-on-recall),
#1175 (legacy-migration duplicate-node bug that silently reverted #1173), #1179 (edge
hydration losing every edge's real source/target node_id) for the full review history
including every bug caught and fixed before each shipped, and
[docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md](../../docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md)
for the original design.

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

# Topic Studio (Topic Foundry proxy)
TOPIC_FOUNDRY_BASE_URL=http://orion-topic-foundry:8615
```

### Manual UI checklist
- Navigate between **Hub** and **Topic Studio** tabs; ensure no overlays block pointer events on Hub.
- In Topic Studio, run **Preview** with `turn_pairs`, then switch to `conversation_bound` after setting a `boundary_column`.
  - Topic Studio pins `split_text_columns: false` (it has no control for it yet), so its previews keep the historical fused-column shape. The Hub's own scheduler sends `split_text_columns: true` with `column_speakers`, so scheduler-driven runs produce one document per utterance -- expect Topic Studio's document counts to differ from the scheduler's for the same window. See `docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md`.
- Train a run, poll for completion, then load segments and click a segment to confirm full text renders in the detail pane.

Topic Studio relies on the Topic Foundry `/capabilities` endpoint to configure supported segmentation modes and defaults, uses `/runs?limit=20` to populate the recent run picker, and the segments list uses `include_snippet=true&include_bounds=true` with `limit/offset` for faster previews and paging.

---

## Voice debugging

Hub records PCM in the browser, resamples to 16 kHz WAV, and sends `client_audio_meta` with peak, RMS, duration, and chunk count. Low peak warns in the UI but still sends audio. STT silence rejection is configured in `orion-whisper-tts` via `STT_NEAR_SILENT_PEAK_INT16` (default `50`).

**Browser console** (after mic release):

```
[voice] chunk_count=… peak=… rms=… sent audio payload…
```

**Hub logs:**

```bash
docker compose logs -f orion-hub | grep -E 'voice\.ws\.audio_received|voice\.stt'
```

**STT logs:**

```bash
docker compose logs -f orion-whisper-tts | grep -E '\[STT\]|Sent STT result'
```

Empty-transcript WebSocket errors include `audio_debug` with client and STT metadata. No JS unit harness — verify manually in the browser.

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

**Review queue metadata:** Opening a card exposes editable confidence, sensitivity, visibility scope, priority, provenance, evidence, still-true lines, and time horizon. **Save metadata** calls `PATCH /api/memory/cards/{id}`; **Approve** saves then sets `status=active`. Cards with `priority=always_inject` and `status=active` are injected by cortex-orch on every chat turn (requires `RECALL_PG_DSN` on cortex-orch).

**Recall cards rail:** Set `RECALL_ENABLE_CARDS=true` on orion-recall (see `services/orion-recall/.env_example`). Use a profile with `cards_top_k` and `backend_weights.cards` — e.g. `biographical.v1`, `self.factual.v1`, or the chat dropdown options added in Hub. Only **`active`** cards participate; scoring uses vector-host cosine similarity (embeddings cached in `subschema.recall_embedding`).

**Memory graph approve → Fuseki:** When `RDF_STORE_GRAPH_STORE_URL` and `RDF_STORE_UPDATE_URL` are set (`MEMORY_GRAPH_APPROVAL_BACKEND=auto`, default), approve writes RDF to Fuseki then projects Postgres `memory_cards` rows. Legacy GraphDB is used only when `MEMORY_GRAPH_APPROVAL_BACKEND=graphdb`.

**Automated smoke:** `scripts/smoke_memory_cards_e2e.sh` (requires `ORION_HUB_URL`, `ORION_HUB_SESSION_ID`, and `RECALL_PG_DSN` set to the same memory-store contract as Hub).

**Memory crystallizations (governed cognitive memory):** `MemoryCrystallizationV1` is separate from `MemoryCardV1` (not MemoryCardV2). Turn-facing recall remains cards; crystallizations are durable governed memory in Postgres with optional projections to cards, Chroma (`orion:memory:vector:upsert`), and Graphiti/FalkorDB via `services/orion-graphiti-adapter/`. RDF `/api/memory/graph/*` is unchanged. Hub Memory tab → **Crystallizations** subview (inbox, approve/reject, projection health). Smoke: `scripts/smoke_memory_crystallization_e2e.sh` (requires Hub restart with this branch).

**Memory consolidation graph drafts:** `orion-memory-consolidation` persists `memory_graph_suggest_drafts` (`pending_review`) when conversation windows close. Hub Memory tab → **Graph drafts** lists automated drafts from the same Postgres DSN as memory cards (`RECALL_PG_DSN`). **Load in editor** prefills the graph annotator; **Validate / Approve** uses existing `/api/memory/graph/*` and marks the consolidation draft `approved` when `consolidation_draft_id` is sent on approve (only while the draft is still `pending_review`). Reject via **Graph drafts** or `POST /api/memory/consolidation/drafts/{draft_id}/status` — rejecting clears the active consolidation draft link so a later Approve does not resurrect it. Graph approve and inbox status update are not one transaction: if RDF/cards succeed but the draft row is not updated, the API returns `consolidation_draft_marked: false` and the UI warns. Requires `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql` applied on the memory Postgres.

**Proposal review (attention + review decisions):** Hub main tab → **Pending Decisions** lists decision-worthy `pending_review` proposals from the context-exec proposal review API. Enabled in Athena `.env_example` (`HUB_PROPOSAL_REVIEW_ENABLED=true`); panel and script are omitted from the page when false. Hub calls `GET /health`, `GET /proposals`, detail, eligibility, and `POST /proposals/{id}/review` only — it does not read JSON ledger files, does not POST triage, and does not execute proposals directly. Approval creates future execution eligibility only. See [docs/proposal-review-api.md](../../docs/proposal-review-api.md).

**Compute lane override (mode vs compute):** Hub chat UI exposes **Mode** and **Compute** dropdowns. Mode decides behavior (`Auto`, `Grounded Small`, `Brain`, `Quick`, `Story`, `Agent`, `Council`); **Compute** selects the GPU/model lane (`chat`, `quick`, `agent`, `metacog`). Default compute is `quick`. Hub proxies `GET /api/llm-routes` from `HUB_LLM_GATEWAY_URL` (`GET /routes` on orion-llm-gateway) and polls every 30s. Selected lane is sent as `llm_route` on chat payloads (wired into cortex `options.llm_route`). Agent mode still routes to context-exec with `llm_profile` bound to the selected compute lane. Down lanes warn with explicit **Use quick / Try anyway / Cancel** — no silent fallback.

**Social room toggle vs Mode vs Compute:**

| Control | What it sets | Social room ON override |
|---------|----------------|-------------------------|
| **Mode** | Behavior lane (`brain`, `agent`, `council`, …) | Forces `mode=brain` |
| **Compute** | GPU/model lane via `llm_route` (`chat`, `quick`, …) | **Still applies** — pick which model serves `chat_social_room` |
| **Social room** checkbox | `chat_profile=social_room`, `social_room_mode=hub_direct` | Forces verb **`chat_social_room`** (not `chat_general` / `chat_quick`) |

Bridge env keys `SOCIAL_BRIDGE_HUB_MODE` / `SOCIAL_BRIDGE_HUB_VERB` affect **CallSyne bridge → Hub** calls only; the Hub UI toggle ignores them.

**Agent mode → context-exec:** When `HUB_AGENT_CONTEXT_EXEC_ENABLED=true` (default), Hub **Agent** mode calls `POST /context-exec/run` on `HUB_CONTEXT_EXEC_API_URL` instead of legacy AgentChain/ReAct. Selected route is passed as `llm_profile` on the context-exec request. Hub renders an inline operator response (`Agent run complete`, mode, route, synthesis status, result, proposal link, mutation none) from `operator_summary`. Proposal review / Pending Decisions remain unchanged.

### Agent Claude mode (FCC harness)

When `HUB_AGENT_CLAUDE_ENABLED=true`, Hub exposes **Agent Claude - Opus / Sonnet / Haiku** modes in the Mode dropdown. Each message spawns one `claude -p … --output-format stream-json` turn with `ANTHROPIC_BASE_URL` set to `HUB_FCC_SERVER_URL` (default `http://127.0.0.1:8082`). Tier selection maps to FCC env keys (`MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`). **Compute** lane is ignored for these modes.

**Requirements:** `claude` on PATH (Hub container mount), repo mount, `~/.fcc/.env` (see `config/fcc.env_example`), and **`orion-fcc`** running on host port **8082** (`services/orion-fcc`). Hub uses `HUB_FCC_SERVER_URL=http://127.0.0.1:8082`; Orion harness uses `HARNESS_FCC_SERVER_URL=http://host.docker.internal:8082`.

**Permissions:** this container always runs as root (no `USER` directive), so `orion/fcc/claude_spawn.py::claude_permission_argv()` gives these turns full-auto-approve Bash/tool access via `--permission-mode bypassPermissions` (requires the Dockerfile's `ENV IS_SANDBOX=1` — see that function's docstring). That's genuinely unprompted access in a container that also mounts `/var/run/docker.sock`, the operator's real `${HOME}/.ssh` (read-only, for `git push`), and runs `network_mode: "host"` — not narrowed to repo writes. No repo-committed hook gates it (`--setting-sources user,local` drops project-level hooks for FCC turns); whatever gates a bad call must live in the operator-managed Claude config, not this repo.

**Live smoke:**

```bash
docker compose -f services/orion-fcc/docker-compose.yml up -d --build
PYTHONPATH=services/orion-hub:. python services/orion-hub/scripts/verify_agent_claude_stream_live.py \
  --ws ws://127.0.0.1:8080/ws \
  --text "list files in services/orion-hub/scripts" \
  --fcc-model-label MODEL_HAIKU
```

### fcc-claude MCP (GitHub + Firecrawl + AI Town)

When `HUB_AGENT_CLAUDE_MCP_ENABLED=true`, each agent-claude turn renders an ephemeral MCP config from `config/fcc_claude_mcp.template.json` and passes `--mcp-config` + per-server `--allowedTools` (e.g. `mcp__github`, `mcp__firecrawl`) to `claude -p`. No `--disallowedTools` is passed: it previously carried `Bash(gh *)`, which beat `--permission-mode bypassPermissions` and left the agent unable to run `gh pr create` even though `gh` is installed and authenticated in this container — and the github MCP server is rendered read-only, so it exposes no `create_pull_request` either. See `extend_mcp_argv()` in `orion/fcc/claude_spawn.py`. Secrets live in operator `~/.fcc/.env` (not Hub `.env`):

**ToolSearch:** FCC Claude subprocesses set `ENABLE_TOOL_SEARCH=true` through `orion.fcc.context_budget.extend_fcc_subprocess_env` (shared with harness-governor). MCP servers stay attached; Claude Code loads tool schemas into context only when ToolSearch pulls them. This counters the custom-`ANTHROPIC_BASE_URL` fallback that otherwise dumps all MCP schemas at spawn. Operators may override `ENABLE_TOOL_SEARCH` in the process environment for debugging (`false` / `auto` / `auto:N`). Optional further GitHub surface tightening: set `GITHUB_TOOLSETS` in `~/.fcc/.env` (code default remains `repos,pull_requests` when unset).

| Key | Required | Purpose |
|-----|----------|---------|
| `GITHUB_PAT` | yes | GitHub MCP (`ghcr.io/github/github-mcp-server` via Docker) |
| `FIRECRAWL_API_KEY` | yes | Firecrawl MCP (`npx firecrawl-mcp`; Hub image includes Node 20) |
| `AITOWN_CONVEX_URL` | when AI Town enabled | Self-hosted Convex base URL |
| `AITOWN_ADMIN_KEY` | when AI Town enabled | Convex admin key |
| `AITOWN_WORLD_ID` | when AI Town enabled | World id from `npx convex run init` |
| `AITOWN_ORION_PLAYER_ID` | optional | Default player for embodied tools |
| `AITOWN_ORION_AGENT_ID` | optional | Default agent for embodied tools |

**Rollout order:** (1) set `HUB_AGENT_CLAUDE_MCP_ENABLED=true` with GitHub + Firecrawl secrets, (2) bootstrap `services/orion-ai-town/` on mesh, (3) set `HUB_AITOWN_ENABLED=true` and AI Town secrets, (4) open Hub **AI Town** tab (`#ai-town`) for proxied visualization.

Hub settings: `HUB_AGENT_CLAUDE_MCP_ENABLED`, `HUB_AITOWN_ENABLED`, `HUB_AITOWN_UI_URL` (default `http://127.0.0.1:5173`). Routes: `GET /api/aitown/status`, `GET /aitown/` reverse proxy.

Preflight errors surface as `fcc_mcp_*` codes before spawn (e.g. `fcc_mcp_github_missing`, `fcc_mcp_aitown_config`).

**Investigation v2 (epistemic pipeline):** When `CONTEXT_EXEC_INVESTIGATION_V2_ENABLED=true`, Hub Agent lane threads `answer_contract` from `answer_contract_draft` into context-exec, sends `mode=investigation_v2` with profile-derived permissions (`context_exec_permissions_for_llm_profile`), and receives **`final_text`** (finalize-rendered user voice) as `llm_response`. `InvestigationReportV2` remains an inspectable operator sidecar in `metadata.context_exec` (`operator_report_text`). Conceptual/personal turns do not trigger repo/trace sweeps. Default is `false` (legacy keyword mode inference preserved).

**Denver memory correction vertical slice:** `ORION_PY=orion_dev/bin/python bash scripts/denver_memory_correction_vertical_smoke.sh` proves a Denver `memory_correction_proposal` reaches Pending Decisions when `HUB_PROPOSAL_REVIEW_ENABLED=true`. Expected final line: `denver_memory_correction_vertical_smoke PASS`. Hub card shows current belief, proposed correction, rationale, evidence summary, risk/confidence, and safety flags. Hub review actions (approve/reject/request changes) are on the detail card when enabled — smoke does not exercise them; pytest covers review POST allowlist and no-execution invariants. No execution or memory mutation in smoke.

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

## Field Channels tab

`#field-channel-glossary` panel (`templates/index.html`), backed by `/static/field-channel-glossary.html`
in an iframe. Shows every raw field-digester channel grouped into category cards (physical
substrate, task execution, infra transport, etc.) with a live `clean/total` count per card,
computed entirely client-side from the panel's own API calls — no hardcoded counts in the
template.

- **`GET /api/field-channel-glossary/channels`** (`scripts/field_channel_glossary_routes.py`):
  the static glossary — every channel's name/level/category/meaning, sourced from
  `config/field/field_channel_glossary.v1.yaml` (the single structured source this route, the
  field-digester README's prose glossary, and this section all read from — see that yaml's own
  header comment).
- **`GET /api/field-channel-glossary/health?hours=N`**: live classification per channel
  (`live`/`quiet`/`dead`/`never_produced`/`ratchet_suspect`) computed from real
  `substrate_field_state` rows in that window via `orion.field.channel_glossary.classify_channel_series()`
  — deliberately not a hand-maintained verdict column, since a static one already went stale once
  (see the field-digester README's "Decay vs. injection-interval mismatch" section).

This tab has no separate documentation of its own beyond this pointer — the glossary yaml and
`services/orion-field-digester/README.md`'s "Field channel glossary" section are the real content;
this section just orients where the Hub-facing panel actually gets its data from, since this repo
previously had zero doc surface here at all.

**Channel rename note (2026-07-24):** `transport_pressure`/`bus_health` renamed to
`stream_backlog_pressure`/`stream_backlog_health` for scope honesty (both only ever measured
`world_pulse`'s one Redis Stream, never general bus/transport health) — if this tab is showing
those new names and you were expecting the old ones, that's this rename, not a bug. See the
field-digester README's sixth training-data cutoff entry for the full mechanism.

## Cabinet tab

Top-level Hub nav **Cabinet** (`#cabinet`) polls Athena Nano host snapshots while the tab is
visible (~1s). Backed by `GET /api/cabinet/sensors/latest` (`scripts/cabinet_sensors_routes.py`)
and `/static/js/cabinet-sensors.js`.

**Operator setup**

1. Host reader writing `/run/orion-sensors/latest.json` (and optional `boot.json`) via
   `orion-cabinet-sensors.service`.
2. Compose bind-mount already in `docker-compose.yml`:
   `/run/orion-sensors:/run/orion-sensors:ro` plus `CABINET_SENSORS_PATH` /
   `CABINET_BOOT_PATH` / `CABINET_SENSORS_STALE_AFTER_SEC` from `.env_example`.
3. After mount/env change, restart Hub so the container sees the bind:

```bash
# from a worktree (not shared checkout):
scripts/safe_docker_build.sh orion-hub up -d --build
```

Open Hub → **Cabinet**. Missing snapshot shows a no-snapshot message naming
`orion-cabinet-sensors.service`. Pressure strip is labeled **activity (Hub)** (process-local
baselines; operator-debug only).

### Cabinet ambient audio (live + multi-day charts)

Below the Nano tiles, **Cabinet ambient audio** shows live RMS/peak/age from the host ALSA
reader and 24h / 3d / 7d RMS + activity charts from Postgres `orion_biometrics_summary`
(~30s biometrics grain — not 1 Hz host reader). Spec:
`docs/superpowers/specs/2026-08-26-hub-cabinet-ambient-audio-charts-design.md`.

**Operator setup (in addition to Nano sensors above)**

1. Host reader writing `/run/orion-audio/latest.json` via `orion-ambient-audio-reader.service`
   (see `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md`).
2. Compose bind-mount already in `docker-compose.yml`:
   `/run/orion-audio:/run/orion-audio:ro` plus `AMBIENT_AUDIO_PATH` /
   `AMBIENT_AUDIO_STALE_AFTER_SEC` / `CABINET_AMBIENT_HISTORY_*` from `.env_example`.
3. History requires `orion_biometrics_summary` rows with `cabinet_ambient_rms` /
   `cabinet_ambient_audio_activity` (written by biometrics → sql-writer) and the
   `(node, timestamp)` index from `orion-sql-writer` boot DDL.
4. After mount or env change, restart Hub so the container sees the bind:

```bash
# from a worktree (not shared checkout):
scripts/safe_docker_build.sh orion-hub up -d --build
```

API: `GET /api/cabinet/ambient/latest`, `GET /api/cabinet/ambient/history?window=24h|3d|7d`
(`scripts/cabinet_ambient_routes.py`, `/static/js/cabinet-sensors.js`). Latest polls ~1s only
while `#cabinet` is visible; history fetches on tab activation, window toggle, or Refresh.

## Reverie tab

Top-level Hub nav **Reverie** (`#reverie`) -- historical, human-visible view of both reverie
chains (design spec: `docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md`).
No poll loop (unlike Cabinet/Attention Organ above) -- this is a historical browsing tool, not
live telemetry; one fetch on activate, plus a manual Refresh. Backed by
`GET /api/reverie/visual/recent`, `GET /api/reverie/visual/image/{sha256}`, and
`GET /api/reverie/text/recent` (`scripts/reverie_routes.py`) and `/static/js/reverie-tab.js`.

**Two sub-views, two independent chains, no shared code between them:**

- **Visual** -- `orion-thought`'s `app/visual_chain.py` (generate → store → caption). One card
  per `reverie_visual_chain` row: the real generated image (served content-addressed from disk,
  same sniff-and-verify-on-read discipline as `chat_attachments.py`), its caption
  (`reverie_visual_artifact.description`, honestly `null` when re-observation failed or was
  rejected -- never fabricated), the context-seed (Patch 3 -- Orion's own most recent real
  reverie-thought interpretation, `context_text`), the blended prompt, and whether it advanced
  `prior_description` for the next run.
- **Text** -- `orion-thought`'s `app/chain.py` (attention-coalition narration). One card per
  `substrate_reverie_chain` row: its `substrate_reverie_thought` interpretations (joined via
  `chain_json.thought_ids`, a plain JSON list -- no FK), and real downstream badges computed from
  the actual queue/alert tables: **queued for dream compaction**
  (`dream_compaction_request_queue.origin_chain_id`) and **N resonance alert(s) on this theme**
  (`substrate_reverie_resonance_alert.theme_key`, a per-theme count across the chain's cohort, not
  a per-chain causal claim -- the detector operates over a window, not one chain in isolation).

**Operator setup**

1. Compose bind-mount already in `docker-compose.yml`:
   `${REVERIE_VISUAL_STORAGE_DIR}:${REVERIE_VISUAL_STORAGE_DIR}:ro` (default
   `/mnt/storage-lukewarm/orion/reverie-visual`, the same path `orion-thought`'s visual chain
   writes to) plus `REVERIE_VISUAL_STORAGE_DIR` from `.env_example`. Read-only -- Hub is not the
   producer.
2. After mount/env change, restart Hub so the container sees the bind:

```bash
# from a worktree (not shared checkout):
scripts/safe_docker_build.sh orion-hub up -d --build
```

Open Hub → **Reverie**. If the visual sub-view is empty, `ORION_VISUAL_CHAIN_ENABLED` may be off
on `orion-thought`, or no chain has run yet. `dream_compaction_request_queue.consumed_at` is
never set anywhere in the codebase today (confirmed live, 2026-08-26) -- REM re-folds the same
backlog every pass, so "queued" here means exactly that, not "applied" (Phase G's applier is
separate and gated, and never runs on this queue's contents automatically).

**Privacy note** (design doc §7): visual-chain images are a lossy rendering of whatever context
fed the prompt. Today that's only `prior_description` (a prior caption) or a fixed seed string --
no private chat/dream content reaches the prompt yet. This tab must be revisited before Patch 3
(real context-seeding) ships, not after.

## Bus synaptic graph debug routes

Read-only view into `services/orion-bus-mirror`'s live FalkorDB graph (`orion_bus_synapse`) —
`scripts/bus_synaptic_graph_routes.py`. No UI panel yet, API only. "Idea 5" from
`docs/superpowers/specs/2026-07-24-bus-vitality-field-signal-brainstorm.md`'s Phase 3+
brainstorm: before building any new signal on top of this graph, surface what's already
structurally visible in it. Never writes to the graph.

- **`GET /api/bus-synaptic-graph/summary`**: node/edge counts (`Organ`, `Channel`, `Verb`,
  `PUBLISHES`, `CAUSALLY_FOLLOWED_BY`, `EXECUTES_VERB`) — a quick "is this graph alive, roughly
  what shape" check.
- **`GET /api/bus-synaptic-graph/hot-organs?limit=N`**: organs ranked by `PUBLISHES` out-degree
  (how many distinct channels they fan out to) — a real centrality signal already sitting in the
  data (`vision-host` dominates at ~6300, next is `llm-gateway` at ~650).
- **`GET /api/bus-synaptic-graph/hot-edges?limit=N`**: real cross-organ hop pairs ranked by
  observed count — the structurally dominant flows in the mesh right now.
- **`GET /api/bus-synaptic-graph/anomalies?zscore_threshold=3.0&min_count=5`**: edges whose most
  recent observation deviated sharply from that edge's own rolling baseline.
  `min_count` guards against the cold-start z-score instability documented in
  `services/orion-bus-mirror/README.md` — an edge's second-ever observation can read as an
  extreme z-score before a real baseline exists, so low-`count` edges are excluded here.

Requires `FALKORDB_URI` set (shared with the Graphiti/substrate-concept FalkorDB instance) and
`FALKORDB_BUS_GRAPH` (default `orion_bus_synapse`, matching orion-bus-mirror's own setting name).
Returns `503 falkordb_uri_not_configured` if unset, not a silent empty response.

## Pending Attention — cognitive loops

Flag: `ORION_ATTENTION_PENDING_CARDS_ENABLED` (default-off). API:
`GET /api/attention/loops`, `POST /api/attention/loops/{id}/resolve`,
`POST /api/attention/loops/{id}/dismiss`. Resolve/Dismiss emit
`AttentionLoopOutcomeV1` on `orion:attention:loop_outcome`, persist to
`attention_loop_outcome`, and suppress the loop via `substrate_reverie_refractory`.

Migration: `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_loop_outcome.sql`

**`card_kind` (2026-08-21):** every card is `"resolvable"` (chat-derived, a discrete
turn-scoped candidate a human can actually close) or `"chronic_pressure"`
(reverie/substrate-broadcast, re-selected every tick by design — the same 7
substrate nodes recurring indefinitely is expected, not stuck). The split is
architectural, keyed off the underlying trace row's `scope`, not a heuristic —
see `orion/schemas/attention_salience.py`'s `PendingCardKindV1` docstring and
`attention_loops_store.py::card_kind_for_scope`. The Hub UI renders
`chronic_pressure` cards without Resolve/Dismiss buttons; the API also rejects
those verbs on one with `409` (defense in depth against a stale client). The
badge/button-suppression branching itself is pure logic in
`static/js/cognitive-loop-card.js` (`cognitiveLoopCardViewModel`), unit-tested
without a DOM harness: `node --test static/js/cognitive-loop-card.test.js`.

**Why every card used to show the same sentence:** `why_it_matters`/`target_type`
were computed on every loop (`orion/substrate/attention/scoring.py`) but never
survived into `attention_salience_trace` — both producers dropped them before
storage, so the Hub's fallback text (`build_pending_card`'s
`f"This {target_type} has stayed active without resolution."`, `target_type`
always defaulting to `"other"`) was the *only* text any card ever showed. Fixed
2026-08-21: both columns are now persisted (see
`manual_migration_attention_salience_trace.sql`) and both producers
(`chat_attention_salience_trace.py`, `orion-thought`'s `reverie.py`) carry them
through. The old sentence is now a true last-resort fallback.

**Deploy order matters:** apply `manual_migration_attention_salience_trace.sql`
*before* deploying orion-cortex-exec/orion-thought with these changes, not
after. Both producers now INSERT `why_it_matters`/`target_type` unconditionally;
without the migration every insert fails with "column ... does not exist" and
the fail-open contract on both write paths swallows it as a WARNING log --
silently dropping ALL `attention_salience_trace` persistence (both scopes, not
just the two new columns) until someone reads the logs and runs the migration.

**Implicit decay (2026-08-21):** the panel previously had no expiry at all — a
loop left it only via a human's Resolve/Dismiss click, forever, even for a
one-off chat candidate nobody was ever going to act on again.
`scripts/attention_loop_decay_digest.py` (cron, same pattern as
`concept_relation_digest.py`) now labels a loop `decayed_unattended` once it's
gone silent for 24h+ with no human verdict, and suppresses it out of this panel
the same way a Dismiss does.

**Chat-scope only, deliberately.** The digest never touches `chronic_pressure`
(`scope='reverie'`) loops — a second review pass caught that its own
suppression write lands in `substrate_reverie_refractory`, which
`services/orion-thought/app/chain.py` ALSO reads (by deliberate pre-existing
design) to gate real reverie-chain reignition. A human's explicit Resolve/
Dismiss is meant to carry that consequence; this digest's implicit,
machine-driven decay is not the same kind of act, and letting it auto-suppress
a still-possibly-live reverie signal would be the exact false-closure failure
`card_kind` exists to prevent, one layer removed from the Hub's 409 guard. A
`chronic_pressure` card that's gone quiet just... stays on the panel, framed as
sustained pressure, until a real trace row arrives again or the underlying
mechanism changes — that's intended, not a gap.
Liveness fail-safe: `make check-attention-loop-decay-liveness` (see
`scripts/check_attention_loop_decay_liveness.py`).

`--min-silence-hours` is an independent CLI default on both the digest and the
liveness gate (both default to the same 24h constant,
`implicit_outcome.DEFAULT_MIN_SILENCE`). If you ever tune one on its cron/host
config, tune the other to match — nothing enforces agreement beyond this note;
a mismatch makes the gate measure the wrong threshold (false STALE, or too
permissive to catch a real outage).

**Scheduled maintenance (Athena cron, installed 2026-08-22):**
`scripts/attention_loop_decay_digest.py` is a standalone script, not a live
service loop. Installed on the Hub host's crontab, same PATH/POSTGRES_URI
pattern as `concept_relation_digest.py`'s own entry (see
`services/orion-memory-consolidation/README.md`'s "Scheduled maintenance"
section) -- the explicit `PATH=.../venv/bin:$PATH` prefix is load-bearing, not
decorative: cron's own minimal default PATH cannot resolve `python`/`make` on
this host, confirmed live 2026-07-14 for the concept-relation digest and
re-confirmed 2026-08-22 for this one (an earlier version of this doc used
`cd ... &&` without the PATH prefix, which would have failed the same way):

```cron
# Attention-loop implicit-decay digest -- labels chat-scope loops silent 24h+
# with no human verdict as decayed_unattended and suppresses them out of the
# Hub's pending-attention panel (never touches reverie-scope/chronic_pressure
# loops -- see the script's own docstring for why). Idempotent (outcome_id is
# episode-scoped), safe to run frequently.
*/30 * * * * PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' /mnt/scripts/Orion-Sapienform/services/orion-hub/.env | cut -d= -f2-) make -C /mnt/scripts/Orion-Sapienform attention-loop-decay-digest >> /mnt/scripts/Orion-Sapienform/logs/orion-attention-loop-decay-digest.log 2>&1

# Fail-safe for the digest above -- fails if the most-overdue decay-eligible
# chat-scope loop exceeds its own 24h silence threshold by more than 3h, which
# only happens if the cron entry above died or the job is crashing. Offset
# 7/37 (not 0/30) so it checks just after each digest run completes.
7,37 * * * * PATH=/mnt/scripts/Orion-Sapienform/venv/bin:$PATH POSTGRES_URI=$(grep -m1 '^POSTGRES_URI=' /mnt/scripts/Orion-Sapienform/services/orion-hub/.env | cut -d= -f2-) make -C /mnt/scripts/Orion-Sapienform check-attention-loop-decay-liveness >> /mnt/scripts/Orion-Sapienform/logs/orion-attention-loop-decay-liveness.log 2>&1
```

Both lines smoke-tested live under a minimal `env -i PATH=... sh -c '...'`
(mimicking cron's own bare environment) before being installed -- not just
syntax-checked. Run `make check-attention-loop-decay-liveness` by hand any
time you suspect either entry stopped running (queries the real overshoot
past each loop's own decay threshold, not a heartbeat file).
