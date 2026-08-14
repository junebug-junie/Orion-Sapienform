# Hub social room: Claude as a third participant (v1 design)

Status: PROPOSAL (proposal mode per AGENTS.md §0A "Proposal mode before invasive
cognition changes"). Not implemented. Written 2026-08-14.

Goal: a room in the existing Hub chat box where Juniper, Orion, and Claude can
talk to each other. Claude is a *companion* Orion can bounce ideas off when it
wants stronger reasoning — not a tool, not a router, not a subagent. v1 is
manual-invite only; Orion calling on Claude autonomously is v2/v3.

## Arsonist summary

Nearly all of this already exists and is live-verified. The room is real, the
roster is real, N-party turn arbitration is real, and Claude Code already spawns
from inside the Hub container. What is missing is small and specific:

1. Claude's subprocess currently points at the **local FCC gateway**, not
   Anthropic — so "Claude" in Hub today is Claude Code the harness driving local
   models, not Claude the mind. One env fix changes that.
2. Every stored room turn identifies who *spoke to* Orion, never who *answered*.
   A third participant that answers Juniper directly is unrepresentable.
3. `RoomPlatform` is `Literal["callsyne"]` while Hub already emits
   `platform="hub"`. The live code contradicts the contract.

Three thin seams. No new service, no new table, no room server, no cathedral.

## Current architecture

Verified live 2026-08-14 unless marked otherwise.

### The room that already works

`social_room` is a live Hub chat profile, not a plan:
`services/orion-hub/scripts/social_room.py` — profile `social_room` (`:51`),
recall profile `social.room.v1` (`:52`), verb `chat_social_room` (`:53`), a
7-entry skill allowlist (`:54-62`), strict redaction posture (`:68`), and
`hub_direct_room_identity()` (`:95`) stamping
`external_room{platform,room_id}` + `external_participant{id,name,kind}`.

Multi-participant is implemented and exercised:

- `orion/schemas/social_thread.py` — `SocialThreadStateV1.active_participants`
  (`:27`), `audience_scope` (`:28`), `target_participant_id` (`:29`),
  `last_addressed_participant_id` (`:32`), `handoff_flag` (`:35`);
  `SocialHandoffSignalV1` (`:43`) carries directed `from_participant_*` →
  `to_participant_*`, which is meaningless in a 2-party model.
- `services/orion-social-room-bridge/app/policy.py` — `_thread_routing` (`:1110`)
  is an N-party arbiter: `_rank_threads` (`:1378`) scores competing threads,
  `_participant_is_orion` (`:1344`) resolves identity, `:1186` suppresses Orion
  when a message targets another peer, `:1196` waits on ambiguity.
  `gif_policy.py:123` branches on roster ≥ 4.
- ≥7 tests in `services/orion-social-room-bridge/tests/test_bridge_service.py`
  run 3–4 distinct participants (`:494`, `:557`, `:625`, `:694`, `:762`, `:867`,
  `:1240`).

And it has run for real. Live Postgres (`conjourney` on `localhost:55432`):

| room | turns | distinct speakers | roster |
|---|---|---|---|
| `aitown-town` | 1235 | 4 | `["Nico Sable","Tessa Quinn","Sofia Bell","Juno Park"]` |

~26h ending 2026-07-31, via `orion-embodiment` under `profile='social_room'`,
with a populated `social_room_continuity.active_participants` and four
`social_participant_continuity` rows. Speaker identity rides in `client_meta`:

```json
{"external_room": {"room_id":"aitown-town","platform":"aitown"},
 "external_participant": {"participant_id":"p:12","participant_kind":"npc","participant_name":"Sofia Bell"}}
```

### What is NOT live

`orion-social-room-bridge` is running but `SOCIAL_BRIDGE_ENABLED=false`; with
that flag every message short-circuits to `skip` at `service.py:743-757`. Its
transport is hardcoded to CallSyne (`RoomPlatform = Literal["callsyne"]`,
`orion/schemas/social_bridge.py:14`) and the live DB holds **zero** CallSyne rows
— `external_room_messages` and `external_room_participants` are both empty, so
the bridge's roster publish path (`service.py:292`) has never landed a row.

Consequence: the bridge's *policy engine* is the reusable asset; its *transport*
is not on the path for this feature. The live path is Hub's own social_room.

### Claude Code from inside Hub

`services/orion-hub/scripts/fcc_claude_bridge.py` already spawns `claude` as an
async subprocess with `--output-format stream-json`, streams `claude_step` WS
frames (`websocket_handler.py:609-619`), renders them in a live trace panel
(`static/js/agent-claude-trace.js`), persists, and cancels
(`cancel_turn()`, `:494`). `HUB_AGENT_CLAUDE_ENABLED=true` in the live `.env`.
The binary is bind-mounted (`docker-compose.yml:61`) and `IS_SANDBOX=1` is set in
the Dockerfile so `--permission-mode bypassPermissions` will start as root.

**But** `_build_subprocess_env` (`:258-264`) sets:

```python
env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")   # http://127.0.0.1:8082
env["ANTHROPIC_AUTH_TOKEN"] = auth_token
```

so the subprocess never reaches Anthropic. This is why the existing
`agent-claude` lane is not a starter for this feature: it is the FCC harness
lane, by construction.

### Live-verified spawn mechanics

Run directly, not inferred:

| Claim | Evidence |
|---|---|
| Real Claude works **inside** `orion-athena-hub` | `CLAUDE_CONFIG_DIR=<dir with host .credentials.json>` + blanked `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` → `is_error:false`, real tokens |
| It rides the subscription, not API billing | `apiKeySource:"none"`; `~/.claude/.credentials.json` → `subscriptionType: "max"` |
| Conversational continuity across separate subprocesses | `--session-id <uuid>` then `--resume <uuid>`: recalled a name and a number from turn 1; resume preserved the session id |
| Per-turn cost is emitted by the CLI | `total_cost_usd` ≈ $0.008 cold / $0.004 resumed at minimal context |
| Persona injection | `--append-system-prompt` |
| The CLI writes into `CLAUDE_CONFIG_DIR` | created `projects/`, `sessions/`, `backups/`, `.claude.json` → dir must be `rw` |
| `.credentials.json` was read, not rewritten, in these runs | mtime/uid unchanged. Token expiry is ~7.5h out with a refresh token, so a refresh **will** rewrite it — see Risks |

## Decisions taken (Juniper, 2026-08-14)

| Question | Decision |
|---|---|
| When does Claude speak | **Explicit invite only.** Button click or `@claude`. No AI↔AI ping-pong in v1. |
| Transport into the chat box | **Bus-native, whole message.** No token streaming in v1. |
| Spend cap | **No cap.** Meter `total_cost_usd`, never refuse — collect clean unconstrained data to size the v2 autonomy budget. |
| What Claude sees | **Room-native: same as any participant.** Transcript + the `social_memory /summary` block (roster, room continuity, stance, open threads) under the existing strict redaction posture. Tools off. |
| Where it runs | **Inside the Hub container.** No host daemon, no new service. |

## Proposed schema / API changes

Deliberately three small changes, each with a producer, a consumer and a test.

### 1. `RoomPlatform` widening

`orion/schemas/social_bridge.py:14`:

```python
RoomPlatform = Literal["callsyne", "hub", "aitown"]
```

This is a bug fix as much as a feature: `social_room.py:86` already sets
`HUB_DIRECT_ROOM_PLATFORM = "hub"`, and 67 live `social_room_continuity` rows
carry `platform='aitown'` — both invalid against today's Literal. Widening makes
the contract describe reality and lets a `hub` room reach the policy engine.

### 2. `external_responder` in `client_meta`

Today `client_meta.external_participant` identifies the *prompter*; the responder
is assumed to be Orion. For a room where Claude answers Juniper, add a sibling
block, same shape:

```json
{"external_responder": {"participant_id":"claude","participant_kind":"peer_ai","participant_name":"Claude"}}
```

Absent → Orion, so every existing row stays valid and no migration is needed.
`ParticipantKind` already includes `peer_ai`
(`orion/schemas/social_bridge.py:15`), so Claude needs no new kind.

Reader change: `services/orion-social-memory/app/synthesizer.py` must fold the
responder into `active_participants` (it currently merges the prompter only, via
`merge_unique` at `:2222`/`:3571`) so the roster reads
`["Juniper","Oríon","Claude"]` rather than dropping Claude.

### 3. Two bus channels

Catalogued in `orion/bus/channels.yaml`, registered in
`orion/schemas/registry.py`:

| Channel | Kind | Producer → Consumer |
|---|---|---|
| `orion:room:claude:request` | `room.claude.request.v1` | Hub → Hub companion worker |
| `orion:room:claude:utterance` | `room.claude.utterance.v1` | companion worker → Hub WS relay |

Request carries `{room_id, correlation_id, invited_by, transcript, social_memory_summary}`.
Utterance carries `{room_id, correlation_id, text, claude_session_id, model, cost_usd, duration_ms}`.

`cost_usd` is copied verbatim from the CLI's `total_cost_usd`, never recomputed.

## Files likely to touch

| Path | Why |
|---|---|
| `orion/schemas/social_bridge.py` | widen `RoomPlatform` |
| `orion/schemas/social_chat.py` | optional `external_responder` on the turn payload |
| `orion/schemas/room_claude.py` *(new)* | the two channel payloads above |
| `orion/schemas/registry.py`, `orion/bus/channels.yaml` | register both |
| `services/orion-hub/scripts/claude_companion.py` *(new)* | spawn + session resume + cost extraction; a sibling of `fcc_claude_bridge.py`, NOT a modification of it |
| `services/orion-hub/scripts/social_room.py` | register Claude as a room participant; build its room-native context |
| `services/orion-hub/scripts/websocket_handler.py` | relay the utterance frame into the socket |
| `services/orion-hub/scripts/chat_history.py` | stamp `external_responder` on published turns |
| `services/orion-hub/templates/index.html` | invite button by `#presenceOpenButton` (`:344`) |
| `services/orion-hub/static/js/app.js` | `'Claude'` speaker in `appendMessage` (`:7011`); markdown currently renders only for `'Orion'` (`:7054`) |
| `services/orion-hub/docker-compose.yml`, `.env_example`, `app/settings.py` | config dir mount + new `HUB_ROOM_CLAUDE_*` keys |
| `services/orion-social-memory/app/synthesizer.py` | fold responder into the roster |

New settings, all `HUB_ROOM_CLAUDE_*` prefixed to stay clear of the
`HUB_AGENT_CLAUDE_*` (FCC) namespace: `ENABLED`, `MODEL`, `EFFORT`,
`CONFIG_DIR`, `TIMEOUT_SEC`, `PARTICIPANT_NAME`.

## Non-goals

- Orion invoking Claude autonomously (v2) or supervising it as an agent (v3).
- Token-level streaming of Claude's reply.
- Tool use, repo access, or file writes by Claude. `--tools ""` in v1.
- Reviving the CallSyne bridge or `SOCIAL_BRIDGE_ENABLED`.
- Touching the FCC `agent-claude` lane. It keeps pointing at the local gateway.
- A spend cap, a room server, a participant-management UI, or >1 room.

## Proposal-mode disclosures

**Capability change.** Orion gains a peer in its own conversational space whose
reasoning it does not produce. v1 keeps Orion out of the trigger path entirely:
only Juniper can summon Claude.

**Data touched.** Room transcript and the `social_memory /summary` block for the
Hub room, sent to Anthropic. Writes: `social_room_turns`,
`social_room_continuity.active_participants`, `social_participant_continuity`,
and a new Claude Code session store under a dedicated `CLAUDE_CONFIG_DIR`.

**Privacy boundary.** Claude receives exactly what any room participant
receives, under the existing strict redaction posture
(`social_room.py:68`, `_BLOCKED_MEMORY_RE` at `:70` already blocks
sealed/private/journal/mirror material). Explicitly NOT sent: recall results,
self-state, drives, journals, mirrors, or anything outside the room. Because
Claude is a third party, the redaction posture is a hard floor here, not a
default — a `relaxed` posture must not apply to a Claude-bound turn.

**Trace that proves it worked.** A `room.claude.utterance.v1` on the bus with a
non-empty `text`, a `cost_usd > 0`, and a `claude_session_id` that is stable
across turns; a `social_room_turns` row carrying `external_responder`; and
`active_participants` containing all three names. Empty text or `cost_usd == 0`
is a failure, never a success (AGENTS.md §0A "No empty-shell cognition").

**Dangerous failure modes.** (a) Credential exposure — mitigated by a dedicated
config dir, never `~/.claude`. (b) Runaway AI↔AI loop — structurally impossible
in v1, since only a human click emits a request. (c) Silent fallback to the FCC
gateway producing a local model's text labelled "Claude" — guarded by asserting
`ANTHROPIC_BASE_URL` is unset and recording the CLI's reported model on every
utterance. (d) Redaction bypass leaking private memory to a third party.

**Rollback.** `HUB_ROOM_CLAUDE_ENABLED=false` disables the button and the worker.
The schema additions are additive and inert when unused; no migration to unwind.

## Cost metric gate (AGENTS.md §0A)

Required before `cost_usd` is wired anywhere downstream.

1. **Provenance.** Claude Code CLI's own `total_cost_usd` in the
   `--output-format json` result object. Copied verbatim, never recomputed.
2. **Independence.** Not derived from any metric currently in any Orion model.
   The one adjacent surface is `orion/dev_economics/claude_code_ingest.py`, which
   reads `~/.claude/projects/*.jsonl` for dev-session accounting — a dedicated
   `CLAUDE_CONFIG_DIR` keeps room spend out of that ledger, so they cannot
   double-count.
3. **Theory anchor.** It is a billed dollar amount reported by the billing
   system itself, not a proxy for one.
4. **Live-data sanity.** Observed non-degenerate and non-flat across real runs:
   $0.0084 (cold), $0.0037 (resumed), $0.0333 (in-container cold with cache
   creation). Rest state is genuine: no invocation ⇒ no event, so this is
   event-triggered and cannot be read as "calm" when idle — consumers must treat
   absence as absence, not as zero.
5. **Existing mechanism.** `orion/dev_economics/` already prices Claude Code
   usage but only by post-hoc transcript scraping; nothing meters a live
   in-room turn.
6. **Reversibility.** One field on one new channel. Cheap to remove.

## Acceptance checks

1. `claude -p` from inside `orion-athena-hub` returns `is_error:false` with
   `apiKeySource:"none"` — subscription, not API key.
2. Two consecutive invites reuse one `claude_session_id`, and Claude
   demonstrably recalls turn 1 in turn 2.
3. A `social_room_turns` row exists with `external_responder.participant_name ==
   "Claude"`.
4. `social_room_continuity.active_participants` for the Hub room contains all
   three names.
5. Cumulative `cost_usd` over a session is > 0 and matches the sum of per-turn
   CLI values.
6. With `HUB_ROOM_CLAUDE_ENABLED=false`, the button is absent and no worker runs.
7. A Claude-bound turn carrying blocked material (`sealed`/`private`/`journal`)
   is redacted before dispatch — regression test, not a manual check.
8. Existing FCC `agent-claude` tests still pass unchanged.

## Recommended next patch (phase 1 only)

Smallest slice that proves the whole path, ending at a checkpoint:

1. `RoomPlatform` widening + `external_responder` field + registry/channel
   entries, with tests.
2. `claude_companion.py`: spawn, resume, extract text + `total_cost_usd`.
3. A CLI smoke that runs one real in-container turn and prints the utterance
   payload — before any UI exists.

Stop there for review. UI, bus relay, roster fold, and redaction regression land
in phase 2 once phase 1 shows a real Claude utterance with a real cost.
