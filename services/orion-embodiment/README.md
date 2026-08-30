# orion-embodiment

Mind-to-sprite bridge: gives Orion a persistent AI Town body driven by its own state. This service is the **sole Convex actuator/perceiver**. Producers only publish semantic `EmbodimentIntentV1` events; this worker arbitrates (deliberate preempts involuntary for a hold window), resolves semantic intents to `{x,y}`, actuates via Convex `sendInput(moveTo)`, and emits outcomes + perception.

**All flags default off.** With `ORION_EMBODIMENT_ENABLED=false` the worker connects nothing and actuates nothing.

Independent of `ORION_EMBODIMENT_ENABLED`, this service always publishes a bus-native
`SystemHealthV1` heartbeat to `orion:system:health` every `HEARTBEAT_INTERVAL_SEC` (default
10s), on its own connection separate from the worker's intent/outcome/perception bus traffic
above -- process liveness, not town actuation.

## Inputs

- `orion:embodiment:intent` — `EmbodimentIntentV1` (from substrate-runtime, harness-governor, cortex-exec)

## Outputs

- `orion:embodiment:outcome` — `EmbodimentOutcomeV1`
- `orion:embodiment:perception` — `WorldPerceptionV1`

## Port

`8130` (`EMBODIMENT_PORT`)

## Bus / Redis

| Env | Default | Purpose |
|-----|---------|---------|
| `ORION_BUS_URL` | `${ORION_BUS_URL}` (pass-through from root `.env`; Tailscale node) | Redis URL for Orion bus |
| `ORION_BUS_ENABLED` | `true` | Disable bus connect/publish when `false` |
| `ORION_EMBODIMENT_ENABLED` | `false` | Master switch — worker does nothing unless `true` |
| `EMBODIMENT_CHANNEL_INTENT` | `orion:embodiment:intent` | Intent subscribe channel |
| `EMBODIMENT_CHANNEL_OUTCOME` | `orion:embodiment:outcome` | Outcome publish channel |
| `EMBODIMENT_CHANNEL_PERCEPTION` | `orion:embodiment:perception` | Perception publish channel |

Never hardcode `bus-core` / `redis://redis`. `ORION_BUS_URL` passes through from the root `.env`.

## Town speech (unified vs quick)

When `EMBODIMENT_SPEECH_ENABLED=true`, Orion generates town utterances via `_request_utterance`, a dispatcher with two lanes:

- **Quick (default):** `chat_quick` on the **chat exec lane** (`EMBODIMENT_CORTEX_REQUEST_CHANNEL=orion:cortex:exec:request:chat`). Never use the legacy `orion:cortex:exec:request` intake — it shares the queue with heavy `chat_general` / harness work and starves town turns.
- **Grounded (optional):** when `EMBODIMENT_SPEECH_UNIFIED_ENABLED=true`, try `chat_general` with `surface=aitown` / `grounded_small` first, then fall back to quick on timeout/error/empty.

The dispatcher tries grounded first only when unified is enabled. With `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false` (default) it calls the quick lane directly.

**Env contract:** `services/orion-embodiment/.env_example` is the Athena operator source of truth. After any `.env_example` edit run `python scripts/sync_local_env_from_example.py orion-embodiment` from repo root — do not maintain a separate override block in `.env`.

| Env | Default | Purpose |
|-----|---------|---------|
| `EMBODIMENT_SPEECH_UNIFIED_ENABLED` | `false` | Optional grounded_small `chat_general` pass before quick fallback |
| `EMBODIMENT_CORTEX_REQUEST_CHANNEL` | `orion:cortex:exec:request:chat` | Chat exec lane intake (not legacy) |
| `EMBODIMENT_SPEECH_HUB_LLM_ROUTE` | `chat` | LLM route when unified grounded pass is enabled |
| `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE` | `quick_background` | LLM route for the live quick-lane path (`_request_utterance_quick`, active when unified is off, the default). Separate from `EMBODIMENT_SPEECH_LANE`, which is cognition "mode" metadata only, not a gateway route -- see below. |

**2026-07-30 fix:** `_request_utterance_quick` previously only set `extra={"lane": EMBODIMENT_SPEECH_LANE}` when calling orion-cortex-exec, but that service's `llm_route` override resolution reads `ctx["llm_route"]`, not `ctx["lane"]` -- so `EMBODIMENT_SPEECH_LANE` never actually changed which gateway route town speech used; the observed "quick" behavior came entirely from cortex-exec's own per-verb default mapping, coincidentally matching. `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE` is now threaded through as the actual override, defaulting to `quick_background` so Orion's own ai-town dialogue shares `atlas-worker-fast-1` without competing evenly with `orion-mind`/`orion-hub`'s own `quick` traffic -- see `services/orion-llm-gateway/README.md`'s "Background-priority routes".

## Spatial grounding: named map landmarks

`EMBODIMENT_LOCATIONS_JSON` (name -> tile `{x,y}`) is a dual-use registry:

- **Movement** (pre-existing): `go_to_location` intents resolve a named destination through it (`orion/embodiment/resolver.py`).
- **Perception/speech grounding** (new): `build_perception` (`orion/embodiment/perception.py`) computes `nearby_landmarks` (nearest 3, sorted by distance -- a hardcoded default, matching how `nearby_players`' cap is also hardcoded, not env-configurable) from the same registry against Orion's live position every tick, and `build_speech_prompt` (`orion/embodiment/speech.py`) includes a short `Nearby: <names>.` clause when non-empty. This exists because live dialogue inspection (2026-07-30) showed Orion using conversation partners' names correctly but with content entirely disconnected from the actual scene -- the prompt had no way to reference anything physically nearby since nothing in perception knew what was actually on the map. The model still chooses what to say; this only ensures the grounding material is present to draw from.

`.env_example` ships this pre-populated with the real landmarks in this world's actual map file (`services/orion-ai-town/upstream/data/gentle.js`'s `animatedsprites`, pixel coords / `tiledim=32`): a campfire, 3 windmills, and a stream/waterfall feature. If the map changes, regenerate this list from the new map file the same way -- see `docs/superpowers/specs/2026-07-30-aitown-spatial-grounding-design.md`.

**Line-of-sight (approximate).** Both `nearby_players` and `nearby_landmarks` entries carry `is_visible`, computed via a grid raycast (`orion/embodiment/perception.py`'s `_has_line_of_sight`) against `orion/embodiment/worldmap.py`'s `walkable_tiles` -- the same movement-collision layer AI Town itself uses for pathfinding, reused here as a stand-in for sight-blocking. This is an honest approximation, not a real visibility layer: the map has no distinct "blocks sight" data, so a walk-blocking-but-see-through tile (a short fence) would be wrongly treated as blocking sight, and the reverse (sight-blocking-but-walkable) can't be represented at all. Fails open (`is_visible=True`) when map data is unavailable, so a missing/failed load never hides something real. Only visible landmarks are surfaced in the speech prompt's `Nearby:` clause. See `docs/superpowers/specs/2026-07-30-aitown-los-and-social-recall-design.md`.

## Social-memory read-back into town speech

`EMBODIMENT_CONVERSATION_MEMORY_ENABLED` also gates the read-back half of conversation memory: `_fetch_participant_continuity` (`app/worker.py`) calls `orion-social-memory`'s `GET /summary` (`EMBODIMENT_SOCIAL_MEMORY_URL`, same tailscale-IP-reachability constraint as `EMBODIMENT_HUB_CHAT_URL`) for the current conversation partner and threads the result into the speech request as `metadata.aitown_participant_continuity`. `orion/cognition/prompts/chat_quick.j2` (the actual live template -- `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false` by default, so `chat_general.j2`'s block is the optional grounded-lane counterpart) renders it as a distinct field from `juniper_relationship_summary`, since a town NPC/human partner isn't Juniper. This closes the gap the earlier conversation-memory patch left: writing durable memory with nothing reading it back before Orion spoke again. Fail-open: any fetch error/timeout/missing row means no continuity, not a crash.

**Participant keying.** Social-memory-facing `participant_id` is the `orion.town_cast` slug (`slug_for_name`), never a Convex `p:*` id. `_publish_conversation_memory` and `_fetch_participant_continuity` skip the social write/fetch (fail-open) when the partner name is unknown. Social `external_room.thread_id` is `thread_id_for("Orion", participant_name)` when both slugs exist (e.g. `orion--sofia-bell`). `chat_history_log` may still store the Convex player_id. A Convex wipe orphans old `p:*` social-memory rows; after wipe, continuity only accumulates under the slug keys. See the operator wipe runbook in `docs/superpowers/specs/2026-08-29-aitown-cast-cull-and-town-continuity-design.md`.

**Room scoping.** `orion-social-memory` keys relationship continuity on `platform:room_id:participant_id` -- both `_publish_conversation_memory`'s social-turn write and `_fetch_participant_continuity`'s read use the fixed constant `SOCIAL_MEMORY_ROOM_ID = "aitown-town"`, NOT the per-conversation `conversation_id`. ai-town mints a brand-new `conversation_id` every time two characters start talking, so keying on it fragmented each NPC's relationship into a new, disconnected row every conversation instead of one accumulating over time -- caught live within hours of first deploying this feature (Sofia Bell already had two orphaned rows). Mirrors `services/orion-hub/scripts/social_room.py`'s `HUB_DIRECT_ROOM_ID = "hub-direct"`, itself a fixed constant across all of hub's direct-chat history -- proof `room_id` was always meant to mean "stable venue," not "this specific exchange." Deliberately NOT `world_id` either: this repo has already used a full world wipe+reseed (`testing:wipeAllTables` + `init`) as a real recovery path, which would reset every relationship if room_id were world-scoped. `chat_history_log`'s `session_id`/`client_meta` stay conversation-scoped -- verbatim recall genuinely should distinguish separate conversation instances; only the social-memory-facing room_id needed to change. See `docs/superpowers/specs/2026-07-30-aitown-social-memory-room-scope-design.md` and `docs/superpowers/specs/2026-07-30-aitown-los-and-social-recall-design.md`.

## Facing the conversation partner

The AI Town engine (`convex/aiTown/conversation.ts` `Conversation.tick`) orients **both** participants toward each other on every tick — but **only for a participant that is NOT pathfinding** (`if (!player.pathfinding) player.facing = v`). Orion is an externally-driven "join" player with no town-AI agent, so if it reaches `participating` still carrying a lingering movement path, the engine never turns it to face the partner.

This service guarantees the `!pathfinding` precondition for Orion:

- **Perception surfaces own state.** `WorldPerceptionV1` now carries Orion's own `facing` ({dx,dy}) and `pathfinding` (bool), read from the raw serialized player (`convex/aiTown/player.ts` `serialize()`: `facing`, optional `pathfinding`). When Orion is `participating`, `active_conversation.facing_partner` reports whether Orion's `facing` vector aligns (dot product ≥ 0.7) with the direction to the partner — `True`/`False`, or `None` when facing/positions are unknown.
- **Worker clears the lingering path.** In `_engage_conversation`, when the active conversation is `participating` **and** `perception.pathfinding` is truthy, the worker issues **one** stop per conversation id — a `moveTo` to Orion's own **exact current position** (a zero-length path the engine resolves to an immediate stop, with no micro-move that would keep `pathfinding` truthy) — so the engine's next `Conversation.tick` orients Orion. It is guarded by `_faced_conversations` (fires at most once per conversation) and does **not** fire when `pathfinding` is falsy, to avoid fighting the engine's own post-transition move / spamming inputs at the shared engine. Fail-open (logged, never crashes the loop).
- **Observability.** The `~30s` `embodiment_heartbeat` INFO line includes `facing_partner=<True|False|None>`, and a one-shot `embodiment_face_partner_stop convo=<id> pos=(x,y)` line records each stop.

**Group chat is N/A** — this engine's conversations are strictly 2-party, so there is no multi-party orientation case to handle.

> **UNVERIFIED at runtime.** The live town is not runnable in CI, so the visual confirmation that Orion actually turns to face its partner is deferred to the operator. The logic (perception fields, `facing_partner`, the one-shot stop, and the heartbeat/log surfaces) is covered by unit tests, but "Orion visibly faces the partner in the live world" has not been observed by this change.

## Memory: journal facts vs conversation content

Two independent flags, both off by default:

- **`EMBODIMENT_MEMORY_ENABLED`** — journals the *fact* that a salient episode happened (partner name, utterance count, or a first-sighting encounter) as a `JournalTriggerV1` (`trigger_kind="town_episode"`) to `orion-actions`'s journal pipeline. No conversation content involved.
- **`EMBODIMENT_CONVERSATION_MEMORY_ENABLED`** — richer capture, on top of the above. Per Orion-reply exchange, publishes to two existing hub-owned bus rails (not new infrastructure):
  - `orion:chat:history:turn` (`chat.history`) → `chat_history_log` — the table `orion-recall` reads verbatim text from. This is genuine, retrievable recall of what was actually said.
  - `orion:chat:social:turn` (`social.turn.v1`) → a *separate* table (`social_room_turns`), whose re-emission feeds `orion-social-memory`'s rolling per-participant relationship synthesis (tone, shared topics, trust) — tagged `platform="aitown"`, keyed so it cannot collide with hub/callsyne relationships for the same participant id.

  Both publishes share one correlation_id per exchange; `EMBODIMENT_MEMORY_ENABLED`'s journal entry for the conversation cross-references it via `spawned_correlation_id`, and is seeded with the real transcript (`JournalTriggerV1.prompt_seed`) instead of only the templated fact — **requires both flags on**, not `EMBODIMENT_CONVERSATION_MEMORY_ENABLED` alone.

  Participants are tagged `participant_kind`: `"human"` or `"npc"` (from ai-town's own `human` player field, forwarded by `perception.py`). **Privacy caveat, stated plainly:** there is no enforced privacy mechanism anywhere in this codebase for this class of content today — `chat_history_log` has no visibility/redaction gate, and the nearest analog (`SocialRedactionScoreV1.recall_safe`/`redaction_level`) is computed elsewhere but never consumed by any reader. A human-participant town conversation gets exactly the same (lack of dedicated) protection as any other Orion-Juniper conversation already flowing through these tables — this flag does not add new protection, and does not reduce what already exists. One concrete consequence worth naming: `orion-social-memory`'s `GET /summary`/`GET /inspection` endpoints take `platform`/`room_id`/`participant_id` as unauthenticated query params — with this flag on, that already-pre-existing exposure now also covers `platform=aitown` relationship summaries derived from human-participant town conversations, same as it already does for `platform=hub`.

  See `docs/superpowers/specs/2026-07-30-aitown-conversation-memory-design.md` for the full design rationale.

## Secrets (`~/.fcc/.env`)

`AITOWN_CONVEX_URL`, `AITOWN_ADMIN_KEY`, `AITOWN_WORLD_ID`, `AITOWN_ORION_PLAYER_ID`, `AITOWN_ORION_AGENT_ID` are loaded from `~/.fcc/.env` (mounted read-only at `/root/.fcc/.env`), **not** from this service `.env`.

## Bootstrap (create/update Orion's body)

```bash
python services/orion-embodiment/scripts/bootstrap_orion_agent.py           # dry-run
python services/orion-embodiment/scripts/bootstrap_orion_agent.py --write    # persist ids to ~/.fcc/.env
```

The persona is a privacy-filtered projection of Orion's live self-model.

## Run

```bash
cd services/orion-embodiment
cp -n .env_example .env
docker compose --env-file ../../.env --env-file .env up -d --build
curl -fsS http://localhost:8130/health
```
