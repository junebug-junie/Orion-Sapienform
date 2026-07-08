# orion-embodiment

Mind-to-sprite bridge: gives Orion a persistent AI Town body driven by its own state. This service is the **sole Convex actuator/perceiver**. Producers only publish semantic `EmbodimentIntentV1` events; this worker arbitrates (deliberate preempts involuntary for a hold window), resolves semantic intents to `{x,y}`, actuates via Convex `sendInput(moveTo)`, and emits outcomes + perception.

**All flags default off.** With `ORION_EMBODIMENT_ENABLED=false` the worker connects nothing and actuates nothing.

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

- **Unified (default):** `POST {EMBODIMENT_HUB_CHAT_URL}` with `{"mode": "orion", "session_id": "<prefix>:<conversation_id>", "messages": [{"role": "user", "content": <prompt>}]}`. The hub runs the full unified-turn saga (full cognition pass) and returns a final frame; its `llm_response` becomes the town utterance. `session_id` uses `EMBODIMENT_UNIFIED_SESSION_PREFIX` + the active AI Town conversation id.
- **Quick (fallback):** the legacy cortex-exec rail (`chat_quick` on lane `quick`) over the bus RPC channel `EMBODIMENT_CORTEX_REQUEST_CHANNEL`.

The dispatcher tries unified first when `EMBODIMENT_SPEECH_UNIFIED_ENABLED=true`. On timeout, HTTP/JSON error, a non-`final` frame (`turn_deferred`/`turn_error`), or an empty `llm_response`, it logs one `embodiment_speech_unified_fallback reason=<...>` INFO line and falls back to the quick lane. With `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false` it calls the quick lane directly (preserves legacy behavior).

**Dependency:** the unified path only works if the hub at `EMBODIMENT_HUB_CHAT_URL` has `ORION_UNIFIED_TURN_ENABLED=true` **and** `ORION_HARNESS_GOVERNOR_ENABLED=true`. Otherwise the hub returns a deferred/error frame and Orion always falls back to the quick lane.

| Env | Default | Purpose |
|-----|---------|---------|
| `EMBODIMENT_SPEECH_UNIFIED_ENABLED` | `true` | Prefer the unified turn; `false` = quick-only legacy path |
| `EMBODIMENT_HUB_CHAT_URL` | `http://100.92.216.81:8080/api/chat` | Hub unified-turn endpoint. Hub is host-networked, so use the node's host-reachable (Tailscale) IP, not the Docker service name |
| `EMBODIMENT_UNIFIED_TIMEOUT_SEC` | `120` | Unified turn HTTP timeout (sec) |
| `EMBODIMENT_UNIFIED_SESSION_PREFIX` | `aitown` | Prefix for the unified `session_id` |

## Facing the conversation partner

The AI Town engine (`convex/aiTown/conversation.ts` `Conversation.tick`) orients **both** participants toward each other on every tick — but **only for a participant that is NOT pathfinding** (`if (!player.pathfinding) player.facing = v`). Orion is an externally-driven "join" player with no town-AI agent, so if it reaches `participating` still carrying a lingering movement path, the engine never turns it to face the partner.

This service guarantees the `!pathfinding` precondition for Orion:

- **Perception surfaces own state.** `WorldPerceptionV1` now carries Orion's own `facing` ({dx,dy}) and `pathfinding` (bool), read from the raw serialized player (`convex/aiTown/player.ts` `serialize()`: `facing`, optional `pathfinding`). When Orion is `participating`, `active_conversation.facing_partner` reports whether Orion's `facing` vector aligns (dot product ≥ 0.7) with the direction to the partner — `True`/`False`, or `None` when facing/positions are unknown.
- **Worker clears the lingering path.** In `_engage_conversation`, when the active conversation is `participating` **and** `perception.pathfinding` is truthy, the worker issues **one** stop per conversation id — a `moveTo` to Orion's own **exact current position** (a zero-length path the engine resolves to an immediate stop, with no micro-move that would keep `pathfinding` truthy) — so the engine's next `Conversation.tick` orients Orion. It is guarded by `_faced_conversations` (fires at most once per conversation) and does **not** fire when `pathfinding` is falsy, to avoid fighting the engine's own post-transition move / spamming inputs at the shared engine. Fail-open (logged, never crashes the loop).
- **Observability.** The `~30s` `embodiment_heartbeat` INFO line includes `facing_partner=<True|False|None>`, and a one-shot `embodiment_face_partner_stop convo=<id> pos=(x,y)` line records each stop.

**Group chat is N/A** — this engine's conversations are strictly 2-party, so there is no multi-party orientation case to handle.

> **UNVERIFIED at runtime.** The live town is not runnable in CI, so the visual confirmation that Orion actually turns to face its partner is deferred to the operator. The logic (perception fields, `facing_partner`, the one-shot stop, and the heartbeat/log surfaces) is covered by unit tests, but "Orion visibly faces the partner in the live world" has not been observed by this change.

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
