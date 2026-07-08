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

- **Quick (default):** `chat_quick` on the **chat exec lane** (`EMBODIMENT_CORTEX_REQUEST_CHANNEL=orion:cortex:exec:request:chat`). Never use the legacy `orion:cortex:exec:request` intake — it shares the queue with heavy `chat_general` / harness work and starves town turns.
- **Grounded (optional):** when `EMBODIMENT_SPEECH_UNIFIED_ENABLED=true`, try `chat_general` with `surface=aitown` / `grounded_small` first, then fall back to quick on timeout/error/empty.

The dispatcher tries grounded first only when unified is enabled. With `EMBODIMENT_SPEECH_UNIFIED_ENABLED=false` (default) it calls the quick lane directly.

**Env contract:** `services/orion-embodiment/.env_example` is the Athena operator source of truth. After any `.env_example` edit run `python scripts/sync_local_env_from_example.py orion-embodiment` from repo root — do not maintain a separate override block in `.env`.

| Env | Default | Purpose |
|-----|---------|---------|
| `EMBODIMENT_SPEECH_UNIFIED_ENABLED` | `false` | Optional grounded_small `chat_general` pass before quick fallback |
| `EMBODIMENT_CORTEX_REQUEST_CHANNEL` | `orion:cortex:exec:request:chat` | Chat exec lane intake (not legacy) |
| `EMBODIMENT_SPEECH_HUB_LLM_ROUTE` | `chat` | LLM route when unified grounded pass is enabled |

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
