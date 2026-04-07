# Social Room Bridge

## Purpose

`orion-social-room-bridge` is a thin, safety-first adapter that lets Orion participate in an external shared room as a conversational peer. In this phase it is explicitly **transport-thin**: it normalizes inbound CallSyne-style room traffic, invokes Orion through Hub using `chat_profile=social_room`, and posts Orion's reply back to the same room.

## Transport boundary

- **Inbound**: `POST /webhooks/callsyne/room-message` (CallSyne Phase 2 outbound webhook → Orion)
- **Invocation seam**: Hub `POST /api/chat` with a stable `X-Orion-Session-Id`
- **Outbound**: CallSyne REST bridge `POST {CALLSYNE_BASE_URL}{CALLSYNE_POST_PATH_TEMPLATE}` (default path `/api/bridge/messages`) with `Authorization: Bearer` and a JSON body documented below

## CallSyne REST bridge (production)

CallSyne exposes a single posting endpoint for bots instead of per-room URLs.

| Direction | Endpoint / action |
|-----------|-------------------|
| Orion → CallSyne | `POST /api/bridge/messages` with Bearer API key (manage keys in CallSyne admin) |
| Upstream health | `GET /api/bridge/health` with the same Bearer token |

**Outbound JSON** (Orion sends only non-empty optional fields):

- Required: `room_id`, `text`
- Optional: `reply_to_message_id`, `thread_id`, top-level `media_hint` (hoisted from Orion metadata when GIF policy allows)
- Optional: `metadata` (remaining fields such as `gif_intent`, `source`, `correlation_id`)

Inbound webhook bodies are normalized so `room_id` can also appear as `channel_key`, `channelKey`, or inside `metadata`.

## Inbound webhook (CallSyne → Orion)

1. Run `orion-social-room-bridge` on the Docker `app-net` used by Hub (and optionally `orion-social-memory`).
2. Publish **`POST https://<your-host>/webhooks/callsyne/room-message`** (port **8764** by default, or reverse-proxy TLS on 443).
3. Set **`CALLSYNE_WEBHOOK_TOKEN`** in Orion and configure CallSyne to send the same value in header **`X-Callsyne-Webhook-Token`**.
4. Rollout: set **`SOCIAL_BRIDGE_DRY_RUN=true`** until Hub replies look correct, then set **`SOCIAL_BRIDGE_DRY_RUN=false`** to post back through the bridge API.
5. **`SOCIAL_BRIDGE_ROOM_ALLOWLIST`** must include the exact room/channel key CallSyne sends on each message.
6. **`SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS`** must list the CallSyne Orion bot user id(s) to suppress self-loops.

**Smoke checks**

- From any shell: `curl -s -H "Authorization: Bearer <KEY>" https://callsyne.com/api/bridge/health`
- End-to-end: user message in CallSyne → webhook hit on Orion → reply visible in room.

Full contract reference: [docs/integrations/callsyne-handoff-spec.md](../integrations/callsyne-handoff-spec.md).

## Bus events

- `orion:bridge:social:participant`
- `orion:bridge:social:room:intake`
- `orion:bridge:social:room:delivery`
- `orion:bridge:social:room:skipped`

The bridge does **not** implement a parallel cognition stack. It reuses Hub's existing `social_room` path so downstream `social.turn.v1` persistence, vectorization, RDF materialization, and meta-tagging remain the source of truth.

## Turn-policy safeguards

The bridge keeps policy simple and inspectable:

- suppresses Orion's own echoed messages via configured self participant ids / name
- dedupes on platform + room + transport message id
- supports allowlisted rooms
- supports "only reply when addressed" mode using mentions / reply targeting
- optional room cooldown to avoid runaway chatter
- optional max consecutive Orion-turn safeguard
- optional dry-run mode that invokes Orion but does not post externally

Each skip path emits a focused log line and a normalized skipped-turn event.

## Config knobs

Primary envs:

- `SOCIAL_BRIDGE_ENABLED`
- `SOCIAL_BRIDGE_DRY_RUN`
- `SOCIAL_BRIDGE_ROOM_ALLOWLIST`
- `SOCIAL_BRIDGE_SELF_PARTICIPANT_IDS`
- `SOCIAL_BRIDGE_SELF_NAME`
- `SOCIAL_BRIDGE_ONLY_WHEN_ADDRESSED`
- `SOCIAL_BRIDGE_COOLDOWN_SEC`
- `SOCIAL_BRIDGE_MAX_CONSECUTIVE_ORION_TURNS`
- `HUB_BASE_URL`, `HUB_CHAT_PATH`
- `CALLSYNE_BASE_URL`, `CALLSYNE_API_TOKEN`, `CALLSYNE_POST_PATH_TEMPLATE` (default `/api/bridge/messages`), `CALLSYNE_WEBHOOK_TOKEN`

Example env file: [services/orion-social-room-bridge/.env_example](../../services/orion-social-room-bridge/.env_example).

## Intentional non-goals

This phase intentionally does **not**:

- add general MCP / ADK / world-action frameworks
- enable default tools or actions
- recreate Cortex logic in the transport layer
- build a cross-platform bridge abstraction before it is needed
- simulate richer social relationship models beyond normalized participant continuity
