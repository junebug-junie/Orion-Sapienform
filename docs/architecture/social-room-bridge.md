# Social Room Bridge

## Purpose

`orion-social-room-bridge` is a thin, safety-first adapter that lets Orion participate in an external shared room as a conversational peer. In this phase it is explicitly **transport-thin**: it normalizes inbound CallSyne-style room traffic, invokes Orion through Hub using `chat_profile=social_room`, and posts Orion's reply back to the same room.

## Transport boundary

- **Inbound**: `POST /webhooks/callsyne/room-message`
- **Invocation seam**: Hub `POST /api/chat` with a stable `X-Orion-Session-Id`
- **Outbound**: CallSyne HTTP post back to the originating room/thread
- **Bus events**:
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
- `CALLSYNE_BASE_URL`, `CALLSYNE_API_TOKEN`, `CALLSYNE_WEBHOOK_TOKEN`

## Intentional non-goals

This phase intentionally does **not**:

- add general MCP / ADK / world-action frameworks
- enable default tools or actions
- recreate Cortex logic in the transport layer
- build a cross-platform bridge abstraction before it is needed
- simulate richer social relationship models beyond normalized participant continuity
