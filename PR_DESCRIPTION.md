## Summary

- Add inbound CallSyne webhook hardening for `orion-social-room-bridge`:
  - enforce `X-Callsyne-Webhook-Token` (existing behavior),
  - add optional HMAC-SHA256 body signature verification with configurable header/prefix.
- Fix Hub social room bus publishing crash by serializing `SocialRoomTurnV1` payloads before building `BaseEnvelope`.
- Update bridge outbound payload shaping for current CallSyne behavior:
  - post top-level snake_case fields to `/api/bridge/messages`,
  - only include numeric `thread_id` / `reply_to_message_id`.
- Add and update focused tests for webhook auth and social room turn publishing.
- Update bridge docs and env templates with new webhook HMAC settings.

## Why

- Live social room bridge traffic was failing due to:
  1. missing inbound webhook signature validation,
  2. runtime validation crash in Hub (`BaseEnvelope.payload` expected dict),
  3. transport contract mismatches during outbound bridge posting.

## Validation

- `services/orion-social-room-bridge` test coverage:
  - `test_webhook_auth.py` (new)
  - `test_bridge_service.py` targeted serializer tests
- `services/orion-hub` regression test:
  - `test_social_room_turn_publish.py` (new)
- Live runtime verification:
  - signed inbound webhook -> bridge accepted and invoked Hub successfully,
  - Hub returned `200` on `/api/chat`,
  - bridge attempted real outbound `POST /api/bridge/messages` when `SOCIAL_BRIDGE_DRY_RUN=false`.

## Current External Blocker

- Outbound delivery is now blocked by upstream CallSyne configuration:
  - response snippet: `Bridge bot user not configured`
- This indicates API key authentication and request transport are reaching CallSyne, but the bridge bot user mapping on CallSyne side still needs to be configured.

## Test Plan

- [x] Unit/regression tests for webhook auth and payload serialization
- [x] Local bridge health check (`GET /health`)
- [x] Signed webhook E2E into bridge endpoint
- [x] Hub route execution and response logging verification
- [ ] End-to-end room message appearance in CallSyne UI (pending upstream bot user config)
