# PR: CallSyne Bridge Integration Hardening + Social Inspection Panel Live Polling

**Branch:** `feat/social-room-bridge-callsyne-integration` → `main`

---

## Summary

- Hardens the `orion-social-room-bridge` → CallSyne webhook contract to survive any casing variant the CallSyne team ships (snake_case, camelCase, PascalCase, nested `message.*` fields).
- Adds graceful Hub-timeout recovery so a slow Cortex response returns a structured `delivery_failed` 200 rather than crashing the ASGI process and silently eating retries.
- Closes the Social Inspection panel blind spot: bridge-routed `@orion` turns now populate the Hub UI panel within 8 s via a new poll endpoint, matching the experience operators get from direct Hub WebSocket turns.

---

## Changes

### `services/orion-social-room-bridge`

**`app/service.py`**
- `normalize_callsyne_message` now tries `room_id` → `roomId` → `RoomId`, `message_id` → `messageId`, `sender_id` → `senderId`, `thread_id` → `threadId`, and `message.id` / `message.text` nested variants before falling back to empty string. Mirrors CallSyne's real-world payload shapes.
- `process_callsyne_message` wraps `hub_client.chat` in `try/except`. On `httpx.TimeoutException` or any other exception:
  - Evicts the dedupe key so the next CallSyne retry is accepted.
  - Publishes `room_turn_skipped` with `skip_reason=hub_timeout` or `hub_request_failed`.
  - Returns HTTP 200 `{ "status": "delivery_failed" }` so CallSyne's retry logic can proceed cleanly.

**`tests/test_bridge_service.py`**
- `test_normalize_camel_case_callsyne_payload_fields` — camelCase field extraction
- `test_normalize_pascal_case_and_nested_message_payload_fields` — PascalCase + nested `message.*`
- `test_hub_timeout_returns_structured_200_payload_and_allows_retry` — `_FailingHubClient` fixture simulates `TimeoutError`; asserts first call returns `delivery_failed` and a retry from the same message succeeds

### `services/orion-hub`

**`scripts/social_room_inspection_cache.py`** _(new)_
- In-memory store (`dict` keyed by `room_id`) holding the latest `routing_debug` + `stored_at` ISO timestamp after each social-room turn.
- `store(room_id, routing_debug)` / `get(room_id)` / `get_latest()`.

**`scripts/api_routes.py`**
- After every social-room `/api/chat` turn, extracts `external_room.room_id` and writes `routing_debug` to the inspection cache.
- New `GET /api/social-room/inspection/latest?room_id=<optional>` endpoint — returns `{ routing_debug, stored_at, room_id }` or null fields when no turn has been seen yet.

**`static/js/app.js`**
- `setInterval(_pollSocialRoomInspection, 8000)` — polls the new endpoint; compares `stored_at` against the last seen value and calls `syncSocialInspectionFromRouteDebug(data.routing_debug)` only when a newer turn arrives. No-ops gracefully on network errors.

---

## Test Plan

- [ ] `pytest services/orion-social-room-bridge/tests/test_bridge_service.py` — all normalization + timeout tests pass
- [ ] Send a synthetic webhook with camelCase fields; confirm bridge returns 200 and Orion replies
- [ ] Kill Hub mid-turn (or lower `HUB_TIMEOUT_SEC`); confirm bridge returns 200 `delivery_failed`, no ASGI crash, log shows `hub_timeout`
- [ ] With bridge running, tag `@orion` in the CallSyne world room; within 8 s the Hub UI Social Inspection panel should populate with routing + gif sections
- [ ] `GET /api/social-room/inspection/latest` before any turn → `{ routing_debug: null, stored_at: null, room_id: null }`
- [ ] `GET /api/social-room/inspection/latest` after a bridge turn → `social_inspection_present: true`, `sections ≥ 1`
