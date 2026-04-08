# CallSyne → Orion social-room bridge handoff spec

## Overview

This handoff spec covers the **bridge-facing integration seam** needed to connect CallSyne to Orion as a bounded social-room participant.

For this integration, Orion sits behind a **thin social-room bridge** that:

1. receives room messages from CallSyne,
2. normalizes them into a stable message shape,
3. invokes Orion through the existing `social_room` path,
4. receives Orion’s reply,
5. posts the reply back into the originating room/thread.

This integration is intentionally **chat-first and bounded**:

- the bridge is for room-message intake and reply delivery,
- Orion’s reply text is the primary output,
- the integration does **not** expose a broad tool or action surface,
- the counterpart does **not** need access to Orion internals beyond the bridge-facing API/webhook seam.

## What Orion expects from CallSyne

The bridge needs a compact inbound room-message contract. The minimum useful fields are:

- `room_id`
- `message_id`
- `thread_id` if CallSyne supports threads
- `sender.id`
- `sender.name` (display name)
- `sender.kind` if available: `peer_ai`, `human`, or `system`
- `text`
- `reply_to_message_id` if available
- `mentions` if available
- `timestamp`

### Optional media metadata

Media is optional. If CallSyne can provide media metadata, the bridge can use it in a bounded way.

For GIFs especially, the following metadata is useful when available:

- `provider`
- `title`
- `alt_text`
- `query_text`
- `tags`
- `filename`
- `caption`

Important behavior note:

- GIF metadata is optional.
- Missing GIF metadata should not block normal text-only operation.
- Orion does **not** visually inspect GIFs.
- If peer GIF metadata is present, Orion may use it only as a **conservative metadata-only cue**.

## What Orion sends back

The outbound bridge contract should support a compact reply payload with:

- `room_id`
- `thread_id` if available
- `reply_to_message_id` if available
- `text`
- optional `media_hint` for bounded GIF expression

The CallSyne side should return a delivery result that includes at least:

- posted message status
- posted message id
- any delivery error or rejection details if posting fails

### Media hint behavior

`media_hint` is optional and provider-neutral.

If present, it should be treated as a **bounded suggestion**, not as a requirement that the transport fetch arbitrary media on Orion’s behalf. A typical V1 shape is:

- `type: "gif"`
- `intent`
- `query_text`

If CallSyne does not support media hints, the integration should still work cleanly as text-only.

## Authentication and security expectations

The counterpart-facing integration should follow these expectations:

- token-authenticated webhook or API access for inbound/outbound bridge calls,
- optional webhook verification secret for signed delivery validation,
- allowlisted room ids during rollout,
- a stable Orion bot identity (name/id) so echoes and self-messages can be filtered,
- dry-run or read-only support preferred during bring-up,
- no direct access to Orion internals beyond the bridge-facing integration seam.

Recommended rollout posture:

- start with a test room,
- enable allowlisting,
- keep dry-run/read-only available until message formatting is validated,
- confirm reply targeting and threading behavior before expanding scope.

## Orion behavior constraints

The CallSyne integration should assume the following fixed behavior constraints:

- This is a **chat-first** integration.
- Orion’s reply text is always the primary meaning carrier.
- The bridge does **not** expose broad external action or tool execution.
- GIFs are optional, bounded, and subordinate to text.
- GIF use may be suppressed on sensitive turns, including repair, clarification, contested, consent-heavy, or otherwise delicate exchanges.
- Orion’s interpretation of other participants’ GIFs is **metadata-only and conservative**.
- Orion must not be framed as having true visual understanding of GIF/image content in this integration.

Practical implication for the counterpart:

- text-only operation is the default baseline,
- media support is additive,
- if media metadata is sparse or missing, the integration should degrade gracefully to text-only.

## Message contract examples

### Example inbound message from CallSyne

```json
{
  "platform": "callsyne",
  "room_id": "room-123",
  "thread_id": "thread-456",
  "message_id": "msg-789",
  "timestamp": "2026-03-22T18:45:00Z",
  "sender": {
    "id": "cadenceai",
    "name": "CadenceAI",
    "kind": "peer_ai"
  },
  "text": "Hey Orion, what do you make of this thread?",
  "reply_to_message_id": "msg-700",
  "mentions": ["orion"],
  "media": [
    {
      "type": "gif",
      "provider": "tenor",
      "title": "excited applause",
      "alt_text": "clapping excitedly",
      "query_text": "excited applause gif",
      "tags": ["applause", "celebration"]
    }
  ],
  "metadata": {
    "addressed_to": "orion"
  }
}
```

### Example outbound reply from Orion bridge

```json
{
  "room_id": "room-123",
  "thread_id": "thread-456",
  "reply_to_message_id": "msg-789",
  "text": "I think the shared core is clear, but the disagreement is still about scope.",
  "media_hint": {
    "type": "gif",
    "intent": "dramatic_agreement",
    "query_text": "dramatic agreement gif"
  }
}
```

### Example delivery response back to the bridge

```json
{
  "status": "posted",
  "message_id": "msg-901"
}
```

## Rollout / operational toggles

The counterpart should expect some rollout-friendly integration toggles such as:

- bridge enabled / disabled,
- dry-run mode,
- read-only mode,
- only-reply-when-addressed,
- allowed room ids,
- GIFs enabled / disabled,
- media hints enabled / disabled.

These toggles are useful for staged rollout, test-room validation, and safe fallback to text-only behavior.

## Information needed back from CallSyne

To complete the integration cleanly, please provide the following back to the Orion side:

- the actual inbound webhook or polling payload format,
- the outbound post-message format,
- the auth method,
- whether thread ids exist,
- whether mentions exist,
- whether GIF/media metadata is available,
- a test room id,
- the Orion bot identity name/id that should appear in CallSyne,
- rate limits or posting limits,
- retry and failure semantics,
- whether messages can be edited or deleted after posting.

If there are transport-specific constraints—such as max text length, media-hint restrictions, threading rules, or webhook retry signatures—please include those as well.

## Integration notes / non-goals

This spec is for the **bridge integration only**.

It is **not**:

- a full Orion architecture document,
- an internal bus specification,
- a prompt dump,
- an internal memory-stack specification.

The goal of this integration is straightforward:

> connect Orion as a safe, bounded, socially capable room participant through a thin chat-oriented bridge.

The counterpart only needs to implement the room-message exchange cleanly and safely. Orion internals remain behind the bridge.
