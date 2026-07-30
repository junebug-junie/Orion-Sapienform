# AI Town social-memory room scoping fix

Status: implemented (2026-07-30, Juniper: "give me your take and design" on the fragmentation
bug found immediately after deploying the social-memory read-back patch).

## Arsonist summary

`SocialParticipantContinuitySQL`'s composite key is `platform:room_id:participant_id`
(`services/orion-social-memory/app/models.py`). The conversation-memory patch populated
`room_id` with ai-town's `conversation_id` -- but ai-town mints a brand-new `conversation_id`
every time two characters start talking, with no persistent "thread" concept. Result, confirmed
live on athena's Postgres: Sofia Bell already has two disconnected continuity rows after one
afternoon (`c:280672`, `c:280580`), each with its own small `evidence_count`, instead of one
accumulating relationship.

The fix mirrors a pattern already live in this exact codebase: `services/orion-hub/scripts/
social_room.py:86-87` defines `HUB_DIRECT_ROOM_ID = "hub-direct"` -- a **fixed constant**, used
across every hub direct-chat interaction ever, proving `room_id` was always meant to mean "stable
venue," not "this specific exchange." `conversation_id` was the wrong value to put there.

`world_id` was considered and rejected: this repo has already used a full world wipe+reseed
(`testing:wipeAllTables` + `init`) as a real recovery path this same day. A world_id-scoped room
would reset every relationship to zero on the next such recovery. A fixed constant survives it.

## Current architecture

- `_publish_conversation_memory` (`services/orion-embodiment/app/worker.py`) builds
  `client_meta = {"external_room": {"platform": "aitown", "room_id": convo_id}, ...}`, used for
  both the `ChatHistoryTurnV1` (verbatim log) and `SocialRoomTurnV1` (relationship synthesis)
  publishes -- one shared `client_meta` for both, which is exactly the coupling causing the bug:
  the SAME `room_id` value serves two different purposes that need different scopes.
- `_fetch_participant_continuity` calls `orion-social-memory`'s `/summary?platform=aitown&room_id=
  {convo_id}&participant_id={partner_id}` -- same wrong value, read side.
- `ChatHistoryTurnV1.session_id` is separately set to `f"aitown:{convo_id}"` (not part of
  `client_meta`) -- this one is fine as-is: verbatim chat-log grouping genuinely should
  distinguish separate conversation instances; only the social-memory-facing `room_id` needs to
  change.
- Confirmed via hub's own code: `HUB_DIRECT_ROOM_PLATFORM = "hub"` / `HUB_DIRECT_ROOM_ID =
  "hub-direct"` are literal fixed strings, not derived from any per-session value.

## Missing questions

None outstanding -- direction (fixed constant, not world_id, no migration of existing orphaned
rows) confirmed with Juniper.

## Proposed schema / API changes

No schema changes -- `room_id` is already a plain string column/field everywhere in this pipeline.
Only the *value* passed changes, from `conversation_id` to a new fixed constant.

## Files likely to touch

- `services/orion-embodiment/app/worker.py`: new constant `SOCIAL_MEMORY_ROOM_ID = "aitown-town"`
  (mirrors `HUB_DIRECT_ROOM_ID`'s naming/placement convention); used in place of `convo_id` for
  `client_meta.external_room.room_id` in `_publish_conversation_memory`'s `SocialRoomTurnV1`
  construction, and in `_fetch_participant_continuity`'s `/summary` call. `ChatHistoryTurnV1`'s
  `session_id` (still `f"aitown:{convo_id}"`) and its own `client_meta` are untouched -- verbatim
  log grouping is a different, correctly-scoped concern.
- Tests: `services/orion-embodiment/tests/test_worker_conversation_memory.py` -- update assertions
  that currently expect `room_id == "conv1"` to expect the fixed constant instead; add a test
  proving two different `convo_id`s for the same partner both resolve to the same `room_id`.
- `services/orion-embodiment/README.md`: note the fixed room-id convention and why (mirrors hub's
  `hub-direct`, survives world wipes, not per-conversation).

## Non-goals

- Not migrating/merging the existing handful of orphaned per-conversation rows (Sofia x2, Nico x2,
  Tessa x1, all `evidence_count` 3-5) -- low value, self-heals under the new key going forward.
- Not changing `ChatHistoryTurnV1`/`chat_history_log`'s per-conversation `session_id` scoping --
  that's correctly conversation-scoped already, a different axis from relationship continuity.
- Not handling the hypothetical of a genuine second, separate ai-town deployment sharing the same
  `orion-social-memory` database -- not a real scenario today (one deployment, confirmed all
  session); would need a real per-deployment room id if that ever changes.

## Acceptance checks

- Two separate conversations with the same NPC (different `conversation_id`s) both write/read
  against the same `social_participant_continuity` row (`platform:aitown-town:{participant_id}`),
  with `evidence_count` accumulating across both rather than each conversation getting its own row.
- `chat_history_log`/verbatim recall is unaffected -- still grouped per conversation as before.
- Existing tests updated to assert the fixed room id, not `conversation_id`.

## Recommended next patch

Implement as scoped above in a single PR (this spec + the code), on
`fix/aitown-social-memory-room-scope` (worktree at
`/mnt/scripts/Orion-Sapienform-aitown-social-memory-room-scope`).
