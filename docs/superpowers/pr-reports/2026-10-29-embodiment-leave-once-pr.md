## Summary

- Orion's body now sends one `leaveConversation` per dead conversation per abandon window (180 s), instead of one every 3 s tick. The mark is set before sending, so a timed-out send is not retried every tick.
- A pending invite is accepted at most once per 20 s, instead of every tick. The same loop shape, found in review.
- A failed `list_conversations` now skips the tick, like a failed `list_players`. Before, it read as "no conversation" and reset the resend guards.
- Fixed the test fixtures behind 14 embodiment tests that were already failing on main.
- Orion's body sends nothing while the town is frozen. Hub's embedded Freeze button calls `testing.stop`, which sets the world to `stoppedByDeveloper` and stops the engine. Inputs sent meanwhile queue unprocessed and replay after Unfreeze. Each tick now reads `engine.running`. A frozen tick produces no perception, so no chat, speech, accept or leave fires. `process_intent` refuses with "town frozen", which covers bus moves and idle wander.
- Root cause (Juniper, 2026-10-29): the town was frozen with that button from about 2026-10-24 16:58. Embodiment kept sending for about 3.5 days. Unfreeze then had to replay about 84k inputs at about 18/s before any NPC action could finish.

## Outcome moved

Failure mode: a stalled engine's queue fills with Orion's resends, and the backlog has to drain before the town can recover. Live evidence, 2026-10-28/29:
- `embodiment_dead_chat_abandoned convo=c:608342` was logged 19,541 times from 07:16 to 23:55 UTC.
- 1,266 of the newest 1,500 engine inputs were `leaveConversation` from p:29.
- The engine had processed up to input 846,016 while the newest was 930,137, a backlog of about 84,000. It drains at about 18 inputs/s.
- Convex logged `Timing out agentDoSomething` for every agent, and `Input … failed: Invalid conversation ID c:608342` for the stale leaves.
- No town line was recorded after 2026-10-24 16:58.

## Current architecture

`_maybe_abandon_dead_chat` asks `should_abandon_dead_chat`. Exit 1 is "last line older than the window". Once that is true it stays true on every tick, until the engine processes the leave and the chat leaves perception. Nothing stopped a resend.

## Architecture touched

`services/orion-embodiment/app/worker.py` and `orion/embodiment/aitown_client.py` (`engine_running`). No env, bus or schema change.

## Files changed

- `services/orion-embodiment/app/worker.py`: `_abandon_leave_sent_ms`; skip a resend inside the abandon window; cleared when there is no active conversation.
- `services/orion-embodiment/tests/test_worker_dead_chat_abandon.py`: 4 ticks with the chat still visible send 1 leave; after the window, a 2nd.
- `services/orion-embodiment/tests/test_worker_conversation_memory.py`, `tests/test_worker_social.py`: fixtures gain the worker's `_abandon_*` attributes (14 failures on main).
- `services/orion-embodiment/README.md`: abandon section.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: at most one leave per conversation per `EMBODIMENT_CONVERSATION_ABANDON_SEC`.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-embodiment/tests -> 101 passed; pytest orion/embodiment -> 100 passed
mutation: freeze gate removed -> test_frozen_town_gets_no_perception_and_no_intents fails (main: 14 failed, 83 passed)
mutation: guard disabled -> test_dead_chat_sends_one_leave_per_abandon_window fails
mutation: mark moved after the send -> test_failed_leave_is_not_resent_every_tick fails
git diff --check -> ok
```

## Evals run

None. The live check: after deploy, no repeated `embodiment_dead_chat_abandoned` lines for one conversation within 180 s, and the engine's input backlog stays near zero.

## Docker/build/smoke checks

Not deployed. The current backlog is draining by itself (embodiment stopped sending at 23:55).

## Review findings fixed

- Finding (medium-high): a leave that raised (client timeout under backlog, while the input still lands) was never recorded, so it was resent every tick.
  - Fix: mark before the send.
  - Evidence: `test_failed_leave_is_not_resent_every_tick`; it fails with the mark moved back after the send.
- Finding (medium): a failed `list_conversations` read as "no conversation" and cleared the guard.
  - Fix: skip the tick (return no perception).
  - Evidence: `test_failed_conversation_lookup_skips_the_tick`.
- Finding (medium): this report claimed the leaves froze the town. The numbers say the engine stalled first.
  - Fix: Summary and Outcome corrected; the stall is listed as open below.
  - Evidence: the numbers above.
- Finding (follow-up, fixed here): `acceptInvite` fired every tick while invited, which is the same loop shape.
  - Fix: one accept per conversation per 20 s (the engine's invite times out at 60 s).
  - Evidence: `test_pending_invite_is_accepted_once_per_window`.
- Noted (low): the guard is in memory, so a restart resends one leave. Bounded.

## Restart required

After merge, from the clean primary main checkout on athena: `git pull --ff-only`. The post-pull rebuild covers orion-embodiment.

## Risks / concerns

- Severity: low. Concern: one extra `worldState` query per 3 s tick for `engine.running`. Mitigation: it is the same query perception already makes twice.
- Severity: low. Concern: a leave the engine drops leaves Orion in the chat for one more window (180 s). Mitigation: it retries after the window.
- Severity: low (follow-up). Concern: other per-tick inputs (`moveTo` at 125 of 1,500, through the 6 s move cooldown) are bounded only by cooldowns. The engine has no backpressure, so any loop can build a queue. Mitigation: follow-up; leaves were the flood.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2943

🤖 Generated with [Claude Code](https://claude.com/claude-code)
