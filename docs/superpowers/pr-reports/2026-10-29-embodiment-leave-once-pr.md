## Summary

- Orion's body now sends one `leaveConversation` per dead conversation per abandon window (180 s), instead of one every 3 s tick.
- That flood froze AI Town on 2026-10-28. Leaves queued faster than the engine could process them. Every NPC action's result waited behind them and timed out, so no one spoke.
- Fixed the test fixtures behind 14 embodiment tests that were already failing on main.

## Outcome moved

Failure mode: town freeze from an input flood. Live evidence, 2026-10-28/29:
- `embodiment_dead_chat_abandoned convo=c:608342` was logged 19,541 times from 07:16 to 23:55 UTC.
- 1,266 of the newest 1,500 engine inputs were `leaveConversation` from p:29.
- The engine had processed up to input 846,016 while the newest was 930,137, a backlog of about 84,000. It drains at about 18 inputs/s.
- Convex logged `Timing out agentDoSomething` for every agent, and `Input … failed: Invalid conversation ID c:608342` for the stale leaves.
- No town line was recorded after 2026-10-24 16:58.

## Current architecture

`_maybe_abandon_dead_chat` asks `should_abandon_dead_chat`. Exit 1 is "last line older than the window". Once that is true it stays true on every tick, until the engine processes the leave and the chat leaves perception. Nothing stopped a resend.

## Architecture touched

`services/orion-embodiment/app/worker.py` only. No env, bus or schema change.

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
pytest services/orion-embodiment/tests -> 97 passed (main: 14 failed, 83 passed)
mutation: guard disabled -> test_dead_chat_sends_one_leave_per_abandon_window fails
git diff --check -> ok
```

## Evals run

None. The live check: after deploy, no repeated `embodiment_dead_chat_abandoned` lines for one conversation within 180 s, and the engine's input backlog stays near zero.

## Docker/build/smoke checks

Not deployed. The current backlog is draining by itself (embodiment stopped sending at 23:55).

## Review findings fixed

(filled after review)

## Restart required

After merge, from the clean primary main checkout on athena: `git pull --ff-only`. The post-pull rebuild covers orion-embodiment.

## Risks / concerns

- Severity: low. Concern: a leave the engine drops leaves Orion in the chat for one more window (180 s). Mitigation: it retries after the window.
- Severity: medium (follow-up). Concern: other per-tick inputs (`moveTo` at 125 of 1,500, `startConversation`) are not throttled. The engine has no backpressure, so any loop can build a queue. Mitigation: follow-up; leaves were the flood.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
