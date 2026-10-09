## Summary

- **Drop an outreach when Juniper spoke while it was being composed.** Composing can take minutes, with the agent lane first and then a chat-lane fallback. A message from her saved after composing began means the conversation has moved on.
- **Recent-chat gate.** No background outreach within `HUB_ENDOGENOUS_OUTREACH_RECENT_CHAT_SEC` (900 s) of Juniper's last message. This is read from `chat_history_log` (Hub sources only), so a Hub restart can't reset it. Door-A (finished curiosity runs) stays exempt, as for the other schedule gates.
- **History lines carry their age** ("(2 days ago) …"), so old turns can't read as tonight.
- **New "Where Juniper is right now" block,** from the situation graph's projection (`orion:situation:latest`). It is the graph's first reader outside shadow, and shows:
  - her current whereabouts, or "nothing says she is away";
  - what she said recently, with its age;
  - away-facts that ended in the last week.

## Outcome moved

Live 2026-10-09. Hub restarted at 01:58:05. An outreach composed at 01:58:20; Juniper wrote 01:58:27–01:59:56; the outreach was sent at 02:04:07 from a prompt that never saw her message. It told her she was "working on the mesh from Chicago tonight": the history it saw was two days old and undated, and she had flown home on 10-07.

Replayed against live data:
- the drop check finds her 01:59:56 message after the 01:58:20 compose start, so that outreach would now be dropped;
- the situation block on the real state reads:

```text
Where Juniper is right now, from your running situation:
- Nothing on record says she is away from home right now.
- She said 2 days ago: Juniper told me she has an early wakeup at 4:30am CDT for a 7:30am flight home.
- No longer true (ended 36 hours ago): Juniper told me she is staying in Chicago for another night and will fly back to SLC at 7:30am.
Trust this over anything older in the conversation history below.
```

## Current architecture

Background outreach gated on in-memory state: a turn in flight (websocket busy), quiet hours, the daily cap and the cooldown. It re-checked only that in-memory gate before sending. "The last thing the two of you said" was the last 3 rows for the session, undated. It had no notion of where Juniper is now.

## Architecture touched

Hub only: `scripts/endogenous_outreach.py`, settings and env. It reads the existing `SituationStateV1` Redis projection from #2539. No bus or schema change.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`:
  - `_seconds_since_juniper_spoke` / `_juniper_spoke_since`: Hub-typed sources only, so dream-cycle and Collapse Mirror automation rows don't count;
  - the `recent_chat` gate, with its last reading cached so `status()` reports it;
  - the drop before delivery (`juniper_spoke_after_generation`);
  - the cheap gates are checked before the DB read;
  - `_age_phrase`, dated history;
  - `_read_situation` and `situation_lines`, which never show an expired whereabouts as current;
  - the `situation` grounding lane.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`: `HUB_ENDOGENOUS_OUTREACH_RECENT_CHAT_SEC=900`.
- `services/orion-hub/tests/test_endogenous_outreach.py`: 14 new tests.

## Schema / bus / API changes

- Added: none. It reads `orion:situation:latest` (`SituationStateV1`, from #2539).
- Behavior changed:
  - new block reasons `recent_chat` and `juniper_spoke_after_generation`;
  - the outreach prompt gains the situation block and dated history;
  - the decision record's grounding gains a `situation` lane.

## Env/config changes

- Added: `HUB_ENDOGENOUS_OUTREACH_RECENT_CHAT_SEC=900` (0 disables).
- Local `.env` synced (`sync_local_env_from_example.py orion-hub`): key added.

## Tests run

```text
services/orion-hub: pytest tests/*outreach*.py   316 passed
```

## Evals run

```text
Live replay of the 10-09 incident (above). No outreach eval harness exists for prompt quality.
```

## Docker/build/smoke checks

```text
Not built from this branch. Deploy Hub from the primary checkout on main after merge.
```

## Review findings fixed

- Finding: a whereabouts fact past its end date was still shown as current. The projection only writes on change, so a stale key is possible.
  - Fix: past `valid_until`, it renders as "No longer true".
  - Evidence: `test_whereabouts_past_its_end_is_never_shown_as_current`.
- Finding: the block ignored what Juniper said recently. On 10-09 the "flight home" fact lived only in `recent`.
  - Fix: `juniper_said` recent facts are listed with their age.
  - Evidence: `test_what_she_said_recently_is_listed_with_its_age`, plus the live replay.
- Finding: dream-cycle and Collapse Mirror automation rows counted as Juniper speaking.
  - Fix: only `hub_orion`, `hub_ws` and `hub_http` count.
- Finding: `status()` and `blocked_reason()` couldn't see `recent_chat`.
  - Fix: the last DB reading is cached and aged into the gate inputs.
  - Evidence: `test_status_reports_recent_chat_from_the_last_read`.
- Finding: the DB was read every 10 s tick even while blocked.
  - Fix: the cheap gates are checked first.
  - Evidence: `test_db_is_not_read_when_a_cheap_gate_already_blocks`.
- Not changed, needs Juniper:
  - Door-A (finished curiosity runs) has no "she spoke while composing" drop and is exempt from `recent_chat`, per her 09-22 schedule-gate ruling. The reviewer argues being mid-conversation is presence, not schedule.
- Accepted as-is:
  - a small race between a turn ending and its row landing;
  - a lagging Hub image silently drops the block (a strict schema; rebuild Hub with durable-runs).

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- **Medium: the situation key can go stale.**
  - Concern: the situation graph only writes on change. If durable-runs stops, the key holds its last state for up to 7 days.
  - Mitigation: expired end dates are never shown as current; a heartbeat is a follow-up.
- **Low: a short-message conversation can delay outreach.**
  - Concern: `recent_chat` delays outreach up to 15 min after each of Juniper's messages.
  - Mitigation: the cooldown is already 45 min, so this rarely binds.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
