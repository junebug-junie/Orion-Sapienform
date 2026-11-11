# PR: office messages are written on the chat lane

## Summary

- The first live office message (11-06, 17:55 local) reached Juniper 5 minutes into her visit. Composing tried the agent lane first; that lane did not answer within its 210 s budget, and Hub then fell back to the chat lane.
- Juniper (2026-11-07): office messages go straight to the chat lane.
- `EndogenousOutreach.compose` gains `lane="chat"`: one attempt on the chat lane with the full `timeout_sec`, no agent-lane first try. The default (`agent_first`) is unchanged for every other caller (reading findings, self findings, the endogenous tick).
- `office_reach_out.py` passes `lane="chat"`.

## Outcome moved

- Before: the office message waited up to 210 s for the agent lane, then started over on chat. After: one chat-lane attempt. The expected end-to-end time is about one chat turn (UNVERIFIED until the next office visit).

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `compose(..., lane=)` and `_generate(..., lane=)`.
- `services/orion-hub/scripts/office_reach_out.py`: chat lane.
- `services/orion-hub/tests/test_office_reach_out.py`: the office composes on the chat lane (the test fails if the office goes back to the agent lane); a chat-lane compose makes one chat attempt with the full budget.
- `tests/test_reading_finding_outreach.py`, `test_self_finding_outreach.py`, `test_office_reach_out_pg.py`: their `_generate` fakes accept `lane`.

## Env/config changes

None.

## Tests run

```text
hub: test_office_reach_out + reading/self finding + endogenous_outreach + physiology_shadow_bid   357 passed
mutation: office lane back to agent_first -> test_office_message_is_composed_on_the_chat_lane fails
hub full suite: see PR checks (compared against main's known failures)
```

## Restart required

```text
No manual step: the post-pull rebuild rebuilds orion-hub.
```
