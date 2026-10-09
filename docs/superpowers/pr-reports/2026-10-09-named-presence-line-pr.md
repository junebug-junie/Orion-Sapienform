## Summary

- The chat situation brief now names the person the camera matched: "Juniper has been in view for 7 minutes (matched by face)." instead of "Someone has been in view for 7 minutes." Unmatched presence still says "Someone".
- New caution when the room read is older than 60 s: "Your last visual read of the room is N min ago, not live. Do not say you see anyone or anything right now; say what you last saw and when."
- `presence_fragment` gains an optional `subject` arg; callers that pass none (endogenous_outreach) keep the exact old wording.

## Outcome moved

2026-10-09 01:59 UTC cockpit record (turn 219e7af9, `/api/chat/turn/<corr>/cockpit`): Juniper asked "can you see me?". Camera had matched her at 0.70 54 s earlier and presence was `present`, but the prompt said "Room (seen 7 min ago): Someone has been in view for 7 minutes..." and Orion answered "I see you, Juniper. Right here, right now." -- the name came from the chat, "right now" from nothing. Now the prompt carries the face-match evidence and a not-live warning.

## Current architecture

vision-window writes `substrate_embodied_presence` (`subject` set on probable/possible match) -> `orion.situational.context._build_perception_context` -> `presence_fragment` -> brief -> compact text -> chat prompt (cockpit `situation` hop records it).

## Files changed

- `orion/situational/perception_reader.py`: `presence_fragment(subject=)`
- `orion/situational/context.py`: pass `presence_subject`; stale-read caution (`_STALE_VISUAL_READ_SECONDS = 60`)
- `orion/situational/tests/test_presence_fragment_subject.py`, `test_hub_situation_perception.py`: 7 new tests

## Schema / bus / API / env

None. No env keys, no contract change.

## Tests run

`pytest orion/situational/tests` -> 128 passed; `services/orion-cortex-exec/tests/test_situation_perception_context.py` -> 56 passed; `services/orion-hub/tests/test_endogenous_outreach.py` -> 220 passed.

## Evals / live

No eval harness for prompt wording. Live check after deploy: stand in front of cam0, send a chat turn, read the `situation` hop at `/api/chat/turn/<corr>/cockpit` -> `raw.compact_text`.

## Restart required

`orion/` is baked into images; rebuild cortex-exec lanes and hub (commands in PR comment / chat).

## Risks / concerns

- Low: +1 caution line (~190 chars) on turns with a >60 s old scene; worst-case fits the 7200 cap test.
- A possible-level match (sim 0.35-0.55) is also named; wording says "matched by face", not "confirmed".
- Not addressed: scene description refresh rate; carbon-vs-cam0 stream_id attribution seen in a replay.
