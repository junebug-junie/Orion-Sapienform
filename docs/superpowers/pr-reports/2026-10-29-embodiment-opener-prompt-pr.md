## Summary

- When Orion opens a town conversation (no partner line yet), it now gets an opener prompt: a greeting plus one concrete thing, in one or two spoken sentences.
- Before, the reply prompt was used with "answer this latest line: (no partner line yet)", plus "If there is no partner line worth answering, reply with an empty string".
- The reply prompt also drops that empty-string escape. The worker's turn-taking gate already decides when Orion stays silent.

## Outcome moved

The root cause of the 2026-10-29 leaks:
- The model reasoned at length over whether to speak. Its reasoning was posted in `c:608777`, `c:609823` and `c:609988`.
- It also answered `[]` (`c:609091`, `c:609537`, `c:609823`, `c:609988`).

#2955/#2961 stop those from being posted. This removes the instruction that produced them.

## Current architecture

`orion/embodiment/speech.py` `build_speech_prompt` has one template for every turn. The worker calls it when the last message is from someone else, or when the conversation is empty and Orion has not opened it.

## Architecture touched

`orion/embodiment/speech.py` only.

## Files changed

- `orion/embodiment/speech.py`: opener branch; reply prompt without the empty-string escape; own lines matched by `author_id` when present.
- `orion/embodiment/tests/test_speech.py`: `test_empty_conversation_gets_an_opener_prompt_not_a_reply_prompt`, `test_reply_prompt_never_offers_an_empty_reply`.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: prompt wording only.
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
pytest orion/embodiment -> 104 passed
pytest services/orion-embodiment/tests -> 102 passed
```

## Evals run

None. The live check after deploy: Orion's opening lines in town are real greetings. No `embodiment_speech_empty_reply_skipped` on an opener, and no reasoning in cortex `reasoning_content` about whether to speak. UNVERIFIED until the town is unfrozen.

## Docker/build/smoke checks

Not deployed.

## Review findings fixed

- Finding (low): `latest_partner_line` skipped any message whose author was named "Orion", whatever its id. So a partner named Orion got an opener mid-conversation.
  - Fix: the id decides when it is present.
  - Evidence: `test_partner_named_orion_is_still_answered`.
- Noted (low): goodbye ping-pong is now bounded only by the partner leaving. The old decline path already failed by posting `[]`/reasoning. Watch live.
- Noted (low): `chat_quick.j2` frames the whole prompt as the partner's line ("what they just said"). The model may answer the opener instruction instead of greeting. Same framing as the reply path; watch live.
- Noted (low): the self-repeat retry nudge reads as a reply instruction, but it cannot reach an opener (empty transcript means no own lines).

## Restart required

After merge, from the clean primary main checkout on athena: `git pull --ff-only` (rebuilds orion-embodiment).

## Risks / concerns

- Severity: low. Concern: without the escape, the reply prompt always produces a line when called. Mitigation: the worker only calls it on Orion's turn, and the self-repeat, echo and `is_injectable` gates still apply.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2971

🤖 Generated with [Claude Code](https://claude.com/claude-code)
