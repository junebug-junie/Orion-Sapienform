## Summary

Orion's slow loop (Prime) can now see and keep a promise Orion made to Juniper in chat. Juniper approved this on 2026-11-13 ("ok fine").

- Prime's thinking frame now shows the promise sentence itself, with who it was made to and when. Example: `PROMISE: a promise I made to Juniper in chat at 2026-11-12 07:08Z: "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"`.
- The concern's own words now begin "a promise I made in a chat with Juniper". They used to begin "a thread I opened", and Prime read that as "a new chat thread opened". The words keep Juniper's opening line. They drop Orion's reply line, because a reply often opens with the promise itself.
- A promise concern now fades after 72 hours if Prime does not keep it. Every other concern still fades after 24 hours. The quiet floor (6 h) does not change.
- A promise concern holds the one turn-concern slot only for the first 24 hours after Prime last kept it, as long as it did before. So a promise that lives longer does not block the next promise.
- Prime's prompt has one new line: when a concern is a promise to look into something, Prime keeps it by asking a curiosity question, because the curiosity run does the looking.

## Outcome moved

The failure: on 11-12, chat turn `a129b72f-e698-425a-a435-3388cd7cd759` promised a "lazy browse through the last 24 hours". Prime opened concern `d8db9d7b1bba4cf7`. That concern did not hold the sentence. Thought `f93d7b7709d14ef5` read it as a new chat thread and chose "no move". The concern would then have faded at 24 h. Now Prime sees what was promised, sees how to keep it, and has 3 days to act.

## Current architecture

- `orion/prime/handback.py` folds the cockpit hops of each turn. When a turn made a promise, it opens a concern with `event_ref = turn:<correlation_id>`. The concern text held a ref to the promise and a gist of the prompt and reply, but never the quote.
- `orion/prime/state.py` `observe` faded every concern after `CONCERN_TTL` (24 h). `QUIET_FLOOR = CONCERN_TTL / 4`.
- `orion/prime/thought.py` builds the frame and `SYSTEM_PROMPT`.
- `orion/prime/block.py` renders `concern.text` into the Prime block, and that block goes into the prefix of every unified turn.

## Architecture touched

- `PrimeConcernV1` has a new optional field, `promise_text`. The saved state leaves it out when it is None, so older state saves byte for byte as before.
- Only the frame line for each concern (`thought.py` `_concern_line`) shows `promise_text`. The field stays out of `concern.text`. Because of that, the quote never reaches the Prime block, the published view `orion:prime:concerns`, `pending_changes` or `concern_record`. This keeps the choice-as-EFE acceptance check S-1.8 true: ordinary turns' motor prefix carries no promise quote.
- The concern kind comes from `event_ref` (`turn:` means a promise concern). There is no new kind field. Concerns saved before this patch get the right TTL with no migration. The one live concern, `d8db9d7b1bba4cf7`, gets 72 h.
- `PROMISE_CONCERN_TTL` (72 h) is a constant. No other concern TTL is set through env, so this is not an env key. A test pins it to the intention ledger's promise horizon (`turn_promise.HORIZON_SEC`).

## Files changed

- `orion/prime/state.py`: adds `promise_text`, `PROMISE_EVENT_PREFIX`, `PROMISE_CONCERN_TTL`, `is_promise_concern` and `concern_ttl`. `observe` now uses a TTL for each concern. The fade note and the track-record words name the TTL.
- `orion/prime/handback.py`: the signal carries the quotes (`promise_quotes`). `promise_text()` builds the labelled sentence. The concern words change from "thread I opened" to "promise I made", and they no longer hold the reply line. The pending-change line says "made a promise, now concern <id>". `holds_turn_slot()` limits the slot to 24 h.
- `orion/prime/evals/run_turn_handback_replay.py`: `bounded` counts only the concerns that hold the slot. A new check, `promise_ttl_costs_no_thread`, compares against a replay with the 24 h TTL. It fails if the 72 h life opens fewer threads.
- `orion/prime/thought.py`: the frame shows a `PROMISE:` line. The prompt fade sentence changes, and the new line on how to keep a promise is added. The take-on paragraph (only when take-on is on) says a promise not taken on ends 3 days after it was made.
- `orion/prime/acts.py`: the `CONCERN_BASIS` fade words and a comment.
- `orion/prime/tests/test_prime_promise_concerns.py`: new regression tests.
- `orion/prime/tests/test_prime_handback.py`, `orion/prime/tests/test_prime_intention.py`, `services/orion-mind/tests/test_prime_turn_handback.py`: these tests expected a turn concern to fade at 24 h. They now expect 72 h.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json` and `test_prime_replay.py`: the golden is re-recorded. 22 parts changed. After the four changed text pieces are mapped back to the old text, 0 parts differ (checked by script).

## Schema / bus / API changes

- Added: optional field `PrimeConcernV1.promise_text` (max 600 characters) in Redis `orion:prime:state`.
- Removed: none.
- Renamed: none.
- Behavior changed: a promise concern fades after 72 h, not 24 h. The words of a promise concern and of its wake line changed. The `CONCERN_BASIS` text changed.
- Compatibility notes: `PrimeConcernV1` is `extra="forbid"`. Hub and the harness governor read `orion:prime:state` (`orion/prime/block.py` `parse_prime_redis`, `orion/claude_tools/readers.py`). An old image cannot read a concern that has `promise_text`. That is the same rule as `event_family`: **rebuild orion-hub and orion-harness-governor before orion-mind**. To roll back mind, remove `promise_text` from `orion:prime:state` first. Old saved state loads and saves back byte for byte. This was checked against the live Redis state on 2026-11-13.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, because no template changed.
- Skipped keys that need operator action: none.

## Tests run

```text
pytest orion/prime orion/mindloop                                  252 passed
pytest services/orion-mind/tests                                   382 passed
pytest orion/orion_day/tests/test_letter_turn.py orion/autonomy/tests/test_state_store.py
       orion/claude_tools/tests/test_registry_and_readers.py orion/curiosity/tests/test_replica_*.py
       tests/scripts/test_schema_skew_discovery.py                 109 passed
pytest orion/dig/tests/test_refs_contract.py scripts/analysis/tests/test_curiosity_*_readout.py  32 passed
pytest services/orion-hub/tests/{test_curiosity_focus,test_turn_orchestrator_cockpit_hops,
       test_room_claude_full_context,test_intention_order}.py      103 passed
orion-static-gates.yml python steps (run locally)                  all rc=0
```

## Evals run

```text
orion/prime/evals/run_prime_frame_eval.py            ok
orion/prime/evals/run_prime_block_eval.py            ok
orion/prime/evals/run_prime_wake_replay.py           ok
python -m orion.prime.evals.run_intention_take_on_replay   "ok": true
python -m orion.prime.evals.run_turn_handback_replay --days 3 (live, 1,378 hops)  all 6 checks true;
    threads_opened 2 (same on the 24 h TTL baseline and on origin/main), threads_refused 5
    mutation: slot held for the full 72 h -> promise_ttl_costs_no_thread = false (opened 1 vs 2)
orion/curiosity/replica/evals/run_replica_frame_eval.py    ok
```

Live read-only check (2026-11-13):

- `harness_turn_trace` row `a129b72f-...` (created 2026-11-12 07:08:40Z) has `promises_made = [{"offer": false, "quote": "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"}]`. In the last 7 days, 17 of 366 turns with a `promises_made` array made at least one promise.
- I fed that real row (and its `chat_history_log` prompt) through the new fold. It gives concern id `d8db9d7b1bba4cf7`, the same id as the live concern. It gives the frame line `PROMISE: a promise I made to Juniper in chat at 2026-11-12 07:08Z: "While you're off in Hyrule I'll take a lazy browse through the last 24 hours"` and a TTL of 3 days. The Prime block does not contain the quote.
- The live `orion:prime:state` loads under the new model and round-trips byte-identical. It still holds `d8db9d7b1bba4cf7`, which now gets the 72 h TTL.

## Docker/build/smoke checks

```text
Not run. This is pure Python in orion/prime, and no dependency or compose change was made. Live behavior after deploy is UNVERIFIED
until the next chat promise opens a concern with promise_text and a Prime thought sees it.
```

## Review findings fixed

The review ran in a subagent and only read the code.

- Finding (material): the 72 h life also held the single turn-concern slot (`TURN_CONCERNS_MAX_OPEN = 1`) for 72 h. On the live 3-day replay, 1 thread opened instead of 2.
  - Fix: `holds_turn_slot()`. A turn concern holds the slot only while it is within 24 h of last being kept, which is exactly as long as it lived before. The replay eval has a new floor check against the 24 h baseline.
  - Evidence: the replay is back to 2 opened / 5 refused, the same as main. A mutation that holds the slot for 72 h fails the new check. New test: `test_an_unkept_promise_older_than_a_day_lives_on_but_does_not_block_the_next_promise`.
- Finding (minor): the reply line in `concern.text` can hold the promise when the reply opens with it, and that text goes into the Prime block (S-1.8). The old block test passed only because its fixture reply began "Have fun!".
  - Fix: a promise concern's text no longer holds the reply line. It keeps Juniper's opening line and the refs. The gist also left the pending-change line.
  - Evidence: `test_a_reply_that_opens_with_the_promise_still_keeps_the_quote_out_of_the_block`.
- Finding (minor): there is a version skew. An old Hub or governor cannot read state that has `promise_text`.
  - Fix: the restart order (Hub and governor before mind) is in this report. This is the same rule as `event_family`.
- Finding (nit): with take-on on, `age_out` ends an un-taken promise at 72 h even if Prime kept it.
  - Fix: the take-on paragraph now says this. The base prompt does not, because without take-on, keeping a promise does extend it.
- Finding (nit): `PROMISE_TO[who]` could raise a KeyError if a new kind of turn is added.
  - Fix: `test_every_kind_of_turn_has_promise_words` pins `PROMISE_TO`, `TurnWho` and `WHO_WORDS` together.
- Finding (nit): the intention test at 73 h no longer isolates `keep_taken_on`.
  - Not changed: the test calls only `keep_taken_on` and `observe`, not `age_out`. At 73 h it still shows that keeping stops the fade, against a plain concern that fades.

## Restart required

Deploy from the clean primary main checkout after merge. Do it in this order:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: low. Concern: Prime's own `last_reading` goes into the block. If Prime quotes the promise in its reading, those words reach the block. These are Prime's own words, not this path.
- Severity: low. Concern: an unkept promise concern takes one of the 3 `MAX_CONCERNS` slots for up to 3 days, not 1. Prime's own concerns have less room in that time. Mitigation: Prime can drop it, and the turn-concern slot itself still frees at 24 h.
- Severity: low. Concern: concerns opened before this deploy keep their old words ("a thread I opened") and have no `promise_text`. The live `d8db9d7b1bba4cf7` is one of them. Mitigation: they get the 72 h TTL. The next promise gets the new words.
- Issues opened/closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
