## Summary

- Prime now refuses to take on a promise that a curiosity run (or consult, letter, game sitting) made. Only a promise made in a chat with Juniper has a chat thread to carry the reply back to.
- The refusal is shown to Prime before any act opens. It says why, and what to do instead: send a `curiosity_focus` question for the concern, or drop it.
- Prime's thinking frame no longer marks such a concern "you can take this on".
- The promise adapter gains an optional `take_on_refusal(concern)` hook. `orion/prime/intention.py` calls it in `take_on_refusal` and in the new `can_take_on`.

## Outcome moved

Failure mode: 2 of the 3 take-ons since 11-17 burned their make act on an order Hub had to refuse (`act_939dac6755330635f7c0b4a1` on 11-20, `act_aff9db0f35262af4ec65673d` on 11-23, both closed 0 "Hub refused the order: no chat history row"). That close was not Orion's choice. Now it cannot happen for a promise whose turn kind is known.

## Current architecture

Proven cause (live DB, read only):

- Curiosity run `1c162ffc7034` finished turn `ab1fc9b0-...` (grammar_acts ask `act_75cbd9a1...`). The turn made a promise ("Check me at 11-23 12:00Z").
- The hand-back (`orion/prime/handback.py` `fold_turn_signals`) opens a promise concern for every turn kind except carry turns, with `event_ref = turn:<corr>`. Prime's 11-22 07:48Z note: "a curiosity run turn (ab1fc9b0...) made a promise, now concern ba9b129bcdd06506".
- `intention.take_on_refusal` only checked "has an adapter" (`turn:` prefix), so the take-on passed and the make act opened.
- Hub's `turn_promise.adapt` joins `harness_turn_trace` to `chat_history_log`. Both turns have a trace row and 0 chat rows, so Hub refused. Turn `63628dbb...` (11-20) is the same story ("a curiosity run turn (63628dbb...) made a promise").
- The regression test builds the concern through the real hand-back fold and gets the live concern id `ba9b129bcdd06506`. On unfixed main the take-on passes (test fails); on this branch it is refused.

## Architecture touched

Prime only (pure code, runs in orion-mind; Hub imports the same modules but its behaviour is unchanged). No bus, schema, env or API change.

## Files changed

- `orion/prime/handback.py`: `PROMISE_CONCERN_LEAD` constant used to write the promise concern text; `promise_concern_who(text)` reads the turn kind back.
- `orion/prime/intention_sources/turn_promise.py`: `THREAD_WHO`, `take_on_refusal(concern)`.
- `orion/prime/intention.py`: `source_refusal`, `can_take_on`; `take_on_refusal` calls the adapter's check; docs for the optional hook.
- `orion/prime/thought.py`: the "you can take this on" marker uses `can_take_on` (one line).
- `orion/prime/tests/test_prime_intention.py`: 3 regression tests.
- `orion/prime/evals/run_intention_take_on_replay.py` + fixture: new `no_thread` check on the live 11-22 concern.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: Prime refuses `take_on` for a non-chat promise concern up front.
- Compatibility notes: concerns with older text (pre-11-12 "a thread I opened ...") are not read and pass as before; Hub still decides those.

## Env/config changes

- None. `.env_example` not touched; no sync needed.

## Tests run

```text
pytest orion/prime/tests -q                          349 passed
pytest services/orion-mind/tests -q                  444 passed
pytest services/orion-hub/tests/test_intention_order.py -q   12 passed
new curiosity-run test against unfixed main          1 failed (move passed) -> as expected
services/orion-mind tests/prime_replay_scenarios.py  golden unchanged
```

## Evals run

```text
python -m orion.prime.evals.run_intention_take_on_replay
frame, decision, adapter, request, no_adapter, no_thread, ages_out: all true; ok=true
```

## Docker/build/smoke checks

```text
Not run: pure Python change, no compose/env/dependency change. New import edge (turn_promise -> handback) checked for cycles in a fresh interpreter by review.
```

## Review findings fixed

- Finding: the turn kind is read back from concern prose; a later rewording would make saved concerns fail open (back to Hub's refusal).
  - Fix: not in this PR. A typed `turn_who` field belongs on `PrimeConcernV1` in `orion/prime/state.py`, which another agent is changing now (#3610 patch 2) and which needs a consumer-first migration. Filed #3785. The round-trip test pins every current `WHO_WORDS` entry both ways.
  - Evidence: `test_the_hand_back_text_names_every_kind_of_turn_and_only_chat_with_juniper_has_a_thread`.
- Finding (low): a concern Prime adds itself with a `turn:` event_ref would pass. Funded event ids do not use `turn:` today; Hub still refuses. No change.
- Finding (low): a chat_juniper turn with no saved chat row still passes here and is refused by Hub. Same as before; noted in risks.
- Nits: refusal suggests curiosity_focus also for game sittings (harmless); eval must run with `-m` (pre-existing).

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

Run from the clean primary main checkout after merge (the post-pull rebuild does this). Hub needs no rebuild for this change.

## Risks / concerns

- Severity: low. Concern: text parse fails open on a future rewording. Mitigation: #3785; round-trip test.
- Severity: low. Concern: a chat_juniper turn with no chat_history_log row can still burn an act. Mitigation: unchanged from before; Hub refuses with a reason.
- Issues opened: #3785 (follow-up), #3786 (verify-live). Issues referenced: #3772 (stays open until #3786 proves it live).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3787
